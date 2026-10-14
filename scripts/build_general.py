#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build Reef's "general decision" training corpus from public train/dev splits.

Every case is formatted like the request the Decision Index (0.2.1/0.3) harness sends for the
corresponding benchmark: same ``state`` layout, question id, instruction wording, criteria keys and
option texts. Only train/dev splits are used, labels are the published human labels (no model or
teacher is involved), and every training row whose normalised input text equals an input of the
benchmark's test split is dropped (see ``norm``).

The request wording, criteria and option-key rules are copied from the Decision Index harness
(MIT licence, https://github.com/apolinario/decision-index, ``decision_index/suite/build/``). The
home-appliance scenario generator and the GSM8K answer-group distractor method are adapted from
the same harness (``home_appliance.py``, ``gsm8k_v3.py``); both are credited inline.

Output (``--out``, default ``data/general``)::

    <source>.jsonl     one Reef case per line: id, workflow, state, questions, gold, teacher, meta
    MANIFEST.json      provenance, licence, split, counts, dedupe drops, mirrored benchmark

Gold probabilities are label-smoothed: the published label gets 0.92 and the remaining 0.08 is
spread uniformly over the other options.

Usage::

    .venv/bin/python scripts/build_general.py                 # everything
    .venv/bin/python scripts/build_general.py --only banking77 clinc150
    .venv/bin/python scripts/build_general.py --validate-only

Downloads are cached under ``<out>/.raw`` and in the Hugging Face cache; ``HF_TOKEN`` is used for
Hub downloads when set (read access only). Requires pyarrow, pandas, huggingface_hub and git.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

SEED = "reef-general-v1"
GOLD_P = 0.92  # probability mass on the published label; the rest is spread uniformly
DEFAULT_CAP = 8000
HARNESS = "github.com/apolinario/decision-index (MIT), decision_index/suite/build/"


# --------------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------------


def norm(text) -> str:
    """Normalised text used for contamination checks: NFKC, casefolded, punctuation removed,
    whitespace collapsed. Deliberately aggressive, so near-verbatim copies are caught too."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def prio(*parts) -> str:
    """Deterministic hash order used for every subsample."""
    return hashlib.sha256(":".join(map(str, (SEED,) + parts)).encode()).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def letter_keys(n: int) -> list[str]:
    """Option keys exactly as the harness' ``layout.choice_row``: A.. or option_i above 26."""
    return [chr(65 + i) if n <= 26 else f"option_{i}" for i in range(n)]


def smooth(keys, gold) -> dict[str, float]:
    assert gold in keys and len(keys) >= 2, (gold, keys)
    rest = (1.0 - GOLD_P) / (len(keys) - 1)
    return {k: GOLD_P if k == gold else rest for k in keys}


def allocate(cap: int, weights: dict, available: dict) -> dict:
    """Largest-remainder proportional allocation (as the harness' ESCI freeze), clamped to what
    each stratum has; any shortfall is redistributed over strata with spare rows."""
    quota = {k: 0 for k in weights}
    left = min(cap, sum(available.values()))
    while left > 0:
        open_ = {k: w for k, w in weights.items() if available[k] > quota[k] and w > 0}
        if not open_:
            break
        total = sum(open_.values())
        raw = {k: left * w / total for k, w in open_.items()}
        add = {k: min(int(v), available[k] - quota[k]) for k, v in raw.items()}
        rem = left - sum(add.values())
        for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), str(k))):
            if rem <= 0:
                break
            if available[k] > quota[k] + add[k]:
                add[k] += 1
                rem -= 1
        if not any(add.values()):
            break
        for k, v in add.items():
            quota[k] += v
        left = min(cap, sum(available.values())) - sum(quota.values())
    return quota


def stratified(items: list, cap: int, stratum, key, source: str, weights: dict | None = None) -> list:
    """Hash-ordered subsample of ``cap`` items, proportional over strata (by ``weights`` or by
    stratum size)."""
    buckets = defaultdict(list)
    for it in items:
        buckets[stratum(it)].append(it)
    for b in buckets.values():
        b.sort(key=lambda it: prio(source, key(it)))
    available = {k: len(v) for k, v in buckets.items()}
    w = {k: (weights or available).get(k, 0) for k in buckets}
    quota = allocate(cap, w, available)
    out = [it for k in sorted(buckets, key=str) for it in buckets[k][: quota[k]]]
    return sorted(out, key=lambda it: prio(source, key(it)))


def capped(items: list, cap: int, key, source: str) -> list:
    return sorted(items, key=lambda it: prio(source, key(it)))[:cap]


# --------------------------------------------------------------------------------------------
# Downloads (cached, pinned)
# --------------------------------------------------------------------------------------------


class Fetch:
    def __init__(self, raw: Path):
        self.raw = raw
        raw.mkdir(parents=True, exist_ok=True)

    def hf(self, repo: str, filename: str, revision: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                    token=os.environ.get("HF_TOKEN") or None))

    def url(self, url: str, name: str, sha256: str | None = None) -> Path:
        dest = self.raw / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            print(f"  downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            tmp.replace(dest)
        if sha256 and sha256_file(dest) != sha256:
            raise ValueError(f"{dest}: sha256 differs from pinned {sha256}")
        return dest

    def git_sparse(self, url: str, revision: str, paths: list[str], name: str) -> Path:
        dest = self.raw / name
        if not (dest / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *paths], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", revision], check=True)
        return dest


def gh_raw(repo: str, rev: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"


# --------------------------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------------------------


@dataclass
class Source:
    key: str
    benchmark: str  # Decision Index family whose request format is mirrored
    adapter: str  # harness file:function the format is copied from
    origin: str  # dataset id / URL
    revision: str
    licence: str
    split: str
    teacher: str
    cap: int = DEFAULT_CAP
    notes: list[str] = field(default_factory=list)


SOURCES: dict[str, Source] = {}
BUILDERS = {}


def source(**kw):
    def wrap(fn):
        s = Source(**kw)
        SOURCES[s.key] = s
        BUILDERS[s.key] = fn
        return fn

    return wrap


class Stats(dict):
    """Per-source counters written to the manifest."""

    def drop(self, reason: str, n: int = 1):
        self.setdefault("dropped_other", Counter())[reason] += n


def make_case(src: Source, sid, state, questions: dict, answers: dict, meta: dict | None = None) -> dict:
    gold = {q: {"type": questions[q]["type"], "probabilities": smooth(list(questions[q]["criteria"]), a), "label": a}
            for q, a in answers.items()}
    return {"id": f"{src.key}:{sid}", "workflow": src.key, "state": state, "questions": questions, "gold": gold,
            "teacher": f"human labels ({src.teacher})", "meta": meta or {}}


def choice_case(src: Source, sid, instructions: str, options: list[str], gold_index: int, meta=None) -> dict:
    """The harness' ``layout.choice_row`` request: empty state, one question ``q1``."""
    assert 2 <= len(options) <= 255 and 0 <= gold_index < len(options)
    keys = letter_keys(len(options))
    q = {"q1": {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, options))}}
    return make_case(src, sid, {}, q, {"q1": keys[gold_index]}, meta)


def dedupe_text(rows, text_of, test_norms: set, stats: Stats, within: bool = True, label_of=None):
    """Drop rows whose normalised text is in the benchmark test inputs, and (``within``) rows that
    repeat inside the training split (conflicting labels drop every copy)."""
    kept, seen, conflict = [], {}, set()
    dropped = 0
    for r in rows:
        t = norm(text_of(r))
        if t in test_norms:
            dropped += 1
            continue
        if within:
            if t in seen:
                if label_of and label_of(seen[t]) != label_of(r):
                    conflict.add(t)
                stats.drop("duplicate within train")
                continue
            seen[t] = r
        kept.append(r)
    if conflict:
        before = len(kept)
        kept = [r for r in kept if norm(text_of(r)) not in conflict]
        stats.drop("conflicting labels for a repeated train text", before - len(kept))
    stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + dropped
    return kept


# --------------------------------------------------------------------------------------------
# 1. BANKING77  (normalize_direct.banking77)
# --------------------------------------------------------------------------------------------

POLYAI = ("PolyAI-LDN/task-specific-datasets", "57ec275d8078af65b7731c2a98be812d844a6d6b")


@source(key="banking77", benchmark="BANKING77", adapter="normalize_direct.py:banking77",
        origin="github.com/PolyAI-LDN/task-specific-datasets banking_data/{train,test}.csv, categories.json",
        revision=POLYAI[1], licence="CC BY 4.0", split="train", teacher="BANKING77 train")
def build_banking77(src: Source, fetch: Fetch, stats: Stats):
    get = lambda p, sha=None: fetch.url(gh_raw(*POLYAI, f"banking_data/{p}"), f"banking77/{p}", sha)
    labels = json.loads(get("categories.json", "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63").read_text())
    read = lambda p: list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    test = read(get("test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"))
    train = read(get("train.csv"))
    stats["source_rows"] = len(train)
    rows = dedupe_text(train, lambda r: r["text"], {norm(r["text"]) for r in test}, stats, label_of=lambda r: r["category"])
    rows = [dict(r, i=i) for i, r in enumerate(rows)]
    rows = stratified(rows, src.cap, lambda r: r["category"], lambda r: r["text"], src.key)
    return [choice_case(src, prio(src.key, r["text"])[:12], "Classify the banking intent of this user request:\n" + r["text"],
                        labels, labels.index(r["category"]), {"label_name": r["category"]}) for r in rows]


# --------------------------------------------------------------------------------------------
# 2. CLINC150 + OOS  (normalize_direct.clinc150)
# --------------------------------------------------------------------------------------------

CLINC = ("clinc/oos-eval", "828f8093932c8fe6ca7936c3d2e52903b1c523de")


@source(key="clinc150", benchmark="CLINC150+OOS", adapter="normalize_direct.py:clinc150",
        origin="github.com/clinc/oos-eval data/data_oos_plus.json (train + oos_train); data_full.json for labels/test",
        revision=CLINC[1], licence="CC BY 3.0", split="plus train + oos_train", notes=['all 250 oos_train rows kept; in-scope rows subsampled proportionally by intent'],
        teacher="CLINC150 plus train")
def build_clinc150(src: Source, fetch: Fetch, stats: Stats):
    full = json.loads(fetch.url(gh_raw(*CLINC, "data/data_full.json"), "clinc150/data_full.json",
                                "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0").read_text())
    plus = json.loads(fetch.url(gh_raw(*CLINC, "data/data_oos_plus.json"), "clinc150/data_oos_plus.json").read_text())
    labels = sorted({pair[1] for pair in full["train"]} | {"oos"})
    assert len(labels) == 151
    descriptions = ["out of scope: none of the listed intents" if x == "oos" else x.replace("_", " ") for x in labels]
    test_norms = {norm(t) for split in ("test", "oos_test") for obj in (full, plus) for t, _ in obj[split]}
    train = [{"text": t, "label": y} for split in ("train", "oos_train") for t, y in plus[split]]
    stats["source_rows"] = len(train)
    rows = dedupe_text(train, lambda r: r["text"], test_norms, stats, label_of=lambda r: r["label"])
    # Keep every out-of-scope row (test is 18% OOS, train under 2%); in-scope rows fill the rest.
    oos = [r for r in rows if r["label"] == "oos"]
    ins = stratified([r for r in rows if r["label"] != "oos"], src.cap - len(oos), lambda r: r["label"], lambda r: r["text"], src.key)
    rows = sorted(oos + ins, key=lambda r: prio(src.key, r["text"]))
    instr = "Classify the intent of this user request, or choose out of scope if none applies:\n"
    return [choice_case(src, prio(src.key, r["text"])[:12], instr + r["text"], descriptions, labels.index(r["label"]),
                        {"label_name": r["label"]}) for r in rows]


# --------------------------------------------------------------------------------------------
# 3. Amazon ESCI  (normalize_text.esci, freeze.py locale x label strata)
# --------------------------------------------------------------------------------------------

ESCI_REV = "7916cdf6ab75a462e77f20ab40428a10923998d5"
ESCI_CRITERIA = {"E": "Exact: the product satisfies the search query.",
                 "S": "Substitute: a product that could substitute for the requested product.",
                 "C": "Complement: a product that complements the requested product.",
                 "I": "Irrelevant: the product does not address the requested product need."}
ESCI_INSTR = "Classify the relevance of this product to the search query using the ESCI categories."
ESCI_PRODUCT = ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]


@source(key="esci", benchmark="Amazon-ESCI", adapter="normalize_text.py:esci",
        origin="github.com/amazon-science/esci-data shopping_queries_dataset_{examples,products}.parquet (git LFS)",
        revision=ESCI_REV, licence="Apache-2.0", split="train", notes=['8,000 pairs allocated over locale x label proportionally to the test split, hash-ordered within strata'],
        teacher="Amazon ESCI train")
def build_esci(src: Source, fetch: Fetch, stats: Stats):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    lfs = lambda f: fetch.url(f"https://media.githubusercontent.com/media/amazon-science/esci-data/{ESCI_REV}/shopping_queries_dataset/{f}", f"esci/{f}")
    ex = pq.read_table(lfs("shopping_queries_dataset_examples.parquet"),
                       columns=["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "split"]).to_pandas()
    test, train = ex[ex.split == "test"], ex[ex.split == "train"]
    stats["source_rows"] = len(train)
    test_queries = {norm(q) for q in test["query"].unique()}
    weights = Counter(zip(test.product_locale, test.esci_label))
    bad = {q for q in train["query"].unique() if norm(q) in test_queries}
    kept = train[~train["query"].isin(bad)]
    stats["dropped_dedupe"] = len(train) - len(kept)
    stats.setdefault("notes", []).append("dedupe: rows dropped when the normalised query equals any test query")
    # Hash order inside each locale x label stratum; over-sample, join product text, cut to quota.
    kept = kept.assign(_p=[prio(src.key, e) for e in kept.example_id]).sort_values("_p")
    kept = kept.groupby(["product_locale", "esci_label"], sort=True).head(src.cap)
    kept = [r for r in kept.drop(columns=["_p", "split"]).to_dict("records")]
    del ex, train
    over = stratified(kept, int(src.cap * 1.15), lambda r: (r["product_locale"], r["esci_label"]), lambda r: r["example_id"], src.key, dict(weights))
    need = {f'{r["product_locale"]}:{r["product_id"]}' for r in over}
    products = {}
    pf = pq.ParquetFile(lfs("shopping_queries_dataset_products.parquet"))
    for batch in pf.iter_batches(batch_size=65536, columns=["product_locale", "product_id"] + ESCI_PRODUCT):
        keys = pc.binary_join_element_wise(batch.column("product_locale"), batch.column("product_id"), ":")
        hit = batch.filter(pc.is_in(keys, value_set=pa.array(sorted(need))))
        for row in hit.to_pylist():
            products.setdefault(f'{row["product_locale"]}:{row["product_id"]}', row)
    have = [r for r in over if (p := products.get(f'{r["product_locale"]}:{r["product_id"]}')) and p.get("product_title")]
    stats.drop("missing product text", len(over) - len(have))
    rows = stratified(have, src.cap, lambda r: (r["product_locale"], r["esci_label"]), lambda r: r["example_id"], src.key, dict(weights))
    out = []
    for r in rows:
        p = products[f'{r["product_locale"]}:{r["product_id"]}']
        product = {k.removeprefix("product_"): p[k] for k in ESCI_PRODUCT if p.get(k) is not None}
        q = {"answer": {"type": "choice", "instructions": ESCI_INSTR, "criteria": dict(ESCI_CRITERIA)}}
        out.append(make_case(src, r["example_id"], {"search_query": r["query"], "product": product}, q, {"answer": r["esci_label"]},
                             {"locale": r["product_locale"], "query_id": r["query_id"], "product_id": r["product_id"]}))
    stats["locale_label_counts"] = {f"{a}:{b}": n for (a, b), n in sorted(Counter((c["meta"]["locale"], c["gold"]["answer"]["label"]) for c in out).items())}
    return out


# --------------------------------------------------------------------------------------------
# 4. When2Call  (adapters_added.when2call)
# --------------------------------------------------------------------------------------------

W2C = ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53")
W2C_INSTR = "Which response should the assistant give to the user's question, given the available tools?"
W2C_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
W2C_KEY = dict(zip(W2C_ORDER, letter_keys(4)))  # A..D in the published order, as the benchmark


def w2c_category(text: str) -> str:
    """Category of a When2Call training response (the train files carry no category field)."""
    if "<TOOLCALL>" in text:
        return "tool_call"
    asks = "?" in text or re.search(r"\b(please provide|please specify|please tell|could you|can you provide|i('ll| will)? need|need to know)\b", text, re.I)
    refuses = re.search(r"\b(unable|can't|cannot|can not|not able|don't have (the )?(ability|access|capabilit)|i'm sorry|apologi[sz]e|apologies)\b", text, re.I)
    if refuses and not ("?" in text and not re.search(r"\b(unable|cannot|can't)\b", text, re.I)):
        return "cannot_answer"
    return "request_for_info" if asks else "direct"


def w2c_text(text: str) -> str:
    """Tool calls as the test MCQ writes them: a bare ``{"name": .., "arguments": ..}`` object."""
    m = re.fullmatch(r"\s*<TOOLCALL>(.*)</TOOLCALL>\s*", text, re.S)
    if not m:
        return text
    try:
        calls = json.loads(m.group(1))
    except json.JSONDecodeError:
        return m.group(1).strip()
    return json.dumps(calls[0] if isinstance(calls, list) and len(calls) == 1 else calls, ensure_ascii=False)


@source(key="when2call", benchmark="When2Call", adapter="adapters_added.py:when2call",
        origin="hf:nvidia/When2Call train/when2call_train_{sft,pref}.jsonl", revision=W2C[1], licence="CC BY 4.0",
        split="train_sft + train_pref", notes=["train files have no 4-way MCQ: options are the pair's chosen + rejected responses (2, rarely 3 of A..D); keys keep the benchmark's category letters (A direct, B tool_call, C request_for_info, D cannot_answer); categories inferred heuristically from response text; SFT-only questions (single response) unused"],
        teacher="When2Call train preference/SFT responses")
def build_when2call(src: Source, fetch: Fetch, stats: Stats):
    rd = lambda f: [json.loads(x) for x in fetch.hf(W2C[0], f, W2C[1]).open(encoding="utf-8")]
    sft, pref = rd("train/when2call_train_sft.jsonl"), rd("train/when2call_train_pref.jsonl")
    test_q = {norm(r["question"]) for r in rd("test/when2call_test_mcq.jsonl")}
    stats["source_rows"] = len(sft) + len(pref)
    groups = {}
    for kind, rows in (("pref", pref), ("sft", sft)):
        for i, r in enumerate(rows):
            q = r["messages"][-1]["content"] if kind == "pref" else r["messages"][0]["content"]
            k = (norm(q), json.dumps(r["tools"], sort_keys=True))
            g = groups.setdefault(k, {"question": q, "tools": r["tools"], "gold": set(), "options": {}, "ids": []})
            g["ids"].append(f"{kind}-{i}")
            good = r["chosen_response"]["content"] if kind == "pref" else r["messages"][-1]["content"]
            gc = w2c_category(good)
            g["gold"].add(gc)
            g["options"].setdefault(gc, w2c_text(good))
            if kind == "pref":
                bad = r["rejected_response"]["content"]
                g.setdefault("bad", {}).setdefault(w2c_category(bad), w2c_text(bad))
    out = []
    for k, g in groups.items():
        if k[0] in test_q:
            stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
            continue
        if len(g["gold"]) != 1:
            stats.drop("conflicting preferred categories")
            continue
        gc = next(iter(g["gold"]))
        opts = {gc: g["options"][gc]} | {c: t for c, t in g.get("bad", {}).items() if c != gc}
        if len(opts) < 2:
            stats.drop("fewer than two candidate responses (SFT-only or same-category pair)")
            continue
        criteria = {W2C_KEY[c]: opts[c] for c in W2C_ORDER if c in opts}
        state = {"tools": [json.loads(t) for t in g["tools"]], "question": g["question"]}
        out.append(make_case(src, g["ids"][0], state, {"q": {"type": "choice", "instructions": W2C_INSTR, "criteria": criteria}},
                             {"q": W2C_KEY[gc]}, {"track": gc, "answer_key_map": {W2C_KEY[c]: c for c in opts}}))
    out = capped(out, src.cap, lambda c: c["id"], src.key)
    stats["gold_tracks"] = dict(Counter(c["meta"]["track"] for c in out))
    stats["options_per_case"] = dict(Counter(len(c["questions"]["q"]["criteria"]) for c in out))
    return out


# --------------------------------------------------------------------------------------------
# 5. BFCL-style tool selection from ToolACE  (adapters_mechanical.build_bfcl)
# --------------------------------------------------------------------------------------------

TOOLACE = ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734")
BFCL = ("gorilla-llm/gorilla", "916260dfc116bf06793a1af79b4ec8195b0453b6")
BFCL_INSTR = ("Given the complete user conversation and the published function schemas in state, mark each "
              "candidate tool yes if it should be invoked to answer the request, or no otherwise. "
              "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n")
BFCL_CRITERIA = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}


def bfcl_key(name: str) -> str:
    return "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def toolace_tools(system: str):
    start = system.find("[{")
    if start < 0:
        return None
    try:
        tools, _ = json.JSONDecoder().raw_decode(system[start:])
    except json.JSONDecodeError:
        return None
    return [t for t in tools if isinstance(t, dict) and t.get("name")] if isinstance(tools, list) else None


@source(key="bfcl_toolace", benchmark="BFCL-tool-selection", adapter="adapters_mechanical.py:build_bfcl",
        origin="hf:Team-ACE/ToolACE data.json", revision=TOOLACE[1], licence="Apache-2.0",
        split="train (only split)", notes=["candidates = the row's own ToolACE tools plus 0-3 distractor tools from other rows; conversation is the first user turn; rows whose first assistant turn is not a tool call are skipped (BFCL test always has a gold tool)"],
        teacher="ToolACE reference tool calls", cap=6000)
def build_bfcl_toolace(src: Source, fetch: Fetch, stats: Stats):
    data = json.loads(fetch.hf(TOOLACE[0], "data.json", TOOLACE[1]).read_text())
    stats["source_rows"] = len(data)
    test_q = set()
    for cat in ("simple", "live_simple", "live_multiple"):
        p = fetch.url(gh_raw(*BFCL, f"berkeley-function-call-leaderboard/data/BFCL_v3_{cat}.json"), f"bfcl/BFCL_v3_{cat}.json")
        for line in p.open(encoding="utf-8"):
            for turn in json.loads(line)["question"]:
                test_q.update(norm(m["content"]) for m in turn if m.get("role") == "user")
    items, pool = [], {}
    for i, x in enumerate(data):
        tools = toolace_tools(x["system"])
        conv = x["conversations"]
        if not tools or len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            stats.drop("unparseable tools or conversation")
            continue
        for t in tools:
            pool.setdefault(t["name"], t)
        call = conv[1]["value"].strip()
        names = [t["name"] for t in tools if re.search(r"(^\[|[\[,]\s*)" + re.escape(t["name"]) + r"\(", call)] if call.startswith("[") else []
        if not names:
            stats.drop("first assistant turn is not a tool call")
            continue
        if norm(conv[0]["value"]) in test_q:
            stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
            continue
        items.append((i, x, tools, set(names)))
    names_sorted = sorted(pool)
    out = []
    for i, x, tools, needed in items:
        rng = random.Random(prio(src.key, i))
        own = {t["name"] for t in tools}
        k = rng.choice([0, 0, 1, 2, 3])
        extra = [pool[n] for n in rng.sample(names_sorted, k + len(own)) if n not in own][:k]
        cands = tools + extra
        rng.shuffle(cands)
        keys = [bfcl_key(t["name"]) for t in cands]
        if len(set(keys)) != len(keys) or "tool_" in keys:
            stats.drop("candidate tool keys collide")
            continue
        qs = {kk: {"type": "choice", "instructions": BFCL_INSTR + f"\n\nCandidate tool: {t['name']}", "criteria": dict(BFCL_CRITERIA)}
              for kk, t in zip(keys, cands)}
        ans = {kk: "yes" if t["name"] in needed else "no" for kk, t in zip(keys, cands)}
        state = {"conversation": [[{"role": "user", "content": x["conversations"][0]["value"]}]], "functions": cands,
                 "task": "BFCL tool-name selection"}
        out.append(make_case(src, i, state, qs, ans, {"gold_tools": sorted(needed), "distractors": [t["name"] for t in extra]}))
    out = capped(out, src.cap, lambda c: c["id"], src.key)
    stats["candidates_per_case"] = dict(sorted(Counter(len(c["questions"]) for c in out).items()))
    return out


# --------------------------------------------------------------------------------------------
# 6. API-Bank tool selection  (adapters_selection.apibank)
# --------------------------------------------------------------------------------------------

APIBANK = ("liminghao1630/API-Bank", "12e8158b7628c168f07e8f31fbbe3445e99f44cf")
DAMO = ("https://github.com/AlibabaResearch/DAMO-ConvAI.git", "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6")
APIBANK_INSTR = ("Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in state. "
                 "Do not infer arguments or use the current/future API result.")
APIBANK_CATALOG = 53  # size of the harness catalog (apis/*.py)


def apibank_call(text: str):
    """``Name(k='v', ...)`` -> (name, params); params {} when the arguments are not a literal."""
    m = re.match(r"\s*([A-Za-z_][\w.]*)\s*\(", text)
    if not m:
        return None, {}
    try:
        node = ast.parse(text.strip(), mode="eval").body
        return m.group(1), {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except (SyntaxError, ValueError, AttributeError, TypeError):
        return m.group(1), {}


def apibank_tool(name, description, inp, out) -> dict:
    return {"name": name, "description": description, "input_parameters": inp or {}, "output_parameters": out or {}}


def apibank_searcher_result(text: str):
    m = re.search(r"API: (?P<name>[^|]+?) \| Description: (?P<desc>.*?) \| Input parameters: (?P<inp>\{.*?\}) \| Output result: (?P<out>\{.*\})", text, re.S)
    if not m:
        return None
    lit = lambda s: (lambda v: v if isinstance(v, dict) else {})(_safe_literal(s))
    return apibank_tool(m.group("name").strip(), m.group("desc").strip(), lit(m.group("inp")), lit(m.group("out")))


def _safe_literal(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None


def apibank_parse(inp: str):
    """Split a training ``input`` into its API descriptions and a dialogue in the shape of the
    harness' lv1-lv2-samples turns (User/AI text turns, API turns with name, params and result)."""
    tools, turns = [], []
    for line in inp.split("\n"):
        if line.startswith('{"apiCode"'):
            d = json.loads(line)
            tools.append(apibank_tool(d["apiCode"], d.get("description", ""), d.get("parameters"), d.get("response")))
        elif line.startswith("Generate API Request"):
            break
        elif line.startswith("User: ") or line.startswith("AI: "):
            role, _, text = line.partition(": ")
            turns.append({"role": role, "text": text})
        elif line.startswith("API-Request: ["):
            call, _, result = line[len("API-Request: ["):].partition("]->")
            name, params = apibank_call(call)
            try:
                output = json.loads(result)
            except json.JSONDecodeError:
                output = result
            turns.append({"role": "API", "api_name": name, "param_dict": params,
                          "result": {"api_name": name, "input": params, "output": output, "exception": None}})
            if name == "ToolSearcher" and isinstance(output, str) and (t := apibank_searcher_result(output)):
                tools.append(t)
        elif turns and line.strip() and turns[-1].get("text") is not None:
            turns[-1]["text"] += "\n" + line
    return tools, turns


@source(key="apibank", benchmark="API-Bank-tool-selection", adapter="adapters_selection.py:apibank",
        origin="hf:liminghao1630/API-Bank training-data/lv{1,2}-api-train.json", revision=APIBANK[1],
        licence="MIT (HF dataset card)", split="training-data lv1 + lv2 (api)", notes=["training dialogues/APIs are synthetic (not the 53-tool harness catalog): catalog = the sample's APIs (lv2: revealed via ToolSearcher results) + distractor APIs from other samples, 53 tools total, sorted by name, option order shuffled; rows whose gold is ToolSearcher are skipped; catalog entries carry larger schemas than the harness catalog (~5.4k vs ~2.9k whitespace tokens)"],
        teacher="API-Bank reference API calls")
def build_apibank(src: Source, fetch: Fetch, stats: Stats):
    # Test dialogues: the harness samples (DAMO-ConvAI lv1-lv2-samples) plus the HF test-data copies.
    repo = fetch.git_sparse(DAMO[0], DAMO[1], ["api-bank/lv1-lv2-samples"], "damo-convai")
    test_users = set()
    for p in sorted((repo / "api-bank/lv1-lv2-samples").rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and (t := json.loads(line)).get("role") == "User":
                test_users.add(norm(t["text"]))
    for f in ("level-1-api.json", "level-2-api.json", "level-1-response.json", "level-2-response.json"):
        for x in json.loads(fetch.hf(APIBANK[0], f"test-data/{f}", APIBANK[1]).read_text()):
            test_users.update(norm(l[6:]) for l in str(x.get("input", "")).split("\n") if l.startswith("User: "))
    test_users = {u for u in test_users if len(u.split()) >= 4}  # "yes", "thank you" recur everywhere
    items, pool = [], {}
    n_rows = 0
    for level in ("lv1", "lv2"):
        for i, x in enumerate(json.loads(fetch.hf(APIBANK[0], f"training-data/{level}-api-train.json", APIBANK[1]).read_text())):
            n_rows += 1
            gold, _ = apibank_call(x["output"].removeprefix("API-Request: [").rstrip("]"))
            tools, turns = apibank_parse(x["input"])
            for t in tools:
                pool.setdefault(t["name"], t)
            if gold == "ToolSearcher":
                stats.drop("gold is ToolSearcher (never gold in the benchmark's level-1 samples)")
                continue
            own = {t["name"]: t for t in tools}
            if not gold or gold not in own or not turns:
                stats.drop("gold tool missing from the sample's descriptions or empty dialogue")
                continue
            if any(norm(t["text"]) in test_users for t in turns if t["role"] == "User"):
                stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
                continue
            items.append((f"{level}-{i}", turns, own, gold))
    stats["source_rows"] = n_rows
    names_sorted = sorted(pool)
    out = []
    for sid, turns, own, gold in items:
        rng = random.Random(prio(src.key, sid))
        extra = [n for n in rng.sample(names_sorted, min(len(names_sorted), APIBANK_CATALOG + len(own))) if n not in own]
        catalog = {**own, **{n: pool[n] for n in extra[: APIBANK_CATALOG - len(own)]}}
        names = sorted(catalog)
        order = list(range(len(names)))
        rng.shuffle(order)
        labels = {f"option_{i}": names[j] for i, j in enumerate(order)}
        goldkey = next(k for k, v in labels.items() if v == gold)
        state = {"dialogue": turns, "available_tools": [catalog[n] for n in names], "task": "API-Bank current API tool selection"}
        out.append(make_case(src, sid, state, {"tool": {"type": "choice", "instructions": APIBANK_INSTR, "criteria": labels}},
                             {"tool": goldkey}, {"gold_tool": gold}))
    return capped(out, src.cap, lambda c: c["id"], src.key)


# --------------------------------------------------------------------------------------------
# 7. Home appliance  (adapted from home_appliance.py, MIT, decision-index)
# --------------------------------------------------------------------------------------------
# The harness builds every household from its index alone (rooms by index % 4, power parity,
# outside temperature 12 + index % 12, ...), so new household names would reproduce the test
# states. We keep its scenario, resolution and question code verbatim and vary the households:
# room/device names, power states, levels, watts, clock, weather, policy. Rows whose state equals
# a harness test state are dropped.

HA_SEED = 2026101207  # harness: 2026091807
HA_HOUSEHOLDS = ["alder", "birch", "cedar", "dogwood", "elm", "fir", "garden", "harbor", "indigo", "juniper", "kestrel", "linden", "maple",
                 "northstar", "orchard", "prairie", "quartz", "redwood", "sequoia", "terrace", "upland", "violet", "willow", "zephyr"]
HA_ROOM_SETS = [["kitchen", "den", "main bedroom", "garage", "patio"], ["galley", "lounge", "guest room", "study", "porch"],
                ["kitchen", "family room", "nursery", "workshop", "deck"], ["breakfast nook", "sitting room", "bedroom", "office", "mudroom"]]
HA_SCENARIOS = ("exact_toggle", "room_except", "named_group", "duplicate_name", "already_satisfied", "range_reject", "atomic_block", "future_schedule",
                "conditional_rule", "power_filter", "context_pronoun", "missing_device", "relative_level", "room_alias_ambiguous", "last_wins",
                "partial_unavailable", "security_policy", "ordered_sequence", "group_exception", "status_query")
HA_ROOMS = ["kitchen", "galley", "kitchenette", "den", "lounge", "living room", "family room", "sitting room", "great room", "main bedroom",
            "guest room", "nursery", "bedroom", "kids room", "loft", "attic room", "garage", "study", "workshop", "office", "library",
            "laundry room", "utility room", "patio", "porch", "deck", "mudroom", "sunroom", "basement", "hallway", "dining room", "playroom"]
HA_LIGHTS = ["Pendant", "Window Lamp", "Reading Lamp", "Ceiling Spots", "Corner Light", "Floor Lamp", "Desk Lamp", "Wall Sconce",
             "Strip Light", "Night Light", "Chandelier", "Track Lights", "Globe Lamp", "Bedside Lamp", "Arc Lamp", "Lantern"]
HA_APPLIANCES = [("Kettle", 1200), ("Air Purifier", 55), ("Heated Throw", 110), ("Workbench Plug", 400), ("Patio Warmer", 1500),
                 ("Coffee Maker", 900), ("Space Heater", 1500), ("Fan", 45), ("Dehumidifier Plug", 300), ("Toaster", 850),
                 ("Humidifier", 30), ("Aquarium Pump", 12), ("Monitor", 35), ("Sewing Machine", 85), ("Slow Cooker", 220),
                 ("Bread Maker", 600), ("Ice Maker", 150), ("Printer", 60), ("Soldering Station", 75), ("Hot Tub Plug", 2000),
                 ("Phone Charger", 10), ("Mini Fridge", 90), ("Rice Cooker", 500), ("Electric Blanket", 100)]
HA_HOME_NAMES = ["aspen", "beacon", "brook", "canyon", "clover", "coral", "crest", "delta", "ember", "fern", "glade", "grove", "hazel",
                 "heather", "hollow", "iris", "ivy", "jasper", "lagoon", "laurel", "meadow", "mesa", "moss", "oak", "olive", "pebble",
                 "pine", "poplar", "quill", "raven", "reed", "ridge", "river", "rowan", "saffron", "sage", "shore", "sierra", "spruce",
                 "summit", "thistle", "timber", "tulip", "vale", "vista", "walnut", "wren", "yarrow", "yew", "zinnia"]


def _ha_canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ha_slug(text):
    return "_".join(text.lower().replace("'", "").split())


def ha_home_harness(index, rng=None):
    """Verbatim harness ``make_home`` (needed to reproduce the test states for dedupe)."""
    rooms = HA_ROOM_SETS[index % len(HA_ROOM_SETS)]
    light_names = ["Pendant", "Window Lamp", "Reading Lamp", "Ceiling Spots", "Corner Light"]
    devices = []
    for i, room in enumerate(rooms):
        devices.extend([
            {"id": f"{_ha_slug(room)}_light", "name": light_names[i], "room": room, "kind": "light", "available": True, "power": bool((index + i) % 2), "brightness": 20 + 15 * i, "watts": 8 + i * 3},
            {"id": f"{_ha_slug(room)}_plug", "name": ["Kettle", "Air Purifier", "Heated Throw", "Workbench Plug", "Patio Warmer"][i], "room": room, "kind": "appliance", "available": True, "power": bool((index + i + 1) % 2), "watts": [1200, 55, 110, 400, 1500][i]},
        ])
    devices[0]["name"] = "Reading Lamp"
    devices[4]["name"] = "Reading Lamp"
    devices.extend([
        {"id": f"{_ha_slug(rooms[0])}_island_light", "name": "Island Lights", "room": rooms[0], "kind": "light", "available": True, "power": bool(index % 2), "brightness": 65, "watts": 16},
        {"id": f"{_ha_slug(rooms[0])}_thermostat", "name": "Thermostat", "room": rooms[0], "kind": "thermostat", "available": True, "power": True, "temperature": 21, "min_temperature": 16, "max_temperature": 26, "watts": 3},
        {"id": f"{_ha_slug(rooms[1])}_speaker", "name": "Bookshelf Speaker", "room": rooms[1], "kind": "speaker", "available": True, "power": False, "volume": 35, "watts": 18},
        {"id": f"{_ha_slug(rooms[3])}_lock", "name": "Side Door", "room": rooms[3], "kind": "lock", "available": True, "locked": index % 2 == 0, "watts": 1},
    ])
    return {"home": HA_HOUSEHOLDS[index], "now": "2026-09-18T20:00:00-07:00", "outside_temperature": 12 + index % 12, "rooms": rooms, "room_aliases": {"back room": [rooms[2], rooms[3]]}, "devices": devices, "groups": {"evening": [devices[1]["id"], devices[3]["id"], devices[-2]["id"]], "downstairs": [d["id"] for d in devices if d["room"] in rooms[:2]]}, "device_context": devices[3]["id"], "policy": {"multi_action": "atomic", "conflict": "last_wins", "allow_partial": False, "security_mode": "away" if index % 3 == 0 else "home", "away_unlock": "blocked", "unavailable_action": "blocked"}, "schedules": [], "rules": []}


def ha_home_varied(name, rng):
    """Same device layout as the harness (the scenarios address devices by position), varied content."""
    rooms = rng.sample(HA_ROOMS, 5)
    lights = rng.sample(HA_LIGHTS, 5)
    apps = rng.sample(HA_APPLIANCES, 5)
    devices = []
    for i, room in enumerate(rooms):
        devices.extend([
            {"id": f"{_ha_slug(room)}_light", "name": lights[i], "room": room, "kind": "light", "available": True, "power": rng.random() < 0.5, "brightness": rng.randrange(10, 95, 5), "watts": rng.choice([5, 6, 8, 9, 10, 12, 14, 18])},
            {"id": f"{_ha_slug(room)}_plug", "name": apps[i][0], "room": room, "kind": "appliance", "available": True, "power": rng.random() < 0.5, "watts": apps[i][1]},
        ])
    devices[0]["name"] = devices[4]["name"]  # the duplicate-name scenario needs two lights with one name
    lo, hi = rng.randint(12, 17), rng.randint(25, 30)
    devices.extend([
        {"id": f"{_ha_slug(rooms[0])}_island_light", "name": rng.choice(["Island Lights", "Counter Lights", "Cabinet Lights", "Bar Lights"]), "room": rooms[0], "kind": "light", "available": True, "power": rng.random() < 0.5, "brightness": rng.randrange(20, 100, 5), "watts": rng.choice([12, 16, 20])},
        {"id": f"{_ha_slug(rooms[0])}_thermostat", "name": "Thermostat", "room": rooms[0], "kind": "thermostat", "available": True, "power": True, "temperature": rng.randint(lo + 2, hi - 2), "min_temperature": lo, "max_temperature": hi, "watts": 3},
        {"id": f"{_ha_slug(rooms[1])}_speaker", "name": rng.choice(["Bookshelf Speaker", "Smart Speaker", "Soundbar", "Radio"]), "room": rooms[1], "kind": "speaker", "available": True, "power": rng.random() < 0.5, "volume": rng.randrange(10, 80, 5), "watts": rng.choice([10, 18, 25, 40])},
        {"id": f"{_ha_slug(rooms[3])}_lock", "name": rng.choice(["Side Door", "Back Door", "Front Door", "Garage Door", "Gate"]), "room": rooms[3], "kind": "lock", "available": True, "locked": rng.random() < 0.5, "watts": 1},
    ])
    powered = [d["id"] for d in devices if "power" in d and d["kind"] in ("light", "appliance")]
    day, hour, minute = rng.randint(1, 28), rng.randint(6, 21), rng.choice([0, 10, 15, 20, 30, 40, 45, 50])
    now = f"2027-{rng.randint(1, 12):02d}-{day:02d}T{hour:02d}:{minute:02d}:00{rng.choice(['-08:00', '-07:00', '-05:00', '+00:00', '+01:00', '+02:00'])}"
    return {"home": name, "now": now, "outside_temperature": rng.randint(-8, 34), "rooms": rooms, "room_aliases": {"back room": [rooms[2], rooms[3]]}, "devices": devices, "groups": {"evening": [devices[1]["id"], devices[3]["id"], devices[-2]["id"]], "downstairs": [d["id"] for d in devices if d["room"] in rooms[:2]]}, "device_context": rng.choice(powered), "policy": {"multi_action": "atomic", "conflict": "last_wins", "allow_partial": False, "security_mode": rng.choice(["away", "home", "home"]), "away_unlock": "blocked", "unavailable_action": "blocked"}, "schedules": [], "rules": []}


def _ha_dev(home, did):
    return next(d for d in home["devices"] if d["id"] == did)


def ha_scenario(home, name, variant):
    """Harness ``scenario``; the only change is that the duplicate-name request names the
    household's duplicated light (always "Reading Lamp" in the harness households)."""
    ds, rooms = home["devices"], home["rooms"]
    l0, l1, l2 = ds[0], ds[2], ds[4]
    a0, a1, a2 = ds[1], ds[3], ds[5]
    therm, speaker, lock = ds[-3], ds[-2], ds[-1]
    if name == "exact_toggle":
        value = not a0["power"]
        req = f"Could you switch the {a0['name'].lower()} {'on' if value else 'off'}?" if variant % 2 else f"I need the {a0['name'].lower()} {'running' if value else 'shut down'}. "
        return {"kind": "execute", "ops": [[a0["id"], "power", value]]}, req.strip()
    if name == "room_except":
        targets = [d["id"] for d in ds if d["room"] == rooms[0] and d["kind"] == "light" and d["id"] != l0["id"]]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"Turn off every light in {rooms[0]} except the {l0['name']}."
    if name == "named_group":
        return {"kind": "execute", "ops": [[x, "power", True] for x in home["groups"]["evening"]]}, "Start the evening group, please."
    if name == "duplicate_name":
        return {"kind": "clarify", "candidates": [l0["id"], l2["id"]]}, "Turn on the Reading Lamp." if l0["name"] == "Reading Lamp" else f"Turn on the {l0['name']}."
    if name == "already_satisfied":
        return {"kind": "no_change", "ops": [[speaker["id"], "power", speaker["power"]]]}, f"Leave the {speaker['name'].lower()} {'on' if speaker['power'] else 'off'}."
    if name == "range_reject":
        value = therm["max_temperature"] + 4
        return {"kind": "unsupported", "ops": [[therm["id"], "temperature", value]], "reason": "outside supported range"}, f"Set the thermostat to {value}°C."
    if name == "atomic_block":
        a1["available"] = False
        return {"kind": "blocked", "ops": [[a0["id"], "power", True], [a1["id"], "power", True]], "reason": "atomic request contains unavailable device"}, f"Turn on both the {a0['name'].lower()} and the {a1['name'].lower()}."
    if name == "future_schedule":
        return {"kind": "schedule", "at": "22:15", "ops": [[a2["id"], "power", False]]}, f"At 10:15 tonight, turn off the {a2['name'].lower()}."
    if name == "conditional_rule":
        return {"kind": "rule", "condition": ["outside_temperature", "below", 15], "ops": [[therm["id"], "temperature", 22]]}, "Whenever it gets below 15°C outside, set the thermostat to 22°C."
    if name == "power_filter":
        targets = [d["id"] for d in ds if d["room"] == rooms[0] and d["kind"] == "appliance" and d["watts"] > 100]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"In {rooms[0]}, shut down appliances rated over 100 watts."
    if name == "context_pronoun":
        target = home["device_context"]
        return {"kind": "execute", "ops": [[target, "power", False]]}, "Turn it off too."
    if name == "missing_device":
        return {"kind": "clarify", "candidates": []}, f"Start the dehumidifier in {rooms[2]}."
    if name == "relative_level":
        value = min(100, l1["brightness"] + 20)
        return {"kind": "execute", "ops": [[l1["id"], "brightness", value]]}, f"Make the {l1['name'].lower()} 20 points brighter."
    if name == "room_alias_ambiguous":
        cands = [d["id"] for d in ds if d["room"] in home["room_aliases"]["back room"] and d["kind"] == "light"]
        return {"kind": "clarify", "candidates": cands}, "Switch off the light in the back room."
    if name == "last_wins":
        home["policy"]["multi_action"] = "ordered"
        return {"kind": "execute", "ops": [[a0["id"], "power", True], [a0["id"], "power", False]]}, f"Turn the {a0['name'].lower()} on, then turn it back off."
    if name == "partial_unavailable":
        a1["available"] = False
        home["policy"]["allow_partial"] = True
        home["policy"]["multi_action"] = "best_effort"
        return {"kind": "execute_partial", "ops": [[a0["id"], "power", True], [a1["id"], "power", True]]}, f"Switch on the {a0['name'].lower()} and the {a1['name'].lower()}. Do whatever is available."
    if name == "security_policy":
        home["policy"]["security_mode"] = "away"
        return {"kind": "blocked", "ops": [[lock["id"], "locked", False]], "reason": "away mode blocks unlock"}, f"Unlock the {lock['name'].lower()}."
    if name == "ordered_sequence":
        home["policy"]["multi_action"] = "ordered"
        return {"kind": "execute", "ops": [[l1["id"], "power", False], [l1["id"], "brightness", 40], [l1["id"], "power", True]]}, f"Switch off the {l1['name'].lower()}, set it to 40%, then switch it on."
    if name == "group_exception":
        targets = [x for x in home["groups"]["downstairs"] if x != a0["id"] and "power" in _ha_dev(home, x)]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"Power down the downstairs group, but keep the {a0['name'].lower()} as it is."
    return {"kind": "query", "target": therm["id"], "property": "temperature"}, "What's the thermostat set to right now?"


def ha_resolve(home, spec):
    """Verbatim harness ``resolve``."""
    final = copy.deepcopy(home)
    status = spec["kind"]
    changed = []
    if status in {"execute", "execute_partial"}:
        for did, prop, value in spec["ops"]:
            d = _ha_dev(final, did)
            if not d.get("available", True):
                if status == "execute_partial" and final["policy"]["allow_partial"]:
                    continue
                return {"status": "blocked", "targets": sorted({x[0] for x in spec["ops"]}), "changed": [], "final": home}
            old = d.get(prop)
            d[prop] = value
            if old != value:
                changed.append([did, prop, value])
        if not changed:
            status = "no_change"
    elif status == "schedule":
        final["schedules"].append({"at": spec["at"], "ops": spec["ops"]})
    elif status == "rule":
        final["rules"].append({"condition": spec["condition"], "ops": spec["ops"]})
    targets = sorted({x[0] for x in spec.get("ops", [])} | set(spec.get("candidates", [])) | ({spec["target"]} if "target" in spec else set()))
    return {"status": status, "targets": targets, "changed": changed, "final": final}


def ha_summarize(result):
    if result["status"] == "schedule":
        return "a future schedule is added; device properties do not change yet"
    if result["status"] == "rule":
        return "an automation rule is saved; device properties do not change immediately"
    if not result["changed"]:
        return "no device property changes"
    return "; ".join(f"{d}.{p} becomes {json.dumps(v)}" for d, p, v in result["changed"])


def ha_choices(rng, values, answer, namespace):
    values = list(dict.fromkeys(values))
    assert answer in values and len(values) >= 2
    rng.shuffle(values)
    criteria, gold = {}, None
    for i, value in enumerate(values):
        key = "k_" + hashlib.sha256(f"{namespace}:{i}:{rng.random()}".encode()).hexdigest()[:8]
        criteria[key] = value
        if value == answer:
            gold = key
    return criteria, gold


def ha_row(home, hidx, offset, seed, sidx, variant):
    """Harness ``make_row`` with the household passed in (questions and wording unchanged)."""
    rng = random.Random(f"home-benchmark:{seed}:{hidx}:{offset}")
    scenario_name = HA_SCENARIOS[sidx]
    spec, request = ha_scenario(home, scenario_name, variant)
    result = ha_resolve(home, spec)
    state = {k: v for k, v in home.items() if k != "home"}
    state["request"] = request
    qdefs = []
    statuses = ["execute now", "execute available parts now", "schedule for later", "save an automation rule", "ask a clarifying question", "make no change because already satisfied", "reject as unsupported", "block under household policy", "answer a state query"]
    status_desc = {"execute": "execute now", "execute_partial": "execute available parts now", "schedule": "schedule for later", "rule": "save an automation rule", "clarify": "ask a clarifying question", "no_change": "make no change because already satisfied", "unsupported": "reject as unsupported", "blocked": "block under household policy", "query": "answer a state query"}[result["status"]]
    qdefs.append(("resolution", "How should the assistant handle the request?", statuses, status_desc))
    executable_targets = set() if result["status"] == "clarify" else set(result["targets"])
    for d in home["devices"]:
        answer = "yes, this device is in the justified target set" if d["id"] in executable_targets else "no, this device is not in the justified target set"
        qdefs.append((f"target_member:{d['id']}", f"Is {d['id']} in the justified device target set for this request?", ["yes, this device is in the justified target set", "no, this device is not in the justified target set"], answer))
    outcome = ha_summarize(result)
    distract = ["no device property changes", "a future schedule is added; device properties do not change yet", "an automation rule is saved; device properties do not change immediately"]
    if result["targets"]:
        did = result["targets"][0]
        distract += [f"{did}.power becomes true", f"{did}.power becomes false"]
    qdefs.append(("outcome", "What is the concrete immediate outcome after applying the correct handling?", [outcome] + [x for x in distract if x != outcome], outcome))
    probe = home["devices"][(hidx + offset) % len(home["devices"])]
    prop = "power" if "power" in probe else "locked" if "locked" in probe else "temperature"
    pval = probe[prop]
    vals = [f"{probe['id']}.{prop} is {json.dumps(pval)}", f"{probe['id']}.{prop} is {json.dumps(not pval) if isinstance(pval, bool) else pval + 1}", "the property is not listed"]
    qdefs.append(("state_read", f"Before handling the request, what is the listed {prop} state of {probe['name']} in {probe['room']}?", vals, vals[0]))
    for n, (did, prop, value) in enumerate(spec.get("ops", [])[:2]):
        actual = _ha_dev(result["final"], did).get(prop)
        vals = [json.dumps(actual), json.dumps(value), json.dumps(_ha_dev(home, did).get(prop)), "not applicable or not listed"]
        qdefs.append((f"post_{n}", f"Immediately after correct handling, what is {prop} for {did}?", vals, json.dumps(actual)))
    questions, expected, roles = {}, {}, {}
    for pos, (role, instruction, vals, answer) in enumerate(qdefs):
        qid = "q_" + hashlib.sha256(f"qid:{seed}:{hidx}:{offset}:{pos}".encode()).hexdigest()[:10]
        criteria, gold = ha_choices(rng, vals, answer, qid)
        questions[qid] = {"type": "choice", "instructions": instruction, "criteria": criteria}
        expected[qid] = gold
        roles[qid] = role
    return state, questions, expected, {"scenario_family": scenario_name, "question_roles": roles, "oracle": {"status": result["status"], "targets": result["targets"]}}


@source(key="home_appliance", benchmark="Home-Appliance", adapter="home_appliance.py:make_row",
        origin="generated (scenario/question code adapted from decision-index home_appliance.py)", revision=f"seed {HA_SEED}",
        licence="MIT (generator adapted from decision-index); generated data", split="generated, households disjoint from the harness",
        notes=['households re-randomised (rooms, device names, states, watts, clock, weather, policy) because harness households are a function of index only; scenario/question code otherwise unchanged'],
        teacher="rule-based simulator oracle", cap=2000)
def build_home_appliance(src: Source, fetch: Fetch, stats: Stats):
    test_states = set()
    for h in range(8, len(HA_HOUSEHOLDS)):  # harness: hidx < 8 is dev, the rest is test
        for off in range(10):
            home = ha_home_harness(h)
            st, *_ = ha_row(home, h, off, 2026091807, (h * 7 + off) % len(HA_SCENARIOS), h + off)
            test_states.add(_ha_canon(st))
    out, seen = [], set()
    n_homes = src.cap // len(HA_SCENARIOS)
    names = [f"{a}-{b}" for a in HA_HOME_NAMES for b in ("east", "west", "north", "south")]
    stats["source_rows"] = n_homes * len(HA_SCENARIOS)
    for h in range(n_homes):
        name = names[h]
        for off, sidx in enumerate(range(len(HA_SCENARIOS))):
            home = ha_home_varied(name, random.Random(f"{SEED}:home:{HA_SEED}:{name}"))
            st, qs, ans, meta = ha_row(home, 100 + h, off, HA_SEED, sidx, h + off)
            c = _ha_canon(st)
            if c in test_states:
                stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
                continue
            if c in seen:
                stats.drop("duplicate generated state")
                continue
            seen.add(c)
            out.append(make_case(src, f"{name}:{off:02d}", st, qs, ans, {"household": name, **meta}))
    stats["test_states_checked"] = len(test_states)
    return out


# --------------------------------------------------------------------------------------------
# 8. Financial entity sentiment  (adapters_mechanical.build_finentity)
# --------------------------------------------------------------------------------------------

SENTFIN = ("tasksource/sentfin", "d6294b768091ae7198051294c7fdeee2fbfbd224")
TFNS = ("zeroshot/twitter-financial-news-sentiment", "ccbe24de388e287beb92dd393a335c376b350ac3")
FINENTITY = ("yixuantt/FinEntity", "3b6cedc5485b669c2ed168f1d949f517636eb7b8")
FIN_LABELS = ["Negative", "Neutral", "Positive"]


def finentity_question(start, end, value):
    return {"type": "choice", "instructions": ("Classify the sentiment toward the supplied financial entity span. "
                                               f"The entity span is given explicitly as {start}:{end} ({value}); "
                                               "do not extract or alter it. Choose exactly one of Negative, Neutral, or Positive."),
            "criteria": {x: x for x in FIN_LABELS}}


def finentity_norms(fetch: Fetch) -> set:
    """FinEntity documents, used only to drop overlapping training text (no FinEntity row is used)."""
    p = fetch.url(gh_raw(*FINENTITY, "data/FinEntity.json"), "finentity/FinEntity.json")
    return {norm(d["content"]) for d in json.loads(p.read_text(encoding="utf-8"))}


def finentity_case(src, sid, text, spans, meta):
    qs, ans = {}, {}
    for j, (start, end, label) in enumerate(spans):
        key = f"entity_{j:04d}_{start}_{end}"
        qs[key] = finentity_question(start, end, text[start:end])
        ans[key] = label
    return make_case(src, sid, {"document": text, "task": "FinEntity entity-given sentiment"}, qs, ans, meta)


@source(key="sentfin", benchmark="FinEntity-entity-given", adapter="adapters_mechanical.py:build_finentity",
        origin="hf:tasksource/sentfin (SEntFiN 1.0, pyRis/SEntFiN@eba4361)", revision=SENTFIN[1], licence="MIT",
        split="train + validation", notes=['entity spans located by first occurrence of the annotated entity string in the headline'],
        teacher="SEntFiN entity sentiment")
def build_sentfin(src: Source, fetch: Fetch, stats: Stats):
    import pandas as pd

    rd = lambda s: pd.read_parquet(fetch.hf(SENTFIN[0], f"data/{s}-00000-of-00001.parquet", SENTFIN[1])).to_dict("records")
    test_norms = {norm(r["sentence1"]) for r in rd("test")} | finentity_norms(fetch)
    docs = {}
    for split in ("train", "validation"):
        for r in rd(split):
            m = json.loads(r["metadata"])
            d = docs.setdefault((split, m["headline_id"]), {"text": r["sentence1"], "ents": []})
            d["ents"].append((r["sentence2"], r["labels"]))
    stats["source_rows"] = len(docs)
    out = []
    for (split, hid), d in docs.items():
        text = d["text"]
        if norm(text) in test_norms:
            stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
            continue
        spans, used = [], set()
        for ent, label in d["ents"]:
            start = text.find(ent)
            if start < 0:
                start = text.lower().find(ent.lower())
            if start < 0 or (start, start + len(ent)) in used:
                stats.drop("entity not found verbatim in the headline (entity skipped)")
                continue
            used.add((start, start + len(ent)))
            spans.append((start, start + len(ent), label.capitalize()))
        if not spans:
            stats.drop("headline without locatable entities")
            continue
        out.append(finentity_case(src, f"{split}-{hid}", text, spans, {"headline_id": hid, "split": split}))
    return capped(out, src.cap, lambda c: c["id"], src.key)


@source(key="twitter_fin", benchmark="FinEntity-entity-given", adapter="adapters_mechanical.py:build_finentity",
        origin="hf:zeroshot/twitter-financial-news-sentiment sent_train.csv", revision=TFNS[1], licence="MIT", split="train",
        notes=["tweet-level sentiment assigned to the tweet's single cashtag span (tweets with zero or several cashtags skipped)"],
        teacher="Twitter financial news sentiment (tweet-level label on its single cashtag)")
def build_twitter_fin(src: Source, fetch: Fetch, stats: Stats):
    rd = lambda f: list(csv.DictReader(fetch.hf(TFNS[0], f, TFNS[1]).open(newline="", encoding="utf-8")))
    test_norms = {norm(r["text"]) for r in rd("sent_valid.csv")} | finentity_norms(fetch)
    rows = rd("sent_train.csv")
    stats["source_rows"] = len(rows)
    label = {"0": "Negative", "1": "Positive", "2": "Neutral"}  # bearish, bullish, neutral
    out, seen = [], set()
    for i, r in enumerate(rows):
        text = r["text"]
        if norm(text) in test_norms:
            stats["dropped_dedupe"] = stats.get("dropped_dedupe", 0) + 1
            continue
        if norm(text) in seen:
            stats.drop("duplicate within train")
            continue
        seen.add(norm(text))
        tags = list(re.finditer(r"\$[A-Za-z]{1,6}(?:\.[A-Za-z]{1,2})?\b", text))
        if len(tags) != 1:
            stats.drop("not exactly one cashtag (entity undefined)")
            continue
        out.append(finentity_case(src, i, text, [(tags[0].start(), tags[0].end(), label[r["label"]])], {"source_label": r["label"]}))
    return capped(out, src.cap, lambda c: c["id"], src.key)


# --------------------------------------------------------------------------------------------
# 9. iSarcasmEval task A English  (normalize_text.isarcasm)
# --------------------------------------------------------------------------------------------

ISARCASM = ("iabufarha/iSarcasmEval", "dfc708b53bde1bb571abfb5692f63231c2232195")


@source(key="isarcasm_a_en", benchmark="iSarcasmEval-A-En", adapter="normalize_text.py:isarcasm",
        origin="github.com/iabufarha/iSarcasmEval train/train.En.csv", revision=ISARCASM[1], licence="MIT",
        split="train (English)", teacher="iSarcasmEval author labels")
def build_isarcasm(src: Source, fetch: Fetch, stats: Stats):
    rd = lambda p: list(csv.DictReader(fetch.url(gh_raw(*ISARCASM, p), f"isarcasm/{p}").open(encoding="utf-8-sig", newline="")))
    test_norms = set()
    for p in ("test/task_A_En_test.csv", "test/task_B_En_test.csv", "test/task_C_En_test.csv"):
        for r in rd(p):
            test_norms.update(norm(r[k]) for k in ("text", "text_0", "text_1") if r.get(k))
    rows = [r for r in rd("train/train.En.csv") if (r.get("tweet") or "").strip() and r["sarcastic"] in {"0", "1"}]
    stats["source_rows"] = len(rows)
    rows = dedupe_text(rows, lambda r: r["tweet"], test_norms, stats, label_of=lambda r: r["sarcastic"])
    q = lambda: {"sarcastic": {"type": "choice", "instructions": "Is this text intended to be sarcastic?", "criteria": {"no": "No", "yes": "Yes"}}}
    out = [make_case(src, r[""] or i, r["tweet"], q(), {"sarcastic": "yes" if r["sarcastic"] == "1" else "no"}) for i, r in enumerate(rows)]
    return capped(out, src.cap, lambda c: c["id"], src.key)


# --------------------------------------------------------------------------------------------
# 10. HellaSwag and WinoGrande  (normalize_direct.knowledge)
# --------------------------------------------------------------------------------------------

HELLASWAG = ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76")
WINOGRANDE = ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5")


def _parquet(fetch, repo, path):
    import pyarrow.parquet as pq

    return pq.read_table(fetch.hf(repo[0], path, repo[1])).to_pylist()


@source(key="hellaswag", benchmark="HellaSwag", adapter="normalize_direct.py:knowledge",
        origin="hf:Rowan/hellaswag data/train parquet", revision=HELLASWAG[1], licence="MIT", split="train",
        teacher="HellaSwag gold endings", cap=6000)
def build_hellaswag(src: Source, fetch: Fetch, stats: Stats):
    clean = lambda s: " ".join(str(s).split())
    test_norms = {norm(clean(r["ctx"])) for s in ("validation", "test") for r in _parquet(fetch, HELLASWAG, f"data/{s}-00000-of-00001.parquet")}
    rows = [dict(r, i=i) for i, r in enumerate(_parquet(fetch, HELLASWAG, "data/train-00000-of-00001.parquet"))]
    stats["source_rows"] = len(rows)
    rows = dedupe_text(rows, lambda r: clean(r["ctx"]), test_norms, stats, within=False)
    rows = capped(rows, src.cap, lambda r: r["i"], src.key)
    return [choice_case(src, f"{r['i']}:{r['ind']}", "Which continuation is most plausible?\n" + clean(r["ctx"]),
                        [clean(x) for x in r["endings"]], int(r["label"]), {"activity_label": r["activity_label"]}) for r in rows]


@source(key="winogrande", benchmark="WinoGrande", adapter="normalize_direct.py:knowledge",
        origin="hf:allenai/winogrande winogrande_xl/train parquet", revision=WINOGRANDE[1],
        licence="Apache-2.0 (github.com/allenai/winogrande LICENSE; the HF card states none)", split="winogrande_xl train",
        teacher="WinoGrande gold options", cap=6000)
def build_winogrande(src: Source, fetch: Fetch, stats: Stats):
    test_norms = {norm(r["sentence"]) for s in ("validation", "test") for r in _parquet(fetch, WINOGRANDE, f"winogrande_xl/{s}-00000-of-00001.parquet")}
    rows = [dict(r, i=i) for i, r in enumerate(_parquet(fetch, WINOGRANDE, "winogrande_xl/train-00000-of-00001.parquet"))]
    stats["source_rows"] = len(rows)
    rows = dedupe_text(rows, lambda r: r["sentence"], test_norms, stats, label_of=lambda r: (r["option1"], r["option2"], r["answer"]))
    rows = capped(rows, src.cap, lambda r: r["i"], src.key)
    return [choice_case(src, r["i"], "Which option correctly fills the blank?\n" + r["sentence"], [r["option1"], r["option2"]],
                        int(r["answer"]) - 1) for r in rows]


# --------------------------------------------------------------------------------------------
# 11. GSM8K multiple choice  (adapters_selection.gsm + gsm8k_v3 answer groups, MIT, decision-index)
# --------------------------------------------------------------------------------------------

GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
GSM_INSTR = "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K."
GSM_K, GSM_WINDOW = 10, 4


def gsm_gold(answer):
    return Decimal(answer.split("####")[-1].strip().replace(",", ""))


def gsm_num(n):
    return format(Decimal(n).normalize(), "f")


def _gsm_max_run(vals):
    best = run = 1
    for a, b in zip(vals, vals[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _gsm_bands(vals, k):
    out, s, n = [], 0, len(vals)
    while s < n:
        m = 1
        while s + k * m <= n and _gsm_max_run(vals[s:s + k * m]) > m:
            m += 1
        if s + k * m > n:
            while True:
                s0 = out.pop()[0] if out else 0
                m = (n - s0) // k
                if _gsm_max_run(vals[s0:n]) <= m:
                    out.append((s0, m))
                    return out
                if s0 == 0:
                    raise ValueError("a value occurs more than n/k times")
        out.append((s, m))
        s += k * m
    return out


def gsm_option_sets(golds: dict) -> dict:
    """The harness' answer-groups-v1 method (gsm8k_v3.py) applied to the training problems: each
    distractor is the gold of another training problem of similar size, never derived from the
    row's own gold; 4-choice sets are ring windows of the 10-choice set."""
    tiebreak = lambda i: prio("gsm8k:order", i)
    order = sorted(golds, key=lambda i: (golds[i], tiebreak(i)))
    r = len(order) % GSM_K
    left, extra = [], []
    if r:
        start = random.Random(prio("gsm8k:leftover")).randrange(len(order))
        scan = order[start:] + order[:start][::-1]
        seen = set()
        for i in scan:
            if golds[i] not in seen:
                left.append(i)
                seen.add(golds[i])
            if len(left) == r:
                break
        top, bottom = max(golds[i] for i in left), min(golds[i] for i in left)
        extra = (sorted({g for g in golds.values() if g > top}) + sorted({g for g in golds.values() if g < bottom}, reverse=True))[: GSM_K - r]
    rest = [i for i in order if i not in set(left)]
    vals = [golds[i] for i in rest]
    groups = [{"cases": [rest[s + t * m + j] for t in range(GSM_K)], "extra": []} for s, m in _gsm_bands(vals, GSM_K) for j in range(m)]
    if left:
        groups.append({"cases": sorted(left, key=lambda i: golds[i]), "extra": extra})
    sets = {}
    for n, g in enumerate(groups):
        values = [golds[i] for i in g["cases"]] + g["extra"]
        assert len(values) == GSM_K and len(set(values)) == GSM_K
        block = list(range(GSM_WINDOW))
        random.Random(prio("gsm8k:shifts", n // GSM_WINDOW)).shuffle(block)
        shift = block[n % GSM_WINDOW]
        ring = sorted(values)
        pos = {v: p for p, v in enumerate(ring)}
        for i in g["cases"]:
            four = [ring[(pos[golds[i]] - shift + t) % GSM_K] for t in range(GSM_WINDOW)]
            sets[i] = {10: values, 4: four, "group": f"G{n:04d}"}
    return sets


@source(key="gsm8k", benchmark="GSM8K-4choice / GSM8K-10choice", adapter="adapters_selection.py:gsm + gsm8k_v3.py",
        origin="hf:openai/gsm8k main/train parquet", revision=GSM8K[1], licence="MIT", split="main train",
        notes=['each training problem appears once, in one track (4- or 10-choice by hash); distractors are golds of other training problems (answer-groups-v1 method)'],
        teacher="GSM8K reference answers", cap=5000)
def build_gsm8k(src: Source, fetch: Fetch, stats: Stats):
    test_norms = {norm(r["question"]) for r in _parquet(fetch, GSM8K, "main/test-00000-of-00001.parquet")}
    rows = [dict(r, i=i) for i, r in enumerate(_parquet(fetch, GSM8K, "main/train-00000-of-00001.parquet"))]
    stats["source_rows"] = len(rows)
    rows = dedupe_text(rows, lambda r: r["question"], test_norms, stats)
    golds = {r["i"]: gsm_gold(r["answer"]) for r in rows}
    sets = gsm_option_sets(golds)
    out = []
    for r in capped(rows, src.cap, lambda r: r["i"], src.key):
        i = r["i"]
        count = 4 if int(prio(src.key, "track", i), 16) % 2 else 10
        opts = [gsm_num(v) for v in sets[i][count]]
        random.Random(prio(src.key, "GSM", i, count)).shuffle(opts)
        correct = opts.index(gsm_num(golds[i]))
        state = {"question": r["question"], "task": f"GSM8K deterministic {count}-choice numeric selection"}
        q = {"answer": {"type": "choice", "instructions": GSM_INSTR, "criteria": {f"option_{j}": v for j, v in enumerate(opts)}}}
        out.append(make_case(src, f"{count}:{i}", state, q, {"answer": f"option_{correct}"},
                             {"gold_numeric": gsm_num(golds[i]), "answer_group": sets[i]["group"], "track": f"GSM8K-{count}"}))
    stats["tracks"] = dict(Counter(c["meta"]["track"] for c in out))
    return out


# --------------------------------------------------------------------------------------------
# Validation and manifest
# --------------------------------------------------------------------------------------------


def validate(path: Path) -> dict:
    """Load with Reef's own loaders and check every question and gold distribution."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from reef.data import load_source
    from reef.packing import normalize_questions

    cases = load_source(str(path))
    ids = set()
    n_q = state_tok = instr_tok = max_opts = 0
    for c in cases:
        assert c.id not in ids, f"duplicate id {c.id}"
        ids.add(c.id)
        qs = normalize_questions(c.questions)
        assert set(c.gold) == set(c.questions), c.id
        for q in qs:
            g = c.gold[q.id]
            keys = [o.id for o in q.options]
            assert set(g["probabilities"]) == set(keys), (c.id, q.id)
            assert abs(sum(g["probabilities"].values()) - 1.0) < 1e-6, (c.id, q.id)
            assert g["label"] in keys and g["label"] == max(g["probabilities"], key=g["probabilities"].get), (c.id, q.id)
            instr_tok += len(q.instructions.split())
            max_opts = max(max_opts, len(keys))
            n_q += 1
        state_tok += len((c.state if isinstance(c.state, str) else json.dumps(c.state, ensure_ascii=False)).split())
    return {"cases": len(cases), "questions": n_q, "avg_state_tokens": round(state_tok / max(1, len(cases)), 1),
            "avg_instruction_tokens": round(instr_tok / max(1, n_q), 1), "max_options": max_opts}


def write_jsonl(path: Path, rows: list[dict]):
    # ASCII-escaped like the harness' dump_ascii; also required because reef's load_jsonl splits
    # on str.splitlines(), which breaks raw U+2028/U+0085 characters found in product text.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/general", type=Path)
    ap.add_argument("--raw", default=None, type=Path, help="download cache (default <out>/.raw)")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="build only these sources")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fetch = Fetch(args.raw or args.out / ".raw")
    manifest_path = args.out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {e["source"]: e for e in manifest.get("sources", [])}
    for key in args.only or sorted(SOURCES):
        src = SOURCES[key]
        path = args.out / f"{key}.jsonl"
        if not args.validate_only:
            print(f"[{key}] building", file=sys.stderr)
            stats = Stats()
            cases = BUILDERS[key](src, fetch, stats)
            assert len(cases) <= src.cap
            write_jsonl(path, cases)
            other = stats.pop("dropped_other", Counter())
            entries[key] = {
                "source": key, "file": path.name, "origin": src.origin, "revision": src.revision, "licence": src.licence,
                "split_used": src.split, "mirrored_benchmark": src.benchmark, "harness_adapter": src.adapter,
                "cap": src.cap, "n_cases": len(cases), "n_source_rows": stats.pop("source_rows", None),
                "n_dropped_dedupe": stats.pop("dropped_dedupe", 0), "n_dropped_other": dict(other),
                "notes": src.notes + stats.pop("notes", []), "details": dict(stats),
            }
        v = validate(path)
        entries.setdefault(key, {"source": key}).update(validation=v, sha256=sha256_file(path))
        print(f"[{key}] {json.dumps(v)}", file=sys.stderr)
    manifest = {
        "name": "reef-general-decisions", "seed": SEED, "gold_smoothing": {"gold": GOLD_P, "others": "uniform share of the rest"},
        "format_credit": f"request wording, criteria and option keys copied from {HARNESS}",
        "contamination_rule": "train/dev splits only; rows whose normalised input text (NFKC, casefold, punctuation stripped, "
                              "whitespace collapsed) equals a test input of the same benchmark source are dropped",
        "builder": "scripts/build_general.py",
        "sources": [entries[k] for k in sorted(entries)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
