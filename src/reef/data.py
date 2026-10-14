"""Loading LocalLLaMA/typed-decisions and turning cases into training windows."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reef.packing import Packer, Question, normalize_questions, subquestion
from reef.text import render_state

DATASET = "LocalLLaMA/typed-decisions"
MAX_QUESTIONS = 8  # per window
MAX_OPTIONS = 8  # per question (see reef.packing.MAX_PART_OPTIONS)
HOIST_AUGMENT = 0.1  # share of single-question training cases whose instructions move to the state


@dataclass
class Case:
    id: str
    workflow: str
    state: str
    questions: dict
    gold: dict  # qid -> {"probabilities": {...}, "label": ..., "type": ...}


def load_cases(split: str, config: str = "all", path: str | None = None, dataset: str = DATASET,
               decode_state: bool = False) -> list[Case]:
    """Load a split of typed-decisions (or another dataset in the same parquet layout) from the Hub
    (cached) or from a local parquet file. ``decode_state`` parses JSON-encoded states back into
    objects (Reef datasets store them as strings so packs with different shapes share one schema)."""
    import pandas as pd

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(dataset, f"{config}/{split}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(Path(path))
    return [
        Case(r.id, r.workflow, json.loads(r.state) if decode_state else r.state, json.loads(r.questions),
             json.loads(r.gold))
        for r in df.itertuples()
    ]


def _maybe_json(v):
    return json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v


def load_jsonl(path: str | Path) -> list[Case]:
    """Load cases written by ``reef label`` (one JSON object per line, same fields as typed-decisions)."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            r = json.loads(ln)
            out.append(Case(r["id"], r["workflow"], r["state"], _maybe_json(r["questions"]), _maybe_json(r["gold"])))
    return out


OPEN_JEV = "ZefanCai/Open-Jev"
# Its card notes that this source's question text comes from TypeSafe docs without a verified licence.
OPEN_JEV_EXCLUDE = ("customer-control-v1",)


def _parse_query(s: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in s.split("&") if "=" in kv) if s else {}


def _strip_preamble(questions: list[str]) -> dict[str, str]:
    """Open-Jev workflow questions start with a long policy preamble shared by every question of a
    source, and end with the actual question. Reef reads only the first ~40 instruction tokens, so
    drop paragraphs that most questions share and put the last paragraph (the question) first."""
    from collections import Counter

    paras = {q: [p.strip() for p in q.split("\n\n") if p.strip()] for q in set(questions)}
    common = Counter(p for ps in paras.values() for p in set(ps))
    n = len(paras)
    out = {}
    for q, ps in paras.items():
        keep = [p for p in ps if n < 4 or common[p] <= n / 2] or ps[-1:]
        if len(keep) > 1:
            keep = keep[-1:] + keep[:-1]
        out[q] = " ".join(" ".join(keep).split())
    return out


def load_open_jev(split: str, config: str = "release-v2-redistributable", exclude=OPEN_JEV_EXCLUDE,
                  cap: int | None = None, caps: dict[str, int] | None = None, max_questions: int = 4,
                  seed: int = 0, path: str | None = None) -> list[Case]:
    """Load `ZefanCai/Open-Jev <https://huggingface.co/datasets/ZefanCai/Open-Jev>`_ (CC0) as Reef cases.

    Open-Jev has one typed question per row; rows with the same state become one case with up to
    ``max_questions`` questions. Questions with more options than the model has slots are skipped.
    ``cap`` / ``caps`` limit rows per source (e.g. to keep the game sources from dominating)."""
    import pandas as pd

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(OPEN_JEV, f"data/{config}/{split}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(Path(path))
    df = df[~df.source.isin(list(exclude or ()))]
    df = df[df.options.apply(len) <= MAX_OPTIONS]
    rng = random.Random(seed)
    keep = []
    for src, d in df.groupby("source", sort=True):
        limit = (caps or {}).get(src, cap)
        idx = list(d.index)
        if limit is not None and len(idx) > limit:
            # cap by whole states so a case keeps all its questions together
            states = list(dict.fromkeys(d.state_json))
            rng.shuffle(states)
            chosen, total = set(), 0
            counts = d.state_json.value_counts()
            for s in states:
                if total >= limit:
                    break
                chosen.add(s)
                total += counts[s]
            idx = list(d.index[d.state_json.isin(chosen)])
        keep += idx
    df = df.loc[sorted(keep)]
    instr = {}
    for _src, d in df.groupby("source"):
        instr.update(_strip_preamble(list(d.question)))

    cases = []
    for (src, state_json), d in df.groupby(["source", "state_json"], sort=False):
        rows = list(d.itertuples())
        for start in range(0, len(rows), max_questions):
            questions, gold = {}, {}
            for i, r in enumerate(rows[start:start + max_questions]):
                qid = f"q{i}"
                opts, target = [str(o) for o in r.options], [float(t) for t in r.target]
                if r.kind == "noul":
                    lower = [o.strip().lower() for o in opts]
                    yes = next((j for j, o in enumerate(lower) if o in ("yes", "true")), 1)
                    p_yes = target[yes] / (sum(target) or 1.0)
                    questions[qid] = {"type": "noul", "instructions": instr[r.question]}
                    probs = {"true": p_yes, "false": 1.0 - p_yes}
                elif r.kind == "choice":
                    crit, ids = {}, []
                    for j, o in enumerate(opts):
                        k, sep, v = o.partition(": ")
                        k = k.strip() if sep else o.strip()
                        while k in crit:  # keep ids unique
                            k = f"{k}_{j}"
                        crit[k] = v.strip() if sep else ""
                        ids.append(k)
                    questions[qid] = {"type": "choice", "instructions": instr[r.question], "criteria": crit}
                    probs = dict(zip(ids, target))
                else:  # score: options are the ordered rubric
                    questions[qid] = {"type": "score", "instructions": instr[r.question], "criteria": opts}
                    probs = {str(j): t for j, t in enumerate(target)}
                s = sum(probs.values()) or 1.0
                probs = {k: v / s for k, v in probs.items()}
                gold[qid] = {"type": r.kind, "probabilities": probs, "label": max(probs, key=probs.get)}
            state = json.loads(state_json)
            cases.append(Case(f"open-jev/{rows[start].id}", f"open-jev/{src}", state, questions, gold))
    return cases


def load_source(spec: str) -> list[Case]:
    """``typed-decisions:train`` / ``typed-decisions:test``; Open-Jev as
    ``open-jev:<split>[:<config>][?cap=N&cap.<source>=N&max_questions=4]``; another Hub dataset in the
    Reef layout as ``org/name:split[:config]`` (e.g. ``seraphimserapis/reef-distill:train``);
    or a ``.jsonl`` path."""
    if spec.startswith("typed-decisions:"):
        return load_cases(spec.split(":", 1)[1])
    if spec.startswith("open-jev:"):
        body, _, query = spec.partition("?")
        _, split, *cfg = body.split(":")
        q = _parse_query(query)
        caps = {k[4:]: int(v) for k, v in q.items() if k.startswith("cap.")}
        return load_open_jev(split, cfg[0] if cfg else "release-v2-redistributable",
                             cap=int(q["cap"]) if "cap" in q else None, caps=caps or None,
                             max_questions=int(q.get("max_questions", 4)))
    bare = spec.removeprefix("hf://")
    if not Path(spec).exists() and ":" in bare and "/" in bare.split(":", 1)[0]:
        repo, split, *cfg = bare.split(":")
        return load_cases(split, cfg[0] if cfg else "all", dataset=repo, decode_state=True)
    return load_jsonl(spec)


def split_holdout(cases: list[Case], frac: float, seed: int) -> tuple[list[Case], list[Case]]:
    """Stratified (by workflow) train/validation split."""
    rng = random.Random(seed)
    by_wf: dict[str, list[Case]] = {}
    for c in cases:
        by_wf.setdefault(c.workflow, []).append(c)
    train, val = [], []
    for wf in sorted(by_wf):
        cs = by_wf[wf][:]
        rng.shuffle(cs)
        k = round(len(cs) * frac)
        val += cs[:k]
        train += cs[k:]
    return train, val


def _augment_questions(qs: list[Question], rng: random.Random) -> list[Question]:
    """Shuffle question order and the order of unordered options (choice, noul).

    Score rubrics stay ordered. Shuffling teaches the model to read option *text* rather than
    memorise option *positions*, which is what lets one network take arbitrary schemas.
    """
    out = []
    for q in qs:
        if q.type in ("choice", "noul"):
            opts = list(q.options)
            rng.shuffle(opts)
            q = Question(q.id, q.type, q.instructions, tuple(opts))
        out.append(q)
    rng.shuffle(out)
    return out


def _fit_options(q: Question, gold: dict, packer: Packer, rng: random.Random) -> Question:
    """At most MAX_OPTIONS options whose block fits: the gold favourite plus random others.

    Questions with dozens of options (intents, tools) are answered at inference as a knockout over
    parts of up to MAX_OPTIONS (see reef.packing), so training on random subsets that contain the
    favourite is the matching task. The gold distribution is renormalised over the subset later.
    """
    if len(q.options) <= MAX_OPTIONS and packer.block_fits(q):
        return q
    probs = (gold or {}).get("probabilities", {})
    order = list(range(len(q.options)))
    rng.shuffle(order)
    best = max(order, key=lambda i: float(probs.get(q.options[i].id, 0.0)))
    keep = [best]
    for i in order:
        if len(keep) >= MAX_OPTIONS:
            break
        if i != best and packer.block_fits(subquestion(q, sorted([*keep, i]))):
            keep.append(i)
    # Original relative order, like the runtime's knockout parts. (Never favourite-first: the model
    # would learn that the first option wins.)
    return subquestion(q, sorted(keep))


@dataclass
class WindowBatch:
    ids: np.ndarray  # [N, L]
    roles: np.ndarray
    levels: np.ndarray
    opt_pos: np.ndarray  # [N, Q, K] int32
    opt_mask: np.ndarray  # [N, Q, K] float32
    gold: np.ndarray  # [N, Q, K] float32
    # bookkeeping for evaluation: (case index, question) per (window, slot)
    meta: list[list[tuple[int, Question]]]

    def __len__(self) -> int:
        return len(self.ids)

    def tf_inputs(self, idx=slice(None)) -> dict:
        return {"ids": self.ids[idx], "roles": self.roles[idx], "levels": self.levels[idx]}


def build_windows(
    cases: list[Case], packer: Packer, rng: random.Random | None = None
) -> WindowBatch:
    """Pack cases into windows with gold targets. ``rng`` enables augmentation."""
    rows = {k: [] for k in ("ids", "roles", "levels", "opt_pos", "opt_mask", "gold")}
    meta = []
    for ci, c in enumerate(cases):
        qs = normalize_questions(c.questions)
        if rng is not None:
            qs = _augment_questions(qs, rng)
        r = rng or random.Random(c.id)  # deterministic option subsets for evaluation
        qs = [_fit_options(q, (c.gold or {}).get(q.id), packer, r) for q in qs]
        force = rng is not None and len(qs) == 1 and rng.random() < HOIST_AUGMENT
        windows = [(gqs, w) for _, gqs, state_ids in packer.prepare_groups(qs, render_state(c.state, rng), force)
                   for w in packer.pack(state_ids, gqs)]
        for qs, w in windows:
            pos = np.zeros((MAX_QUESTIONS, MAX_OPTIONS), np.int32)
            msk = np.zeros((MAX_QUESTIONS, MAX_OPTIONS), np.float32)
            gld = np.zeros((MAX_QUESTIONS, MAX_OPTIONS), np.float32)
            wmeta = []
            for si, (qi, positions) in enumerate(w.slots[:MAX_QUESTIONS]):
                q = qs[qi]
                probs = c.gold.get(q.id, {}).get("probabilities", {}) if c.gold else {}
                for oi, (o, p) in enumerate(zip(q.options[:MAX_OPTIONS], positions)):
                    pos[si, oi] = p
                    msk[si, oi] = 1.0
                    gld[si, oi] = float(probs.get(o.id, 0.0))
                s = gld[si].sum()
                if s > 0:
                    gld[si] /= s
                wmeta.append((ci, q))
            rows["ids"].append(w.ids)
            rows["roles"].append(w.roles)
            rows["levels"].append(w.levels)
            rows["opt_pos"].append(pos)
            rows["opt_mask"].append(msk)
            rows["gold"].append(gld)
            meta.append(wmeta)
    return WindowBatch(**{k: np.stack(v) for k, v in rows.items()}, meta=meta)
