"""Hugging Face Hub integration: run, pull and push Reef models.

Anywhere Reef takes a model directory it also takes a Hub repo id::

    reef serve seraphimserapis/reef        # downloads once, then runs from the HF cache
    Reef("seraphimserapis/reef")

    reef pull seraphimserapis/reef models/reef
    reef push artifacts/reef seraphimserapis/reef --teacher "Gemma 4 E4B"

Only the files the runtime needs (plus the model card and eval reports) are transferred.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ARTIFACT_FILES = [
    "config.json",
    "tokenizer.json",
    "embeddings.npz",
    "reef_int8.tflite",
    "reef_int8_edgetpu.tflite",
]
FLOAT_FILES = ["reef_float.weights.h5"]  # float training weights: only needed to fine-tune
REPORT_FILES = ["eval_*.json", "train_report.json", "train_history.json", "edgetpu_compiler.log", "README.md"]
_REPO_RE = re.compile(r"^(hf://)?[A-Za-z0-9][\w.-]*/[\w.-]+(@[\w./-]+)?$")


def is_repo_id(spec: str | Path) -> bool:
    s = str(spec)
    return not Path(s).exists() and bool(_REPO_RE.match(s)) and not s.startswith(("./", "../", "/"))


def resolve(spec: str | Path, with_float: bool = False) -> Path:
    """A local directory as-is, or ``org/name[@revision]`` / ``hf://org/name`` → a local snapshot."""
    if not is_repo_id(spec):
        return Path(spec)
    repo, _, rev = str(spec).removeprefix("hf://").partition("@")
    return Path(_download(repo, revision=rev or None, with_float=with_float))


def _download(repo_id: str, revision: str | None = None, local_dir: str | Path | None = None,
              with_float: bool = False) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - it is a core dependency, but stay friendly
        raise SystemExit("pip install huggingface_hub to load models from the Hub") from e
    from reef import __version__

    return snapshot_download(
        repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=ARTIFACT_FILES + REPORT_FILES + (FLOAT_FILES if with_float else []),
        library_name="reef",
        library_version=__version__,
    )


def pull(repo_id: str, dest: str | Path | None = None, with_float: bool = False) -> Path:
    repo, _, rev = repo_id.removeprefix("hf://").partition("@")
    return Path(_download(repo, revision=rev or None, local_dir=dest, with_float=with_float))


# -- model card ---------------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def _metrics_rows(art: Path) -> list[tuple[str, dict]]:
    rows = []
    rep = art / "train_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        for spec, res in (r.get("tests") or {}).items():
            rows.append((f"float · {spec}", res.get("overall", res)))
        if not r.get("tests") and r.get("test"):
            rows.append(("float · typed-decisions:test", r["test"]))
    for f in sorted(art.glob("eval_*.json")):
        e = json.loads(f.read_text())
        rows.append((f"int8 · {e.get('device', f.stem[5:])} · {e.get('split', 'test')}", e.get("overall", {})))
    return rows


def model_card(art: str | Path, repo_id: str, teacher: str | None = None, summary: str | None = None) -> str:
    art = Path(art)
    cfg = json.loads((art / "config.json").read_text())
    m, ta = cfg["model"], cfg.get("train_args", {})
    sources = ta.get("train_data") or ["typed-decisions:train"]
    tests = ta.get("test_data") or ["typed-decisions:test"]
    uses_td = any(str(s).startswith("typed-decisions") for s in sources + tests)
    packs = sorted({Path(s).stem for s in sources if str(s).endswith(".jsonl")})
    tags = ["edge-tpu", "coral", "tflite", "int8", "typed-decisions", "systemone", "reef", "homelab", "text-classification"]
    meta = ["---", "license: apache-2.0", "library_name: reef", "pipeline_tag: text-classification", "language:", "- en",
            "tags:", *[f"- {t}" for t in tags]]
    if uses_td:
        meta += ["datasets:", "- LocalLLaMA/typed-decisions"]
    meta.append("---")

    rows = _metrics_rows(art)
    table = ["| Model / test set | Accuracy | KL ↓ | Brier ↓ | ECE ↓ |", "|---|---|---|---|---|"]
    for name, o in rows:
        table.append(f"| {name} | {_fmt(o.get('accuracy'))} | {_fmt(o.get('kl'))} | {_fmt(o.get('brier'))} | "
                     f"{_fmt(o.get('ece'))} |")
    tpu_size = (art / "reef_int8_edgetpu.tflite").stat().st_size / 2**20 if (art / "reef_int8_edgetpu.tflite").exists() else None

    lines = [
        *meta, "",
        f"# {cfg.get('name', repo_id.split('/')[-1])}", "",
        summary or ("A tiny typed-decision model that runs on a **Google Coral Edge TPU** (or any CPU). Give it a JSON "
                    "state and typed questions (yes/no, choice, 0–4 score); it returns calibrated probabilities for "
                    "every option in a few milliseconds."), "",
        "Part of **[Reef](https://github.com/SeraphimSerapis/reef)**: give your retired Coral a new job.", "",
        "## Quickstart", "",
        "```bash",
        "pip install \"reef-decide @ git+https://github.com/SeraphimSerapis/reef\"",
        f"reef serve {repo_id} --host 0.0.0.0 --port 8080      # Edge TPU if attached, else CPU",
        "```", "",
        "```bash",
        "curl -s localhost:8080/v1/systemone -d '{",
        "  \"state\": {\"source\": \"uptime-kuma\", \"monitor\": \"nextcloud\", \"status\": \"down\", \"down_for\": \"14m\"},",
        "  \"questions\": {\"page\": {\"type\": \"noul\", \"instructions\": \"Should someone be woken up for this?\"}}",
        "}'",
        "```", "",
        "Or with Docker: `docker run --device /dev/bus/usb -p 8080:8080 ghcr.io/seraphimserapis/reef serve "
        f"{repo_id} --host 0.0.0.0`.", "",
        "## Files", "",
        "| File | What |", "|---|---|",
        f"| `reef_int8_edgetpu.tflite` | encoder compiled for the Edge TPU (one subgraph{f', {tpu_size:.2f} MiB' if tpu_size else ''}) |",
        "| `reef_int8.tflite` | the same int8 encoder for CPU |",
        "| `embeddings.npz` | token / role / level embedding tables (looked up on the host) |",
        "| `tokenizer.json`, `config.json` | tokenizer, shapes and per-type calibration temperatures |",
        *(["| `reef_float.weights.h5` | float training weights (Keras), for fine-tuning |"] if (art / "reef_float.weights.h5").exists() else []), "",
        "## Model", "",
        f"- {cfg.get('encoder_params', 0) / 1e6:.1f}M-parameter convolutional encoder (width {m['width']}, depth "
        f"{m['depth']}, expansion {m['expansion']}{', max-pool state summary' if m.get('global_max') else ''}); "
        f"no attention, because the Edge TPU compiler cannot map it.",
        f"- Input: one {m['state_len']}-token state region + {m['schema_len']}-token question region laid out as a "
        f"{(m['state_len'] + m['schema_len']) // m['grid_width']}×{m['grid_width']} grid; long states are split into windows.",
        f"- Token embeddings initialised from [{cfg.get('embeddings_source', 'minishlab/potion-base-8M')}]"
        f"(https://huggingface.co/{cfg.get('embeddings_source', 'minishlab/potion-base-8M')}) (MIT).",
        "- Full-integer int8 quantisation; outputs are temperature-scaled per question type.", "",
        *(["## Fine-tune it", "",
            "Label your own events (or edit a pack's house policy and relabel), then start from these weights:", "",
            "```bash",
            "pip install \"reef-decide[train] @ git+https://github.com/SeraphimSerapis/reef\"",
            f"reef train --init {repo_id} --data my-events.jsonl --epochs 10 --lr 5e-4 --out runs/mine",
            "reef export runs/mine artifacts/mine && reef eval artifacts/mine --device edgetpu",
            "```", ""] if (art / "reef_float.weights.h5").exists() else []),
        "## Training data", "",
        *[f"- `{s}`" for s in sources], "",
    ]
    if packs:
        lines += [f"Homelab packs: {', '.join(f'`{p}`' for p in packs)}. Pack states are synthetic (no real home data); "
                  "soft labels come from a teacher LLM's answer-token probabilities.", ""]
    if teacher:
        lines += [f"**Teacher model:** {teacher}. Pack labels were distilled from its option probabilities "
                  "(one forward pass per question, no sampling). Thank you to the Gemma team.", ""]
    if uses_td:
        lines += ["Benchmark data: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).", ""]
    if rows:
        lines += ["## Results", "", *table, "",
                  "Metrics follow the typed-decisions card: accuracy of the top option, KL(gold ‖ predicted), multi-class "
                  "Brier and 15-bin ECE.", ""]
    lines += [
        "## Limitations", "",
        "- A specialist: good at the question shapes and domains it was trained on, not a general reasoner.",
        "- Distilled labels inherit the teacher's judgement and blind spots. Use it to *filter and rank*, and keep a "
        "human in the loop for anything safety-critical.",
        "- The Edge TPU does not make it faster than a modern laptop CPU; it makes it cheap to run on a Raspberry Pi "
        "or an old NUC next to Home Assistant.", "",
        "## License", "",
        "Apache-2.0. See [NOTICE](https://github.com/SeraphimSerapis/reef/blob/main/NOTICE) for third-party credits.", "",
    ]
    return "\n".join(lines)


def push(art: str | Path, repo_id: str, private: bool = False, teacher: str | None = None,
         card: str | Path | None = None, message: str | None = None) -> str:
    from huggingface_hub import HfApi

    art = Path(art)
    missing = [f for f in ARTIFACT_FILES if f != "reef_int8_edgetpu.tflite" and not (art / f).exists()]
    if missing:
        raise SystemExit(f"{art} is not a Reef artifact (missing {', '.join(missing)})")
    text = Path(card).read_text() if card else model_card(art, repo_id, teacher=teacher)
    (art / "README.md").write_text(text)
    api = HfApi()
    url = api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(art), repo_id=repo_id, allow_patterns=ARTIFACT_FILES + REPORT_FILES + FLOAT_FILES,
                      commit_message=message or f"Upload {art.name}")
    return str(url)


# -- datasets -----------------------------------------------------------------------------------
def _read_rows(files: list[str | Path]) -> list[dict]:
    rows = []
    for f in files:
        rows += [json.loads(ln) for ln in Path(f).read_text(encoding="utf-8").split("\n") if ln.strip()]
    return rows


def _dataset_stats(rows: list[dict]) -> dict:
    """Per pack: cases, questions per type, top-option confidence, and (for merged data) how often the
    teachers' top answers agree."""
    out: dict = {}
    for r in rows:
        p = out.setdefault(r["workflow"], {"cases": 0, "types": {}})
        p["cases"] += 1
        for qid, g in r["gold"].items():
            t = p["types"].setdefault(g["type"], {"n": 0, "conf": 0.0, "agree": 0, "agree_n": 0})
            t["n"] += 1
            t["conf"] += max(g["probabilities"].values())
            if len(r.get("teachers") or []) > 1:
                tops = {max(tp["probabilities"][qid], key=tp["probabilities"][qid].get) for tp in r["teachers"]}
                t["agree_n"] += 1
                t["agree"] += len(tops) == 1
    return out


def dataset_card(repo_id: str, splits: dict[str, list[dict]], teachers: list[str], summary: str | None = None) -> str:
    from reef import packs as pack_registry

    total = sum(len(v) for v in splits.values())
    size = "1K<n<10K" if total < 10_000 else "10K<n<100K"
    meta = ["---", "license: apache-2.0", "language:", "- en", "task_categories:", "- text-classification",
            "size_categories:", f"- {size}", "pretty_name: Reef homelab distillation set",
            "tags:", *[f"- {t}" for t in ["typed-decisions", "systemone", "distillation", "soft-labels", "homelab",
                                          "home-assistant", "synthetic", "reef"]],
            "configs:", "- config_name: all", "  data_files:",
            *[f"  - split: {s}\n    path: all/{s}-*.parquet" for s in splits], "---"]
    stats = {s: _dataset_stats(rows) for s, rows in splits.items()}
    names = sorted({p for st in stats.values() for p in st})
    pack_rows = []
    for n in names:
        try:
            desc = pack_registry.get(n).description
        except Exception:  # noqa: BLE001 - a pack that is not built in
            desc = ""
        counts = " / ".join(str(stats[s].get(n, {}).get("cases", 0)) for s in splits)
        pack_rows.append(f"| `{n}` | {desc} | {counts} |")
    agree_rows = []
    test = stats.get("test") or next(iter(stats.values()))
    for n in names:
        types = test.get(n, {}).get("types", {})
        cells = []
        for t in ("noul", "choice", "score"):
            v = types.get(t)
            if not v:
                cells.append("–")
            elif v["agree_n"]:
                cells.append(f"{v['agree'] / v['agree_n']:.2f} ({v['conf'] / v['n']:.2f})")
            else:
                cells.append(f"– ({v['conf'] / v['n']:.2f})")
        agree_rows.append(f"| `{n}` | " + " | ".join(cells) + " |")
    multi = len(teachers) > 1
    lines = [
        *meta, "",
        "# Reef homelab distillation set", "",
        summary or ("Synthetic homelab and smart-home events with **soft labels** for typed questions (yes/no, "
                    "choice, 0–4 score), distilled from Gemma 4. It is the training data for "
                    "[seraphimserapis/reef](https://huggingface.co/seraphimserapis/reef), a tiny model that "
                    "answers these questions in milliseconds on a Google Coral Edge TPU, and it has the same row format "
                    "as [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)."), "",
        "## Packs", "",
        f"| Pack | What | Cases ({' / '.join(splits)}) |", "|---|---|---|", *pack_rows, "",
        "Every state is generated by a pack in [Reef](https://github.com/SeraphimSerapis/reef) (`src/reef/packs/`). "
        "Event shapes mirror the real tools (Alertmanager, Uptime Kuma, Home Assistant, Frigate, UniFi Protect, UniFi "
        "Network system log keys), but **no real home data is included**: names are invented, IP addresses are private "
        "or RFC 5737 documentation ranges, MAC addresses are made-up locally administered ones. Train and test use "
        "different seeds and share no states.", "",
        "## Row format", "",
        "| Field | Type | What |", "|---|---|---|",
        "| `id` | string | `<pack>-<seed>-<index>` (regenerate the state with `reef label` and the same seed) |",
        "| `workflow` | string | pack name |",
        "| `state` | JSON string | the event / situation, as the model sees it |",
        "| `questions` | JSON string | typed questions: `{qid: {type, instructions, criteria?}}` |",
        "| `gold` | JSON string | `{qid: {type, probabilities: {option: p}, label}}`: the soft label to train on |",
        "| `teacher` | string | provenance |",
        *(["| `teachers` | JSON string | each teacher's own probabilities, before averaging |"] if multi else []), "",
        "## How the labels were made", "",
        "- Each question is asked as a multiple-choice prompt; the label is the teacher's probability for each option, "
        "read from the log-probabilities of the first answer token (one forward pass, no sampling, nothing to parse).",
        "- Each pack adds a short glossary and house policy to the teacher's system prompt (`teacher_notes` in the pack), "
        "e.g. what a UniFi Protect `alrmSpeak` event is, or that a stranger at the door at 03:00 is worth a push.",
        "- Yes/no and choice questions are asked twice, with the options in forward and reversed order, and averaged; this "
        "cancels most letter-position bias. Scores are asked once as a digit scale 0–4 in natural order.",
        *([f"- **Teacher ensemble:** every case was labelled independently by {' and '.join(teachers)}; `gold` is the "
           "average. Where the teachers agree the label stays sharp, where they disagree it is soft, so a student learns "
           "the uncertainty instead of one model's mistake."] if multi else []), "",
        f"**Teacher model{'s' if multi else ''}:** {', '.join(teachers)} (Apache-2.0). Thank you to the Gemma team.", "",
        "## Label statistics (test split)", "",
        ("Teacher agreement on the top answer, and in brackets the mean top-option probability of `gold`:"
         if multi else "Mean top-option probability of `gold`:"), "",
        "| Pack | yes/no | choice | score |", "|---|---|---|---|", *agree_rows, "",
        *(["Scores are ordinal, so an off-by-one between teachers counts as a disagreement here.", ""] if multi else []),
        "## Use it", "",
        "```bash",
        "pip install \"reef-decide[train] @ git+https://github.com/SeraphimSerapis/reef\"",
        f"reef train --data {repo_id}:train --test {repo_id}:test --out runs/mine",
        "```", "",
        "```python",
        "from datasets import load_dataset",
        f"ds = load_dataset(\"{repo_id}\")",
        "```", "",
        "## Limitations", "",
        "- Synthetic states cover the situations the generators know about; real events are messier.",
        "- Labels are LLM judgement under one house policy (see each pack's notes). Your household may disagree: edit the "
        "notes, relabel, and fine-tune.",
        "- Not for safety-critical decisions without a human in the loop.", "",
        "## License", "",
        "Apache-2.0, like the generators and the teacher models.", "",
    ]
    return "\n".join(lines)


def push_dataset(repo_id: str, splits: dict[str, list[str | Path]], private: bool = True,
                 teachers: list[str] | None = None, message: str | None = None, out_dir: str | Path | None = None,
                 upload: bool = True) -> str:
    """Write ``all/<split>-00000-of-00001.parquet`` (typed-decisions layout, nested fields as JSON strings)
    plus a dataset card, and upload them as a dataset repo."""
    import tempfile

    import pandas as pd
    from huggingface_hub import HfApi

    rows = {s: _read_rows(files) for s, files in splits.items()}
    if teachers is None:
        teachers = sorted({t["teacher"].split(" (")[0] for rs in rows.values() for r in rs
                           for t in (r.get("teachers") or [{"teacher": r.get("teacher", "unknown")}])})
    tmp = Path(out_dir or tempfile.mkdtemp(prefix="reef-ds-"))
    (tmp / "all").mkdir(parents=True, exist_ok=True)
    for s, rs in rows.items():
        df = pd.DataFrame([{
            "id": r["id"], "workflow": r["workflow"],
            "state": json.dumps(r["state"], ensure_ascii=False),
            "questions": json.dumps(r["questions"], ensure_ascii=False),
            "gold": json.dumps(r["gold"], ensure_ascii=False),
            "teacher": r.get("teacher", ""),
            **({"teachers": json.dumps(r["teachers"], ensure_ascii=False)} if "teachers" in r else {}),
        } for r in rs])
        df.to_parquet(tmp / "all" / f"{s}-00000-of-00001.parquet", index=False)
    (tmp / "README.md").write_text(dataset_card(repo_id, rows, teachers))
    if not upload:
        return str(tmp)
    api = HfApi()
    url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="dataset",
                      commit_message=message or "Upload Reef distillation data")
    return str(url)
