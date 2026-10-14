"""Generating teacher-labelled training data for a pack.

    reef label alert-triage --teacher http://gpu-box:8080 --n 5000 --out data/alert-triage.jsonl

Each line of the output is one case in the same shape as typed-decisions (``id``, ``workflow``,
``state``, ``questions``, ``gold``), so ``reef train --data`` can mix packs with each other and
with the benchmark. Labelling is resumable: existing ids in the output file are skipped.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def make_cases(pack, n: int, seed: int) -> list[dict]:
    rng = random.Random(f"{pack.name}:{seed}")
    out = []
    for i in range(n):
        out.append({
            "id": f"{pack.name}-{seed}-{i:06d}",
            "workflow": pack.name,
            "state": pack.generate(rng),
            "questions": pack.sample_questions(rng),
        })
    return out


def label(pack, teacher, n: int, out: str | Path, seed: int = 0, workers: int = 2, progress: bool = True,
          max_consecutive_failures: int = 25) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").split("\n"):
            if ln.strip():
                done.add(json.loads(ln)["id"])
    todo = [c for c in make_cases(pack, n, seed) if c["id"] not in done]
    lock = threading.Lock()
    t0, ok, failed = time.time(), 0, 0

    def work(case):
        case["gold"] = teacher.label(case["state"], case["questions"], seed=zlib.crc32(case["id"].encode()),
                                     notes=getattr(pack, "teacher_notes", ""))
        case["teacher"] = teacher.name
        return case

    streak = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, c) for c in todo]
        for fut in as_completed(futures):
            try:
                case = fut.result()
            except Exception as e:  # noqa: BLE001 - keep labelling; report at the end
                failed += 1
                streak += 1
                if failed <= 3:
                    print(f"teacher error: {e}", file=sys.stderr)
                if streak >= max_consecutive_failures:
                    for other in futures:
                        other.cancel()
                    raise RuntimeError(f"{pack.name}: {streak} consecutive teacher failures, giving up "
                                       f"({ok + len(done)}/{n} labelled so far; re-run to resume)") from e
                continue
            streak = 0
            with lock:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            if progress and ok % 50 == 0:
                rate = ok / (time.time() - t0)
                print(f"{pack.name}: {ok + len(done)}/{n} labelled ({rate:.1f} cases/s, "
                      f"eta {(len(todo) - ok) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
    print(f"{pack.name}: wrote {ok} new cases to {out} ({failed} failed, {len(done)} already there)")
    return out


def merge(inputs: list[str | Path], out: str | Path, weights: list[float] | None = None) -> dict:
    """Ensemble teachers: average the soft labels of the same cases labelled by different teachers
    (``reef label`` with the same pack, seed and n gives the same ids and states). Where the teachers
    agree the label stays sharp; where they disagree the student learns the uncertainty instead of
    one model's mistake. Only ids present in every input are written. Returns agreement stats."""
    weights = weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise ValueError("one weight per input")
    tables = []
    for p in inputs:
        rows = [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").split("\n") if ln.strip()]
        tables.append({r["id"]: r for r in rows})
    ids = [i for i in tables[0] if all(i in t for t in tables[1:])]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agree = total = 0
    with out.open("w") as f:
        for i in ids:
            rows = [t[i] for t in tables]
            states = {json.dumps(r["state"], sort_keys=True) for r in rows}
            if len(states) > 1:
                raise ValueError(f"{i}: inputs disagree on the state; were they labelled with the same pack version?")
            gold = {}
            for qid, g0 in rows[0]["gold"].items():
                probs = {k: sum(w * r["gold"][qid]["probabilities"][k] for w, r in zip(weights, rows)) / sum(weights)
                         for k in g0["probabilities"]}
                gold[qid] = {"type": g0["type"], "probabilities": {k: round(v, 6) for k, v in probs.items()},
                             "label": max(probs, key=probs.get)}
                total += 1
                agree += len({r["gold"][qid]["label"] for r in rows}) == 1
            merged = {**rows[0], "gold": gold, "teacher": " + ".join(dict.fromkeys(r["teacher"] for r in rows)),
                      "teachers": [{"teacher": r["teacher"], "weight": w,
                                    "probabilities": {q: g["probabilities"] for q, g in r["gold"].items()}}
                                   for r, w in zip(rows, weights)]}
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    return {"cases": len(ids), "questions": total, "teachers_agree": agree / max(total, 1)}
