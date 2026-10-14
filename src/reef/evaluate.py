"""Evaluating an exported artifact (int8 CPU or Edge TPU) on typed-decisions or any source, and benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from reef.metrics import Decision, breakdown, score


def evaluate(artifact: str, device: str = "auto", split: str = "test", limit: int | None = None) -> dict:
    from reef.data import load_cases, load_source
    from reef.runtime import Reef

    reef = Reef(artifact, device=device)
    # a bare split name means typed-decisions; anything else is a load_source spec or .jsonl path
    cases = (load_source(split) if (":" in split or split.endswith(".jsonl")) else load_cases(split))[:limit]
    decisions, lat = [], []
    for c in cases:
        r = reef.decide(c.state, c.questions)
        lat.append(r["usage"]["total_ms"])
        for qid, a in r["answers"].items():
            g = c.gold[qid]
            decisions.append(Decision(g["type"], qid, c.workflow, a["probabilities"], g["probabilities"],
                                      g.get("label")))
    return {
        "artifact": str(artifact),
        "device": reef.device,
        "split": split,
        "overall": score(decisions),
        "by_type": breakdown(decisions, "question_type"),
        "by_workflow": breakdown(decisions, "workflow"),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
    }


def prior_baseline(split: str = "test") -> dict:
    """The card's 'Prior' reference: answer every case with the train-split mean distribution."""
    from collections import defaultdict

    from reef.data import load_cases

    sums: dict = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(int)
    for c in load_cases("train"):
        for qid, g in c.gold.items():
            counts[(c.workflow, qid)] += 1
            for k, v in g["probabilities"].items():
                sums[(c.workflow, qid)][k] += v
    ds = []
    for c in load_cases(split):
        for qid, g in c.gold.items():
            n = counts[(c.workflow, qid)]
            pred = {k: v / n for k, v in sums[(c.workflow, qid)].items()}
            ds.append(Decision(g["type"], qid, c.workflow, pred, g["probabilities"], g.get("label")))
    return {"overall": score(ds), "by_type": breakdown(ds, "question_type")}


def bench(artifact: str, n: int = 200, threads: int = 1) -> dict:
    """Per-window encoder latency on CPU (int8, XNNPACK) and on the Edge TPU, plus end to end."""
    from reef.data import load_cases
    from reef.packing import normalize_questions
    from reef.runtime import Reef
    from reef.text import render_state

    case = load_cases("test")[0]
    out: dict = {}
    for device in ("cpu", "edgetpu"):
        try:
            reef = Reef(artifact, device=device, num_threads=threads)
        except RuntimeError as e:
            out[device] = {"error": str(e)}
            continue
        win = reef.packer.pack(reef.packer.encode(render_state(case.state)), normalize_questions(case.questions))[0]
        for _ in range(10):
            reef.run_window(win.ids, win.roles, win.levels)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            reef.run_window(win.ids, win.roles, win.levels)
            ts.append((time.perf_counter() - t) * 1e3)
        e2e = []
        for _ in range(max(20, n // 5)):
            e2e.append(reef.decide(case.state, case.questions)["usage"]["total_ms"])
        out[device] = {
            "window_ms_p50": round(float(np.median(ts)), 3),
            "window_ms_p95": round(float(np.percentile(ts, 95)), 3),
            "decide_ms_p50": round(float(np.median(e2e)), 3),
            "threads": threads if device == "cpu" else None,
        }
    return out


def dump(obj: dict, path: str | None) -> None:
    txt = json.dumps(obj, indent=2)
    print(txt)
    if path:
        Path(path).write_text(txt)
