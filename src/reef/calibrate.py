"""Refit an exported artifact's temperatures on its int8 outputs.

Training fits one temperature per question type on the float model. Post-training int8
quantisation changes the logits a little (the output scale clips the largest ones), which shows
up as worse calibration on the exported model. This module rebuilds the run's validation split
from ``config.json`` (same sources, ``val_frac``, seed and ``select_on`` as training), answers it
with the int8 interpreter at temperature 1, and picks the temperature per type that minimises
KL from gold. With ``p = softmax(z)``, ``softmax(z / t)`` is ``p ** (1 / t)`` renormalised, so one
pass over the data is enough.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from reef.metrics import Decision, score

QTYPES = ("noul", "choice", "score")


def _validation_cases(train_args: dict, select: list[str] | None = None) -> list:
    from reef.data import load_source, split_holdout

    cases = [c for spec in train_args.get("train_data") or [] for c in load_source(spec)]
    if not cases:
        raise SystemExit("config.json has no train_args.train_data; pass --data to calibrate on")
    _, val = split_holdout(cases, train_args.get("val_frac", 0.2), train_args.get("seed", 0))
    prefixes = select or train_args.get("select_on")
    if prefixes:
        val = [c for c in val if c.workflow.startswith(tuple(prefixes))]
    return val


def calibrate(artifact: str, device: str = "cpu", limit: int | None = 4000, data: list[str] | None = None,
              write: bool = True, select: list[str] | None = None) -> dict:
    """Return ``{"before", "after", "temperatures", "float_temperatures", "n"}``; with ``write`` the
    new temperatures go into ``config.json`` (the float ones are kept as ``float_temperatures``).

    ``select``: workflow prefixes of the validation cases to fit on (default: the run's
    ``select_on``). Fit on soft (teacher) labels where calibration matters; one-hot human labels
    of benchmark-style data pull the temperature the other way."""
    from reef.data import load_source
    from reef.runtime import Reef

    path = Path(artifact) / "config.json"
    meta = json.loads(path.read_text())
    if data:
        cases = [c for spec in data for c in load_source(spec)]
    else:
        cases = _validation_cases(meta.get("train_args", {}), select)
    if limit and len(cases) > limit:
        cases = random.Random(0).sample(cases, limit)
    reef = Reef(artifact, device=device)
    float_temps = meta.get("float_temperatures") or meta["temperatures"]
    reef.temperatures = {t: 1.0 for t in QTYPES}
    raw: list[tuple[Decision, list[str], np.ndarray]] = []
    for c in cases:
        out = reef.decide(c.state, c.questions)
        for qid, a in out["answers"].items():
            g = (c.gold or {}).get(qid)
            if not g or "probabilities" not in g:
                continue
            keys = list(a["probabilities"])
            p = np.array([a["probabilities"][k] for k in keys], np.float64)
            raw.append((Decision(g["type"], qid, c.workflow, {}, g["probabilities"], g.get("label")), keys, p))

    def decisions(temps: dict[str, float]) -> list[Decision]:
        out = []
        for d, keys, p in raw:
            q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / temps.get(d.question_type, 1.0))
            q /= q.sum()
            out.append(Decision(d.question_type, d.question_id, d.workflow, dict(zip(keys, q.tolist())), d.gold,
                                d.gold_label))
        return out

    grid = np.exp(np.linspace(math.log(0.25), math.log(6.0), 120))
    temps = dict(float_temps)
    for qtype in QTYPES:
        if not any(d.question_type == qtype for d, _, _ in raw):
            continue
        kls = [(score([d for d in decisions({qtype: float(t)}) if d.question_type == qtype])["kl"], float(t))
               for t in grid]
        temps[qtype] = round(min(kls)[1], 4)
    before, after = score(decisions(float_temps)), score(decisions(temps))
    if write:
        meta["float_temperatures"] = float_temps
        meta["temperatures"] = temps
        meta["int8_calibration"] = {"n_cases": len(cases), "n_questions": len(raw), "device": reef.device,
                                    "select": select or meta.get("train_args", {}).get("select_on"),
                                    "data": data, "before": before, "after": after}
        path.write_text(json.dumps(meta, indent=2))
    return {"n": len(raw), "float_temperatures": float_temps, "temperatures": temps, "before": before,
            "after": after}
