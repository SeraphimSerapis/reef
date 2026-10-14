"""Scoring typed decisions against soft gold distributions.

The typed-decisions benchmark reports accuracy, KL from gold, Brier and ECE over every decision
(case × question). KL and Brier below reproduce the card's "Uniform" reference row exactly
(0.444 / 0.238, see ``tests/test_metrics.py``). Accuracy depends on how argmax ties are broken
and the card does not spell out its ECE binning, so those two can differ in the third decimal
for degenerate predictors like Uniform:

* accuracy: the predicted argmax equals the gold label
* KL: ``sum_k gold_k * log(gold_k / pred_k)``, averaged over decisions
* Brier: ``sum_k (pred_k - gold_k)^2``, averaged over decisions
* ECE: confidence = max predicted probability, 10 equal-width bins, weighted
  ``|accuracy - confidence|``
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass
class Decision:
    question_type: str  # noul | choice | score
    question_id: str
    workflow: str
    pred: dict[str, float]
    gold: dict[str, float]
    gold_label: str | None = None


def _vec(d: dict[str, float], keys: list[str]) -> np.ndarray:
    v = np.array([float(d.get(k, 0.0)) for k in keys], np.float64)
    s = v.sum()
    return v / s if s > 0 else np.full(len(keys), 1.0 / len(keys))


def score(decisions: list[Decision], n_bins: int = 10) -> dict[str, float]:
    if not decisions:
        return {}
    accs, kls, briers, confs = [], [], [], []
    for d in decisions:
        keys = list(d.gold)
        g, p = _vec(d.gold, keys), _vec(d.pred, keys)
        label = d.gold_label if d.gold_label is not None else keys[int(np.argmax(g))]
        accs.append(float(keys[int(np.argmax(p))] == label))
        pc = np.clip(p, EPS, 1.0)
        kls.append(float(np.sum(np.where(g > 0, g * (np.log(np.clip(g, EPS, 1)) - np.log(pc)), 0.0))))
        briers.append(float(np.sum((p - g) ** 2)))
        confs.append(float(p.max()))
    accs_a, confs_a = np.array(accs), np.array(confs)
    bins = np.minimum((confs_a * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(accs_a[m].mean() - confs_a[m].mean())
    return {
        "n": len(decisions),
        "accuracy": float(accs_a.mean()),
        "kl": float(np.mean(kls)),
        "brier": float(np.mean(briers)),
        "ece": float(ece),
    }


def breakdown(decisions: list[Decision], key: str) -> dict[str, dict[str, float]]:
    """Scores grouped by ``question_type``, ``workflow`` or ``question_id``."""
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        groups[getattr(d, key)].append(d)
    return {k: score(v) for k, v in sorted(groups.items())}
