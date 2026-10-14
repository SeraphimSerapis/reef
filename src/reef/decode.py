"""From per-position logits to SystemOne answers. Shared by training, eval and the runtime."""

from __future__ import annotations

import numpy as np

from reef.packing import Question

# Default temperatures per question type. Overwritten by calibration (stored in config.json).
DEFAULT_TEMPERATURES = {"noul": 1.0, "choice": 1.0, "score": 1.0}


def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(x, np.float64) / max(t, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def confidence(p: np.ndarray) -> float:
    """How far the top option sits above uniform, rescaled to [0, 1].

    This matches the ``confidence`` field in the typed-decisions gold, ``(max - 1/K) / (1 - 1/K)``.
    For a yes/no question it is ``|2p - 1|``.
    """
    k = len(p)
    return float((p.max() - 1.0 / k) / (1.0 - 1.0 / k)) if k > 1 else 1.0


def make_answer(q: Question, probs: np.ndarray) -> dict:
    """A SystemOne-shaped answer for one question."""
    ids = [o.id for o in q.options]
    pd = {i: round(float(p), 6) for i, p in zip(ids, probs)}
    conf = round(confidence(probs), 6)
    if q.type == "noul":
        return {"type": "noul", "noul": pd["true"], "confidence": conf, "probabilities": pd}
    if q.type == "choice":
        return {"type": "choice", "choice": ids[int(np.argmax(probs))], "confidence": conf, "probabilities": pd}
    levels = np.array([int(i) for i in ids], np.float64)
    return {
        "type": "score",
        "score": round(float(np.dot(levels, probs)), 6),
        "confidence": conf,
        "probabilities": pd,
        "legend": {o.id: o.text for o in q.options},
    }


def question_probs(
    logits: np.ndarray, positions: list[int], qtype: str, temperatures: dict[str, float]
) -> np.ndarray:
    return softmax(logits[positions], temperatures.get(qtype, 1.0))
