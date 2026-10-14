import math
import os

import numpy as np
import pytest

from reef.decode import confidence, make_answer, softmax
from reef.metrics import Decision, breakdown, score
from reef.packing import normalize_question


def test_perfect_prediction():
    d = Decision("choice", "q", "w", {"a": 0.7, "b": 0.3}, {"a": 0.7, "b": 0.3}, "a")
    s = score([d])
    assert s["accuracy"] == 1.0
    assert s["kl"] == pytest.approx(0.0, abs=1e-9)
    assert s["brier"] == pytest.approx(0.0, abs=1e-12)
    assert s["ece"] == pytest.approx(0.3)  # confident at 0.7, right 100% of the time


def test_kl_and_brier_by_hand():
    d = Decision("noul", "q", "w", {"true": 0.5, "false": 0.5}, {"true": 0.9, "false": 0.1}, "true")
    s = score([d])
    assert s["kl"] == pytest.approx(0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5))
    assert s["brier"] == pytest.approx(0.4**2 * 2)


def test_missing_prediction_keys_are_zero_and_renormalised():
    d = Decision("choice", "q", "w", {"a": 2.0}, {"a": 0.5, "b": 0.5}, "a")
    s = score([d])
    assert s["accuracy"] == 1.0 and s["brier"] == pytest.approx(0.5)


def test_breakdown_groups():
    ds = [Decision("noul", "q1", "w1", {"true": 1, "false": 0}, {"true": 1, "false": 0}),
          Decision("score", "q2", "w1", {"0": 1, "1": 0}, {"0": 0, "1": 1})]
    b = breakdown(ds, "question_type")
    assert b["noul"]["accuracy"] == 1.0 and b["score"]["accuracy"] == 0.0


def test_softmax_and_confidence():
    p = softmax(np.array([2.0, 0.0]), t=1.0)
    assert p.sum() == pytest.approx(1.0) and p[0] > p[1]
    assert softmax(np.array([2.0, 0.0]), t=100.0)[0] == pytest.approx(0.5, abs=0.01)
    assert confidence(np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert confidence(np.array([0.9, 0.1])) == pytest.approx(0.8)
    assert confidence(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_make_answer_shapes():
    noul = make_answer(normalize_question("x", {"type": "noul"}), np.array([0.8, 0.2]))
    assert noul["noul"] == pytest.approx(0.8)
    choice = make_answer(normalize_question("x", {"type": "choice", "criteria": ["a", "b"]}), np.array([0.3, 0.7]))
    assert choice["choice"] == "b"
    sc = make_answer(normalize_question("x", {"type": "score", "criteria": ["lo", "mid", "hi"]}),
                     np.array([0.0, 0.5, 0.5]))
    assert sc["score"] == pytest.approx(1.5) and sc["legend"] == {"0": "lo", "1": "mid", "2": "hi"}


@pytest.mark.skipif(not os.environ.get("REEF_TEST_DATA"), reason="set REEF_TEST_DATA=1 to download the dataset")
def test_uniform_reference_row():
    """The dataset card reports Uniform: KL 0.444, Brier 0.238 on the test split."""
    from reef.data import load_cases

    ds = []
    for c in load_cases("test"):
        for qid, g in c.gold.items():
            k = len(g["probabilities"])
            ds.append(Decision(g["type"], qid, c.workflow, {o: 1 / k for o in g["probabilities"]},
                               g["probabilities"], g.get("label")))
    s = score(ds)
    assert round(s["kl"], 3) == 0.444
    assert round(s["brier"], 3) == 0.238


def test_int8_temperature_refit_matches_softmax_with_temperature():
    # calibrate.py rescales probabilities computed at T=1: p ** (1/t), renormalised.
    import numpy as np

    from reef.decode import softmax

    z = np.array([2.0, -1.0, 0.5, 3.0])
    p1 = softmax(z, 1.0)
    for t in (0.5, 0.8, 2.5):
        q = p1 ** (1 / t)
        np.testing.assert_allclose(q / q.sum(), softmax(z, t), rtol=1e-6)
