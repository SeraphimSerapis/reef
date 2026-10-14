"""The built-in policy cases must reference real packs, real questions and valid labels."""

from reef import packs, policy
from reef.packing import normalize_question


def test_policy_cases_are_valid():
    cases = policy.load()
    assert len(cases) >= 30
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["pack"] for c in cases} == set(packs.names())
    for c in cases:
        spec = packs.get(c["pack"]).canonical_questions()[c["ask"]]
        ids = {o.id for o in normalize_question(c["ask"], spec).options}
        assert c["expect"] and set(c["expect"]) <= ids, c["id"]


def test_policy_runner_with_a_fake_model():
    cases = policy.load()

    def always_first_expected(pack, state, questions):
        (qid,) = questions
        c = next(c for c in cases if c["state"] is state)
        return {qid: {c["expect"][0]: 0.9, "zzz": 0.1}}

    res = policy.run(always_first_expected, cases)
    assert res["passed"] == res["total"] and not res["failures"]
