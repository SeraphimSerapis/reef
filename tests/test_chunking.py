"""Long states (overlapping chunks) and many-option questions (knockout over parts)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from reef.data import Case, build_windows
from reef.decode import softmax
from reef.packing import (
    HOIST_STUB,
    MAX_PART_OPTIONS,
    OPT,
    CapacityError,
    Packer,
    hoist_groups,
    hoist_instructions,
    normalize_question,
    normalize_questions,
    subquestion,
)
from reef.runtime import Reef
from reef.server import make_handler


def test_state_regions_cover_a_long_state(tokenizer):
    p = Packer(tokenizer, state_len=34, schema_len=64)  # chunks of 32 tokens, stride 24
    short, gaps = p.state_regions(list(range(10, 40)))
    assert len(short) == 1 and not gaps
    state = list(range(10, 210))  # 200 tokens
    regions, gaps = p.state_regions(state, max_windows=16)
    assert not gaps and len(regions) == 8  # starts 0, 24, ..., 168
    seen = set()
    for ids, _ in regions:
        assert len(ids) == 34 and ids[0] == tokenizer.token_to_id("[CLS]")
        seen |= set(ids[1:33].tolist())
    assert seen == set(state)  # every token is read at least once
    few, gaps = p.state_regions(state, max_windows=3)
    assert gaps and len(few) == 3
    assert few[0][0][1] == 10 and few[-1][0][32] == 209  # first and last tokens still read


def test_split_options_respects_part_size_and_window(tokenizer):
    p = Packer(tokenizer, state_len=8, schema_len=40)
    q = normalize_question("intent", {"type": "choice", "criteria": [f"intent{i} a b" for i in range(77)]})
    chunks = p.split_options(q, list(range(77)))
    assert sorted(i for c in chunks for i in c) == list(range(77))
    assert all(1 <= len(c) <= MAX_PART_OPTIONS for c in chunks)
    assert all(p.block_fits(subquestion(q, c)) for c in chunks)


def test_strict_refuses_instead_of_truncating(tokenizer):
    q = normalize_question("q", {"type": "noul", "instructions": " ".join(["word"] * 50)})
    assert Packer(tokenizer).block(q).option_offsets[0] == 1 + 40  # lenient: instructions truncated
    with pytest.raises(CapacityError, match="maximum context length"):
        Packer(tokenizer, strict=True).block(q)


def test_structured_instructions_are_rendered():
    q = normalize_question("q", {"type": "noul", "instructions": {"ask": "is it on fire?"}})
    assert "fire" in q.instructions


class FakeTok:
    """Whitespace tokenizer whose option words ``vN`` carry a logit of N / 10 (see FakeReef)."""

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.words = {}

    def token_to_id(self, tok):
        return self.vocab.get(tok)

    def encode(self, text, add_special_tokens=False):
        class E:
            pass

        e = E()
        e.ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.lower().replace(":", " ").split()]
        for w, i in self.vocab.items():
            self.words[i] = w
        return e


def _fake_reef(state_len=34, schema_len=48, **kw):
    r = Reef.__new__(Reef)
    tok = FakeTok()
    r.name, r.device, r.state_len, r.schema_len = "fake", "cpu", state_len, schema_len
    r.temperatures = {"noul": 1.0, "choice": 1.0, "score": 1.0}
    r.max_state_windows, r.strict = kw.get("max_state_windows", 8), kw.get("strict", False)
    r.packer = Packer(tok, state_len, schema_len, strict=r.strict)
    r.hoist, r.instruction_cap = kw.get("hoist", False), 12
    r.hoist_empty = kw.get("hoist_empty", False)
    calls = []

    def run_window(ids, roles, levels):  # an option marker's logit is read off the option's word
        calls.append(ids.copy())
        out = np.zeros(len(ids), np.float32)
        for i in np.flatnonzero(roles == OPT):
            for w in (tok.words.get(int(t), "") for t in ids[i + 1 : i + 3]):
                if w.startswith("v") and w[1:].isdigit():
                    out[i] = float(w[1:]) / 10
                    break
        return out

    r.run_window = run_window
    return r, calls


def test_few_options_match_plain_softmax():
    r, calls = _fake_reef()
    out = r.decide("short state", {"c": {"type": "choice", "criteria": ["v1", "v5", "v3"]}})
    p = out["answers"]["c"]["probabilities"]
    np.testing.assert_allclose([p["v1"], p["v5"], p["v3"]], softmax(np.array([0.1, 0.5, 0.3])), atol=1e-6)
    assert out["answers"]["c"]["type"] == "choice"
    assert out["usage"]["rounds"] == 1 and len(calls) == 1


def test_many_options_knockout():
    r, _ = _fake_reef()
    values = [(i * 37) % 151 for i in range(151)]  # a shuffled 0..150
    crit = [f"v{v}" for v in values]
    out = r.decide("short state", {"intent": {"type": "choice", "criteria": crit}})
    a = out["answers"]["intent"]
    assert a["choice"] == "v150"
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert out["usage"]["rounds"] >= 2
    probs = a["probabilities"]
    assert probs["v150"] == max(probs.values())


def test_split_options_is_even(tokenizer):
    p = Packer(tokenizer, state_len=8, schema_len=384)
    q = normalize_question("c", {"type": "choice", "criteria": [f"o{i}" for i in range(9)]})
    assert [len(c) for c in p.split_options(q, list(range(9)))] == [5, 4]
    q = normalize_question("c", {"type": "choice", "criteria": [f"o{i}" for i in range(77)]})
    sizes = [len(c) for c in p.split_options(q, list(range(77)))]
    assert len(sizes) == 10 and max(sizes) - min(sizes) <= 1


def test_knockout_winner_is_the_final_round_winner():
    # Nine options: the best (v9) must win even though the weakest option used to sit alone in an
    # 8 + 1 split and pass round one unopposed with probability 1.
    r, _ = _fake_reef(schema_len=96)
    crit = ["v2", "v3", "v9", "v4", "v5", "v6", "v7", "v8", "v1"]
    a = r.decide("short state", {"c": {"type": "choice", "criteria": crit}})["answers"]["c"]
    assert a["choice"] == "v9"
    probs = a["probabilities"]
    assert probs["v9"] == max(probs.values())
    assert probs["v1"] == min(probs.values())


def test_long_state_runs_every_chunk_and_averages():
    r, calls = _fake_reef()
    long_state = " ".join(f"w{i}" for i in range(150))
    out = r.decide(long_state, {"q": {"type": "noul", "criteria": {"true": "v9", "false": "v1"}}})
    assert out["usage"]["state_windows"] > 1 and not out["usage"]["state_truncated"]
    assert len(calls) == out["usage"]["state_windows"]
    assert out["answers"]["q"]["noul"] == pytest.approx(softmax(np.array([0.9, 0.1]))[0], abs=1e-5)


def test_strict_refuses_a_state_it_cannot_cover():
    r, _ = _fake_reef(max_state_windows=2, strict=True)
    with pytest.raises(CapacityError):
        r.decide(" ".join(f"w{i}" for i in range(300)), {"q": {"type": "noul"}})


class _Boom:
    name, device = "boom", "cpu"

    def __init__(self, exc):
        self.exc = exc

    def systemone(self, body):
        raise self.exc


@pytest.mark.parametrize("exc,code", [(CapacityError("over Reef's maximum context length"), 413),
                                      (AttributeError("x"), 400), (RuntimeError("bug"), 500)])
def test_server_always_answers(exc, code):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_Boom(exc)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone",
                                     data=json.dumps({"state": 1, "questions": {}}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == code and "error" in json.loads(e.value.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def _count(t):
    return len(t.split())


def test_hoist_single_question_keeps_the_trailing_question():
    story = " ".join(["once upon a time"] * 50)
    qs = normalize_questions({"q": {"type": "choice", "instructions": f"{story}\n\nWho did it?",
                                    "criteria": ["the butler", "the cook"]}})
    out, state = hoist_instructions(qs, "none", _count, cap=20)
    assert out[0].instructions == "Who did it?" and state == story
    same, _ = hoist_instructions(qs, "none", _count, cap=1000)
    assert same is qs  # fits: untouched
    one = normalize_questions({"q": {"type": "noul", "instructions": story}})
    out, state = hoist_instructions(one, "a: 1", _count, cap=20)
    assert out[0].instructions == HOIST_STUB and state == story + "\n\na: 1"


def test_hoist_shared_prefix_of_several_questions():
    task = " ".join(["mark each tool yes if it should be called"] * 5)
    qs = normalize_questions({f"t{i}": {"type": "noul", "instructions": f"{task}\n\nCandidate tool: t{i}"}
                              for i in range(3)})
    out, state = hoist_instructions(qs, "user: book a table", _count, cap=10)
    assert [q.instructions for q in out] == [f"Candidate tool: t{i}" for i in range(3)]
    assert state.startswith(task) and state.endswith("user: book a table")


def test_hoist_empty_state_moves_the_input():
    text = "Classify the banking intent of this user request:\nWhere is my card?"
    one = normalize_questions({"q": {"type": "choice", "instructions": text, "criteria": ["a", "b"]}})
    same, state = hoist_instructions(one, "none", _count, cap=100)
    assert same is one and state == "none"  # fits, and the model was not trained for this
    out, state = hoist_instructions(one, "none", _count, cap=100, empty=True)
    assert out[0].instructions == HOIST_STUB and state == text
    kept, state = hoist_instructions(one, "user: hi", _count, cap=100, empty=True)
    assert kept is one and state == "user: hi"  # a real state: nothing moves
    # several questions: the shared lead moves; with nothing shared, each gets its own state
    qs = normalize_questions({f"t{i}": {"type": "noul", "instructions": f"Request: book a table\n\nTool: t{i}"}
                              for i in range(2)})
    groups = hoist_groups(qs, "none", _count, cap=100, empty=True)
    assert len(groups) == 1 and groups[0][2] == "Request: book a table"
    qs = normalize_questions({"a": {"type": "noul", "instructions": "is x red?"},
                              "b": {"type": "noul", "instructions": "is y blue?"}})
    groups = hoist_groups(qs, "none", _count, cap=100, empty=True)
    assert [g[2] for g in groups] == ["is x red?", "is y blue?"]
    assert all(g[1][0].instructions == HOIST_STUB for g in groups)


def test_opaque_option_keys_are_dropped(tokenizer):
    q = normalize_question("q", {"type": "choice", "criteria": {"option_0": "card arrival", "B": "pin blocked",
                                                                "escalate": "page someone"}})
    assert [o.bare for o in q.options] == ["card arrival", "pin blocked", None]
    old, new = Packer(tokenizer), Packer(tokenizer, drop_opaque_keys=True)
    first = lambda p, i: p.block(q).ids[p.block(q).option_offsets[i] + 1]  # noqa: E731
    ids = lambda w: tokenizer.encode(w).ids[0]  # noqa: E731
    assert first(old, 0) == ids("option") and first(new, 0) == ids("card")
    assert first(new, 2) == ids("escalate:")  # a meaningful key stays


def test_training_subsamples_many_options_around_the_favourite(tokenizer):
    p = Packer(tokenizer, state_len=32, schema_len=128)
    crit = {f"i{k}": f"intent {k}" for k in range(77)}
    gold = {"q": {"type": "choice", "probabilities": {f"i{k}": (0.9 if k == 41 else 0.1 / 76) for k in range(77)}}}
    case = Case("c1", "BANKING77", "none", {"q": {"type": "choice", "instructions": "intent?", "criteria": crit}}, gold)
    wb = build_windows([case], p)
    assert wb.opt_mask[0, 0].sum() == MAX_PART_OPTIONS
    assert wb.gold[0, 0].max() == pytest.approx(0.9 / (0.9 + 7 * 0.1 / 76))
    assert len(build_windows([case], p, __import__("random").Random(0))) == 1


def test_runtime_hoists_long_instructions_into_the_state():
    r, calls = _fake_reef(hoist=True)
    story = " ".join(f"w{i}" for i in range(150))
    out = r.decide({}, {"q": {"type": "choice", "instructions": f"{story}\n\nWhich one?", "criteria": ["v2", "v7"]}})
    assert out["usage"]["state_windows"] > 1 and out["answers"]["q"]["choice"] == "v7"


def test_runtime_answers_long_per_question_texts_one_by_one():
    # ToolRet-style: a shared task, then one long candidate document per question.
    r, calls = _fake_reef(hoist=True, strict=True)
    task = "Is this document relevant to the query?"
    docs = {f"d{i}": " ".join(f"x{i}w{k}" for k in range(60)) for i in range(3)}
    qs = {k: {"type": "choice", "instructions": f"{task}\n\n{doc}", "criteria": ["v2", "v7"]}
          for k, doc in docs.items()}
    qs["short"] = {"type": "choice", "instructions": task, "criteria": ["v8", "v1"]}
    out = r.decide({}, qs)
    assert out["usage"]["groups"] == 4
    assert [out["answers"][k]["choice"] for k in docs] == ["v7"] * 3 and out["answers"]["short"]["choice"] == "v8"
    assert list(out["answers"]) == [*docs, "short"]  # the request's order is kept


def test_training_gives_long_per_question_texts_their_own_state(tokenizer):
    p = Packer(tokenizer, state_len=64, schema_len=64, max_instruction_tokens=16, hoist=True)
    doc = " ".join(["the cat sat on the mat"] * 8)
    qs = {f"d{i}": {"type": "noul", "instructions": f"Relevant?\n\n{doc} {i}"} for i in range(2)}
    gold = {k: {"type": "noul", "probabilities": {"true": 0.8, "false": 0.2}} for k in qs}
    wb = build_windows([Case("c", "toolret", "query: cats", qs, gold)], p)
    assert len(wb) == 2 and [m[0][1].instructions for m in wb.meta] == ["Relevant?", "Relevant?"]
    one = normalize_questions({"q": {"type": "noul", "instructions": "Relevant?\n\n" + doc}})
    out, state = hoist_instructions(one, "query: cats", _count, cap=4)
    assert out[0].instructions == "Relevant?" and state == doc + "\n\nquery: cats"


def test_training_subsample_does_not_put_the_favourite_first(tokenizer):
    # Regression: the gold favourite used to be the first option of every subsampled question.
    import random

    from reef.data import _fit_options

    p = Packer(tokenizer, state_len=32, schema_len=128)
    q = normalize_questions({"q": {"type": "choice", "criteria": {f"i{k}": f"intent {k}" for k in range(77)}}})[0]
    firsts = set()
    for seed in range(40):
        gold = {"probabilities": {"i50": 1.0}}
        sub = _fit_options(q, gold, p, random.Random(seed))
        assert "i50" in [o.id for o in sub.options]
        firsts.add(sub.options[0].id == "i50")
        assert [int(o.id[1:]) for o in sub.options] == sorted(int(o.id[1:]) for o in sub.options)
    assert firsts == {False, True} or firsts == {False}
