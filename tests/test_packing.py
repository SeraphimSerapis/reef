import numpy as np
import pytest

from reef.packing import (
    BOUNDARY,
    OPT,
    PAD,
    Q_CHOICE,
    Q_NOUL,
    Q_SCORE,
    STATE,
    Packer,
    normalize_question,
    normalize_questions,
)


def test_normalize_types(questions):
    qs = normalize_questions(questions)
    assert [q.type for q in qs] == ["noul", "choice", "score"]
    noul, choice, score = qs
    assert [o.id for o in noul.options] == ["true", "false"]
    assert [o.id for o in choice.options] == ["billing", "tech_support"]
    assert choice.options[1].text == "tech support: something is broken"
    assert [o.id for o in score.options] == ["0", "1", "2", "3", "4"]
    assert [o.level for o in score.options] == [1, 2, 3, 4, 5]
    assert all(o.level == 0 for o in choice.options)


def test_score_levels_bucket_long_rubrics():
    q = normalize_question("s", {"type": "score", "criteria": [str(i) for i in range(11)]})
    levels = [o.level for o in q.options]
    assert levels[0] == 1 and levels[-1] == 5 and levels == sorted(levels)


def test_instructions_default_to_humanized_id():
    q = normalize_question("needs_refund", {"type": "noul"})
    assert q.instructions == "needs refund"


@pytest.mark.parametrize(
    "q",
    [
        {"type": "nope"},
        {"type": "choice"},
        {"type": "choice", "criteria": ["only_one"]},
        {"type": "score", "criteria": ["one"]},
    ],
)
def test_invalid_questions(q):
    with pytest.raises(ValueError):
        normalize_question("q", q)


def test_window_layout(tokenizer, questions):
    p = Packer(tokenizer, state_len=64, schema_len=128)
    state = p.encode("the printer is on fire")
    (w,) = p.pack(state, normalize_questions(questions))
    assert w.ids.shape == w.roles.shape == w.levels.shape == (192,)

    # State region: [CLS] state [SEP] tiled cyclically, so it has no padding.
    unit = [p.cls_id, *state, p.sep_id]
    assert list(w.ids[:64]) == (unit * 10)[:64]
    assert set(w.roles[:64]) == {BOUNDARY, STATE}
    assert not w.state_truncated

    # Schema region: question markers, then option markers at the slot positions.
    assert [w.roles[64 + 0]] == [Q_NOUL]
    q_roles = [r for r in w.roles[64:] if r in (Q_NOUL, Q_CHOICE, Q_SCORE)]
    assert q_roles == [Q_NOUL, Q_CHOICE, Q_SCORE]
    assert [qi for qi, _ in w.slots] == [0, 1, 2]
    assert [len(pos) for _, pos in w.slots] == [2, 2, 5]
    for _, positions in w.slots:
        assert all(w.roles[i] == OPT for i in positions)
        assert all(w.ids[i] == p.marker_id for i in positions)
    # Trailing positions are padding.
    assert w.roles[-1] == PAD


def test_long_state_is_truncated(tokenizer, questions):
    p = Packer(tokenizer, state_len=32, schema_len=128)
    state = p.encode(" ".join(f"w{i}" for i in range(100)))
    (w,) = p.pack(state, normalize_questions(questions))
    assert w.state_truncated
    assert w.ids[0] == p.cls_id and w.ids[31] == p.sep_id


def test_schema_overflow_spills_into_more_windows(tokenizer, questions):
    p = Packer(tokenizer, state_len=32, schema_len=32)
    qs = normalize_questions(questions)
    windows = p.pack(p.encode("hi"), qs)
    assert len(windows) >= 2
    used = [sum(len(p.block(qs[qi])) for qi, _ in w.slots) for w in windows]
    assert all(u <= 32 for u in used)
    assert sorted(qi for w in windows for qi, _ in w.slots) == [0, 1, 2]
    for w in windows:  # every window carries the same state region
        np.testing.assert_array_equal(w.ids[:32], windows[0].ids[:32])


def test_block_too_long_raises(tokenizer):
    p = Packer(tokenizer, state_len=32, schema_len=8)
    with pytest.raises(ValueError):
        p.pack([], normalize_questions({"x": {"type": "score", "criteria": ["a b c"] * 4}}))


def test_banded_layout_one_question_per_band(tokenizer, questions):
    p = Packer(tokenizer, state_len=64, schema_len=128, bands=2)
    qs = normalize_questions(questions)
    windows = p.pack(p.encode("disk sda failed"), qs)
    assert len(windows) == -(-len(qs) // 2)
    for w in windows:
        for j, (_, positions) in enumerate(w.slots):
            # every question starts at its band's fixed offset and stays inside its band
            start = 64 + j * 64
            assert w.roles[start] in (Q_NOUL, Q_CHOICE, Q_SCORE)
            assert all(start < pos < start + 64 for pos in positions)


def test_banded_block_limit_is_the_band(tokenizer):
    from reef.packing import CapacityError

    p = Packer(tokenizer, state_len=32, schema_len=64, bands=2, max_instruction_tokens=60)
    q = normalize_questions({"q": {"type": "noul", "instructions": "word " * 50}})[0]
    assert not p.block_fits(q)
    with pytest.raises(CapacityError):
        p.block(q)
