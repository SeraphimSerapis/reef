import json

import pytest

pd = pytest.importorskip("pandas")

from reef.data import load_open_jev  # noqa: E402
from reef.packing import normalize_questions  # noqa: E402

PRE = "Apply this local, explicitly stated workflow policy to the supplied facts."


def _rows():
    rows = []
    for s in range(4):
        state = json.dumps({"invoice": s})
        for a in ("PAY", "HOLD", "REJECT", "ESCALATE", "NOTE"):
            rows.append({"id": f"r{s}{a}", "source": "workflow", "kind": "noul",
                         "question": f"{PRE}\n\nRule: {a.lower()} rule.\n\nShould action {a} be selected now?",
                         "options": ["no", "yes"], "target": [0.25, 0.75], "state_json": state})
    rows.append({"id": "c1", "source": "game", "kind": "choice", "question": "Pick a move.",
                 "options": ["left: go left", "right: go right"], "target": [0.0, 1.0], "state_json": '"board"'})
    rows.append({"id": "c2", "source": "game", "kind": "choice", "question": "Too many options.",
                 "options": [f"o{i}" for i in range(9)], "target": [1.0] + [0.0] * 8, "state_json": '"big"'})
    rows.append({"id": "s1", "source": "skip-me", "kind": "score", "question": "How bad?",
                 "options": ["low", "mid", "high"], "target": [0.0, 0.5, 0.5], "state_json": '"x"'})
    return pd.DataFrame(rows)


def test_open_jev_conversion(tmp_path):
    p = tmp_path / "train.parquet"
    _rows().to_parquet(p)
    cases = load_open_jev("train", path=str(p), exclude=("skip-me",), max_questions=4)
    wf = [c for c in cases if c.workflow == "open-jev/workflow"]
    assert len(wf) == 8  # 4 states x 5 questions, chunked by 4
    q = wf[0].questions["q0"]["instructions"]
    assert q.startswith("Should action") and PRE not in q and "Rule:" in q
    assert wf[0].gold["q0"]["probabilities"] == {"true": 0.75, "false": 0.25}
    game = [c for c in cases if c.workflow == "open-jev/game"]
    assert len(game) == 1 and game[0].gold["q0"]["label"] == "right"  # 9-option question skipped
    assert not any(c.workflow == "open-jev/skip-me" for c in cases)
    for c in cases:
        normalize_questions(c.questions)


def test_open_jev_caps_keep_states_whole(tmp_path):
    p = tmp_path / "train.parquet"
    _rows().to_parquet(p)
    cases = load_open_jev("train", path=str(p), caps={"workflow": 6}, exclude=())
    states = {json.dumps(c.state) for c in cases if c.workflow == "open-jev/workflow"}
    assert len(states) == 2  # 5 rows per state: the cap of 6 rows takes two whole states
