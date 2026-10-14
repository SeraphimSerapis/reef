import json

from reef import packs
from reef.distill import make_cases, merge


def _write(path, cases, probs_for):
    with open(path, "w") as f:
        for c in cases:
            gold = {}
            for qid, q in c["questions"].items():
                p = probs_for(q)
                gold[qid] = {"type": q["type"], "probabilities": p, "label": max(p, key=p.get)}
            f.write(json.dumps({**c, "gold": gold, "teacher": path.stem}) + "\n")


def test_merge_averages_teachers(tmp_path):
    cases = make_cases(packs.get("notify-gate"), 5, seed=3)
    noul_only = [{**c, "questions": {k: v for k, v in c["questions"].items() if v["type"] == "noul"}} for c in cases]
    _write(tmp_path / "a.jsonl", noul_only, lambda q: {"true": 0.9, "false": 0.1})
    _write(tmp_path / "b.jsonl", noul_only[1:], lambda q: {"true": 0.3, "false": 0.7})
    stats = merge([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], tmp_path / "m.jsonl")
    rows = [json.loads(line) for line in open(tmp_path / "m.jsonl")]
    assert stats["cases"] == 4 == len(rows)  # only ids present in both
    assert stats["teachers_agree"] == 0.0
    g = next(iter(rows[0]["gold"].values()))
    assert abs(g["probabilities"]["true"] - 0.6) < 1e-9 and g["label"] == "true"
    assert rows[0]["teacher"] == "a + b"


def test_dataset_roundtrip(tmp_path):
    import pytest

    pytest.importorskip("pandas")
    from reef.data import load_cases
    from reef.hub import push_dataset

    cases = make_cases(packs.get("alert-triage"), 4, seed=5)
    _write(tmp_path / "a.jsonl", cases, lambda q: _uniform(q))
    _write(tmp_path / "b.jsonl", cases, lambda q: _uniform(q))
    merge([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], tmp_path / "m.jsonl")
    out = push_dataset("me/ds", {"train": [tmp_path / "m.jsonl"], "test": [tmp_path / "m.jsonl"]},
                       out_dir=tmp_path / "ds", upload=False)
    card = (tmp_path / "ds" / "README.md").read_text()
    assert "a, b" in card and "Teacher agreement" in card and "1.00 (" in card
    back = load_cases("test", path=f"{out}/all/test-00000-of-00001.parquet", decode_state=True)
    assert [c.state for c in back] == [c["state"] for c in cases]


def _uniform(q):
    from reef.packing import normalize_questions

    opts = [o.id for o in normalize_questions({"q": q})[0].options]
    return {o: 1 / len(opts) for o in opts}


def test_load_jsonl_keeps_unicode_line_separators(tmp_path):
    # U+2028 / U+0085 are line breaks for str.splitlines() but legal inside JSON strings.
    from reef.data import load_jsonl

    row = {"id": "a", "workflow": "w", "state": "x\u2028y\x85z",
           "questions": {"q": {"type": "noul"}}, "gold": {"q": {"type": "noul", "noul": 0.5}}}
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_jsonl(p)[0].state == "x\u2028y\x85z"
