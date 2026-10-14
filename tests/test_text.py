import random

from reef.text import humanize, render_state


def test_humanize():
    assert humanize("days_until_due") == "days until due"
    assert humanize("first-name") == "first name"


def test_render_plain_text_passthrough():
    assert render_state("  hello world ") == "hello world"
    assert render_state("{not json") == "{not json"


def test_render_json_flattens_paths():
    s = {"invoice": {"amount_due": 120.0, "paid": False, "tags": ["a", "b"]}, "note": None}
    assert render_state(s) == "invoice amount due: 120 ; invoice paid: false ; invoice tags: a, b ; note: none"


def test_render_json_string_equals_dict():
    assert render_state('{"a": 1, "b": {"c": 2}}') == render_state({"a": 1, "b": {"c": 2}})


def test_render_messages_as_dialogue():
    s = {"thread": [{"role": "customer", "text": "it broke"}, {"role": "agent", "text": "sorry"}]}
    assert render_state(s) == "customer: it broke ; agent: sorry"


def test_shuffle_keeps_fields():
    s = {f"k{i}": i for i in range(10)}
    plain = set(render_state(s).split(" ; "))
    shuffled = render_state(s, random.Random(0))
    assert set(shuffled.split(" ; ")) == plain
    assert shuffled != render_state(s)


def test_float_formatting():
    assert render_state({"x": 0.12345}) == "x: 0.123"
    assert render_state({"x": 1e-5}) == "x: 1e-05"
