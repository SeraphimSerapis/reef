"""Turning a decision ``state`` into text.

States are often JSON (an invoice, an alert, a ticket thread). Fed to a WordPiece tokenizer as-is,
roughly a third of the tokens are quotes, braces and commas. Reef renders JSON as compact
``path: value`` fields instead, which roughly halves the token count and gives the encoder
"key: value" pairs it can read in a short receptive field.
"""

from __future__ import annotations

import json
import random
from typing import Any

FIELD_SEP = " ; "


def humanize(key: str) -> str:
    """``days_until_due`` -> ``days until due``."""
    return str(key).replace("_", " ").replace("-", " ").strip()


def _scalar(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{round(v, 3)}"
    return str(v).strip()


def _flatten(obj: Any, path: list[str], out: list[str], rng: random.Random | None) -> None:
    if isinstance(obj, dict):
        items = list(obj.items())
        if rng is not None:
            rng.shuffle(items)
        if not items:
            out.append(f"{' '.join(path)}: none" if path else "none")
        for k, v in items:
            _flatten(v, path + [humanize(k)], out, rng)
    elif isinstance(obj, list):
        if not obj:
            out.append(f"{' '.join(path)}: none")
        elif all(not isinstance(x, (dict, list)) for x in obj):
            out.append(f"{' '.join(path)}: {', '.join(_scalar(x) for x in obj)}")
        else:
            for i, x in enumerate(obj, 1):
                # A list of messages ({"role": ..., "text": ...}) reads better as "customer: ...".
                if isinstance(x, dict) and set(x) <= {"role", "text", "content", "author"} and (
                    "text" in x or "content" in x
                ):
                    who = humanize(x.get("role") or x.get("author") or "message")
                    out.append(f"{who}: {_scalar(x.get('text', x.get('content')))}")
                else:
                    _flatten(x, path + [str(i)], out, rng)
    else:
        out.append(f"{' '.join(path)}: {_scalar(obj)}" if path else _scalar(obj))


def render_state(state: Any, rng: random.Random | None = None) -> str:
    """Render a state (str, JSON string, dict or list) to model text.

    ``rng`` shuffles JSON key order. It is used as training-time augmentation, since key order
    carries no meaning.
    """
    if isinstance(state, str):
        s = state.strip()
        if s[:1] in "{[":
            try:
                state = json.loads(s)
            except json.JSONDecodeError:
                return s
        else:
            return s
    if isinstance(state, (dict, list)):
        out: list[str] = []
        _flatten(state, [], out, rng)
        return FIELD_SEP.join(out)
    return _scalar(state)
