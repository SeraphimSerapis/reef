"""Packs: a homelab decision domain = question specs + a state generator.

A pack's questions are sampled per case (a random subset, a random phrasing per question, and
sometimes a dropped option) so the student learns to *read* the schema rather than memorise one.
Users can take a pack's canonical questions (``Pack.canonical_questions()``) and edit them freely.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionSpec:
    id: str
    type: str  # noul | choice | score
    phrasings: list[str]
    criteria: Any = None  # dict for choice, list for score, optional dict for noul
    droppable: bool = True  # choice only: may drop one option at sampling time

    def canonical(self) -> dict:
        q = {"type": self.type, "instructions": self.phrasings[0]}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return q

    def sample(self, rng: random.Random) -> dict:
        q = {"type": self.type, "instructions": rng.choice(self.phrasings)}
        crit = self.criteria
        if self.type == "choice" and isinstance(crit, dict) and self.droppable and len(crit) >= 4 and rng.random() < 0.3:
            drop = rng.choice(list(crit))
            crit = {k: v for k, v in crit.items() if k != drop}
        if crit is not None:
            q["criteria"] = crit
        return q


@dataclass
class Pack:
    name: str
    description: str
    questions: list[QuestionSpec]
    generate: Callable[[random.Random], Any]
    min_questions: int = 2
    tags: list[str] = field(default_factory=list)
    # Glossary + default house policy shown to the *teacher* only (appended to its system prompt).
    # Vendor codes like Protect's ``alrmSpeak`` mean nothing to a 4B LLM without it.
    teacher_notes: str = ""

    def canonical_questions(self) -> dict:
        return {q.id: q.canonical() for q in self.questions}

    def sample_questions(self, rng: random.Random) -> dict:
        k = rng.randint(min(self.min_questions, len(self.questions)), len(self.questions))
        chosen = rng.sample(self.questions, k)
        return {q.id: q.sample(rng) for q in chosen}


# -- helpers shared by generators -------------------------------------------------------------
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def clock(rng: random.Random, night_bias: float = 0.25) -> tuple[str, int]:
    """A local time string like 'Sat 03:12' and its hour. Night hours are over-sampled a bit
    because that is where most of the interesting decisions are."""
    if rng.random() < night_bias:
        h = rng.choice([22, 23, 0, 1, 2, 3, 4, 5])
    else:
        h = rng.randint(0, 23)
    return f"{rng.choice(WEEKDAYS)} {h:02d}:{rng.randint(0, 59):02d}", h


def pick(rng: random.Random, weighted: dict[Any, float]) -> Any:
    items = list(weighted)
    return rng.choices(items, weights=[weighted[i] for i in items])[0]


def maybe(rng: random.Random, p: float, value: Any, default: Any = None) -> Any:
    return value if rng.random() < p else default


def human_duration(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d"
