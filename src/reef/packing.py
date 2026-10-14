"""Packing a (state, questions) request into fixed-length token windows.

The Edge TPU needs static shapes, so every forward pass sees exactly ``state_len + schema_len``
positions. A window has two regions::

    ├──────────── state region (state_len) ────────────┤├────────── schema region (schema_len) ──────────┤
    [CLS] state ... [SEP] [CLS] state ... [SEP] [CLS] st ⟨Q⟩ instructions ⟨O⟩ option ⟨O⟩ option ⟨Q⟩ ... [PAD]

The state is **repeated cyclically** to fill its region. On the device the encoder summarises the
state with a plain mean over the region, and with cyclic filling that mean equals the mean over
the state itself. No mask is needed, and masks are a problem on the Edge TPU (an
activation×activation multiply crashes its compiler).

``⟨Q⟩`` and ``⟨O⟩`` are marker positions. They carry no word embedding, only a learned *role*
embedding (plus a *level* embedding for ordered ``score`` options). The model emits one logit per
position, and the logits at a question's ``⟨O⟩`` markers, softmaxed together, are its answer
distribution. This is the "joint schema" idea behind Clef and Laya, scaled down to a CNN.

If the schema does not fit in one schema region, questions are spread over several windows that
share the same state region. Question blocks depend only on the schema, so they are tokenised
once and cached.

The runtime goes further for requests that do not fit one window at all:

* **Long states** are cut into overlapping chunks (:meth:`Packer.state_regions`); every schema
  window runs once per chunk and the option logits are averaged over chunks.
* **Questions with many options** (more than :data:`MAX_PART_OPTIONS`, or too long for one
  question's block) are split into near-equal parts (:meth:`Packer.split_options`) and answered as
  a knockout: each part's winner goes through to the next round, and the final-round winner wins.
  A knocked-out option scores its part winner's probability times its ratio to that winner.
  Every comparison happens inside one window, as in training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from reef.text import humanize

# Roles: what kind of position this is.
PAD, BOUNDARY, STATE, Q_NOUL, Q_CHOICE, Q_SCORE, Q_TEXT, OPT, OPT_TEXT = range(9)
NUM_ROLES = 9
Q_ROLE = {"noul": Q_NOUL, "choice": Q_CHOICE, "score": Q_SCORE}
# Levels: 0 = not an ordered option, 1..5 = relative position on a score rubric (bucketed).
SCORE_BUCKETS = 5
NUM_LEVELS = SCORE_BUCKETS + 1
# Options compared within one window at inference; training never sees more (data.MAX_OPTIONS).
MAX_PART_OPTIONS = 8

DEFAULT_NOUL = {"true": "yes, the statement is true", "false": "no, the statement is false"}


@dataclass(frozen=True)
class Option:
    id: str
    text: str
    level: int = 0
    # The description alone, when the key carries no meaning ("A", "3", "option_12"). Packers of
    # models trained with ``drop_opaque_keys`` render just this: an opaque key between the option
    # marker and its text only pushes the words that matter away from the marker.
    bare: str | None = None


_OPAQUE_KEY = re.compile(r"(?i)^(?:[a-z]|\d+|(?:option|opt|choice|answer|candidate|label|class|item)[ _-]?\d+)$")


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    options: tuple[Option, ...]


def _level(i: int, n: int) -> int:
    if n <= 1:
        return 1
    return 1 + round(i / (n - 1) * (SCORE_BUCKETS - 1))


def normalize_question(qid: str, q: dict[str, Any]) -> Question:
    """Validate one SystemOne question and turn it into a list of options with readable text."""
    qtype = q.get("type")
    if qtype not in Q_ROLE:
        raise ValueError(f"question {qid!r}: type must be noul, choice or score, got {qtype!r}")
    instructions = q.get("instructions") or humanize(qid)
    if not isinstance(instructions, str):  # SystemOne allows structured instructions
        from reef.text import render_state

        instructions = render_state(instructions)
    instructions = instructions.strip()
    criteria = q.get("criteria")
    if qtype == "noul":
        crit = dict(DEFAULT_NOUL)
        if isinstance(criteria, dict):
            crit.update({str(k).lower(): v for k, v in criteria.items()})
        opts = tuple(Option(k, f"{k}: {crit[k]}") for k in ("true", "false"))
    elif qtype == "choice":
        if isinstance(criteria, dict):
            items = [(str(k), str(v)) for k, v in criteria.items()]
        elif isinstance(criteria, list):
            items = [(str(k), "") for k in criteria]
        else:
            raise ValueError(f"choice question {qid!r} needs criteria")
        if len(items) < 2:
            raise ValueError(f"choice question {qid!r} needs at least two options")
        opts = tuple(Option(k, f"{humanize(k)}: {v}" if v else humanize(k),
                            bare=v if v and _OPAQUE_KEY.match(k) else None) for k, v in items)
    else:  # score
        if isinstance(criteria, list):
            texts = [str(c) for c in criteria]
        elif isinstance(criteria, dict):  # {"0": "...", "1": "..."}
            texts = [str(criteria[k]) for k in sorted(criteria, key=lambda s: int(s))]
        else:
            raise ValueError(f"score question {qid!r} needs a criteria list")
        if len(texts) < 2:
            raise ValueError(f"score question {qid!r} needs at least two levels")
        n = len(texts)
        opts = tuple(Option(str(i), t, _level(i, n)) for i, t in enumerate(texts))
    return Question(qid, qtype, instructions, opts)


def normalize_questions(questions: dict[str, dict]) -> list[Question]:
    if not questions:
        raise ValueError("questions must be a non-empty mapping")
    return [normalize_question(k, v) for k, v in questions.items()]


@dataclass
class Block:
    """The token span for one question, independent of the state."""

    ids: list[int]
    roles: list[int]
    levels: list[int]
    option_offsets: list[int]  # offsets of the ⟨O⟩ markers within the block

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Window:
    ids: np.ndarray  # int32 [L]
    roles: np.ndarray  # int32 [L]
    levels: np.ndarray  # int32 [L]
    # (question index in the request, absolute positions of its option markers)
    slots: list[tuple[int, list[int]]] = field(default_factory=list)
    state_truncated: bool = False


class CapacityError(ValueError):
    """A request Reef cannot read in full. The wording ("maximum context length") is what SystemOne
    clients and benchmark harnesses recognise as a capacity refusal rather than a failure."""


class Packer:
    def __init__(
        self,
        tokenizer,
        state_len: int = 384,
        schema_len: int = 384,
        max_instruction_tokens: int = 40,
        max_option_tokens: int = 28,
        strict: bool = False,
        hoist: bool = False,
        bands: int = 0,
        hoist_empty: bool = False,
        drop_opaque_keys: bool = False,
    ):
        """``strict``: never truncate instructions or option text; raise :class:`CapacityError`
        when they are longer than the limits (training packs leniently and truncates).
        ``hoist``: move over-long instructions into the state (:func:`hoist_instructions`).
        ``bands``: split the schema region into this many fixed bands of ``schema_len // bands``
        positions, one question block per band (models with per-question summaries; 0 = blocks
        packed back to back).
        ``hoist_empty``: with ``hoist``, also move the input of empty-state requests into the state.
        ``drop_opaque_keys``: render options whose keys carry no meaning (``Option.bare``) as
        their description alone."""
        self.tok = tokenizer
        self.state_len = state_len
        self.schema_len = schema_len
        self.max_instruction_tokens = max_instruction_tokens
        self.max_option_tokens = max_option_tokens
        self.strict = strict
        self.hoist = hoist
        self.hoist_empty = hoist and hoist_empty
        self.drop_opaque_keys = drop_opaque_keys
        self.bands = bands
        if bands and schema_len % bands:
            raise ValueError(f"schema_len={schema_len} is not divisible into {bands} bands")
        self.block_len = schema_len // bands if bands else schema_len  # positions one block may use
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.marker_id = tokenizer.token_to_id("[PAD]") or 0
        self._block = lru_cache(maxsize=4096)(self._build_block)

    @property
    def seq_len(self) -> int:
        return self.state_len + self.schema_len

    # -- tokenisation ---------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def prepare(self, questions: list[Question], state_text: str, force_hoist: bool = False
                ) -> tuple[list[Question], list[int]]:
        """Hoist over-long instructions (if enabled) and tokenise the state."""
        if self.hoist:
            questions, state_text = hoist_instructions(questions, state_text, lambda t: len(self.encode(t)),
                                                       self.max_instruction_tokens, force_hoist, self.hoist_empty)
        return questions, self.encode(state_text)

    def prepare_groups(self, questions: list[Question], state_text: str, force_hoist: bool = False
                       ) -> list[tuple[list[int], list[Question], list[int]]]:
        """Like :meth:`prepare`, but questions with long texts of their own get their own state
        (:func:`hoist_groups`). Returns ``(question indices, questions, state ids)`` groups."""
        if not self.hoist:
            return [(list(range(len(questions))), questions, self.encode(state_text))]
        groups = hoist_groups(questions, state_text, lambda t: len(self.encode(t)),
                              self.max_instruction_tokens, force_hoist, self.hoist_empty)
        return [(idx, qs, self.encode(st)) for idx, qs, st in groups]

    def _clip(self, ids: list[int], limit: int, what: str) -> list[int]:
        if self.strict and len(ids) > limit:
            raise CapacityError(f"{what} has {len(ids)} tokens, over Reef's maximum context length of {limit}")
        return ids[:limit]

    def _build_block(self, q: Question) -> Block:
        ids, roles, levels, offs = [self.marker_id], [Q_ROLE[q.type]], [0], []
        instr = self._clip(self.encode(q.instructions), self.max_instruction_tokens, f"question {q.id!r}")
        ids += instr
        roles += [Q_TEXT] * len(instr)
        levels += [0] * len(instr)
        for o in q.options:
            offs.append(len(ids))
            text = o.bare if self.drop_opaque_keys and o.bare else o.text
            t = self._clip(self.encode(text), self.max_option_tokens, f"option {o.id!r} of {q.id!r}")
            ids += [self.marker_id] + t
            roles += [OPT] + [OPT_TEXT] * len(t)
            levels += [o.level] * (1 + len(t))
        return Block(ids, roles, levels, offs)

    def block(self, q: Question) -> Block:
        b = self._block(q)
        if len(b) > self.block_len:
            raise CapacityError(
                f"question {q.id!r} needs {len(b)} positions, over Reef's maximum context length of {self.block_len}"
            )
        return b

    def block_fits(self, q: Question) -> bool:
        return len(self._block(q)) <= self.block_len

    def split_options(self, q: Question, idx: list[int], max_options: int = MAX_PART_OPTIONS) -> list[list[int]]:
        """Split the options ``idx`` of ``q`` into consecutive chunks of at most ``max_options`` whose
        blocks (instructions repeated) each fit one question's block.

        Parts are kept as even as possible (9 options give 5 + 4, not 8 + 1): an option alone in a
        tiny part would go through a knockout round unopposed. The greedy split gives the number
        of parts; if an even split with that many parts does not fit, the greedy split is used.
        """
        chunks: list[list[int]] = [[]]
        for i in idx:
            cand = chunks[-1] + [i]
            if chunks[-1] and (len(cand) > max_options or not self.block_fits(subquestion(q, cand))):
                chunks.append([i])
            else:
                chunks[-1] = cand
        k = len(chunks)
        if k < 2:
            return chunks
        base, extra = divmod(len(idx), k)
        even, start = [], 0
        for j in range(k):
            size = base + (j < extra)
            even.append(list(idx[start:start + size]))
            start += size
        if all(self.block_fits(subquestion(q, c)) for c in even):
            return even
        return chunks

    # -- packing --------------------------------------------------------------------------------
    def state_region(self, state_ids: list[int]) -> tuple[np.ndarray, np.ndarray, bool]:
        """``[CLS] state [SEP]`` tiled to exactly ``state_len`` positions."""
        S = self.state_len
        truncated = len(state_ids) > S - 2
        unit_ids = [self.cls_id] + state_ids[: S - 2] + [self.sep_id]
        unit_roles = [BOUNDARY] + [STATE] * (len(unit_ids) - 2) + [BOUNDARY]
        reps = -(-S // len(unit_ids))
        ids = np.array((unit_ids * reps)[:S], np.int32)
        roles = np.array((unit_roles * reps)[:S], np.int32)
        return ids, roles, truncated

    def state_regions(self, state_ids: list[int], max_windows: int = 8, overlap: float = 0.25
                      ) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
        """State regions covering a long state with overlapping chunks.

        A state that fits gives one region. A longer one is cut into chunks of ``state_len - 2``
        tokens that overlap by ``overlap``; if that needs more than ``max_windows`` chunks, the
        chunks are spread evenly over the state instead and the second value (gaps) is True.
        """
        n, c = len(state_ids), self.state_len - 2
        if n <= c or max_windows <= 1:
            ids, roles, truncated = self.state_region(state_ids)
            return [(ids, roles)], truncated
        stride = max(1, int(c * (1 - overlap)))
        starts = sorted({min(s, n - c) for s in range(0, n - c + stride, stride)})
        gaps = len(starts) > max_windows
        if gaps:
            starts = sorted({int(round(s)) for s in np.linspace(0, n - c, max_windows)})
        return [self.state_region(state_ids[s : s + c])[:2] for s in starts], gaps

    def pack(self, state_ids: list[int], questions: list[Question]) -> list[Window]:
        """One state region (truncated if long) and every question in order. Used for training."""
        s_ids, s_roles, truncated = self.state_region(state_ids)
        return self.pack_blocks(s_ids, s_roles, [(qi, self.block(q)) for qi, q in enumerate(questions)],
                                truncated)

    def pack_blocks(self, s_ids, s_roles, blocks: list[tuple[Any, Block]], truncated: bool = False
                    ) -> list[Window]:
        """Lay keyed question blocks after a state region, starting a new window when the schema
        region is full (banded: when every band holds a block). Each window's ``slots`` hold
        ``(key, option positions)``."""
        if self.bands:
            groups = [list(range(i, min(i + self.bands, len(blocks)))) for i in range(0, len(blocks), self.bands)]
            return [self._window(s_ids, s_roles, [blocks[i] for i in g], truncated) for g in groups]
        groups: list[list[int]] = [[]]
        used = 0
        for i, (_, b) in enumerate(blocks):
            if used + len(b) > self.schema_len and groups[-1]:
                groups.append([])
                used = 0
            groups[-1].append(i)
            used += len(b)
        return [self._window(s_ids, s_roles, [blocks[i] for i in g], truncated) for g in groups]

    def _window(self, s_ids, s_roles, blocks: list[tuple[Any, Block]], truncated: bool) -> Window:
        L, S = self.seq_len, self.state_len
        ids = np.zeros(L, np.int32)
        roles = np.zeros(L, np.int32)
        levels = np.zeros(L, np.int32)
        ids[:S], roles[:S] = s_ids, s_roles
        pos = S
        slots = []
        for j, (key, b) in enumerate(blocks):
            if self.bands:  # band j starts at a fixed offset; the rest of the band stays PAD
                pos = S + j * self.block_len
            n = len(b)
            ids[pos : pos + n] = b.ids
            roles[pos : pos + n] = b.roles
            levels[pos : pos + n] = b.levels
            slots.append((key, [pos + o for o in b.option_offsets]))
            pos += n
        return Window(ids, roles, levels, slots, truncated)

    def with_state(self, w: Window, s_ids: np.ndarray, s_roles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``w``'s ids and roles with its state region replaced (the schema region is unchanged)."""
        ids, roles = w.ids.copy(), w.roles.copy()
        ids[: self.state_len], roles[: self.state_len] = s_ids, s_roles
        return ids, roles


def subquestion(q: Question, idx: list[int]) -> Question:
    """``q`` restricted to the options at ``idx`` (levels are kept from the full rubric)."""
    return Question(q.id, q.type, q.instructions, tuple(q.options[i] for i in idx))


HOIST_STUB = "Answer the question described in the state."


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text]


def _blank(state_text: str) -> bool:
    return state_text.strip() in ("", "none", ": none", "null")


def hoist_instructions(questions: list[Question], state_text: str, count, cap: int, force: bool = False,
                       empty: bool = False) -> tuple[list[Question], str]:
    """Move instruction text that is too long for a question block into the state.

    Many SystemOne clients put the whole input in the instructions (an empty state and a 1,000
    token story in the question). A question block holds ``cap`` instruction tokens, so:

    * one question: its leading paragraphs move to the state and the trailing paragraphs that fit
      (usually the actual question) stay; if the last paragraph alone is too long (a task followed
      by a long document), the leading paragraphs that fit stay and the rest moves; if nothing
      fits, a short stub replaces the instructions;
    * several questions: the paragraphs they all start with move to the state (a shared task
      description followed by a per-question line, e.g. "Candidate tool: x").

    With ``empty`` (models trained with ``hoist_empty_state``) and an empty state, the input
    moves even when it fits: a single question's whole instructions become the state (and the
    question gets the stub), several questions move their shared leading paragraphs. The state
    summary then carries the input to every option, as it does for ordinary states.

    The state can then be read in chunks like any long state. Training applies the same function
    (and ``force`` as augmentation), so the model learns the convention. ``count`` maps text to a
    token count.
    """
    if empty and questions and _blank(state_text):
        if len(questions) == 1:
            q = questions[0]
            return [Question(q.id, q.type, HOIST_STUB, q.options)], q.instructions
        force = True
    if not force and all(count(q.instructions) <= cap for q in questions):
        return questions, state_text
    paras = [_paragraphs(q.instructions) for q in questions]
    if len(questions) == 1:
        p, keep = paras[0], []
        for para in reversed(p):
            if count("\n\n".join([para, *keep])) > cap:
                break
            keep.insert(0, para)
        if len(keep) == len(p):
            if not force:
                return questions, state_text
            keep = keep[1:] if len(keep) > 1 else []
        hoisted, rest = p[: len(p) - len(keep)], ["\n\n".join(keep)]
        if not keep:  # e.g. a task paragraph followed by a long document: keep the leading task
            lead: list[str] = []
            for para in p[:-1]:
                if count("\n\n".join([*lead, para])) > cap:
                    break
                lead.append(para)
            if lead:
                hoisted, rest = p[len(lead):], ["\n\n".join(lead)]
    else:
        n = 0
        while all(len(pp) > n for pp in paras) and len({pp[n] for pp in paras}) == 1:
            n += 1
        if n == 0:
            return questions, state_text
        hoisted, rest = paras[0][:n], ["\n\n".join(pp[n:]) for pp in paras]
    text = "\n\n".join(hoisted)
    if state_text.strip() and state_text.strip() != "none":
        text += "\n\n" + state_text
    return [Question(q.id, q.type, r or HOIST_STUB, q.options) for q, r in zip(questions, rest)], text


def hoist_groups(questions: list[Question], state_text: str, count, cap: int, force: bool = False,
                 empty: bool = False) -> list[tuple[list[int], list[Question], str]]:
    """Plan hoisting as groups of ``(question indices, questions, state text)``.

    :func:`hoist_instructions` runs first. If several questions remain whose own instructions are
    still over ``cap`` (e.g. one long candidate document per question), each of them becomes its
    own group, hoisted on its own from its original instructions and state (so a task paragraph
    can stay in the question while its document moves to the state); the questions that fit stay
    together in one group. With ``empty`` and an empty state that stays empty (several questions
    share no leading paragraph), every question becomes its own group with its input as the state.
    """
    hq, hs = hoist_instructions(questions, state_text, count, cap, force, empty)
    if empty and len(hq) > 1 and _blank(state_text) and _blank(hs):
        return [([i], *hoist_instructions([questions[i]], state_text, count, cap, empty=True))
                for i in range(len(hq))]
    long = [i for i, q in enumerate(hq) if count(q.instructions) > cap]
    if len(hq) < 2 or not long:
        return [(list(range(len(hq))), hq, hs)]
    rest = [i for i in range(len(hq)) if i not in long]
    groups = [([i], *hoist_instructions([questions[i]], state_text, count, cap)) for i in long]
    return ([(rest, [hq[i] for i in rest], hs)] if rest else []) + groups


def schema_key(questions: dict) -> str:
    return json.dumps(questions, sort_keys=True)
