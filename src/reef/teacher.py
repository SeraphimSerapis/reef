"""Teachers: big models that label states with soft answer distributions for distillation.

Two kinds, both reachable over HTTP so the teacher can live on another box in the homelab:

* ``ChatTeacher``: any OpenAI-compatible chat endpoint that returns ``logprobs`` (llama.cpp
  ``llama-server``, Ollama, MLX-LM, vLLM, LM Studio). Each question is asked as a lettered
  multiple-choice prompt, and the probability of each option is read from the log-probabilities
  of the first answer token. One forward pass gives a full soft label: no sampling, nothing to
  parse. ``permutations > 1`` re-asks with the option order shuffled and averages, which cancels
  most of an LLM's letter-position bias.
* ``SystemOneTeacher``: anything that already speaks ``POST /v1/systemone`` (Laya's
  ``laya-serve``, TypeSafe Jev, or another Reef).

Check the licence of the model you distil from before publishing a student trained on its
outputs.
"""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from reef.packing import DEFAULT_NOUL, normalize_questions

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
FLOOR = 1e-4  # probability given to an option the endpoint did not list in its top logprobs

SYSTEM_PROMPT = (
    "You are an experienced, level-headed operator of a home lab and smart home. You read the "
    "situation, then answer the question by picking the single best option. Answer with the "
    "option's letter only."
)
# The answer must be the first generated token, so reasoning models (Gemma 4, Qwen3, ...) are asked
# not to think. Endpoints whose template ignores the flag are unaffected.
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _post(url: str, body: dict, timeout: float, headers: dict | None = None, retries: int = 6) -> dict:
    """POST JSON, retrying transient failures (5xx, 429, connection errors) with exponential backoff,
    so a teacher restart or a busy slot pauses labelling instead of failing every queued case."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            e.close()  # an unread error response keeps its socket open otherwise
            if attempt == retries or not (e.code >= 500 or e.code == 429):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(min(60.0, 2.0**attempt))
    raise AssertionError("unreachable")


def _state_text(state: Any) -> str:
    if isinstance(state, (dict, list)):
        return json.dumps(state, indent=1, ensure_ascii=False)
    return str(state)


@dataclass
class ChatTeacher:
    base_url: str  # e.g. http://gpu-box:8080  (``/v1/chat/completions`` is appended)
    model: str = "default"
    permutations: int = 1
    top_logprobs: int = 10  # enough for <= 10 options; MLX-LM accepts at most 11
    timeout: float = 120.0
    api_key: str | None = None
    system_prompt: str = SYSTEM_PROMPT
    # Merged over THINKING_OFF and passed through to the endpoint.
    extra_body: dict = field(default_factory=dict)
    display_name: str | None = None  # recorded instead of ``model``, e.g. "Gemma 4 26B-A4B" for MLX's "default_model"

    @property
    def name(self) -> str:
        # Provenance recorded in every labelled case; no endpoint URL, since datasets get published.
        who = self.display_name or self.model
        return f"{who} (option logprobs, permutations={self.permutations}, scores as digits)"

    def _ask(self, state_text: str, instructions: str, option_texts: list[str], notes: str = "",
             labels: str = LETTERS) -> list[float]:
        k = len(option_texts)
        opts = "\n".join(f"{labels[i]}) {t}" for i, t in enumerate(option_texts))
        what = "number" if labels[0].isdigit() else "letter"
        prompt = (
            f"Situation:\n{state_text}\n\nQuestion: {instructions}\nOptions:\n{opts}\n\n"
            f"Answer with one {what} ({labels[0]}-{labels[k - 1]})."
        )
        system = self.system_prompt + (f"\n\n{notes.strip()}" if notes else "")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **THINKING_OFF,
            **self.extra_body,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        out = _post(self.base_url.rstrip("/") + "/v1/chat/completions", body, self.timeout, headers)
        lp = out["choices"][0].get("logprobs") or {}
        if not lp.get("content"):
            raise ValueError("teacher endpoint returned no logprobs (LM Studio does not support them; "
                             "use llama-server, MLX-LM, vLLM or Ollama)")
        first = lp["content"][0]["top_logprobs"]
        mass = [0.0] * k
        for t in first:
            tok = t["token"].replace("\u2581", "").strip().rstrip(").:").upper()
            if len(tok) == 1 and tok in labels[:k]:
                mass[labels.index(tok)] += math.exp(t["logprob"])
        if sum(mass) < 0.5:  # e.g. a thinking/channel token came first: a silent uniform label is worse
            raise ValueError(f"first answer token is not an option ({first[0]['token']!r}, "
                             f"{sum(mass):.2f} option mass); is thinking still on?")
        mass = [max(m, FLOOR) for m in mass]
        s = sum(mass)
        return [m / s for m in mass]

    def label(self, state: Any, questions: dict, seed: int = 0, notes: str = "") -> dict:
        """Soft-label one case. ``notes`` (a pack's glossary / house policy) is added to the system
        prompt: it steers the teacher only, the student learns the resulting behaviour from data.

        Yes/no and choice questions are lettered options; with ``permutations > 1`` they are re-asked
        in reversed / shuffled order and averaged. Score questions are an ordinal scale: they are
        always asked in natural order with digit answers, because a reversed scale is not the same
        question to an LLM."""
        rng = random.Random(seed)
        st = _state_text(state)
        gold = {}
        for q in normalize_questions(questions):
            k = len(q.options)
            texts = [_teacher_text(q, o) for o in q.options]
            if q.type == "score" and k <= 10:
                gold[q.id] = _gold(q, self._ask(st, q.instructions, texts, notes, labels=DIGITS))
                continue
            acc = [0.0] * k
            for p in range(max(1, self.permutations)):
                order = list(range(k))
                if p == 1:
                    order.reverse()
                elif p > 1:
                    rng.shuffle(order)
                probs = self._ask(st, q.instructions, [texts[i] for i in order], notes)
                for pos, i in enumerate(order):
                    acc[i] += probs[pos]
            probs = [a / max(1, self.permutations) for a in acc]
            gold[q.id] = _gold(q, probs)
        return gold


@dataclass
class SystemOneTeacher:
    url: str  # full URL of the endpoint, e.g. http://localhost:8000/v1/systemone
    timeout: float = 120.0
    api_key: str | None = None

    @property
    def name(self) -> str:
        return "systemone endpoint"

    def label(self, state: Any, questions: dict, seed: int = 0, notes: str = "") -> dict:  # notes: n/a
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        out = _post(self.url, {"state": state, "questions": questions}, self.timeout, headers)
        gold = {}
        for q in normalize_questions(questions):
            p = out["answers"][q.id]["probabilities"]
            gold[q.id] = _gold(q, [float(p.get(o.id, 0.0)) for o in q.options])
        return gold


def _teacher_text(q, o) -> str:
    """How an option is shown to an LLM teacher. Default yes/no options read as plain Yes / No."""
    if q.type == "noul" and o.text == f"{o.id}: {DEFAULT_NOUL[o.id]}":
        return "Yes" if o.id == "true" else "No"
    return o.text


def _gold(q, probs: list[float]) -> dict:
    s = sum(probs) or 1.0
    pd = {o.id: round(p / s, 6) for o, p in zip(q.options, probs)}
    return {"type": q.type, "probabilities": pd, "label": max(pd, key=pd.get)}


def from_spec(spec: str, model: str | None = None, **kw) -> ChatTeacher | SystemOneTeacher:
    """``http://host:port`` → ChatTeacher; a URL ending in ``/v1/systemone`` → SystemOneTeacher."""
    if spec.rstrip("/").endswith("/v1/systemone"):
        return SystemOneTeacher(spec, **{k: v for k, v in kw.items() if k in ("timeout", "api_key")})
    return ChatTeacher(spec, model=model or "default", **kw)
