"""Reef inference runtime: numpy + tokenizers + a TFLite interpreter, with the Edge TPU optional.

    from reef import Reef

    reef = Reef("artifacts/reef-tiny")          # uses the Edge TPU if one is attached
    out = reef.decide(state, questions)          # SystemOne-shaped response
    out["answers"]["urgency"]["score"]

The runtime has no TensorFlow dependency. Per request it renders and tokenises the state, packs
windows, sums three small embedding lookups per token, quantises to int8, runs the encoder (on
the Edge TPU or on CPU), then softmaxes the option-marker logits.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from reef.decode import make_answer, softmax
from reef.packing import (
    MAX_PART_OPTIONS,
    CapacityError,
    Packer,
    hoist_groups,
    normalize_questions,
    subquestion,
)
from reef.text import render_state


class Reef:
    def __init__(
        self,
        path: str | Path,
        device: str = "auto",
        edgetpu_lib: str | Path | None = None,
        num_threads: int | None = None,
        max_state_windows: int = 8,
        strict: bool = False,
    ):
        """Load an exported model directory, or a Hub repo id like ``seraphimserapis/reef``.

        ``device``: ``"auto"`` (Edge TPU if available, else CPU), ``"edgetpu"`` or ``"cpu"``.
        ``max_state_windows``: how many overlapping chunks of a long state are read (1 = truncate).
        ``strict``: refuse (:class:`~reef.packing.CapacityError`) rather than truncate anything:
        a state longer than ``max_state_windows`` chunks cover, or text longer than a window.
        """
        from tokenizers import Tokenizer

        from reef.hub import resolve

        self.path = resolve(path)
        self.max_state_windows = max_state_windows
        meta = json.loads((self.path / "config.json").read_text())
        self.meta = meta
        self.name = meta.get("name", self.path.name)
        m = meta["model"]
        self.state_len, self.schema_len = m["state_len"], m["schema_len"]
        self.grid = (self.state_len + self.schema_len) // m["grid_width"], m["grid_width"]
        self.temperatures = meta["temperatures"]
        tables = np.load(self.path / "embeddings.npz")
        self.tok_table = tables["tok"].astype(np.float32)
        self.role_table = tables["role"].astype(np.float32)
        self.level_table = tables["level"].astype(np.float32)
        self.strict = strict
        # Lenient: the limits the model was trained with (truncate). Strict: anything that fits a
        # window is read in full, and the rest is refused. Over-long instructions are first moved
        # into the state if the model was trained to read them there.
        lim_i, lim_o = m.get("max_instruction_tokens", 40), m.get("max_option_tokens", 28)
        bands = int(m.get("bands", 0) or 0)
        block = self.schema_len // bands if bands else self.schema_len
        self.packer = Packer(Tokenizer.from_file(str(self.path / "tokenizer.json")), self.state_len,
                             self.schema_len, block if strict else lim_i,
                             block if strict else lim_o, strict=strict, bands=bands,
                             drop_opaque_keys=bool(m.get("drop_opaque_keys", False)))
        self.hoist = bool(m.get("hoist_instructions", False))
        self.hoist_empty = self.hoist and bool(m.get("hoist_empty_state", False))
        self.instruction_cap = lim_i
        self.device, self._interp = self._load_interpreter(device, edgetpu_lib, num_threads)
        try:
            self._interp.allocate_tensors()
        except RuntimeError:
            if self.device != "cpu":
                raise
            # XNNPACK rejects some valid int8 models (requantisation scales outside its range,
            # e.g. an MLP whose ReLU never fires); the builtin kernels run them, only slower.
            from ai_edge_litert.interpreter import Interpreter, OpResolverType

            self._interp = Interpreter(
                str(self.path / "reef_int8.tflite"), num_threads=num_threads,
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            self._interp.allocate_tensors()
            self.device = "cpu-builtin"
        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]
        self._in_idx, self._out_idx = inp["index"], out["index"]
        self._in_scale, self._in_zp = inp["quantization"]
        self._out_scale, self._out_zp = out["quantization"]
        self._in_dtype = inp["dtype"]
        self._lock = threading.Lock()

    # -- setup ----------------------------------------------------------------------------------
    def _load_interpreter(self, device, edgetpu_lib, num_threads):
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        tpu_model = self.path / "reef_int8_edgetpu.tflite"
        cpu_model = self.path / "reef_int8.tflite"
        if device in ("auto", "edgetpu") and tpu_model.exists():
            from reef import edgetpu

            lib = Path(edgetpu_lib) if edgetpu_lib else edgetpu.find_library()
            err = None
            if lib is not None:
                try:
                    delegate = load_delegate(str(lib))
                    return "edgetpu", Interpreter(str(tpu_model), experimental_delegates=[delegate])
                except (ValueError, OSError) as e:
                    err = e
            if device == "edgetpu":
                raise RuntimeError(
                    f"could not open the Edge TPU (library: {lib}). Run `reef doctor`. {err or ''}"
                )
        return "cpu", Interpreter(str(cpu_model), num_threads=num_threads)

    # -- inference ------------------------------------------------------------------------------
    def embed(self, ids: np.ndarray, roles: np.ndarray, levels: np.ndarray) -> np.ndarray:
        x = self.tok_table[ids] + self.role_table[roles] + self.level_table[levels]
        x[roles == 0] = 0.0
        return x

    def run_window(self, ids, roles, levels) -> np.ndarray:
        """Encoder logits for one packed window, shape ``[seq_len]``."""
        x = self.embed(ids, roles, levels).reshape(1, *self.grid, -1)
        if self._in_dtype != np.float32:
            x = np.clip(np.round(x / self._in_scale + self._in_zp), -128, 127).astype(self._in_dtype)
        with self._lock:
            self._interp.set_tensor(self._in_idx, x)
            self._interp.invoke()
            y = self._interp.get_tensor(self._out_idx)
        if y.dtype != np.float32:
            y = (y.astype(np.float32) - self._out_zp) * self._out_scale
        return y.reshape(-1)

    def _score_parts(self, regions, parts: dict) -> tuple[dict, int, float]:
        """Probabilities for keyed question parts: every schema window runs once per state region
        and each part's option logits are averaged over regions before the softmax."""
        keys = list(parts)
        windows = self.packer.pack_blocks(*regions[0], [(k, self.packer.block(parts[k])) for k in keys])
        acc: dict = {}
        t_dev = 0.0
        for w in windows:
            for s_ids, s_roles in regions:
                ids, roles = self.packer.with_state(w, s_ids, s_roles)
                t1 = time.perf_counter()
                logits = self.run_window(ids, roles, w.levels)
                t_dev += time.perf_counter() - t1
                for key, positions in w.slots:
                    acc[key] = acc.get(key, 0.0) + logits[positions]
        probs = {k: softmax(acc[k] / len(regions), self.temperatures.get(parts[k].type, 1.0)) for k in keys}
        return probs, len(windows) * len(regions), t_dev

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        """Answer every question about ``state``. Returns a SystemOne-style response body.

        Long states are read in overlapping chunks (up to ``max_state_windows``), and questions
        with more options than one window compares are answered as a knockout over parts
        (see :mod:`reef.packing`). Plain requests take exactly the single-window path.

        With instruction hoisting, a shared instruction prefix moves into the state; questions
        whose own instructions are still too long (e.g. one candidate document per question)
        are then answered one at a time, each with its instructions hoisted into its own state.
        """
        t0 = time.perf_counter()
        qs = normalize_questions(questions)
        state_text = render_state(state)
        groups = [(list(range(len(qs))), qs, state_text)]
        if self.hoist:
            groups = hoist_groups(qs, state_text, lambda t: len(self.packer.encode(t)), self.instruction_cap,
                                  empty=self.hoist_empty)
        answers: dict[int, dict] = {}
        usage = {"state_tokens": 0, "state_truncated": False, "state_windows": 0, "windows": 0, "rounds": 0}
        t_dev = 0.0
        for idx, gqs, gstate in groups:
            got, u, td = self._answer(gqs, gstate)
            answers.update(zip(idx, got))
            t_dev += td
            for k in ("state_tokens", "state_windows", "rounds"):
                usage[k] = max(usage[k], u[k])
            usage["windows"] += u["windows"]
            usage["state_truncated"] |= u["state_truncated"]
        if len(groups) > 1:
            usage["groups"] = len(groups)
        return {
            "model": self.name,
            "answers": {qs[i].id: answers[i] for i in range(len(qs))},
            "usage": {
                **usage,
                "device": self.device,
                "device_ms": round(t_dev * 1e3, 3),
                "total_ms": round((time.perf_counter() - t0) * 1e3, 3),
            },
        }

    def _answer(self, qs: list, state_text: str) -> tuple[list[dict], dict, float]:
        """Answer already-hoisted questions about one state: chunked state, knockout rounds."""
        state_ids = self.packer.encode(state_text)
        regions, gaps = self.packer.state_regions(state_ids, self.max_state_windows)
        if gaps and self.strict:
            raise CapacityError(f"the state has {len(state_ids)} tokens, over Reef's maximum context length "
                                f"of about {self.max_state_windows * (self.state_len - 2) * 3 // 4} with "
                                f"max_state_windows={self.max_state_windows}")
        current = {qi: list(range(len(q.options))) for qi, q in enumerate(qs)}
        rounds: dict[int, list] = {qi: [] for qi in current}  # per question: [(chunks, chunk probs)]
        final: dict[int, tuple[list[int], np.ndarray]] = {}
        n_windows, n_rounds, t_dev = 0, 0, 0.0
        while current:
            parts, plan = {}, {}
            for qi, idx in current.items():
                sub = subquestion(qs[qi], idx)
                if len(idx) <= MAX_PART_OPTIONS and self.packer.block_fits(sub):
                    plan[qi] = None
                    parts[(qi, 0)] = sub
                else:
                    plan[qi] = self.packer.split_options(qs[qi], idx)
                    if len(plan[qi]) >= len(idx):  # not even two options fit one window
                        raise CapacityError(f"question {qs[qi].id!r} does not fit Reef's maximum context length")
                    for ci, chunk in enumerate(plan[qi]):
                        parts[(qi, ci)] = subquestion(qs[qi], chunk)
            probs, nw, td = self._score_parts(regions, parts)
            n_windows, n_rounds, t_dev = n_windows + nw, n_rounds + 1, t_dev + td
            nxt = {}
            for qi, chunks in plan.items():
                if chunks is None:
                    final[qi] = (current[qi], probs[(qi, 0)])
                else:
                    within = [probs[(qi, ci)] for ci in range(len(chunks))]
                    rounds[qi].append((chunks, within))
                    nxt[qi] = [c[int(np.argmax(p))] for c, p in zip(chunks, within)]
            current = nxt
        answers = []
        for qi, q in enumerate(qs):
            idx, p = final[qi]
            dist = dict(zip(idx, p))
            # Walk back through the rounds: an option knocked out in a part gets its part winner's
            # mass scaled by how it compared with that winner. The final-round winner stays on top.
            for chunks, within in reversed(rounds[qi]):
                dist = {o: dist[c[int(np.argmax(w))]] * pw / max(float(np.max(w)), 1e-12)
                        for c, w in zip(chunks, within) for o, pw in zip(c, w)}
            p = np.array([dist[i] for i in range(len(q.options))], np.float64)
            answers.append(make_answer(q, p / p.sum()))
        usage = {"state_tokens": len(state_ids), "state_truncated": gaps, "state_windows": len(regions),
                 "windows": n_windows, "rounds": n_rounds}
        return answers, usage, t_dev

    def systemone(self, body: dict) -> dict:
        """Handle a ``POST /v1/systemone`` request body."""
        if "state" not in body or "questions" not in body:
            raise ValueError("request needs 'state' and 'questions'")
        return self.decide(body["state"], body["questions"])
