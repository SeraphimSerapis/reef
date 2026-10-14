"""Training a Reef model on typed decisions.

Recipe:
* Soft targets. The loss is cross-entropy against the full gold distribution, which equals KL
  up to a constant. The gold is a three-sample teacher average, so it is soft where the decision
  is ambiguous, and calibration is half the benchmark.
* Fresh augmentation every epoch: JSON key order, option order and question order are reshuffled,
  and whole token vectors are dropped.
* Model selection on a stratified validation split, then per-question-type temperature scaling
  fitted on the same split.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reef.data import WindowBatch, build_windows, load_source, split_holdout
from reef.decode import DEFAULT_TEMPERATURES, question_probs
from reef.embeddings import DEFAULT_EMBEDDINGS, load_static_embeddings
from reef.metrics import Decision, breakdown, score


@dataclass
class TrainArgs:
    out: str = "runs/reef-tiny"
    epochs: int = 40
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.06
    val_frac: float = 0.2
    seed: int = 0
    embeddings: str = DEFAULT_EMBEDDINGS
    full: bool = False  # no validation split: train on everything (temperatures stay at 1)
    # Sources: "typed-decisions:train" / "typed-decisions:test" or paths to `reef label` JSONL files.
    train_data: list[str] = field(default_factory=lambda: ["typed-decisions:train"])
    test_data: list[str] = field(default_factory=lambda: ["typed-decisions:test"])
    # Fine-tune from an exported artifact dir / Hub repo (uses its shape, embeddings and float weights).
    init: str | None = None
    # Workflow-name prefixes whose validation cases pick the best epoch and fit the temperatures
    # (default: all). Lets auxiliary data (e.g. Open-Jev's hard labels) help training without
    # calibrating the model to it.
    select_on: list[str] | None = None
    # Workflow-name prefix -> sampling weight for training cases (e.g. {"alert-triage": 2.0}). Applied
    # after the validation split; fractional parts are sampled.
    weights: dict[str, float] = field(default_factory=dict)
    # Packing settings applied on top of the model config (also after --init), e.g.
    # {"max_instruction_tokens": 128, "hoist_instructions": True}. They need no new weights.
    packing: dict = field(default_factory=dict)


def _tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _apply_weights(cases: list, weights: dict[str, float], rng: random.Random) -> list:
    """Repeat (or subsample) training cases by workflow prefix; the longest matching prefix wins."""
    if not weights:
        return cases
    out = []
    for c in cases:
        match = max((p for p in weights if c.workflow.startswith(p)), key=len, default=None)
        w = weights[match] if match is not None else 1.0
        n = int(w) + (rng.random() < w - int(w))
        out += [c] * n
    rng.shuffle(out)
    return out


def _torch_device() -> str:
    from keras.src.backend.torch.core import DEFAULT_DEVICE

    return DEFAULT_DEVICE


def _release_cache() -> None:
    """Hand cached Apple-GPU memory back to the system (no-op on other backends/devices)."""
    import keras

    if keras.backend.backend() == "torch" and _torch_device() == "mps":
        import torch

        torch.mps.empty_cache()


def soft_target_loss(logits, pos, msk, gold):
    """Cross-entropy against the teacher's soft distribution over each question's option markers.
    Written in keras.ops so it runs on any backend."""
    from keras import ops

    B = ops.shape(pos)[0]
    flat = ops.reshape(ops.cast(pos, "int64"), (B, -1))
    sel = ops.reshape(ops.take_along_axis(logits, flat, axis=1), ops.shape(pos))
    sel = sel + (1.0 - msk) * -1e9
    logp = ops.log_softmax(sel, axis=-1)
    per_q = -ops.sum(gold * logp * msk, axis=-1)
    valid = ops.cast(ops.sum(msk, axis=-1) > 0, "float32")
    return ops.sum(per_q * valid) / ops.maximum(ops.sum(valid), 1.0)


def _make_step(model, opt):
    """One optimisation step for the active Keras backend. Returns a function -> float loss."""
    import keras
    from keras import ops

    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        @tf.function(reduce_retracing=True)
        def tf_step(inputs, pos, msk, gold):
            with tf.GradientTape() as tape:
                loss = soft_target_loss(model(inputs, training=True), pos, msk, gold)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        return lambda inputs, pos, msk, gold: float(tf_step(inputs, pos, msk.astype(np.float32),
                                                            gold.astype(np.float32)))
    if backend == "torch":
        import torch

        def torch_step(inputs, pos, msk, gold):
            inputs = {k: ops.convert_to_tensor(v) for k, v in inputs.items()}
            logits = model(inputs, training=True)
            loss = soft_target_loss(logits, ops.convert_to_tensor(pos), ops.convert_to_tensor(msk, "float32"),
                                    ops.convert_to_tensor(gold, "float32"))
            model.zero_grad()
            loss.backward()
            weights = list(model.trainable_weights)
            grads = [v.value.grad for v in weights]
            with torch.no_grad():
                opt.apply(grads, weights)
            return float(loss.detach().cpu())

        return torch_step
    raise SystemExit(f"training supports the tensorflow and torch Keras backends, not {backend!r}")


def predict_logits(model, wb: WindowBatch, batch_size: int = 64) -> np.ndarray:
    import contextlib

    import keras

    ctx = contextlib.nullcontext()
    if keras.backend.backend() == "torch":
        import torch

        ctx = torch.no_grad()
    out = []
    with ctx:
        for i in range(0, len(wb), batch_size):
            sl = slice(i, i + batch_size)
            out.append(keras.ops.convert_to_numpy(model(wb.tf_inputs(sl), training=False)))
    return np.concatenate(out)


def decisions_from_logits(cases, wb: WindowBatch, logits: np.ndarray, temps: dict) -> list[Decision]:
    out = []
    for w, wmeta in enumerate(wb.meta):
        for si, (ci, q) in enumerate(wmeta):
            k = len(q.options)
            probs = question_probs(logits[w], list(wb.opt_pos[w, si, :k]), q.type, temps)
            c = cases[ci]
            g = c.gold[q.id]
            out.append(
                Decision(q.type, q.id, c.workflow, dict(zip([o.id for o in q.options], probs)),
                         g["probabilities"], g.get("label"))
            )
    return out


def fit_temperatures(cases, wb: WindowBatch, logits: np.ndarray) -> dict[str, float]:
    """Per question type, the temperature that minimises KL from gold on held-out windows."""
    grid = np.exp(np.linspace(math.log(0.25), math.log(6.0), 80))
    temps = {}
    for qtype in ("noul", "choice", "score"):
        best, best_kl = 1.0, float("inf")
        for t in grid:
            ds = [d for d in decisions_from_logits(cases, wb, logits, {qtype: float(t)})
                  if d.question_type == qtype]
            if not ds:
                break
            kl = score(ds)["kl"]
            if kl < best_kl:
                best, best_kl = float(t), kl
        temps[qtype] = round(best, 4)
    return temps


def train(args: TrainArgs, cfg=None) -> Path:
    import keras

    from reef.model import ReefConfig, ReefModel

    cfg = cfg or ReefConfig()
    init_weights = None
    if args.init:
        from reef.export import FLOAT_WEIGHTS
        from reef.hub import resolve

        init_dir = resolve(args.init, with_float=True)
        init_meta = json.loads((init_dir / "config.json").read_text())
        cfg = ReefConfig.from_dict(init_meta["model"])
        args.embeddings = init_meta.get("embeddings_source") or init_meta.get("embeddings") or args.embeddings
        init_weights = next((p for p in (init_dir / FLOAT_WEIGHTS, init_dir / "best.weights.h5") if p.exists()), None)
        if init_weights is None:
            raise SystemExit(f"{args.init} has no float weights ({FLOAT_WEIGHTS}); it cannot be fine-tuned")
    for k, v in args.packing.items():
        if k not in ("max_instruction_tokens", "max_option_tokens", "hoist_instructions", "mlp_leak", "act_clip"):
            raise ValueError(f"unknown packing setting {k!r}")
        setattr(cfg, k, v)
    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok_path, table = load_static_embeddings(args.embeddings)
    cfg.vocab_size, cfg.embed_dim = table.shape
    packer = cfg.packer(_tokenizer(tok_path))

    all_train = [c for spec in args.train_data for c in load_source(spec)]
    if args.full:
        tr, val = all_train, []
    else:
        tr, val = split_holdout(all_train, args.val_frac, args.seed)
    tr = _apply_weights(tr, args.weights, random.Random(args.seed))
    if args.select_on and val:
        sel = [c for c in val if c.workflow.startswith(tuple(args.select_on))]
        if not sel:
            raise SystemExit(f"--select-on {args.select_on} matches no validation cases")
        val = sel
    val_wb = build_windows(val, packer) if val else None
    print(f"train cases={len(tr)} val cases={len(val)} grid={cfg.grid} "
          f"windows/case≈{len(build_windows(tr[:200], packer)) / min(200, len(tr)):.2f}")

    model = ReefModel(cfg)
    probe = build_windows(tr[:2], packer)
    model(probe.tf_inputs(), training=False)
    model.embed.tok.set_weights([table])
    if init_weights is not None:
        model.load_weights(init_weights)
        print(f"fine-tuning from {args.init}")
    n_enc = model.encoder.count_params()
    print(f"encoder params (on the Edge TPU): {n_enc / 1e6:.2f}M")

    steps_per_epoch = math.ceil(len(tr) * 1.05 / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=total - int(total * args.warmup_frac), alpha=0.02,
        warmup_target=args.lr, warmup_steps=int(total * args.warmup_frac))
    opt = keras.optimizers.AdamW(sched, weight_decay=args.weight_decay, clipnorm=1.0)
    step = _make_step(model, opt)
    backend = keras.backend.backend()
    print(f"keras backend: {backend}" + (f" ({_torch_device()})" if backend == "torch" else ""))

    rng = random.Random(args.seed)
    history, best = [], {"kl": float("inf"), "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        wb = build_windows(tr, packer, rng)
        order = np.random.permutation(len(wb))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            losses.append(float(step(wb.tf_inputs(idx), wb.opt_pos[idx], wb.opt_mask[idx], wb.gold[idx])))
            if len(losses) % 50 == 0:
                # Reference cycles keep each step's tensors alive until Python's cyclic GC runs,
                # which it rarely does here: Apple-GPU memory grew by ~10 GB per few hundred steps.
                gc.collect()
            if len(losses) % 500 == 0:  # long epochs: show progress
                print(json.dumps({"epoch": epoch, "step": len(losses), "of": -(-len(order) // args.batch_size),
                                  "loss_500": round(float(np.mean(losses[-500:])), 4),
                                  "sec": round(time.time() - t0, 1)}), flush=True)
                _release_cache()
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}
        if val_wb is not None:
            m = score(decisions_from_logits(val, val_wb, predict_logits(model, val_wb), DEFAULT_TEMPERATURES))
            rec.update({f"val_{k}": round(v, 4) for k, v in m.items() if k != "n"})
            if m["kl"] < best["kl"]:
                best = {"kl": m["kl"], "epoch": epoch}
                model.save_weights(out / "best.weights.h5")
        history.append(rec)
        print(json.dumps(rec), flush=True)

    if val_wb is not None:
        model.load_weights(out / "best.weights.h5")
        val_logits = predict_logits(model, val_wb)
        temps = fit_temperatures(val, val_wb, val_logits)
        val_scores = score(decisions_from_logits(val, val_wb, val_logits, temps))
    else:
        model.save_weights(out / "best.weights.h5")
        temps, val_scores = dict(DEFAULT_TEMPERATURES), {}
    print("temperatures:", temps, "calibrated val:", val_scores)

    # Score each test source (never used for selection or calibration).
    report: dict = {"val": val_scores, "best_epoch": best["epoch"], "encoder_params": n_enc, "tests": {}}
    for spec in args.test_data:
        test = load_source(spec)
        test_wb = build_windows(test, packer)
        test_dec = decisions_from_logits(test, test_wb, predict_logits(model, test_wb), temps)
        report["tests"][spec] = {
            "overall": score(test_dec),
            "by_type": breakdown(test_dec, "question_type"),
            "by_workflow": breakdown(test_dec, "workflow"),
            "by_question": breakdown(test_dec, "question_id"),
        }
        print(f"TEST {spec}:", json.dumps(report["tests"][spec]["overall"]))
    if args.test_data:  # the first test source doubles as the headline number
        report["test"] = report["tests"][args.test_data[0]]["overall"]
    (out / "config.json").write_text(json.dumps(
        {"model": cfg.to_dict(), "temperatures": temps, "embeddings": args.embeddings,
         "train_args": asdict(args)}, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return out
