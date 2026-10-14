"""Export: training checkpoint → runtime artifact directory.

    artifacts/reef-tiny/
      config.json                  model shape, calibrated temperatures, provenance
      tokenizer.json               WordPiece tokenizer (from the static-embedding model)
      embeddings.npz               host tables: tok [V, input_dim] (projection folded in), role, level
      reef_int8.tflite             fully int8-quantised encoder (runs anywhere on CPU)
      reef_int8_edgetpu.tflite     the same, compiled for the Edge TPU
      edgetpu_compiler.log

The encoder is quantised post-training with windows from the training split as the calibration
set. ``reef eval`` reports float vs int8 vs Edge TPU so any quantisation loss is visible.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

COMPILER_IMAGE = "reef-edgetpu-compiler"
FLOAT_WEIGHTS = "reef_float.weights.h5"


def _load_training_model(run_dir: Path):
    from tokenizers import Tokenizer

    from reef.data import build_windows, load_cases
    from reef.embeddings import load_static_embeddings
    from reef.model import ReefConfig, ReefModel

    meta = json.loads((run_dir / "config.json").read_text())
    cfg = ReefConfig.from_dict(meta["model"])
    tok_path, _ = load_static_embeddings(meta["embeddings"])
    packer = cfg.packer(Tokenizer.from_file(str(tok_path)))
    model = ReefModel(cfg)
    model(build_windows(load_cases("train")[:1], packer).tf_inputs(), training=False)
    model.load_weights(run_dir / "best.weights.h5")
    return meta, cfg, model, packer, tok_path


def host_tables(model) -> dict[str, np.ndarray]:
    e = model.embed
    tok = e.tok.get_weights()[0] @ e.proj.get_weights()[0]
    return {
        "tok": tok.astype(np.float16),
        "role": e.role.get_weights()[0].astype(np.float32),
        "level": e.level.get_weights()[0].astype(np.float32),
    }


def export_encoder(cfg, model):
    """A batch-1, dropout-free copy of the encoder with the trained weights.

    A BatchNorm in front of an MLP (``mlp_norm``) is folded into the MLP's first 1x1 conv: at
    inference it is a per-channel affine map ``a*x + c``, so ``W' = diag(a) W`` and ``b' = b + c W``
    give the same outputs with fewer ops and parameters on the Edge TPU.
    """
    from dataclasses import replace

    from reef.model import build_encoder

    fold = cfg.mlp_norm and not cfg.mlp_rank  # the low-rank conv has no bias to absorb c
    enc = build_encoder(replace(cfg, dropout=0.0, mlp_norm=cfg.mlp_norm and not fold), batch_size=1)
    src = {layer.name: layer for layer in model.encoder.layers}
    for layer in enc.layers:
        if not layer.weights:
            continue
        weights = src[layer.name].get_weights()
        bn = src.get(layer.name.removesuffix("_up") + "_mlp_bn") if fold and layer.name.endswith("_up") else None
        if bn is not None:
            gamma, beta, mean, var = bn.get_weights()
            a = gamma / np.sqrt(var + bn.epsilon)
            c = beta - a * mean
            k, b = weights
            weights = [k * a[None, None, :, None], b + c @ k[0, 0]]
        layer.set_weights(weights)
    return enc


def compile_edgetpu(tflite: Path, image: str = COMPILER_IMAGE) -> Path | None:
    """Run edgetpu_compiler (natively if on PATH, else in Docker). Returns the compiled path."""
    out_dir = tflite.parent.resolve()
    if shutil.which("edgetpu_compiler"):
        cmd = ["edgetpu_compiler", "-s", "-o", str(out_dir), str(tflite.resolve())]
    elif shutil.which("docker"):
        cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{out_dir}:/work", image,
               "edgetpu_compiler", "-s", "-o", "/work", f"/work/{tflite.name}"]
    else:
        print("edgetpu_compiler not found and docker unavailable: skipping Edge TPU compilation")
        return None
    res = subprocess.run(cmd, capture_output=True, text=True)
    log = res.stdout + res.stderr
    (out_dir / "edgetpu_compiler.log").write_text(log)
    print(log[log.find("Model"):] if "Model" in log else log)
    compiled = out_dir / tflite.name.replace(".tflite", "_edgetpu.tflite")
    if res.returncode != 0 or not compiled.exists():
        print("Edge TPU compilation failed; see edgetpu_compiler.log")
        return None
    if "will run on CPU" in log or "subgraphs: 1" not in log:
        print("WARNING: not every op mapped to the Edge TPU")
    stray = out_dir / tflite.name.replace(".tflite", "_edgetpu.log")
    if stray.exists():
        stray.rename(out_dir / "edgetpu_compiler.ops.log")
    return compiled


def export(run_dir: str, out: str, n_calibration: int = 256, compile_tpu: bool = True, name: str | None = None,
           int8_temperatures: bool = True, calibrate_on: list[str] | None = None) -> Path:
    import tensorflow as tf

    from reef.data import build_windows

    run_dir, out_dir = Path(run_dir), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, cfg, model, packer, tok_path = _load_training_model(run_dir)
    tables = host_tables(model)
    np.savez_compressed(out_dir / "embeddings.npz", **tables)
    shutil.copyfile(tok_path, out_dir / "tokenizer.json")

    enc = export_encoder(cfg, model)
    H, W = cfg.grid
    # Calibration windows: augmented windows from the run's own training sources (not just the
    # benchmark), so int8 activation ranges cover the text the model will actually see.
    from reef.data import load_source

    sources = meta.get("train_args", {}).get("train_data") or ["typed-decisions:train"]
    pool = [c for spec in sources for c in load_source(spec)]
    random.Random(1234).shuffle(pool)
    calib = build_windows(pool[: max(n_calibration * 4, 1000)], packer, random.Random(1234))
    idx = np.random.default_rng(0).permutation(len(calib))[:n_calibration]

    def embed(i):
        ids, roles, levels = calib.ids[i], calib.roles[i], calib.levels[i]
        x = tables["tok"][ids].astype(np.float32) + tables["role"][roles] + tables["level"][levels]
        x[roles == 0] = 0.0
        return x.reshape(1, H, W, cfg.input_dim)

    def representative():
        for i in idx:
            yield {"emb": embed(i)}

    conv = tf.lite.TFLiteConverter.from_keras_model(enc)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = out_dir / "reef_int8.tflite"
    tfl.write_bytes(conv.convert())
    print(f"wrote {tfl} ({tfl.stat().st_size / 1e6:.2f} MB)")

    compiled = compile_edgetpu(tfl) if compile_tpu else None
    out_meta = {
        "name": name or out_dir.name,
        "format": "reef-v1",
        "model": cfg.to_dict(),
        "temperatures": meta["temperatures"],
        "embeddings_source": meta["embeddings"],
        "encoder_params": int(enc.count_params()),
        "edgetpu": compiled is not None,
        "train_args": meta.get("train_args", {}),
    }
    (out_dir / "config.json").write_text(json.dumps(out_meta, indent=2))
    for extra in ("report.json", "history.json"):
        if (run_dir / extra).exists():
            shutil.copyfile(run_dir / extra, out_dir / f"train_{extra}")
    # Float training weights, so others can fine-tune: `reef train --init <this artifact or repo>`.
    shutil.copyfile(run_dir / "best.weights.h5", out_dir / FLOAT_WEIGHTS)
    if int8_temperatures and out_meta["train_args"].get("train_data"):
        from reef.calibrate import calibrate

        res = calibrate(str(out_dir), device="cpu", select=calibrate_on)
        print(f"int8 temperatures {res['temperatures']} (float {res['float_temperatures']}); validation ECE "
              f"{res['before']['ece']:.4f} -> {res['after']['ece']:.4f}, KL {res['before']['kl']:.4f} -> "
              f"{res['after']['kl']:.4f}")
    return out_dir
