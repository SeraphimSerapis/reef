"""Build a random (untrained) Reef v2 artifact for the Space parity test. CPU only.

    .venv/bin/python space/tools/make_v2_artifact.py <run_dir> <out_dir>

Random weights are fine for parity: the point is a v2 graph (bands, gate: MUL/TANH; leaky MLPs:
scalar MUL + MAXIMUM; pre-MLP BatchNorm folded at export; activation clips as int8 ranges) and a
v2 config.json (bands, hoist_empty_state, drop_opaque_keys, state_avgpool, mlp_leak, mlp_norm,
act_clip), the release-candidate format. zero_init=False so the logits vary.
"""

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from reef.embeddings import load_static_embeddings  # noqa: E402
from reef.export import export  # noqa: E402
from reef.model import ReefConfig, ReefModel  # noqa: E402


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    emb = "minishlab/potion-base-8M"
    tok_path, table = load_static_embeddings(emb)
    depth = 2
    clips = {"stem_q": [-3.5, 4.0]}
    for i in range(depth):
        clips[f"polyp{i}_mixq"] = [-12.0 * (i + 1), 11.0 * (i + 1)]
        clips[f"polyp{i}_resq"] = [-6.0 * (i + 1), 6.5 * (i + 1)]
    cfg = ReefConfig(bands=2, gate=True, hoist_instructions=True, hoist_empty_state=True, drop_opaque_keys=True,
                     state_avgpool=True, mlp_leak=0.1, mlp_norm=True, act_clip=clips,
                     depth=depth, width=64, zero_init=False, dropout=0.0)
    cfg.vocab_size, cfg.embed_dim = table.shape
    keras.utils.set_random_seed(0)
    model = ReefModel(cfg)
    z = {k: np.zeros((1, cfg.seq_len), np.int32) for k in ("ids", "roles", "levels")}
    model(z, training=False)
    model.embed.tok.set_weights([table])
    rng = np.random.default_rng(0)
    for layer in model.encoder.layers:  # non-trivial BatchNorm statistics, so the export fold matters
        if layer.name.endswith("_mlp_bn"):
            gamma, beta, mean, var = layer.get_weights()
            layer.set_weights([rng.uniform(0.5, 1.5, gamma.shape).astype(np.float32),
                               rng.normal(0, 0.2, beta.shape).astype(np.float32),
                               rng.normal(0, 0.5, mean.shape).astype(np.float32),
                               rng.uniform(0.5, 2.0, var.shape).astype(np.float32)])
    run.mkdir(parents=True, exist_ok=True)
    model.save_weights(run / "best.weights.h5")
    (run / "config.json").write_text(json.dumps({
        "model": cfg.to_dict(), "temperatures": {"noul": 1.1, "choice": 0.9, "score": 1.3},
        "embeddings": emb, "train_args": {}}, indent=2))
    export(str(run), str(out), n_calibration=64, compile_tpu=False, name="reef-v2-parity", int8_temperatures=False)


if __name__ == "__main__":
    main()
