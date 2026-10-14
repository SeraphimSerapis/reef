"""Pretrained static token embeddings (model2vec "potion") used as Reef's frozen vocabulary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_EMBEDDINGS = "minishlab/potion-base-8M"


def load_static_embeddings(model_id: str = DEFAULT_EMBEDDINGS) -> tuple[Path, np.ndarray]:
    """Download (cached) a model2vec model. Returns (tokenizer.json path, scaled table).

    potion tables carry Zipf-style weighting, so row norms differ a lot (mean ~95). We rescale
    the whole table to unit per-entry std, which keeps the relative norms (they encode token
    importance) while matching the scale of the learned role and level embeddings.
    """
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file

    root = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))
    table = next(iter(load_file(root / "model.safetensors").values())).astype(np.float32)
    table /= table.std()
    return root / "tokenizer.json", table
