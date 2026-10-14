# Contributing to Reef

Thanks for helping. Reef is small on purpose. Most contributions fall into four areas: model
recipes, runtime and hosts (Raspberry Pi, Linux), docs, and new datasets.

## Setup

```bash
uv venv -p 3.12 && uv pip install -e ".[train]" --group dev   # or: pip install -e ".[train]" pytest ruff
make test        # unit tests (offline)
make test-data   # also checks metrics against the dataset card (downloads ~2 MB)
make lint
```

Exporting for the Edge TPU needs `edgetpu_compiler`, which is x86-64 Linux only. On other hosts,
build the Docker image once with `make compiler`. `reef export` picks it up automatically.

## Ground rules

* **Keep the runtime TensorFlow-free.** `reef.runtime`, `reef.packing`, `reef.text`,
  `reef.decode` and `reef.server` may import only numpy, tokenizers and ai-edge-litert. Put
  anything that needs TF in `train.py`, `export.py` or `model.py`.
* **Keep the encoder on the Edge TPU.** If you change `reef/model.py`, run
  `pytest tests/test_edgetpu_ops.py`, then `reef export` and check that `edgetpu_compiler.log`
  reports one subgraph with every op mapped. The rules and the reasons behind them are in
  [docs/edge-tpu-notes.md](docs/edge-tpu-notes.md).
* **Report results honestly.** Score on the public `test` split, which is never used for
  selection or calibration. Say which reference rows of the dataset card are comparable.
  Models fitted on `train` are specialists, and the card's first table is zero-shot.
* Code style: `ruff check` clean, type hints on public functions, and docstrings that say *why*.

## Pull requests

Small focused PRs are easiest to review. For a model change, include the `report.json` numbers
and a `reef bench` before/after if latency moved.
