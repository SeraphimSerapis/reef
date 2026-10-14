"""Build space/examples.json and (optionally) a local copy of a model for testing the Space.

    .venv/bin/python space/tools/prepare.py                       # examples + space/model/ from artifacts/release-reef
    .venv/bin/python space/tools/prepare.py --model artifacts/reef
    .venv/bin/python space/tools/prepare.py --no-model            # examples only

The browser reads the model files exactly as exported (config.json, tokenizer.json,
embeddings.npz, reef_int8.tflite), so the "export" is a plain copy. space/model/ is gitignored and
must not be uploaded to the Space; the published Space loads the model from the Hub instead.
States and questions are stored as JSON *text* so key order and number formatting survive.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from reef import packs
from reef.distill import make_cases

SPACE = Path(__file__).resolve().parents[1]
REPO = SPACE.parent
MODEL_FILES = ["config.json", "tokenizer.json", "embeddings.npz", "reef_int8.tflite"]


def build_examples(per_pack: int, seed: int) -> list[dict]:
    out = []
    for name in packs.names():
        pack = packs.get(name)
        for case in make_cases(pack, per_pack, seed=seed):
            out.append({
                "id": case["id"],
                "pack": name,
                "description": pack.description,
                "state": json.dumps(case["state"], indent=2, ensure_ascii=False),
                "questions": json.dumps(case["questions"], indent=2, ensure_ascii=False),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(REPO / "artifacts" / "release-reef"), help="exported model dir to copy")
    ap.add_argument("--no-model", action="store_true", help="only write examples.json")
    ap.add_argument("--per-pack", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    examples = build_examples(args.per_pack, args.seed)
    (SPACE / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples to {SPACE / 'examples.json'}")

    if not args.no_model:
        src, dst = Path(args.model), SPACE / "model"
        dst.mkdir(exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy2(src / f, dst / f)
        total = sum((dst / f).stat().st_size for f in MODEL_FILES)
        print(f"copied {len(MODEL_FILES)} model files from {src} to {dst} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
