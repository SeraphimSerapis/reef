"""Policy tests: hand-written, clear-cut cases with the answers a reasonable homelabber expects.

    reef policy seraphimserapis/reef              # a Reef model (dir or Hub id)
    reef policy --teacher http://gpu-box:8080 --model gemma4   # the teacher, with each pack's notes

Each line of ``reef/evals/policy.jsonl`` names a pack, a state, one of the pack's canonical question ids
(``ask``) and the acceptable labels (``expect``). A case passes when the top answer is acceptable.
These are a floor, not a benchmark: a model that fails them should not be published.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "evals" / "policy.jsonl"


def load(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_FILE
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]


def run(answer, cases: list[dict]) -> dict:
    """``answer(pack, state, questions) -> {qid: {label: prob}}``; returns per-pack pass rates."""
    from reef import packs

    results, by_pack = [], {}
    for c in cases:
        pack = packs.get(c["pack"])
        q = {c["ask"]: pack.canonical_questions()[c["ask"]]}
        probs = answer(pack, c["state"], q)[c["ask"]]
        top = max(probs, key=probs.get)
        ok = top in c["expect"]
        mass = sum(probs.get(e, 0.0) for e in c["expect"])
        results.append({"id": c["id"], "pack": c["pack"], "ask": c["ask"], "ok": ok, "top": top,
                        "p_top": round(probs[top], 3), "p_expected": round(mass, 3), "expect": c["expect"]})
        by_pack.setdefault(c["pack"], []).append(ok)
    return {
        "passed": sum(r["ok"] for r in results), "total": len(results),
        "by_pack": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_pack.items())},
        "failures": [r for r in results if not r["ok"]],
    }


def model_answerer(artifact: str, device: str = "auto"):
    from reef.runtime import Reef

    reef = Reef(artifact, device=device)

    def answer(pack, state, questions):
        out = reef.decide(state, questions)["answers"]
        return {qid: a["probabilities"] for qid, a in out.items()}

    return answer


def teacher_answerer(url: str | list[str], model: str | list[str] | None = None):
    """One teacher, or several (lists of URLs and models): their probabilities are averaged, which is
    what ``reef merge`` does to their labelled data."""
    from reef.teacher import from_spec

    urls = [url] if isinstance(url, str) else list(url)
    models = model if isinstance(model, list) else [model] * len(urls)
    if len(models) == 1 and len(urls) > 1:
        models = models * len(urls)
    teachers = [from_spec(u, model=m) for u, m in zip(urls, models)]

    def answer(pack, state, questions):
        golds = [t.label(state, questions, notes=pack.teacher_notes) for t in teachers]
        return {qid: {k: sum(g[qid]["probabilities"][k] for g in golds) / len(golds) for k in g0["probabilities"]}
                for qid, g0 in golds[0].items()}

    return answer
