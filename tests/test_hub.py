"""Hub helpers, offline: repo-id detection and model-card generation."""

import json

from reef.hub import is_repo_id, model_card, resolve


def test_repo_id_detection(tmp_path):
    assert is_repo_id("seraphimserapis/reef")
    assert is_repo_id("hf://seraphimserapis/reef")
    assert is_repo_id("seraphimserapis/reef@v0.1")
    assert not is_repo_id("./artifacts/reef")
    assert not is_repo_id("/model")
    assert not is_repo_id("reef")
    d = tmp_path / "org"
    (d / "name").mkdir(parents=True)
    assert not is_repo_id(str(d / "name"))  # an existing path always wins
    assert resolve(str(d / "name")) == d / "name"


def test_model_card(tmp_path):
    cfg = {
        "name": "reef-test", "encoder_params": 3_100_000, "embeddings_source": "minishlab/potion-base-8M",
        "model": {"state_len": 384, "schema_len": 384, "grid_width": 32, "width": 256, "depth": 10,
                  "expansion": 3, "global_max": True},
        "train_args": {"train_data": ["data/gemma4/protect-triage.jsonl", "typed-decisions:train"],
                       "test_data": ["typed-decisions:test"]},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "eval_edgetpu.json").write_text(json.dumps(
        {"device": "edgetpu", "split": "test", "overall": {"accuracy": 0.62, "kl": 0.23, "brier": 0.13, "ece": 0.1}}))
    card = model_card(tmp_path, "me/reef-test", teacher="Gemma 4 E4B")
    assert card.startswith("---\nlicense: apache-2.0")
    assert "LocalLLaMA/typed-decisions" in card
    assert "`protect-triage`" in card
    assert "Gemma 4 E4B" in card
    assert "| int8 · edgetpu · test | 0.620 |" in card
    assert "reef serve me/reef-test" in card
