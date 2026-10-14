from __future__ import annotations

import pytest


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Whitespace tokenizer with a growing vocab and BERT-style special tokens."""

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    def token_to_id(self, tok):
        return self.vocab.get(tok)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.lower().split():
            ids.append(self.vocab.setdefault(w, len(self.vocab)))
        return _Enc(ids)


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def questions():
    return {
        "is_spam": {"type": "noul", "instructions": "Is this message spam?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle it?",
            "criteria": {"billing": "money matters", "tech_support": "something is broken"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is it?",
            "criteria": ["not urgent", "somewhat", "moderately", "very", "critical"],
        },
    }
