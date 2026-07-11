"""Tests for the logit-gradient baseline token handling."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logit_gradient import SYC_TOKENS, NON_SYC_TOKENS, resolve_token_ids


class MockTokenizer:
    """Maps known words to fixed single token ids; unknown -> empty."""
    VOCAB = {"Yes": 100, " Yes": 101, "No": 200, " No": 201,
             "Correct": 300, " Correct": 301, "zzz": []}

    def encode(self, text, add_special_tokens=False):
        if text in self.VOCAB:
            v = self.VOCAB[text]
            return v if isinstance(v, list) else [v]
        # default: 2-token for anything else (simulates multi-token)
        return [999, 998]


def test_resolve_maps_known_tokens():
    ids, skipped = resolve_token_ids(MockTokenizer(), ["Yes", "No"])
    assert len(ids) >= 1
    assert all(isinstance(i, int) for i in ids)


def test_resolve_prefers_single_token():
    ids, skipped = resolve_token_ids(MockTokenizer(), ["Yes"])
    # " Yes" or "Yes" both single-token in mock -> should pick a single-token id
    assert 100 in ids or 101 in ids


def test_resolve_dedupes():
    ids, skipped = resolve_token_ids(MockTokenizer(), ["Yes", "Yes"])
    assert len(ids) == len(set(ids))


def test_token_sets_nonempty():
    assert len(SYC_TOKENS) > 0
    assert len(NON_SYC_TOKENS) > 0


def test_resolve_returns_ints_and_list():
    ids, skipped = resolve_token_ids(MockTokenizer(), SYC_TOKENS)
    assert isinstance(ids, list)
    assert isinstance(skipped, list)
    # at least one syc token resolves
    assert len(ids) >= 1
