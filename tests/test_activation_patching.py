"""Tests for activation patching (mock TransformerLens adapter)."""
import sys
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_adapters import TransformerLensAdapter
from src.patching import (
    compute_patched_behavior_margin,
    patch_selected_layers_from_reference,
    run_patching_behavior_validation,
)


class _MockTokenizer:
    def encode(self, text, return_tensors=None, truncation=False, max_length=None,
               add_special_tokens=True):
        ids = [(ord(c) % 50) + 1 for c in text.split()[0]][:8] or [1]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids


class _MockTLModel:
    """Minimal stand-in exposing .hooks() and __call__ returning logits."""
    def __init__(self, n_layers, d_vocab=64):
        self.n_layers = n_layers
        self.d_vocab = d_vocab

    @contextmanager
    def hooks(self, fwd_hooks=None):
        yield  # we don't actually run hooks in the mock

    def __call__(self, input_ids):
        T = input_ids.shape[1]
        torch.manual_seed(int(input_ids.sum().item()) % 1000)
        return torch.randn(1, T, self.d_vocab)


class MockTLAdapter(TransformerLensAdapter):
    """Bypass __init__ to avoid loading a real model."""
    def __init__(self, n_layers=4, d_model=8):
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = torch.device("cpu")
        self.tokenizer = _MockTokenizer()
        self.model = _MockTLModel(n_layers)


def test_patch_returns_finite():
    adapter = MockTLAdapter()
    mean_acts = {0: np.zeros(8, dtype=np.float32), 1: np.zeros(8, dtype=np.float32)}
    lp = patch_selected_layers_from_reference(
        adapter, "hello world", "yes indeed", [0, 1], mean_acts, max_length=16)
    assert np.isfinite(lp)


def test_patched_margin_finite():
    adapter = MockTLAdapter()
    mean_acts = {0: np.zeros(8, dtype=np.float32)}
    margin = compute_patched_behavior_margin(
        adapter, "hello world", "yes good", "no bad", [0], mean_acts, max_length=16)
    assert np.isfinite(margin)


def test_run_patching_validation_structure():
    adapter = MockTLAdapter(n_layers=4)
    pairs = pd.DataFrame({
        "prompt": ["alpha beta", "gamma delta"],
        "sycophantic_response": ["yes a", "yes b"],
        "non_sycophantic_response": ["no a", "no b"],
    })
    mean_acts = np.zeros((4, 8), dtype=np.float32)
    res = run_patching_behavior_validation(
        adapter, pairs, top_layers=[0, 1, 2], mean_acts=mean_acts,
        k=2, random_trials=1, max_examples=2, seed=0)
    assert "probe_gradient" in res and "random" in res
    for v in res.values():
        assert "behavior_margin_delta" in v


def test_patching_rejects_non_tl():
    class NotTL:
        pass
    with pytest.raises(NotImplementedError):
        patch_selected_layers_from_reference(NotTL(), "p", "c", [0], {0: np.zeros(8)})
