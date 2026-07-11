"""Tests for behavior metrics: logprob computation and margin calculation."""
import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior_metrics import (
    compute_continuation_logprob,
    compute_pair_logprob_margin,
    compute_behavior_margins_for_dataset,
    summarize_behavior_margins,
)


class MockAdapter:
    """Minimal mock adapter for testing logprob computation."""

    def __init__(self, logprob_fn=None):
        self._logprob_fn = logprob_fn or (lambda p, c, **kw: -len(c.split()) * 2.0)

    def compute_logprob(self, prompt, continuation, max_length=128, normalize=False):
        val = self._logprob_fn(prompt, continuation, max_length=max_length)
        if normalize:
            n = max(len(continuation.split()), 1)
            return val / n
        return val


def test_continuation_logprob_finite():
    adapter = MockAdapter(logprob_fn=lambda p, c, **kw: -7.5)
    lp = compute_continuation_logprob(adapter, "prompt", "some text", normalize=False)
    assert np.isfinite(lp)
    assert lp == pytest.approx(-7.5)


def test_continuation_logprob_normalization():
    # raw = -len(words)*2; normalized = raw / num_words = -2.0 per token
    adapter = MockAdapter()
    raw = compute_continuation_logprob(adapter, "p", "a b c d", normalize=False)
    norm = compute_continuation_logprob(adapter, "p", "a b c d", normalize=True)
    assert raw == pytest.approx(-8.0)
    assert norm == pytest.approx(-2.0)


def test_compute_pair_margin_basic():
    # normalize=False to assert exact raw difference logic
    adapter = MockAdapter(logprob_fn=lambda p, c, **kw: -10.0 if "Yes" in c else -15.0)
    result = compute_pair_logprob_margin(
        adapter, "Is the sky blue?", "Yes, the sky is blue.", "No, actually it is green.",
        normalize=False,
    )
    assert "syc_logprob" in result
    assert "non_syc_logprob" in result
    assert "behavior_margin" in result
    assert "prefers_sycophancy" in result
    assert result["behavior_margin"] == pytest.approx(-10.0 - (-15.0))
    assert result["prefers_sycophancy"] == 1


def test_compute_pair_margin_prefers_honest():
    adapter = MockAdapter(logprob_fn=lambda p, c, **kw: -20.0 if "Yes" in c else -5.0)
    result = compute_pair_logprob_margin(adapter, "prompt", "Yes syc", "No honest", normalize=False)
    assert result["behavior_margin"] < 0
    assert result["prefers_sycophancy"] == 0


def test_compute_pair_margin_graceful_failure():
    def bad_fn(p, c, **kw):
        raise RuntimeError("model failed")
    adapter = MockAdapter(logprob_fn=bad_fn)
    result = compute_pair_logprob_margin(adapter, "prompt", "syc", "non")
    assert np.isnan(result["behavior_margin"])
    assert result["prefers_sycophancy"] == -1


def test_compute_behavior_margins_for_dataset_paired():
    adapter = MockAdapter(logprob_fn=lambda p, c, **kw: -8.0 if "Yes" in c else -12.0)
    df = pd.DataFrame({
        "syc_response": ["Yes A", "Yes B", "Yes C"],
        "non_syc_response": ["No A", "No B", "No C"],
        "prompt": ["p1", "p2", "p3"],
    })
    result = compute_behavior_margins_for_dataset(adapter, df)
    assert len(result) == 3
    assert (result["behavior_margin"] > 0).all()


def test_compute_behavior_margins_for_dataset_flat():
    adapter = MockAdapter(logprob_fn=lambda p, c, **kw: -8.0 if "Absolutely" in c else -12.0)
    df = pd.DataFrame({
        "prompt": ["p1", "p1", "p2", "p2"],
        "response": ["Absolutely agree", "No actually", "Absolutely yes", "No correct"],
        "label": [1, 0, 1, 0],
    })
    result = compute_behavior_margins_for_dataset(adapter, df)
    assert len(result) >= 1


def test_summarize_behavior_margins():
    margins_df = pd.DataFrame({
        "behavior_margin": [0.5, -0.3, 1.2, 0.1, -0.8],
    })
    summary = summarize_behavior_margins(margins_df)
    assert "mean_margin" in summary
    assert "frac_prefers_syc" in summary
    assert summary["n"] == 5
    assert 0 <= summary["frac_prefers_syc"] <= 1


def test_summarize_empty():
    margins_df = pd.DataFrame({"behavior_margin": [float("nan"), float("nan")]})
    summary = summarize_behavior_margins(margins_df)
    assert np.isnan(summary["mean_margin"])
    assert summary["n"] == 0
