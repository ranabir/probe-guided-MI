"""Tests for bootstrap confidence intervals."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistics import bootstrap_mean_ci, bootstrap_metric_ci, pearson_on_pairs


def test_ci_brackets_mean():
    vals = list(np.random.default_rng(0).normal(5.0, 1.0, size=200))
    out = bootstrap_mean_ci(vals, n_boot=500, seed=0)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert out["n"] == 200


def test_ci_deterministic_with_seed():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = bootstrap_mean_ci(vals, n_boot=300, seed=123)
    b = bootstrap_mean_ci(vals, n_boot=300, seed=123)
    assert a == b


def test_empty_returns_nan():
    out = bootstrap_mean_ci([], seed=0)
    assert np.isnan(out["mean"]) and out["n"] == 0


def test_single_value():
    out = bootstrap_mean_ci([3.5], seed=0)
    assert out["mean"] == 3.5 and out["ci_low"] == 3.5 and out["ci_high"] == 3.5


def test_nan_filtered():
    out = bootstrap_mean_ci([1.0, float("nan"), 3.0], n_boot=100, seed=0)
    assert out["n"] == 2


def test_metric_ci_pearson():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = x * 0.8 + rng.normal(size=100) * 0.3
    pairs = list(zip(x, y))
    out = bootstrap_metric_ci(pearson_on_pairs, pairs, n_boot=300, seed=0)
    assert out["ci_low"] <= out["value"] <= out["ci_high"]
    assert out["value"] > 0.5
