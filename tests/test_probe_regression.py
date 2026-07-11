"""Tests for regression probes (behavior_margin target)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes import LinearProbe, train_all_layer_probes


def _make_regression_data(n=120, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    direction = rng.normal(size=(d,))
    direction /= np.linalg.norm(direction)
    y = (X @ direction + rng.normal(size=n) * 0.1).astype(np.float32)
    return X, y, direction


def test_regression_probe_trains():
    X, y, _ = _make_regression_data()
    probe = LinearProbe(task="regression")
    probe.fit(X[:90], y[:90])
    preds = probe.predict(X[90:])
    assert preds.shape == (30,)


def test_regression_metrics_present():
    X, y, _ = _make_regression_data()
    probe = LinearProbe(task="regression")
    probe.fit(X[:90], y[:90])
    metrics = probe.evaluate(X[90:], y[90:])
    for key in ("mse", "mae", "r2", "pearson", "spearman"):
        assert key in metrics, f"missing {key}"


def test_regression_recovers_signal():
    X, y, _ = _make_regression_data(n=200, d=16)
    probe = LinearProbe(task="regression")
    probe.fit(X[:150], y[:150])
    metrics = probe.evaluate(X[150:], y[150:])
    # Strong linear signal -> high Pearson
    assert metrics["pearson"] > 0.7, f"Expected >0.7, got {metrics['pearson']}"


def test_predict_score_regression_returns_value():
    X, y, _ = _make_regression_data()
    probe = LinearProbe(task="regression")
    probe.fit(X, y)
    scores = probe.predict_score(X[:10])
    assert scores.shape == (10,)
    # regression scores are not bounded to [0,1]
    assert scores.dtype.kind == "f"


def test_train_all_layer_probes_regression(tmp_path):
    from unittest.mock import patch
    N, L, D = 100, 5, 16
    rng = np.random.default_rng(1)
    direction = rng.normal(size=(D,)); direction /= np.linalg.norm(direction)
    X = rng.normal(size=(N, L, D)).astype(np.float32)
    y = (X[:, 3, :] @ direction).astype(np.float32)  # signal in layer 3

    with patch("src.probes.artifacts_dir", return_value=tmp_path):
        metrics, best = train_all_layer_probes(
            X[:70], y[:70], X[70:85], y[70:85], X[85:], y[85:],
            model_name="testreg", probe_position="prompt_final",
            input_format="prompt_preferences", task="regression",
        )
    assert len(metrics) == L
    assert "val_pearson" in metrics[0]
    assert best is not None
    # best should be near layer 3 where the signal lives
    assert best.layer_idx == 3
