"""Tests for linear probe training."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes import LinearProbe


def _make_data(n=100, d=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Simple separable labels
    direction = rng.normal(size=(d,))
    direction /= np.linalg.norm(direction)
    y = (X @ direction > 0).astype(int)
    return X, y


def test_probe_fit_predict():
    X, y = _make_data(200, 32)
    probe = LinearProbe(standardize=True)
    probe.fit(X[:150], y[:150])
    preds = probe.predict(X[150:])
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})


def test_probe_accuracy_above_chance():
    X, y = _make_data(400, 32)
    probe = LinearProbe(standardize=True)
    probe.fit(X[:300], y[:300])
    metrics = probe.evaluate(X[300:], y[300:])
    assert metrics["accuracy"] > 0.55, f"Expected > 0.55, got {metrics['accuracy']}"
    assert 0 <= metrics["auroc"] <= 1


def test_probe_proba_range():
    X, y = _make_data(100, 16)
    probe = LinearProbe()
    probe.fit(X[:80], y[:80])
    proba = probe.predict_proba(X[80:])
    assert proba.shape == (20,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_probe_no_standardize():
    X, y = _make_data(100, 16)
    probe = LinearProbe(standardize=False)
    probe.fit(X[:80], y[:80])
    metrics = probe.evaluate(X[80:], y[80:])
    assert "accuracy" in metrics


def test_probe_coef_shape():
    X, y = _make_data(100, 32)
    probe = LinearProbe()
    probe.fit(X, y)
    # coef_ should be [1, D] for binary classification
    assert probe.coef_.shape[1] == 32
