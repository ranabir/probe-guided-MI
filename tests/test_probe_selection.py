"""Tests for attribution probe selection policies."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes import LinearProbe, save_probe, select_attribution_probe


def _save_real_probes(model_name, n_layers, tmp_path, probe_position="test_pos", seed=0):
    """Fit and save a real (picklable) LinearProbe per layer."""
    rng = np.random.default_rng(seed)
    with patch("src.probes.artifacts_dir", return_value=tmp_path):
        for i in range(n_layers):
            X = rng.normal(size=(40, 6))
            y = (X[:, 0] > 0).astype(int)
            p = LinearProbe().fit(X, y)
            p.layer_idx = i
            save_probe(p, model_name, i, probe_position=probe_position)


def _metrics(n_layers, planted):
    """Build a metrics list with planted val_auroc values."""
    rows = []
    for i in range(n_layers):
        rows.append({"layer": i, "val_auroc": planted.get(i, 0.5),
                     "val_accuracy": 0.6, "val_f1": 0.6, "train_accuracy": 1.0,
                     "train_auroc": 1.0, "train_f1": 1.0, "test_accuracy": 0.6,
                     "test_auroc": 0.6, "test_f1": 0.6})
    return rows


def test_best_any_policy(tmp_path):
    n_layers = 12
    _save_real_probes("test-model", n_layers, tmp_path)
    metrics = _metrics(n_layers, {3: 0.99, 9: 0.95})
    with patch("src.probes.artifacts_dir", return_value=tmp_path):
        _, meta = select_attribution_probe(metrics, n_layers, "test-model",
                                           probe_position="test_pos", policy="best_any")
    assert meta["selected_layer"] == 3
    assert meta["selected_policy"] == "best_any"


def test_best_late_policy(tmp_path):
    n_layers = 12
    _save_real_probes("test-model", n_layers, tmp_path)
    metrics = _metrics(n_layers, {3: 0.99, 9: 0.95})
    with patch("src.probes.artifacts_dir", return_value=tmp_path):
        _, meta = select_attribution_probe(metrics, n_layers, "test-model",
                                           probe_position="test_pos", policy="best_late",
                                           min_probe_layer_frac=0.65)
    # layers >= 0.65*12 = 7.8 -> >= 8; best among those is layer 9
    assert meta["selected_layer"] >= 8
    assert meta["selected_policy"] == "best_late"


def test_final_layer_policy(tmp_path):
    n_layers = 6
    _save_real_probes("test-model2", n_layers, tmp_path, probe_position="test_pos2")
    metrics = _metrics(n_layers, {i: 0.6 + i * 0.05 for i in range(n_layers)})
    with patch("src.probes.artifacts_dir", return_value=tmp_path):
        _, meta = select_attribution_probe(metrics, n_layers, "test-model2",
                                           probe_position="test_pos2", policy="final_layer")
    assert meta["selected_layer"] == n_layers - 1
    assert meta["selected_policy"] == "final_layer"


def test_invalid_policy():
    with pytest.raises(ValueError):
        select_attribution_probe([{"layer": 0, "val_auroc": 0.5}], 12, "x", policy="nonexistent_policy")
