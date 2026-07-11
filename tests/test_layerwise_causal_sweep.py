"""Tests for layerwise causal sweep outputs (model-free, via plotting contract)."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import plotting

EXPECTED_COLUMNS = {
    "model_name", "layer", "layer_frac", "intervention_type",
    "test_pearson", "test_spearman", "test_r2",
    "behavior_margin_delta", "bootstrap_ci_low", "bootstrap_ci_high",
    "answer_flip_rate", "accuracy_change", "n_examples",
}


def _fake_sweep(n_layers=8):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "model_name": ["m"] * n_layers,
        "layer": range(n_layers),
        "layer_frac": [i / (n_layers - 1) for i in range(n_layers)],
        "intervention_type": ["activation_patching"] * n_layers,
        "test_pearson": np.linspace(0.2, 0.5, n_layers),
        "test_spearman": np.linspace(0.15, 0.4, n_layers),
        "test_r2": np.linspace(-0.5, 0.1, n_layers),
        "behavior_margin_delta": rng.normal(0, 0.02, n_layers),
        "bootstrap_ci_low": rng.normal(-0.05, 0.01, n_layers),
        "bootstrap_ci_high": rng.normal(0.05, 0.01, n_layers),
        "answer_flip_rate": rng.uniform(0, 0.1, n_layers),
        "accuracy_change": rng.normal(0, 0.02, n_layers),
        "n_examples": [25] * n_layers,
    })


def test_sweep_columns_contract():
    df = _fake_sweep()
    assert EXPECTED_COLUMNS.issubset(set(df.columns))


def test_decodability_vs_causal_plot(tmp_path):
    df = _fake_sweep()
    with patch("src.plotting.get_root", return_value=tmp_path):
        entry = plotting.plot_decodability_vs_causal(df, "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_decodability_vs_causal_effect.png").exists()
    assert "source" in entry


def test_layerwise_behavior_delta_plot(tmp_path):
    df = _fake_sweep()
    with patch("src.plotting.get_root", return_value=tmp_path):
        plotting.plot_layerwise_behavior_delta(df, "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_layerwise_behavior_delta.png").exists()


def test_layerwise_answer_flip_plot(tmp_path):
    df = _fake_sweep()
    with patch("src.plotting.get_root", return_value=tmp_path):
        plotting.plot_layerwise_answer_flip(df, "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_layerwise_answer_flip_rate.png").exists()
