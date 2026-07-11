"""Tests for the plotting module (uses tiny fake CSVs, patched plots dir)."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import plotting


def _metrics_csv(tmp_path):
    df = pd.DataFrame({
        "layer": range(6),
        "val_pearson": [0.1, 0.2, 0.15, 0.25, 0.3, 0.28],
        "test_pearson": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        "test_spearman": [0.18, 0.22, 0.28, 0.3, 0.36, 0.4],
    })
    p = tmp_path / "metrics.csv"
    df.to_csv(p, index=False)
    return p


def _causal_csv(tmp_path):
    df = pd.DataFrame({
        "method": ["probe_gradient", "random", "logit_gradient"],
        "probe_delta": [-0.15, -0.03, 0.0],
        "behavior_margin_delta": [-0.05, 0.02, 0.01],
    })
    p = tmp_path / "causal.csv"
    df.to_csv(p, index=False)
    return p


def test_probe_regression_plot(tmp_path):
    with patch("src.plotting.get_root", return_value=tmp_path):
        entry = plotting.plot_probe_regression_by_layer(_metrics_csv(tmp_path), "gpt2-small", 300)
    assert entry is not None
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_probe_regression_by_layer.png").exists()
    assert "source" in entry and "read" in entry


def test_margin_distribution_plot(tmp_path):
    pref = tmp_path / "pref.csv"
    pd.DataFrame({"behavior_margin": np.random.randn(50)}).to_csv(pref, index=False)
    with patch("src.plotting.get_root", return_value=tmp_path):
        entry = plotting.plot_behavior_margin_distribution(pref, "gpt2-small")
    assert entry is not None
    assert Path(tmp_path / "plots" / "gpt2-small" / "gpt2-small_behavior_margin_distribution.png").exists()


def test_causal_plots(tmp_path):
    c = _causal_csv(tmp_path)
    with patch("src.plotting.get_root", return_value=tmp_path):
        e1 = plotting.plot_causal_probe_delta(c, "gpt2-small")
        e2 = plotting.plot_causal_behavior_margin_delta(c, "gpt2-small")
    assert e1 is not None and e2 is not None


def test_missing_source_returns_none(tmp_path):
    with patch("src.plotting.get_root", return_value=tmp_path):
        entry = plotting.plot_probe_regression_by_layer(tmp_path / "nope.csv", "gpt2-small")
    assert entry is None


def test_update_readme(tmp_path):
    entries = [{"file": "plots/x.png", "model": "gpt2-small", "experiment": "exp",
                "shows": "stuff", "read": "how", "source": "results/tables/x.csv"}]
    with patch("src.plotting.get_root", return_value=tmp_path):
        readme = plotting.update_plots_readme(entries)
    assert readme.exists()
    text = readme.read_text()
    assert "plots/x.png" in text
    assert "| Plot file |" in text
