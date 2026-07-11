"""Tests that causal-control plotting functions create files from tiny fake CSVs."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import plotting


def _results_df():
    return pd.DataFrame({
        "model_name": ["m"] * 6,
        "layer_selection": ["causal_topk", "causal_topk", "decodable_topk", "decodable_topk", "random", "random"],
        "layers": ["[1, 2, 3]"] * 2 + ["[6, 7, 8]"] * 2 + ["[0, 4, 5]"] * 2,
        "intervention": ["probe_steering", "mean_ablation"] * 3,
        "setting": ["alpha=+3", "mean"] * 3,
        "behavior_margin_delta": [-0.05, -0.01, 0.02, -0.03, 0.01, -0.02],
        "answer_flip_rate": [0.1, 0.0, 0.0, 0.05, 0.0, 0.0],
        "targeted_syc_to_non_syc_flip_rate": [0.2, 0.0, 0.0, 0.1, 0.0, 0.0],
        "ci_low": [-0.1, -0.05, -0.02, -0.08, -0.03, -0.05],
        "ci_high": [0.0, 0.03, 0.06, 0.02, 0.05, 0.01],
    })


def _sweep_df():
    return pd.DataFrame({
        "layer": range(9),
        "test_pearson": np.linspace(0.2, 0.5, 9),
        "behavior_margin_delta": np.linspace(-0.2, 0.0, 9),
    })


def test_contrastive_causal_plots(tmp_path):
    with patch("src.plotting.get_root", return_value=tmp_path):
        entries = plotting.plot_contrastive_causal_results(_results_df(), "gpt2-small")
    assert len(entries) == 2
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_contrastive_causal_answer_flip.png").exists()
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_contrastive_causal_behavior_delta.png").exists()


def test_causal_vs_decodable_plot(tmp_path):
    with patch("src.plotting.get_root", return_value=tmp_path):
        e = plotting.plot_causal_vs_decodable_intervention(_sweep_df(), _results_df(), "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_causal_vs_decodable_layer_intervention.png").exists()
    assert e is not None


def test_side_effect_plot(tmp_path):
    metrics = {"side_effect_score": 0.3, "weirdness_rate": 0.4, "repetition_increase": 0.1,
               "qa_accuracy_drop": 0.0, "output_length_ratio": 1.2}
    with patch("src.plotting.get_root", return_value=tmp_path):
        plotting.plot_side_effect_summary(metrics, "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_side_effect_summary.png").exists()


def test_search_plots(tmp_path):
    ranked = pd.DataFrame({
        "rank": [1, 2, 3], "layer_selection_type": ["causal_topk", "decodable_topk", "causal_topk"],
        "direction_type": ["regression", "diff_of_means", "regression"],
        "intervention_type": ["probe_steering", "activation_capping", "probe_steering"],
        "alpha_or_cap": [3.0, 0.75, -1.0],
        "targeted_flip_rate": [0.5, 0.1, 0.3], "side_effect_score": [0.4, 0.05, 0.2],
        "objective_score": [0.3, 0.075, 0.2],
    })
    with patch("src.plotting.get_root", return_value=tmp_path):
        plotting.plot_intervention_search_pareto(ranked, "gpt2-small")
        plotting.plot_best_interventions_ranked(ranked, "gpt2-small")
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_intervention_search_pareto.png").exists()
    assert (tmp_path / "plots" / "gpt2-small" / "gpt2-small_best_interventions_ranked.png").exists()
