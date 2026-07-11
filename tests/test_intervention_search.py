"""Tests for intervention search grid + ranking."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import build_intervention_grid, compute_objective, rank_interventions


def test_grid_creation():
    grid = build_intervention_grid(
        {"causal_topk": [1, 2], "decodable_topk": [8, 9]},
        directions=["regression", "diff_of_means"],
        interventions=["probe_steering", "activation_capping"],
        alphas=[-1, 1], cap_quantiles=[0.5, 0.9],
    )
    # 2 selections * 2 directions * (2 alphas + 2 cap_quantiles) = 16
    assert len(grid) == 16
    assert all("intervention" in c for c in grid)


def test_grid_contrastive_single_per_combo():
    grid = build_intervention_grid(
        {"causal_topk": [1]}, directions=["regression"],
        interventions=["contrastive_patching"], alphas=[-1, 1], cap_quantiles=[0.5],
    )
    assert len(grid) == 1
    assert grid[0]["alpha_or_cap"] is None


def test_objective_formula():
    # objective = flip - lambda * side_effect
    assert compute_objective(0.8, 0.4, 0.5) == pytest.approx(0.8 - 0.2)
    assert compute_objective(0.0, 0.0, 0.5) == 0.0


def test_objective_handles_nan():
    assert compute_objective(float("nan"), 0.5, 0.5) == pytest.approx(-0.25)
    assert compute_objective(0.5, float("nan"), 0.5) == pytest.approx(0.5)


def test_ranking_orders_descending():
    rows = [
        {"targeted_flip_rate": 0.2, "side_effect_score": 0.1},
        {"targeted_flip_rate": 0.8, "side_effect_score": 0.7},
        {"targeted_flip_rate": 0.5, "side_effect_score": 0.0},
    ]
    ranked = rank_interventions(rows, lambda_side_effect=0.5)
    assert list(ranked["rank"]) == [1, 2, 3]
    # objective: 0.15, 0.45, 0.50 -> best is row3 (0.5)
    assert ranked.iloc[0]["objective_score"] == pytest.approx(0.5)
    assert ranked["objective_score"].is_monotonic_decreasing


def test_ranking_empty():
    ranked = rank_interventions([], lambda_side_effect=0.5)
    assert ranked.empty
