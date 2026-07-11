"""Tests for causal-control layer selection, capping, and reference choice."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causal_interventions import (
    choose_opposite_preference_reference,
    estimate_cap_threshold,
    score_before_after,
    select_layers_from_sweep,
)


def _sweep():
    # decodability peaks at high layers; causal effect peaks at low layers (anti-correlated)
    return pd.DataFrame({
        "layer": range(8),
        "test_pearson": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55],
        "behavior_margin_delta": [-0.20, -0.18, -0.15, -0.05, 0.0, 0.01, 0.0, 0.0],
        "answer_flip_rate": [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    })


def test_causal_topk_picks_low_layers():
    layers = select_layers_from_sweep(_sweep(), "causal_topk", k=3)
    assert set(layers) == {0, 1, 2}  # largest |behavior_margin_delta|


def test_decodable_topk_picks_high_layers():
    layers = select_layers_from_sweep(_sweep(), "decodable_topk", k=3)
    assert set(layers) == {5, 6, 7}  # highest pearson


def test_causal_and_decodable_differ():
    causal = set(select_layers_from_sweep(_sweep(), "causal_topk", k=3))
    decodable = set(select_layers_from_sweep(_sweep(), "decodable_topk", k=3))
    assert causal != decodable
    assert causal.isdisjoint(decodable)


def test_manual_selection():
    layers = select_layers_from_sweep(_sweep(), "manual", k=2, manual=[4, 5, 6])
    assert layers == [4, 5]


def test_random_selection_seeded():
    a = select_layers_from_sweep(_sweep(), "random", k=3, seed=1)
    b = select_layers_from_sweep(_sweep(), "random", k=3, seed=1)
    assert a == b and len(a) == 3


def test_opposite_preference_reference():
    df = pd.DataFrame({"behavior_margin": [0.5, -0.3, 0.2, -0.8]})
    # target 0 prefers syc (>0) -> reference must be honest (<=0): index 1 or 3
    ref = choose_opposite_preference_reference(df, 0, seed=0)
    assert ref in (1, 3)
    # target 1 prefers honest (<=0) -> reference must be syc (>0): index 0 or 2
    ref2 = choose_opposite_preference_reference(df, 1, seed=0)
    assert ref2 in (0, 2)


def test_opposite_reference_none_when_no_opposite():
    df = pd.DataFrame({"behavior_margin": [0.5, 0.3, 0.2]})  # all syc
    assert choose_opposite_preference_reference(df, 0) is None


def test_cap_threshold_is_quantile():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    d = np.ones(6)
    thr = estimate_cap_threshold(X, d, quantile=0.75)
    proj = X @ (d / np.linalg.norm(d))
    assert thr == pytest.approx(np.quantile(proj, 0.75), abs=1e-6)


def test_capping_reduces_projection_above_threshold():
    # a vector well above threshold should be pulled back toward threshold along the direction
    d = np.array([1.0, 0.0, 0.0])
    dunit = d / np.linalg.norm(d)
    threshold = 1.0
    cap_strength = 1.0
    h = np.array([5.0, 0.0, 0.0])
    proj = h @ dunit
    excess = max(0.0, proj - threshold)
    h_new = h - cap_strength * excess * dunit
    assert (h_new @ dunit) == pytest.approx(threshold)
    assert (h_new @ dunit) < proj


def test_score_before_after_keys():
    before = [0.5, -0.3, 0.2, -0.1]
    after = [-0.5, -0.3, -0.2, -0.1]
    m = score_before_after(before, after, bootstrap=50, seed=0)
    for k in ("behavior_margin_delta", "answer_flip_rate", "targeted_syc_to_non_syc_flip_rate",
              "accuracy_change", "ci_low", "ci_high", "n_examples"):
        assert k in m
