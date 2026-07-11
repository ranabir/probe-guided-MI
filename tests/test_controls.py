"""Tests for control probes (confound checks)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls import (
    build_static_feature_targets,
    layerwise_decodability,
    run_all_controls,
    shuffle_target,
    topic_codes,
)


def _pref_df(n=20):
    return pd.DataFrame({
        "prompt": [f"I think thing {i}. Do you agree?" if i % 2 else f"Statement {i}." for i in range(n)],
        "sycophantic_response": [f"(A) Agree {i}" if i % 2 else f"Yes {i}" for i in range(n)],
        "non_sycophantic_response": [f"(B) Disagree {i}" for i in range(n)],
        "subset": (["philpapers", "nlp_survey"] * (n // 2 + 1))[:n],
    })


def test_shuffle_target_permutes():
    y = np.arange(10).astype(float)
    sh = shuffle_target(y, seed=0)
    assert sorted(sh) == sorted(y)        # same multiset
    assert not np.array_equal(sh, y)      # but reordered


def test_static_features_built():
    feats = build_static_feature_targets(_pref_df())
    for key in ("prompt_length_tokens", "contains_i_think", "contains_do_you_agree",
                "contains_am_i_right", "syc_response_starts_A"):
        assert key in feats
        assert len(feats[key]) == 20


def test_contains_do_you_agree_detects():
    feats = build_static_feature_targets(_pref_df())
    # odd-index prompts contain "Do you agree"
    assert feats["contains_do_you_agree"].sum() > 0


def test_topic_codes():
    codes = topic_codes(_pref_df())
    assert codes is not None
    assert len(np.unique(codes)) == 2


def test_topic_codes_none_when_missing():
    df = pd.DataFrame({"prompt": ["a", "b"], "sycophantic_response": ["x", "y"],
                       "non_sycophantic_response": ["p", "q"]})
    assert topic_codes(df) is None


def test_layerwise_decodability_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4, 8)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    rows = layerwise_decodability(X[:30], y[:30], X[30:], y[30:], task="regression")
    assert len(rows) == 4
    assert "test_metric" in rows[0]


def test_run_all_controls_handles_missing_topic():
    rng = np.random.default_rng(0)
    splits = {
        "train": {"hidden_states": rng.normal(size=(30, 3, 8)).astype(np.float32),
                  "behavior_margin": rng.normal(size=30).astype(np.float32)},
        "test": {"hidden_states": rng.normal(size=(10, 3, 8)).astype(np.float32),
                 "behavior_margin": rng.normal(size=10).astype(np.float32)},
    }
    pref = {"train": _pref_df(30).iloc[:30].drop(columns=["subset"]),
            "test": _pref_df(10).iloc[:10].drop(columns=["subset"])}
    df = run_all_controls(splits, pref, controls=["random_label", "static_token", "topic"])
    # topic skipped gracefully (no subset col); random_label + static present
    assert "random_label" in set(df["control_name"])
    assert "topic" not in set(df["control_name"])
