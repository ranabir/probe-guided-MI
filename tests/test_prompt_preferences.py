"""Tests for prompt-level preference dataset construction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior_metrics import build_prompt_preference_dataset, _coerce_to_pairs


class MockAdapter:
    """Returns a deterministic logprob based on content, so margins are predictable."""
    def compute_logprob(self, prompt, continuation, max_length=128, normalize=True):
        base = -2.0 if "Yes" in continuation else -3.0
        return base  # already "mean per token" style constant


def test_paired_rows_collapse_to_one_row_per_prompt():
    df = pd.DataFrame({
        "id": ["p0", "p1"],
        "prompt": ["Is A true?", "Is B true?"],
        "sycophantic_response": ["Yes A", "Yes B"],
        "non_sycophantic_response": ["No A", "No B"],
        "source_dataset": ["synthetic", "synthetic"],
        "subset": ["x", "y"],
    })
    out = build_prompt_preference_dataset(MockAdapter(), df)
    assert len(out) == 2  # one row per prompt
    assert set(["syc_logprob", "non_syc_logprob", "behavior_margin", "prefers_sycophancy"]).issubset(out.columns)


def test_behavior_margin_is_difference():
    df = pd.DataFrame({
        "id": ["p0"],
        "prompt": ["Is A true?"],
        "sycophantic_response": ["Yes A"],
        "non_sycophantic_response": ["No A"],
    })
    out = build_prompt_preference_dataset(MockAdapter(), df)
    row = out.iloc[0]
    assert row["behavior_margin"] == pytest.approx(row["syc_logprob"] - row["non_syc_logprob"])
    # Yes -> -2.0, No -> -3.0, margin = +1.0
    assert row["behavior_margin"] == pytest.approx(1.0)


def test_prefers_sycophancy_threshold():
    df = pd.DataFrame({
        "id": ["p0", "p1"],
        "prompt": ["q1", "q2"],
        "sycophantic_response": ["Yes good", "Nope bad"],  # second has no "Yes" -> -3.0
        "non_sycophantic_response": ["No good", "Yes alt"],  # second non has "Yes" -> -2.0
    })
    out = build_prompt_preference_dataset(MockAdapter(), df)
    # p0: syc -2.0, non -3.0, margin +1 -> prefers 1
    # p1: syc -3.0, non -2.0, margin -1 -> prefers 0
    prefs = dict(zip(out["prompt"], out["prefers_sycophancy"]))
    assert prefs["q1"] == 1
    assert prefs["q2"] == 0


def test_coerce_flat_to_pairs():
    flat = pd.DataFrame({
        "id": ["a_syc", "a_non", "b_syc", "b_non"],
        "prompt": ["pa", "pa", "pb", "pb"],
        "response": ["Yes a", "No a", "Yes b", "No b"],
        "label": [1, 0, 1, 0],
    })
    pairs = _coerce_to_pairs(flat)
    assert len(pairs) == 2
    assert "sycophantic_response" in pairs.columns
    assert "non_sycophantic_response" in pairs.columns


def test_one_row_per_prompt_no_duplicates():
    df = pd.DataFrame({
        "id": ["p0", "p1", "p2"],
        "prompt": ["q1", "q2", "q3"],
        "sycophantic_response": ["Yes 1", "Yes 2", "Yes 3"],
        "non_sycophantic_response": ["No 1", "No 2", "No 3"],
    })
    out = build_prompt_preference_dataset(MockAdapter(), df)
    assert out["prompt"].nunique() == len(out)
