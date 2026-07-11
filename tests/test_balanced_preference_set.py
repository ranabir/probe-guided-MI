"""Tests for the balanced preference diagnostic set."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_balanced_subset


def _df(margins):
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(len(margins))],
        "prompt": [f"q{i}" for i in range(len(margins))],
        "behavior_margin": margins,
        "prefers_sycophancy": [1 if m > 0 else 0 for m in margins],
    })


def test_balanced_equal_when_symmetric():
    df = _df([0.5, 0.3, -0.2, -0.4, 0.1, -0.9])
    bal, info = build_balanced_subset(df, seed=0)
    n_pos = int((bal["behavior_margin"] > 0).sum())
    n_neg = int((bal["behavior_margin"] <= 0).sum())
    assert n_pos == n_neg
    assert info["n_each"] == 3


def test_balanced_few_positives():
    # 2 positives, 8 negatives -> balanced should be 2 + 2
    df = _df([0.5, 0.2] + [-0.1 * (i + 1) for i in range(8)])
    bal, info = build_balanced_subset(df, seed=0)
    n_pos = int((bal["behavior_margin"] > 0).sum())
    n_neg = int((bal["behavior_margin"] <= 0).sum())
    assert n_pos == 2 and n_neg == 2
    assert info["n_pos_total"] == 2
    assert info["n_neg_total"] == 8


def test_balanced_records_true_imbalance():
    df = _df([0.5] + [-0.1] * 19)
    bal, info = build_balanced_subset(df, seed=0)
    assert info["frac_pos_natural"] == pytest.approx(1 / 20)
    assert len(bal) == 2  # 1 pos + 1 neg


def test_balanced_empty_class():
    df = _df([-0.1, -0.2, -0.3])  # no positives
    bal, info = build_balanced_subset(df, seed=0)
    assert len(bal) == 0
    assert info["n_each"] == 0


def test_balanced_handles_nan_margins():
    df = _df([0.5, -0.3, 0.2, -0.4])
    df.loc[0, "behavior_margin"] = np.nan
    bal, info = build_balanced_subset(df, seed=0)
    # one positive dropped (nan) -> 1 pos, 2 neg -> balanced 1+1
    assert info["n_pos_total"] == 1
    assert len(bal) == 2
