"""Tests for side-effect evaluation metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.side_effects import (
    compute_repetition_score,
    compute_side_effect_score,
    compute_weirdness_flags,
    load_basic_prompts,
)


def test_basic_prompts_load():
    df = load_basic_prompts()
    assert len(df) >= 15
    assert "prompt" in df.columns


def test_repetition_score_range():
    assert compute_repetition_score("the the the the") > 0.5
    assert compute_repetition_score("a fully unique diverse sentence here") == pytest.approx(0.0)
    assert compute_repetition_score("") == 0.0


def test_weirdness_flags_repeated():
    flags = compute_weirdness_flags("go go go go go go go go")
    assert flags["repeated"] == 1


def test_weirdness_flags_empty():
    flags = compute_weirdness_flags("")
    assert flags["empty"] == 1


def test_weirdness_flags_incoherent():
    flags = compute_weirdness_flags("aaaaaaaaaaaa")
    assert flags["incoherent"] == 1


def test_weirdness_flags_clean():
    flags = compute_weirdness_flags("Paris is the capital of France.")
    assert all(v == 0 for v in flags.values())


def test_side_effect_score_finite_and_bounded():
    base = ["Paris is the capital of France.", "Two plus two is four."]
    interv = ["Paris is the capital of France.", "Two plus two is four."]
    out = compute_side_effect_score(base, interv)
    assert np.isfinite(out["side_effect_score"])
    assert 0.0 <= out["side_effect_score"] <= 1.0


def test_side_effect_detects_degradation():
    base = ["Paris is the capital of France and a lovely city."]
    interv = ["go go go go go go go go go go"]  # repetitive/weird
    out = compute_side_effect_score(base, interv)
    assert out["side_effect_score"] > 0.2
    assert out["weirdness_rate"] == 1.0


def test_side_effect_qa_drop():
    base = ["The answer is paris."]
    interv = ["I do not know anything useful here."]
    out = compute_side_effect_score(base, interv, answers=["paris"])
    assert out["qa_accuracy_drop"] == pytest.approx(1.0)
