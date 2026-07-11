"""Tests for A/B answer-flip and accuracy-change metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (
    accuracy_change,
    accuracy_from_margins,
    answer_flip_rate,
    behavioral_intervention_metrics,
    targeted_syc_to_non_syc_flip_rate,
)


def test_answer_flip_rate_basic():
    before = [0.5, -0.3, 0.2, -0.1]   # prefers: syc, non, syc, non
    after = [-0.5, -0.3, 0.2, 0.1]    # prefers: non, non, syc, syc
    # flips at index 0 (syc->non) and index 3 (non->syc) => 2/4
    assert answer_flip_rate(before, after) == pytest.approx(0.5)


def test_no_flip():
    before = [0.5, -0.3]
    after = [0.4, -0.2]
    assert answer_flip_rate(before, after) == 0.0


def test_targeted_flip_only_counts_syc_to_non():
    before = [0.5, 0.3, -0.1]   # two start syc
    after = [-0.5, 0.3, 0.4]    # idx0 syc->non (counts), idx1 stays syc, idx2 non->syc (ignored)
    # started syc = 2; flipped syc->non = 1 => 0.5
    assert targeted_syc_to_non_syc_flip_rate(before, after) == pytest.approx(0.5)


def test_targeted_flip_nan_when_none_syc():
    before = [-0.5, -0.3]
    after = [0.5, 0.2]
    assert np.isnan(targeted_syc_to_non_syc_flip_rate(before, after))


def test_accuracy_from_margins():
    # accuracy = fraction with margin <= 0 (prefers honest)
    assert accuracy_from_margins([-0.1, -0.2, 0.3, 0.4]) == pytest.approx(0.5)


def test_accuracy_change_positive_means_less_syc():
    before = [0.5, 0.3, -0.1, -0.2]   # acc = 0.5
    after = [-0.5, -0.3, -0.1, -0.2]  # acc = 1.0
    out = accuracy_change(before, after)
    assert out["before_accuracy"] == pytest.approx(0.5)
    assert out["after_accuracy"] == pytest.approx(1.0)
    assert out["accuracy_change"] == pytest.approx(0.5)


def test_behavioral_intervention_metrics_keys():
    out = behavioral_intervention_metrics([0.5, -0.3], [-0.5, -0.3])
    for k in ("answer_flip_rate", "targeted_syc_to_non_syc_flip_rate",
              "before_accuracy", "after_accuracy", "accuracy_change"):
        assert k in out
