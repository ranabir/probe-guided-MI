"""Tests for direction construction."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.directions import (
    compute_layer_directions,
    diff_of_means_direction,
    margin_weighted_direction,
    random_direction,
    regression_direction,
    _unit,
)
from src.probes import LinearProbe


def test_unit_norm():
    v = _unit(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_diff_of_means_shape_and_direction():
    X = np.vstack([np.ones((10, 4)), -np.ones((10, 4))]).astype(np.float32)
    margins = np.array([1.0] * 10 + [-1.0] * 10)
    d = diff_of_means_direction(X, margins)
    assert d.shape == (4,)
    # positive group mean - negative group mean points toward +1 ... +1
    assert np.all(d > 0)


def test_diff_of_means_handles_single_class():
    X = np.ones((5, 4), dtype=np.float32)
    margins = np.ones(5)
    d = diff_of_means_direction(X, margins)
    assert d.shape == (4,)
    assert np.allclose(d, 0)


def test_margin_weighted_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6)).astype(np.float32)
    margins = rng.normal(size=20)
    d = margin_weighted_direction(X, margins)
    assert d.shape == (6,)


def test_random_direction_unit():
    d = random_direction(8, seed=1)
    assert d.shape == (8,)
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_regression_direction_from_probe():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y = (X @ np.array([1.0, 0, 0, 0, 0])).astype(np.float32)
    probe = LinearProbe(task="regression").fit(X, y)
    d = regression_direction(probe)
    assert d.shape == (5,)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_compute_layer_directions_keys():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    margins = rng.normal(size=40)
    probe = LinearProbe(task="regression").fit(X, margins)
    dirs = compute_layer_directions(X, margins, regression_probe=probe)
    assert "diff_of_means" in dirs and "random" in dirs and "regression" in dirs
    for v in dirs.values():
        assert v.shape == (6,)
