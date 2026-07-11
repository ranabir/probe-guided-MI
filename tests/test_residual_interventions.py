"""Tests for backend-agnostic residual interventions (edit math, no model needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import residual_interventions as ri


def _vec(x):
    return torch.tensor([x], dtype=torch.float32)  # [1, D]


def test_projection_ablation_removes_component():
    # h=[3,4], d̂=[1,0] -> component 3 removed -> [0,4]  (norm_preserve off for exact check)
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_projection_ablation(ud, target=0.0, norm_preserve=False)
    out = edit(_vec([3.0, 4.0]), 0)[0].numpy()
    assert np.allclose(out, [0.0, 4.0], atol=1e-5)


def test_projection_ablation_to_target():
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_projection_ablation(ud, target=-1.0, norm_preserve=False)
    out = edit(_vec([3.0, 4.0]), 0)[0].numpy()
    assert np.allclose(out, [-1.0, 4.0], atol=1e-5)  # component set to -1


def test_mean_shift_sets_component_to_honest():
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_mean_shift(ud, honest_proj={0: -2.0}, norm_preserve=False)
    out = edit(_vec([3.0, 4.0]), 0)[0].numpy()
    assert np.allclose(out, [-2.0, 4.0], atol=1e-5)


def test_additive_inflates_along_direction():
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_additive(ud, alpha=-5.0, resid_scale={0: 1.0}, norm_preserve=False)
    out = edit(_vec([3.0, 4.0]), 0)[0].numpy()
    assert np.allclose(out, [-2.0, 4.0], atol=1e-5)  # 3 + (-5) = -2


def test_norm_preserve_keeps_magnitude():
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_additive(ud, alpha=5.0, resid_scale={0: 1.0}, norm_preserve=True)
    v = _vec([3.0, 4.0])
    out = edit(v, 0)
    assert np.isclose(float(out.norm()), float(v.norm()), atol=1e-4)


def test_cap_only_clips_above_threshold():
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    edit = ri.make_cap(ud, threshold={0: 1.0}, strength=1.0, norm_preserve=False)
    # component 3 > threshold 1 -> reduced to 1
    hi = edit(_vec([3.0, 4.0]), 0)[0].numpy()
    assert np.allclose(hi, [1.0, 4.0], atol=1e-5)
    # component 0.5 < threshold -> unchanged
    lo = edit(_vec([0.5, 4.0]), 0)[0].numpy()
    assert np.allclose(lo, [0.5, 4.0], atol=1e-5)


def test_honest_mean_projection():
    hs = np.zeros((4, 1, 2), dtype=np.float32)
    hs[:, 0, :] = np.array([[2, 0], [4, 0], [-1, 0], [-3, 0]])
    margins = np.array([1.0, 1.0, -1.0, -1.0])  # last two honest
    ud = {0: np.array([1.0, 0.0], dtype=np.float32)}
    hp = ri.honest_mean_projection(hs, margins, ud)
    assert hp[0] == pytest.approx(-2.0)  # mean of -1 and -3


def test_unit_directions_are_unit():
    rng = np.random.default_rng(0)
    hs = rng.normal(size=(30, 3, 8)).astype(np.float32)
    margins = rng.normal(size=30)
    ud = ri.unit_directions_for_layers(hs, margins, [0, 2], method="diff_of_means")
    for L, v in ud.items():
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_projection_ablation_preserves_orthogonal():
    # orthogonal component must be untouched
    ud = {0: np.array([0.0, 1.0], dtype=np.float32)}
    edit = ri.make_projection_ablation(ud, norm_preserve=False)
    out = edit(_vec([7.0, 2.0]), 0)[0].numpy()
    assert out[0] == pytest.approx(7.0)  # x untouched
    assert out[1] == pytest.approx(0.0)  # y component removed
