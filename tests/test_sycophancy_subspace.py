"""Tests for sycophancy-subspace construction and subspace ablation."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sycophancy_subspace import (
    build_sycophancy_subspace,
    build_subspaces_for_layers,
    subspace_honest_target,
)
from src import residual_interventions as ri


def _planted_subspace_data(N=400, D=10, k=2, seed=0):
    """Plant a genuine k-mode sycophancy subspace: each prompt belongs to one 'mode' whose
    margin signal lives along its own orthonormal direction. Random balanced splits then recover
    different directions, so the SVD spans the full k-D subspace (mirrors multi-topic sycophancy)."""
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.normal(size=(D, k)))[0][:, :k].T  # [k, D] orthonormal
    mode = rng.integers(0, k, size=N)
    z = rng.normal(size=N)                      # per-prompt sycophancy strength
    margins = z.copy()
    X = rng.normal(size=(N, D)) * 0.2
    for i in range(k):
        sel = mode == i
        X[sel] += z[sel, None] * V[i][None, :]  # mode i's signal lives along V[i]
    return X, margins, V


def test_subspace_is_orthonormal():
    X, m, _ = _planted_subspace_data()
    V = build_sycophancy_subspace(X, m, rank=3, n_splits=20)
    assert V.shape == (3, X.shape[1])
    assert np.allclose(V @ V.T, np.eye(3), atol=1e-4)


def test_subspace_recovers_planted_directions():
    X, m, Vtrue = _planted_subspace_data(k=2)
    V = build_sycophancy_subspace(X, m, rank=2, n_splits=40)
    # planted subspace should be well captured: projection of Vtrue onto V near identity energy
    P = V.T @ V                      # [D, D] projector onto estimated subspace
    captured = np.mean([vt @ P @ vt for vt in Vtrue])   # ~1 if fully captured
    assert captured > 0.8


def test_rank_caps_at_available():
    X, m, _ = _planted_subspace_data(D=6)
    V = build_sycophancy_subspace(X, m, rank=50, n_splits=10)
    assert V.shape[0] <= 6


def test_subspace_ablation_removes_all_k_dims():
    X, m, _ = _planted_subspace_data(k=3)
    V = build_sycophancy_subspace(X, m, rank=3, n_splits=20)
    edit = ri.make_subspace_ablation({0: V}, norm_preserve=False)
    h = torch.tensor(X[:8], dtype=torch.float32)
    out = edit(h, 0).numpy()
    residual_energy = np.abs(out @ V.T).mean()
    assert residual_energy < 1e-4  # subspace fully removed


def test_subspace_ablation_preserves_orthogonal_complement():
    # rank-1 subspace = e_y; ablation must leave e_x, e_z untouched
    V = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    edit = ri.make_subspace_ablation({0: V}, norm_preserve=False)
    out = edit(torch.tensor([[5.0, 9.0, -2.0]]), 0)[0].numpy()
    assert out[0] == pytest.approx(5.0)
    assert out[2] == pytest.approx(-2.0)
    assert out[1] == pytest.approx(0.0)


def test_build_subspaces_for_layers():
    rng = np.random.default_rng(1)
    hs = rng.normal(size=(60, 4, 8)).astype(np.float32)
    m = rng.normal(size=60)
    subs = build_subspaces_for_layers(hs, m, [0, 2], rank=2, n_splits=15)
    assert set(subs) == {0, 2}
    assert subs[0].shape == (2, 8)


def test_honest_target_shape():
    rng = np.random.default_rng(2)
    hs = rng.normal(size=(50, 3, 8)).astype(np.float32)
    m = rng.normal(size=50)
    subs = build_subspaces_for_layers(hs, m, [1], rank=3, n_splits=10)
    tgt = subspace_honest_target(hs, m, subs)
    assert tgt[1].shape == (3,)


# --- interpretation ---------------------------------------------------------

def test_captured_energy_bounds_and_extremes():
    from src.sycophancy_subspace import captured_energy
    V = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)   # subspace = x-axis
    assert captured_energy(np.array([1.0, 0.0, 0.0]), V) == pytest.approx(1.0)  # in subspace
    assert captured_energy(np.array([0.0, 1.0, 0.0]), V) == pytest.approx(0.0)  # orthogonal
    e = captured_energy(np.array([1.0, 1.0, 0.0]), V)
    assert 0.0 < e < 1.0  # partial


def test_subtype_directions_per_group():
    from src.sycophancy_subspace import subtype_directions
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    margins = rng.normal(size=40)
    subsets = np.array(["a", "b"] * 20)
    dirs = subtype_directions(X, margins, subsets)
    assert set(dirs) <= {"a", "b"}
    for v in dirs.values():
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_subtype_capture_vs_rank_runs():
    from src.sycophancy_subspace import subtype_capture_vs_rank
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 8)).astype(np.float32)
    margins = rng.normal(size=80)
    subsets = np.array(["p", "q", "r"] * 27)[:80]
    rows, cos_df, sv = subtype_capture_vs_rank(X, margins, subsets, ranks=[1, 2, 3], n_splits=10)
    assert len(rows) > 0 and "captured_energy" in rows[0]
    assert abs(sum(sv) - 1.0) < 1e-6  # relative energy sums to 1
    assert cos_df.shape[1] == 3  # 3 basis dims at max rank
