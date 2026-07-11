"""Build low-rank 'sycophancy subspaces' from activations.

Motivation: single-direction ablation is clean but weak — if sycophancy is spread across several
directions, removing one leaves the rest. This module estimates a rank-k subspace that captures the
main directions along which activations separate sycophantic-leaning from honest-leaning prompts,
so we can ablate the whole band at once (see src.residual_interventions.make_subspace_ablation).

Method: over many random balanced splits, compute the difference-of-means direction
(mean[margin>0] − mean[margin<=0]); stack these bootstrap directions and take the top-k right
singular vectors (SVD) → a stable orthonormal rank-k basis V [k, D]. k=1 recovers a single direction.
"""
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _diff_of_means(X: np.ndarray, margins: np.ndarray, idx: np.ndarray) -> np.ndarray:
    m = margins[idx]
    pos, neg = X[idx][m > 0], X[idx][m <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    return pos.mean(0) - neg.mean(0)


def build_sycophancy_subspace(layer_X: np.ndarray, margins: np.ndarray, rank: int = 3,
                              n_splits: int = 40, seed: int = 42) -> np.ndarray:
    """Return an orthonormal basis V [rank, D] for the sycophancy subspace at one layer.

    layer_X: [N, D] activations at one layer. margins: [N] behavior margins.
    """
    N, D = layer_X.shape
    margins = np.asarray(margins)
    rng = np.random.default_rng(seed)
    X = layer_X.astype(np.float64)

    vecs = []
    # global direction always included
    vecs.append(_diff_of_means(X, margins, np.arange(N)))
    # bootstrap balanced splits
    pos_idx = np.where(margins > 0)[0]
    neg_idx = np.where(margins <= 0)[0]
    if len(pos_idx) and len(neg_idx):
        k = min(len(pos_idx), len(neg_idx))
        for _ in range(n_splits):
            p = rng.choice(pos_idx, size=max(2, k // 2), replace=True)
            n = rng.choice(neg_idx, size=max(2, k // 2), replace=True)
            idx = np.concatenate([p, n])
            v = _diff_of_means(X, margins, idx)
            if np.linalg.norm(v) > 0:
                vecs.append(v)

    M = np.vstack(vecs)  # [n_vecs, D]
    # SVD → top-k right singular vectors
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    rank = min(rank, Vt.shape[0])
    V = Vt[:rank]  # [rank, D], orthonormal rows
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)  # be safe
    return V.astype(np.float32)


def build_subspaces_for_layers(train_hs: np.ndarray, margins: np.ndarray, layers: List[int],
                               rank: int = 3, n_splits: int = 40, seed: int = 42) -> Dict[int, np.ndarray]:
    """Return {layer: V[rank, D]} for each requested layer."""
    return {L: build_sycophancy_subspace(train_hs[:, L, :], margins, rank=rank,
                                         n_splits=n_splits, seed=seed) for L in layers}


def subspace_honest_target(train_hs: np.ndarray, margins: np.ndarray,
                           subspaces: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Mean coordinates of honest-preferring (margin<=0) activations within each subspace: [rank]."""
    honest = np.asarray(margins) <= 0
    out = {}
    for L, V in subspaces.items():
        X = train_hs[honest, L, :] if honest.any() else train_hs[:, L, :]
        out[L] = (X @ V.T).mean(0)  # [rank]
    return out
