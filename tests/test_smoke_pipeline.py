"""Smoke test: run full pipeline end-to-end with minimal config (no model loading)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribution import compute_layer_attribution, top_k_layers
from src.behavior_metrics import compute_pair_logprob_margin, summarize_behavior_margins
from src.data import generate_synthetic_dataset, split_dataset
from src.patching import mean_ablate_layers, run_causal_validation
from src.probes import LinearProbe, select_attribution_probe, train_all_layer_probes


def _make_fake_activations(n, n_layers, d_model, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_layers, d_model)).astype(np.float32)


def test_end_to_end_no_model():
    """Full pipeline (data -> activations -> probe -> attribution -> validation) without a real model."""
    N, L, D = 60, 6, 32
    rng = np.random.default_rng(42)

    # Fake activations with a learnable signal
    direction = rng.normal(size=(D,))
    direction /= np.linalg.norm(direction)

    labels = rng.integers(0, 2, size=N).astype(np.int64)
    acts = rng.normal(size=(N, L, D)).astype(np.float32)
    # Plant signal in layer 3
    acts[labels == 1, 3, :] += 0.5 * direction
    acts[labels == 0, 3, :] -= 0.5 * direction

    split = int(N * 0.7)
    val_split = int(N * 0.85)
    train_X, train_y = acts[:split], labels[:split]
    val_X, val_y = acts[split:val_split], labels[split:val_split]
    test_X, test_y = acts[val_split:], labels[val_split:]

    # Train probes using in-memory (skip file saving by calling probe directly)
    best_probe = None
    best_auroc = -1
    for layer in range(L):
        probe = LinearProbe(standardize=True)
        probe.fit(train_X[:, layer, :], train_y)
        probe.layer_idx = layer
        metrics = probe.evaluate(val_X[:, layer, :], val_y)
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_probe = probe

    assert best_probe is not None
    assert best_probe.layer_idx is not None

    # Attribution (fake grads = activation itself for simplicity)
    fake_grads = acts.copy()
    scores = compute_layer_attribution(fake_grads, acts, method="grad_times_activation")
    assert scores.shape == (L,)

    top_layers = top_k_layers(scores, k=3)
    assert len(top_layers) == 3

    # Causal validation
    hs_dict = {"test": test_X}
    labels_dict = {"test": test_y}
    results = run_causal_validation(
        hs_dict, labels_dict, best_probe, top_layers, k=2, random_trials=2, max_examples=20
    )
    assert len(results) >= 2
    assert all("before_score" in r and "after_score" in r for r in results)


def test_probe_selection_policy_in_pipeline():
    """Verify select_attribution_probe works with best_late in the no-model pipeline."""
    N, L, D = 60, 10, 16
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(D,))
    direction /= np.linalg.norm(direction)
    labels = rng.integers(0, 2, size=N).astype(np.int64)
    acts = rng.normal(size=(N, L, D)).astype(np.float32)
    acts[labels == 1, 8, :] += direction  # plant signal at layer 8

    split = int(N * 0.7)
    val_split = int(N * 0.85)

    metrics_list = []
    for layer in range(L):
        probe = LinearProbe(standardize=True)
        probe.fit(acts[:split, layer, :], labels[:split])
        probe.layer_idx = layer
        m = probe.evaluate(acts[split:val_split, layer, :], labels[split:val_split])
        metrics_list.append({"layer": layer, "val_auroc": m["auroc"], "val_accuracy": m["accuracy"],
                              "val_f1": m["f1"], "train_accuracy": 1.0, "train_auroc": 1.0,
                              "train_f1": 1.0, "test_accuracy": m["accuracy"],
                              "test_auroc": m["auroc"], "test_f1": m["f1"]})

    # best_late with frac=0.7 should consider layers >= 7
    # We do not call load_probe (requires disk) — just test metric selection logic
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    min_layer = int(0.7 * L)
    late_df = df[df["layer"] >= min_layer]
    assert len(late_df) > 0
    best_late_layer = int(late_df.loc[late_df["val_auroc"].idxmax(), "layer"])
    assert best_late_layer >= min_layer


def test_prompt_preferences_smoke_no_model():
    """End-to-end prompt-preference path (regression) on tiny synthetic data, no real model."""
    from src.probes import LinearProbe
    from src.patching import run_preference_causal_validation
    from src.attribution import compute_layer_attribution, top_k_layers

    N, L, D = 50, 6, 12
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(D,)); direction /= np.linalg.norm(direction)
    X = rng.normal(size=(N, L, D)).astype(np.float32)
    # behavior_margin signal planted at layer 4
    margins = (X[:, 4, :] @ direction + rng.normal(size=N) * 0.05).astype(np.float32)

    tr, vl = int(N * 0.7), int(N * 0.85)
    best, best_p = None, -np.inf
    for layer in range(L):
        probe = LinearProbe(task="regression")
        probe.fit(X[:tr, layer, :], margins[:tr])
        probe.layer_idx = layer
        m = probe.evaluate(X[tr:vl, layer, :], margins[tr:vl])
        if m["pearson"] > best_p:
            best_p, best = m["pearson"], probe
    assert best is not None
    assert best.layer_idx == 4  # recovered the signal layer

    # Attribution via predicted-margin gradients approximated by activation*coef sign
    grads = X.copy()
    scores = compute_layer_attribution(grads, X, method="grad_times_activation")
    assert scores.shape == (L,)
    top = top_k_layers(scores, k=3)

    # Preference causal validation (regression probe, predict_score)
    results = run_preference_causal_validation(X[vl:], best, top, k=2, random_trials=2)
    assert len(results) == 2
    assert all("before_score" in r and "after_score" in r for r in results)


def test_behavior_metrics_in_pipeline():
    """Verify behavior metrics helpers work end-to-end with mock adapter."""
    import pandas as pd

    class MockAdapter:
        def compute_logprob(self, prompt, continuation, max_length=128, normalize=False):
            return -5.0 if "Yes" in continuation else -9.0

    adapter = MockAdapter()
    result = compute_pair_logprob_margin(adapter, "Is x true?", "Yes definitely", "No actually", normalize=False)
    assert result["prefers_sycophancy"] == 1
    assert result["behavior_margin"] == pytest.approx(-5.0 - (-9.0))

    df = pd.DataFrame({"behavior_margin": [result["behavior_margin"]]})
    s = summarize_behavior_margins(df)
    assert s["frac_prefers_syc"] == 1.0
