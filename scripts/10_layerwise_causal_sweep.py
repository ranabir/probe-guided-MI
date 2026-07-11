#!/usr/bin/env python3
"""Step 10: Layerwise decodability vs causal effect (reviewer feedback #8).

For EVERY layer (not just the best-decoding one), report:
  - decodability: test Pearson / Spearman / R2 (from the step-03 per-layer probe metrics)
  - causal effect: patch that single layer's final prompt-token residual and measure
    behavior_margin_delta, answer_flip_rate, accuracy_change (with bootstrap 95% CI).

This answers: are the high-decodability layers also the high causal-effect layers?

Outputs:
  results/tables/{sn}_layerwise_decodability_causal_sweep.csv
  plots/{sn}/{sn}_decodability_vs_causal_effect.png
  plots/{sn}/{sn}_layerwise_behavior_delta.png
  plots/{sn}/{sn}_layerwise_answer_flip_rate.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.metrics import behavioral_intervention_metrics
from src.model_loader import load_adapter
from src.patching import compute_patched_behavior_margin
from src.plotting import (
    plot_decodability_vs_causal,
    plot_layerwise_answer_flip,
    plot_layerwise_behavior_delta,
)
from src.statistics import bootstrap_mean_ci
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Layerwise decodability + causal sweep")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--max_examples", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    tag = f"{inf}_{pp}" if inf == "prompt_preferences" else pp

    from src.model_adapters import TransformerLensAdapter
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    tl = isinstance(adapter, TransformerLensAdapter)
    if not tl:
        logger.warning("Causal sweep needs TransformerLens; %s is HF — decodability only, "
                       "causal columns = NaN.", args.model_name)

    # Decodability from step 03
    metrics_csv = Path(results_dir("tables")) / f"{sn}_{tag}_layer_probe_metrics.csv"
    if not metrics_csv.exists():
        logger.error("Missing %s. Run step 03 first.", metrics_csv); sys.exit(1)
    mdf = pd.read_csv(metrics_csv)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    mean_acts = tr["hidden_states"].mean(axis=0)  # [L, D]
    n_layers = mean_acts.shape[0]

    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    pairs_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv")
    if len(pairs_df) > args.max_examples:
        pairs_df = pairs_df.sample(n=args.max_examples, random_state=args.seed).reset_index(drop=True)
    syc_col = "sycophantic_response"
    non_col = "non_sycophantic_response"

    # Baseline (unpatched) per-prompt margins — computed once
    base = []
    if tl:
        for _, r in pairs_df.iterrows():
            try:
                base.append(compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                            [], {}, max_length=args.max_length))
            except Exception:
                base.append(float("nan"))

    rows = []
    for layer in range(n_layers):
        dm = mdf[mdf["layer"] == layer]
        decod = {
            "test_pearson": float(dm["test_pearson"].iloc[0]) if "test_pearson" in dm and len(dm) else float("nan"),
            "test_spearman": float(dm["test_spearman"].iloc[0]) if "test_spearman" in dm and len(dm) else float("nan"),
            "test_r2": float(dm["test_r2"].iloc[0]) if "test_r2" in dm and len(dm) else float("nan"),
        }
        if tl:
            after = []
            pv = {layer: mean_acts[layer]}
            for _, r in pairs_df.iterrows():
                try:
                    after.append(compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                                [layer], pv, max_length=args.max_length))
                except Exception:
                    after.append(float("nan"))
            deltas = [a - b for a, b in zip(after, base) if np.isfinite(a) and np.isfinite(b)]
            ci = bootstrap_mean_ci(deltas, n_boot=args.bootstrap, seed=args.seed)
            bm = behavioral_intervention_metrics(base, after)
            causal = {
                "behavior_margin_delta": ci["mean"], "bootstrap_ci_low": ci["ci_low"],
                "bootstrap_ci_high": ci["ci_high"], "answer_flip_rate": bm["answer_flip_rate"],
                "accuracy_change": bm["accuracy_change"], "n_examples": ci["n"],
            }
        else:
            causal = {"behavior_margin_delta": float("nan"), "bootstrap_ci_low": float("nan"),
                      "bootstrap_ci_high": float("nan"), "answer_flip_rate": float("nan"),
                      "accuracy_change": float("nan"), "n_examples": 0}
        rows.append({"model_name": args.model_name, "layer": layer,
                     "layer_frac": layer / max(n_layers - 1, 1),
                     "intervention_type": "activation_patching", **decod, **causal})
        logger.info("layer %2d | pearson=%.3f | Δmargin=%.4f flip=%.2f",
                    layer, decod["test_pearson"], causal["behavior_margin_delta"], causal["answer_flip_rate"])

    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    plot_decodability_vs_causal(df, args.model_name)
    plot_layerwise_behavior_delta(df, args.model_name)
    plot_layerwise_answer_flip(df, args.model_name)

    # Correlation between decodability and causal effect across layers
    valid = df.dropna(subset=["test_pearson", "behavior_margin_delta"])
    if len(valid) > 2 and tl:
        from scipy.stats import pearsonr
        r = pearsonr(valid["test_pearson"], valid["behavior_margin_delta"].abs())[0]
        print(f"\nCorrelation(decodability, |causal effect|) across layers = {r:.3f}")
        print("Near 0 ⇒ high-decoding layers are NOT the high-causal layers (decoding ≠ causation).")
    logger.info("=== Step 10 complete. ===")


if __name__ == "__main__":
    main()
