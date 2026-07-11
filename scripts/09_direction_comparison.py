#!/usr/bin/env python3
"""Step 09: Compare steering directions — do simple difference-of-means directions steer
sycophancy better than learned probe directions? (reviewer feedback #6)

At the selected probe layer we build several candidate directions (regression probe, logistic
probe, difference-of-means, margin-weighted, random), then steer along each over an alpha sweep
and measure behavior_margin change, answer-flip rate, and accuracy change.

TransformerLens only (steering uses forward-pass hooks).

Outputs:
  results/tables/{sn}_direction_comparison.csv
  plots/{sn}/{sn}_direction_comparison_behavior_delta.png
  plots/{sn}/{sn}_direction_comparison_answer_flip.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.directions import compute_layer_directions, steer_margins_for_prompts
from src.metrics import behavioral_intervention_metrics
from src.model_loader import load_adapter
from src.plotting import plot_direction_comparison
from src.probes import load_best_probe
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Direction comparison / steering")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--max_examples", type=int, default=20)
    p.add_argument("--alphas", type=float, nargs="+", default=[-3, -2, -1, 0, 1, 2, 3])
    p.add_argument("--layer", type=int, default=None, help="Steering layer (default: probe layer)")
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position

    from src.model_adapters import TransformerLensAdapter
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    if not isinstance(adapter, TransformerLensAdapter):
        logger.error("Direction steering is TransformerLens-only. %s uses HuggingFace; "
                     "directions saved as pending.", args.model_name)
        # Still record that we attempted
        pd.DataFrame([{"model_name": args.model_name, "status": "steering_pending_hf"}]).to_csv(
            Path(results_dir("tables")) / f"{sn}_direction_comparison.csv", index=False)
        sys.exit(0)

    probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
    layer = args.layer if args.layer is not None else probe.layer_idx
    logger.info("=== Step 09: Direction comparison @ layer %d (%s) ===", layer, args.model_name)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    train_X = tr["hidden_states"][:, layer, :]
    train_margins = np.asarray(tr["behavior_margin"])

    directions = compute_layer_directions(train_X, train_margins, regression_probe=probe, seed=args.seed)
    # Scale unit directions by the typical residual norm at this layer, so alpha is in
    # "residual-norm" units (alpha=1 ≈ add one residual-norm's worth along the direction).
    resid_scale = float(np.linalg.norm(train_X, axis=1).mean())
    logger.info("Directions: %s | residual-norm scale=%.2f", list(directions.keys()), resid_scale)
    directions = {k: v * resid_scale for k, v in directions.items()}

    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    pairs_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv")

    # Baseline (alpha=0): steer any direction by 0 -> unperturbed margins
    any_dir = next(iter(directions.values()))
    base = steer_margins_for_prompts(adapter, pairs_df, layer, any_dir, 0.0,
                                     max_examples=args.max_examples, seed=args.seed, max_length=args.max_length)

    rows = []
    for dname, dvec in directions.items():
        for alpha in args.alphas:
            if alpha == 0:
                margins = base
            else:
                margins = steer_margins_for_prompts(adapter, pairs_df, layer, dvec, float(alpha),
                                                    max_examples=args.max_examples, seed=args.seed,
                                                    max_length=args.max_length)
            valid = [m for m in margins if np.isfinite(m)]
            mean_margin = float(np.mean(valid)) if valid else float("nan")
            bm = behavioral_intervention_metrics(base, margins)
            rows.append({
                "model_name": args.model_name, "layer": layer, "direction": dname, "alpha": alpha,
                "mean_behavior_margin": mean_margin,
                "behavior_margin_delta": mean_margin - (np.nanmean(base) if base else float("nan")),
                "answer_flip_rate": bm["answer_flip_rate"],
                "targeted_syc_to_non_syc_flip_rate": bm["targeted_syc_to_non_syc_flip_rate"],
                "accuracy_change": bm["accuracy_change"],
                "n_examples": len(valid),
            })
        logger.info("  %s done", dname)

    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / f"{sn}_direction_comparison.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    plot_direction_comparison(df, args.model_name)

    print(f"\n=== Direction steering @ layer {layer} — {args.model_name} ===")
    # peak |behavior_margin_delta| per direction
    for d, g in df.groupby("direction"):
        i = g["behavior_margin_delta"].abs().idxmax()
        r = g.loc[i]
        print(f"  {d:16s}: max |Δmargin|={r['behavior_margin_delta']:+.3f} @ alpha={r['alpha']:+.0f} "
              f"| flip_rate={r['answer_flip_rate']:.2f} acc_change={r['accuracy_change']:+.2f}")
    print("\nIf diff_of_means matches/beats regression, learned probe directions aren't adding "
          "causal value (the reviewer #6).")
    logger.info("=== Step 09 complete. ===")


if __name__ == "__main__":
    main()
