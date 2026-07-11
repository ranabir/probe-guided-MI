#!/usr/bin/env python3
"""Step 15: Clean causal control — can we flip sycophancy WITHOUT breaking the model?

Compares intervention families on the same causal layers, measuring BOTH answer-level control
(targeted syc→honest flip) and capability side-effects (generation weirdness):

  - additive              : blunt steering  h += α·d̂        (baseline; inflates norm)
  - additive_normpreserve : same, rescaled back to ‖h‖
  - projection_ablation   : remove the sycophancy component  h -= (h·d̂)d̂   (surgical)
  - mean_shift            : set the component to the honest-mean value       (in-distribution)

Works for TransformerLens AND HuggingFace models (Problem 1: instruct models are now steerable).

Outputs:
  results/tables/{sn}_clean_causal_control.csv
  plots/{sn}/{sn}_clean_control_pareto.png
  plots/{sn}/{sn}_clean_control_flip_vs_sideeffect.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.causal_interventions import select_layers_from_sweep
from src.metrics import behavioral_intervention_metrics
from src.model_loader import load_adapter
from src.plotting import plot_clean_control
from src.probes import load_best_probe
from src.side_effects import compute_side_effect_score, load_basic_prompts
from src.statistics import bootstrap_mean_ci
from src import residual_interventions as ri
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Clean causal control: control vs side-effects")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--direction", type=str, default="diff_of_means",
                   choices=["diff_of_means", "regression", "logistic", "margin_weighted"])
    p.add_argument("--top_k_layers", type=int, default=3)
    p.add_argument("--layer_selection", type=str, default="causal_topk",
                   choices=["causal_topk", "decodable_topk", "manual"])
    p.add_argument("--manual_layers", type=str, default=None)
    p.add_argument("--alphas", nargs="+", type=float, default=[-6, -4, -2, 2, 4, 6])
    p.add_argument("--max_examples", type=int, default=30)
    p.add_argument("--side_effect_prompts", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    logger.info("=== Step 15: Clean Causal Control — %s ===", args.model_name)

    # --- layer selection --------------------------------------------------------
    sweep_path = Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv"
    if args.layer_selection == "manual" and args.manual_layers:
        layers = [int(x) for x in args.manual_layers.split(",")][:args.top_k_layers]
    elif sweep_path.exists():
        layers = select_layers_from_sweep(pd.read_csv(sweep_path), args.layer_selection,
                                          args.top_k_layers, seed=args.seed)
    else:
        logger.warning("No layerwise sweep; defaulting to early layers.")
        layers = list(range(1, 1 + args.top_k_layers))
    logger.info("Intervention layers (%s): %s", args.layer_selection, layers)

    # --- data + directions ------------------------------------------------------
    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    hs, mg = tr["hidden_states"], np.asarray(tr["behavior_margin"])
    try:
        probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
    except Exception:
        probe = None
    ud = ri.unit_directions_for_layers(hs, mg, layers, method=args.direction, regression_probe=probe, seed=args.seed)
    scale = ri.resid_scale_for_layers(hs, layers)
    hproj = ri.honest_mean_projection(hs, mg, ud)

    test_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv").dropna(subset=["behavior_margin"]).reset_index(drop=True)
    se_prompts = load_basic_prompts().head(args.side_effect_prompts)["prompt"].tolist()
    baseline_gen = ri.generate_with_edit(adapter, se_prompts, layers, None,
                                         max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    # --- build the method list --------------------------------------------------
    methods = []
    for a in args.alphas:
        methods.append((f"additive(α={a:+.0f})", ri.make_additive(ud, a, scale)))
        methods.append((f"additive_normpres(α={a:+.0f})", ri.make_additive(ud, a, scale, norm_preserve=True)))
    methods.append(("projection_ablation", ri.make_projection_ablation(ud, target=0.0)))
    methods.append(("mean_shift", ri.make_mean_shift(ud, hproj)))

    rows = []
    for name, ef in methods:
        before, after = ri.run_edit_on_prompts(adapter, test_df, layers, ef,
                                               max_examples=args.max_examples, seed=args.seed,
                                               max_length=args.max_length)
        deltas = [a - b for a, b in zip(after, before) if np.isfinite(a) and np.isfinite(b)]
        m = behavioral_intervention_metrics(before, after)
        ci = bootstrap_mean_ci(deltas, n_boot=args.bootstrap, seed=args.seed) if deltas else {"ci_low": np.nan, "ci_high": np.nan}
        # side effects
        try:
            gen = ri.generate_with_edit(adapter, se_prompts, layers, ef,
                                        max_new_tokens=args.max_new_tokens, max_length=args.max_length)
            se = compute_side_effect_score(baseline_gen, gen)["side_effect_score"]
        except Exception as e:
            logger.warning("side-effect for %s failed: %s", name, e)
            se = float("nan")
        family = ("projection_ablation" if name.startswith("projection") else
                  "mean_shift" if name.startswith("mean_shift") else
                  "additive_normpres" if "normpres" in name else "additive")
        rows.append({
            "model_name": args.model_name, "layers": str(layers), "direction": args.direction,
            "method": name, "family": family,
            "behavior_margin_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "answer_flip_rate": m["answer_flip_rate"],
            "targeted_flip_rate": m["targeted_syc_to_non_syc_flip_rate"],
            "accuracy_change": m["accuracy_change"], "side_effect_score": se,
            "ci_low": ci.get("ci_low", np.nan), "ci_high": ci.get("ci_high", np.nan),
            "n_examples": len(deltas),
        })
        logger.info("  %-26s flip=%.2f side_effect=%.2f Δmargin=%+.3f",
                    name, m["targeted_syc_to_non_syc_flip_rate"], se, rows[-1]["behavior_margin_delta"])

    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / f"{sn}_clean_causal_control.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    plot_clean_control(df, args.model_name)

    # highlight the best "clean" point: highest flip among low-side-effect (<0.25) methods
    clean = df[(df["side_effect_score"] < 0.25) & (df["targeted_flip_rate"] > 0)]
    print(f"\n=== Clean Causal Control — {args.model_name} (layers {layers}) ===")
    cols = ["method", "family", "targeted_flip_rate", "side_effect_score", "behavior_margin_delta"]
    print(df[cols].to_string(index=False, float_format="%.3f"))
    if not clean.empty:
        best = clean.sort_values("targeted_flip_rate", ascending=False).iloc[0]
        print(f"\nBest CLEAN lever: {best['method']} → flip {best['targeted_flip_rate']:.2f} "
              f"at side-effect {best['side_effect_score']:.2f}")
    else:
        print("\nNo clean lever found (no method with flip>0 and side-effect<0.25).")
    logger.info("=== Step 15 complete. ===")


if __name__ == "__main__":
    main()
