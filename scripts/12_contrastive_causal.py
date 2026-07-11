#!/usr/bin/env python3
"""Step 12: Contrastive causal intervention at high-causal vs high-decodable layers.

Tests the hypothesis from the layerwise sweep: intervening at the layers with the largest CAUSAL
effect (not the most DECODABLE ones) should give stronger answer-level control. Compares
contrastive_patching / probe_steering / activation_capping / mean_ablation across
causal_topk / decodable_topk / random layer selections.

Outputs:
  results/tables/{sn}_contrastive_causal_results.csv
  results/tables/{sn}_best_causal_intervention.json
  plots/{sn}/{sn}_contrastive_causal_answer_flip.png
  plots/{sn}/{sn}_contrastive_causal_behavior_delta.png
  plots/{sn}/{sn}_causal_vs_decodable_layer_intervention.png
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.causal_interventions import (
    apply_activation_capping,
    apply_contrastive_patch,
    apply_probe_steering,
    cache_prompt_final_residuals,
    choose_opposite_preference_reference,
    estimate_cap_threshold,
    score_before_after,
    select_layers_from_sweep,
)
from src.directions import compute_layer_directions
from src.model_loader import load_adapter
from src.patching import patched_margins_for_layers
from src.plotting import (
    plot_causal_vs_decodable_intervention,
    plot_contrastive_causal_results,
)
from src.probes import load_best_probe
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Contrastive causal interventions")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--dataset_variant", type=str, default="natural", choices=["natural", "balanced"])
    p.add_argument("--layer_selection", nargs="+", default=["causal_topk", "decodable_topk", "random"])
    p.add_argument("--top_k_layers", type=int, default=3)
    p.add_argument("--manual_layers", type=str, default=None, help="comma-separated for manual selection")
    p.add_argument("--intervention", nargs="+",
                   default=["contrastive_patching", "probe_steering", "activation_capping", "mean_ablation"])
    p.add_argument("--alphas", nargs="+", type=float, default=[-5, -3, -1, 1, 3, 5])
    p.add_argument("--cap_quantile", type=float, default=0.75)
    p.add_argument("--cap_strength", type=float, default=1.0)
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    from src.model_adapters import TransformerLensAdapter
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    is_tl = isinstance(adapter, TransformerLensAdapter)
    logger.info("=== Step 12: Contrastive Causal (%s, TL=%s) ===", args.model_name, is_tl)

    tl_only = {"contrastive_patching", "probe_steering", "activation_capping", "mean_ablation"}
    if not is_tl and set(args.intervention).issubset(tl_only):
        logger.warning("All requested interventions are TransformerLens-only; %s is HuggingFace. "
                       "Hook-based residual edits for HF are not implemented — writing pending marker.",
                       args.model_name)
        pd.DataFrame([{"model_name": args.model_name, "status": "causal_intervention_pending_hf",
                       "note": "TL-only residual hooks; HF steering/patching not implemented"}]).to_csv(
            Path(results_dir("tables")) / f"{sn}_contrastive_causal_results.csv", index=False)
        sys.exit(0)

    # Layerwise sweep table drives layer selection
    sweep_path = Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv"
    if not sweep_path.exists():
        logger.error("Missing %s. Run scripts/10_layerwise_causal_sweep.py first.", sweep_path)
        sys.exit(1)
    sweep_df = pd.read_csv(sweep_path)

    # Preference dataset (natural or balanced) + cached activations
    variant_suffix = "_balanced" if args.dataset_variant == "balanced" else ""
    pref_all = pd.read_csv(processed / f"{sn}_prompt_preferences{variant_suffix}.csv").dropna(
        subset=["behavior_margin"]).reset_index(drop=True)
    test_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv").dropna(
        subset=["behavior_margin"]).reset_index(drop=True)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    train_hs = tr["hidden_states"]
    mean_acts = train_hs.mean(axis=0)
    train_margins = np.asarray(tr["behavior_margin"])

    # Directions per layer (use regression probe where available)
    try:
        probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
    except Exception:
        probe = None

    manual = [int(x) for x in args.manual_layers.split(",")] if args.manual_layers else None

    # Precompute reference residuals + opposite-preference reference index per test prompt (for patching)
    ref_residuals = ref_indices = None
    if "contrastive_patching" in args.intervention and is_tl:
        logger.info("Caching reference residuals for contrastive patching...")
        ref_residuals = cache_prompt_final_residuals(adapter, test_df["prompt"].tolist(), args.max_length)
        ref_indices = [choose_opposite_preference_reference(test_df, i, seed=args.seed)
                       for i in range(len(test_df))]

    rows = []
    for selection in args.layer_selection:
        layers = select_layers_from_sweep(sweep_df, selection, args.top_k_layers, manual=manual, seed=args.seed)
        logger.info("Selection=%s -> layers %s", selection, layers)
        primary_layer = layers[0]
        layer_X = train_hs[:, primary_layer, :]
        dirs = compute_layer_directions(layer_X, train_margins, regression_probe=probe, seed=args.seed)
        resid_scale = float(np.linalg.norm(layer_X, axis=1).mean())
        direction = dirs.get("regression", dirs.get("logistic", dirs["diff_of_means"]))

        for intervention in args.intervention:
            try:
                if intervention == "mean_ablation":
                    before, after = patched_margins_for_layers(
                        adapter, test_df, layers, mean_acts, max_examples=args.max_examples,
                        seed=args.seed, max_length=args.max_length)
                    settings = [("mean", None)]
                    res_list = [(before, after, "mean")]
                elif intervention == "contrastive_patching":
                    if not is_tl:
                        logger.warning("Skipping contrastive_patching (HF model).")
                        continue
                    before, after = apply_contrastive_patch(
                        adapter, test_df, ref_residuals, ref_indices, layers,
                        max_examples=args.max_examples, seed=args.seed, max_length=args.max_length)
                    res_list = [(before, after, "contrastive")]
                elif intervention == "probe_steering":
                    if not is_tl:
                        logger.warning("Skipping probe_steering (HF model).")
                        continue
                    res_list = []
                    for alpha in args.alphas:
                        b, a = apply_probe_steering(
                            adapter, test_df, direction, primary_layer, alpha, resid_scale,
                            max_examples=args.max_examples, seed=args.seed, max_length=args.max_length)
                        res_list.append((b, a, f"alpha={alpha:+.0f}"))
                elif intervention == "activation_capping":
                    if not is_tl:
                        logger.warning("Skipping activation_capping (HF model).")
                        continue
                    thr = estimate_cap_threshold(layer_X, direction, args.cap_quantile)
                    b, a = apply_activation_capping(
                        adapter, test_df, direction, primary_layer, thr, args.cap_strength,
                        max_examples=args.max_examples, seed=args.seed, max_length=args.max_length)
                    res_list = [(b, a, f"cap_q={args.cap_quantile}")]
                else:
                    logger.warning("Unknown intervention %s", intervention)
                    continue
            except NotImplementedError as e:
                logger.warning("Skipping %s: %s", intervention, e)
                continue

            for before, after, setting in res_list:
                m = score_before_after(before, after, bootstrap=args.bootstrap, seed=args.seed)
                rows.append({
                    "model_name": args.model_name, "layer_selection": selection,
                    "layers": str(layers), "intervention": intervention, "setting": setting,
                    "top_k_layers": args.top_k_layers, **m,
                })
                logger.info("  %s/%s/%s: Δmargin=%.4f flip=%.3f targeted=%.3f",
                            selection, intervention, setting, m["behavior_margin_delta"],
                            m["answer_flip_rate"], m["targeted_syc_to_non_syc_flip_rate"])

    if not rows:
        logger.error("No interventions ran (HF model with TL-only methods?). Nothing to save.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / f"{sn}_contrastive_causal_results.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    # Best config by targeted flip rate (tie-break: |behavior_margin_delta|)
    df["_abs_delta"] = df["behavior_margin_delta"].abs()
    best = df.sort_values(["targeted_syc_to_non_syc_flip_rate", "_abs_delta"], ascending=False).iloc[0]
    best_cfg = {"model_name": args.model_name, "layer_selection": best["layer_selection"],
                "layers": best["layers"], "intervention": best["intervention"],
                "setting": best["setting"], "targeted_flip_rate": float(best["targeted_syc_to_non_syc_flip_rate"]),
                "behavior_margin_delta": float(best["behavior_margin_delta"]),
                "cap_quantile": args.cap_quantile, "cap_strength": args.cap_strength}
    with open(Path(results_dir("tables")) / f"{sn}_best_causal_intervention.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    plot_contrastive_causal_results(df, args.model_name)
    plot_causal_vs_decodable_intervention(sweep_df, df, args.model_name)

    print(f"\n=== Contrastive Causal — {args.model_name} ===")
    cols = ["layer_selection", "intervention", "setting", "behavior_margin_delta",
            "answer_flip_rate", "targeted_syc_to_non_syc_flip_rate"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print(f"\nBest by targeted flip: {best_cfg['layer_selection']}/{best_cfg['intervention']}"
          f"/{best_cfg['setting']} (flip={best_cfg['targeted_flip_rate']:.3f})")
    logger.info("=== Step 12 complete. ===")


if __name__ == "__main__":
    main()
