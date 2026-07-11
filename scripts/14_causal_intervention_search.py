#!/usr/bin/env python3
"""Step 14: Search for the best causal intervention (control vs side-effect trade-off).

Grid over layer-selection × direction × intervention × strength. For each config: measure targeted
answer-flip (control) and a side-effect score (capability harm), then rank by
  objective = targeted_flip - lambda_side_effect * side_effect_score.

Outputs:
  results/tables/{sn}_causal_intervention_search.csv
  plots/{sn}/{sn}_intervention_search_pareto.png
  plots/{sn}/{sn}_best_interventions_ranked.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.causal_interventions import (
    apply_activation_capping,
    apply_probe_steering,
    estimate_cap_threshold,
    score_before_after,
    select_layers_from_sweep,
)
from src.directions import compute_layer_directions
from src.model_loader import load_adapter
from src.plotting import plot_best_interventions_ranked, plot_intervention_search_pareto
from src.probes import load_best_probe
from src.search import build_intervention_grid, rank_interventions
from src.side_effects import compute_side_effect_score, generate_or_score_outputs, load_basic_prompts
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Causal intervention search")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--dataset_variant", type=str, default="natural", choices=["natural", "balanced"])
    p.add_argument("--directions", nargs="+", default=["regression", "diff_of_means"])
    p.add_argument("--interventions", nargs="+", default=["probe_steering", "activation_capping"])
    p.add_argument("--alphas", nargs="+", type=float, default=[-3, -1, 1, 3])
    p.add_argument("--cap_quantiles", nargs="+", type=float, default=[0.5, 0.75, 0.9])
    p.add_argument("--top_k_layers", type=int, default=3)
    p.add_argument("--lambda_side_effect", type=float, default=0.5)
    p.add_argument("--max_examples", type=int, default=40)
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

    from src.model_adapters import TransformerLensAdapter
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    if not isinstance(adapter, TransformerLensAdapter):
        logger.warning("Intervention search is TL-only; %s is HF. Writing pending marker.", args.model_name)
        pd.DataFrame([{"model_name": args.model_name, "status": "search_pending_hf"}]).to_csv(
            Path(results_dir("tables")) / f"{sn}_causal_intervention_search.csv", index=False)
        sys.exit(0)

    logger.info("=== Step 14: Intervention Search — %s ===", args.model_name)
    sweep_df = pd.read_csv(Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv")
    variant_suffix = "_balanced" if args.dataset_variant == "balanced" else ""
    test_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv").dropna(
        subset=["behavior_margin"]).reset_index(drop=True)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    train_hs = tr["hidden_states"]
    mean_acts = train_hs.mean(axis=0)
    train_margins = np.asarray(tr["behavior_margin"])
    try:
        probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
    except Exception:
        probe = None

    layers_by_selection = {
        "causal_topk": select_layers_from_sweep(sweep_df, "causal_topk", args.top_k_layers, seed=args.seed),
        "decodable_topk": select_layers_from_sweep(sweep_df, "decodable_topk", args.top_k_layers, seed=args.seed),
    }
    grid = build_intervention_grid(layers_by_selection, args.directions, args.interventions,
                                   args.alphas, args.cap_quantiles)
    logger.info("Grid size: %d configs", len(grid))

    # Side-effect baseline (generate once)
    se_prompts = load_basic_prompts().head(args.side_effect_prompts)["prompt"].tolist()
    baseline_out = generate_or_score_outputs(adapter, se_prompts, intervention=None,
                                             max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    # cache directions + resid_scale per primary layer
    dir_cache = {}
    def get_dir(layer, dname):
        if layer not in dir_cache:
            X = train_hs[:, layer, :]
            dirs = compute_layer_directions(X, train_margins, regression_probe=probe, seed=args.seed)
            dir_cache[layer] = (dirs, float(np.linalg.norm(X, axis=1).mean()))
        dirs, scale = dir_cache[layer]
        return dirs.get(dname, dirs.get("diff_of_means")), scale

    rows = []
    for i, cfg in enumerate(grid):
        layers = cfg["layers"]; primary = layers[0]
        direction, scale = get_dir(primary, cfg["direction"])
        intervention = cfg["intervention"]; val = cfg["alpha_or_cap"]
        try:
            if intervention == "probe_steering":
                before, after = apply_probe_steering(adapter, test_df, direction, primary, val, scale,
                                                     max_examples=args.max_examples, seed=args.seed,
                                                     max_length=args.max_length)
                se_dir = direction * scale; alpha = val; threshold = 0.0
            else:  # activation_capping
                threshold = estimate_cap_threshold(train_hs[:, primary, :], direction, val)
                before, after = apply_activation_capping(adapter, test_df, direction, primary, threshold,
                                                        1.0, max_examples=args.max_examples, seed=args.seed,
                                                        max_length=args.max_length)
                se_dir = direction; alpha = 0.0
        except Exception as e:
            logger.warning("config %d failed: %s", i, e); continue

        m = score_before_after(before, after, bootstrap=args.bootstrap, seed=args.seed)

        # side-effect for this config
        interv = {"intervention": intervention, "layers": layers, "direction": se_dir,
                  "alpha": alpha, "threshold": threshold, "cap_strength": 1.0, "mean_acts": mean_acts}
        try:
            se_out = generate_or_score_outputs(adapter, se_prompts, intervention=interv,
                                               max_new_tokens=args.max_new_tokens, max_length=args.max_length)
            se = compute_side_effect_score(baseline_out, se_out)
            side_effect_score = se["side_effect_score"]
        except Exception as e:
            logger.warning("side-effect for config %d failed: %s", i, e)
            side_effect_score = float("nan")

        rows.append({
            "model_name": args.model_name, "layer": primary, "layers": str(layers),
            "layer_selection_type": cfg["layer_selection"], "direction_type": cfg["direction"],
            "intervention_type": intervention, "alpha_or_cap": val,
            "targeted_flip_rate": m["targeted_syc_to_non_syc_flip_rate"],
            "answer_flip_rate": m["answer_flip_rate"], "behavior_margin_delta": m["behavior_margin_delta"],
            "accuracy_change": m["accuracy_change"], "side_effect_score": side_effect_score,
            "ci_low": m["ci_low"], "ci_high": m["ci_high"], "n_examples": m["n_examples"],
        })
        if (i + 1) % 5 == 0:
            logger.info("  %d/%d configs done", i + 1, len(grid))

    ranked = rank_interventions(rows, lambda_side_effect=args.lambda_side_effect)
    out = Path(results_dir("tables")) / f"{sn}_causal_intervention_search.csv"
    ranked.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    plot_intervention_search_pareto(ranked, args.model_name)
    plot_best_interventions_ranked(ranked, args.model_name)

    print(f"\n=== Top 5 interventions by objective — {args.model_name} ===")
    cols = ["rank", "layer_selection_type", "direction_type", "intervention_type", "alpha_or_cap",
            "targeted_flip_rate", "side_effect_score", "objective_score"]
    print(ranked[cols].head(5).to_string(index=False, float_format="%.4f"))
    logger.info("=== Step 14 complete. ===")


if __name__ == "__main__":
    main()
