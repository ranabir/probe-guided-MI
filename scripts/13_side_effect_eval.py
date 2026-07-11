#!/usr/bin/env python3
"""Step 13: Side-effect evaluation of the best causal intervention.

Generates short continuations on basic prompts with and without the intervention active, and
checks whether the intervention harms general capabilities (length, repetition, weirdness, basic QA).

Outputs:
  results/tables/{sn}_side_effect_eval.csv
  results/tables/{sn}_side_effect_samples.csv
  plots/{sn}/{sn}_side_effect_summary.png
  docs/side_effect_eval_notes.md (appended)
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
from src.directions import compute_layer_directions
from src.causal_interventions import estimate_cap_threshold
from src.model_loader import load_adapter
from src.plotting import plot_side_effect_summary
from src.probes import load_best_probe
from src.side_effects import (
    compute_side_effect_score,
    generate_or_score_outputs,
    load_basic_prompts,
    save_side_effect_samples,
)
from src.utils import get_root, load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Side-effect evaluation")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--intervention_config", type=str, default=None,
                   help="JSON from step 12 (best config). Defaults to {sn}_best_causal_intervention.json")
    p.add_argument("--num_prompts", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=30)
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

    from src.model_adapters import TransformerLensAdapter
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    if not isinstance(adapter, TransformerLensAdapter):
        logger.warning("Side-effect generation is TL-only; %s is HF. Writing pending marker.", args.model_name)
        pd.DataFrame([{"model_name": args.model_name, "status": "side_effect_pending_hf"}]).to_csv(
            Path(results_dir("tables")) / f"{sn}_side_effect_eval.csv", index=False)
        sys.exit(0)

    # Load best intervention config
    cfg_path = Path(args.intervention_config) if args.intervention_config else \
        Path(results_dir("tables")) / f"{sn}_best_causal_intervention.json"
    if not cfg_path.exists():
        logger.error("Missing intervention config %s. Run scripts/12 first.", cfg_path)
        sys.exit(1)
    with open(cfg_path) as f:
        best = json.load(f)
    import ast
    layers = ast.literal_eval(best["layers"]) if isinstance(best["layers"], str) else best["layers"]
    intervention = best["intervention"]
    setting = best.get("setting", "")
    alpha = float(setting.split("=")[1]) if setting.startswith("alpha=") else 0.0
    logger.info("=== Step 13: Side-Effect Eval | %s | %s @ layers %s (%s) ===",
                args.model_name, intervention, layers, setting)

    # Build direction + threshold + mean_acts as needed
    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    train_hs = tr["hidden_states"]
    mean_acts = train_hs.mean(axis=0)
    primary_layer = layers[0]
    direction = None
    threshold = 0.0
    if intervention in ("probe_steering", "activation_capping"):
        try:
            probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
        except Exception:
            probe = None
        dirs = compute_layer_directions(train_hs[:, primary_layer, :],
                                        np.asarray(tr["behavior_margin"]), regression_probe=probe, seed=args.seed)
        direction = dirs.get("regression", dirs.get("logistic", dirs["diff_of_means"]))
        resid_scale = float(np.linalg.norm(train_hs[:, primary_layer, :], axis=1).mean())
        if intervention == "probe_steering":
            direction = direction * resid_scale  # match step 12 scaling
        else:
            threshold = estimate_cap_threshold(train_hs[:, primary_layer, :], direction,
                                               best.get("cap_quantile", 0.75))

    prompts_df = load_basic_prompts().head(args.num_prompts)
    prompts = prompts_df["prompt"].tolist()
    answers = prompts_df["answer"].tolist() if "answer" in prompts_df.columns else None

    logger.info("Generating baseline outputs...")
    baseline = generate_or_score_outputs(adapter, prompts, intervention=None,
                                         max_new_tokens=args.max_new_tokens, max_length=args.max_length)
    logger.info("Generating intervened outputs...")
    interv = {"intervention": intervention, "layers": layers, "direction": direction,
              "alpha": alpha, "threshold": threshold,
              "cap_strength": best.get("cap_strength", 1.0), "mean_acts": mean_acts}
    intervened = generate_or_score_outputs(adapter, prompts, intervention=interv,
                                           max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    metrics = compute_side_effect_score(baseline, intervened, answers)
    metrics["model_name"] = args.model_name
    metrics["intervention"] = intervention
    metrics["setting"] = setting

    pd.DataFrame([metrics]).to_csv(Path(results_dir("tables")) / f"{sn}_side_effect_eval.csv", index=False)
    save_side_effect_samples(prompts, baseline, intervened,
                             Path(results_dir("tables")) / f"{sn}_side_effect_samples.csv")
    plot_side_effect_summary(metrics, args.model_name)

    # Notes doc
    notes = get_root() / "docs" / "side_effect_eval_notes.md"
    with open(notes, "a") as f:
        f.write(f"\n## {args.model_name} — {intervention} ({setting})\n")
        f.write(f"- side_effect_score: {metrics['side_effect_score']:.3f}\n")
        f.write(f"- weirdness_rate: {metrics.get('weirdness_rate', float('nan')):.3f}\n")
        f.write(f"- output_length_ratio: {metrics.get('output_length_ratio', float('nan')):.3f}\n")
        f.write(f"- repetition_increase: {metrics.get('repetition_increase', float('nan')):.3f}\n")
        f.write(f"- qa_accuracy_drop: {metrics.get('qa_accuracy_drop', float('nan')):.3f}\n")
        f.write(f"- Sample (prompt -> baseline | intervened):\n")
        for i in range(min(3, len(prompts))):
            f.write(f"  - `{prompts[i][:40]}` -> `{baseline[i][:50]}` | `{intervened[i][:50]}`\n")

    print(f"\n=== Side-Effect Eval — {args.model_name} ===")
    print(f"  side_effect_score : {metrics['side_effect_score']:.3f}  (0=clean, 1=broken)")
    print(f"  weirdness_rate    : {metrics.get('weirdness_rate', float('nan')):.3f}")
    print(f"  output_length_ratio: {metrics.get('output_length_ratio', float('nan')):.3f}")
    print(f"  qa_accuracy_drop  : {metrics.get('qa_accuracy_drop', float('nan')):.3f}")
    logger.info("=== Step 13 complete. ===")


if __name__ == "__main__":
    main()
