#!/usr/bin/env python3
"""Step 11: Control probes — test whether the sycophancy probe follows trivial confounds.

Compares the real sycophancy-margin probe against:
  - random_label  (shuffled behavior_margin)
  - static_token  (surface features: length, "I think", A/B format, first token, ...)
  - topic         (predict the source subset)

Outputs:
  results/tables/{safe_model_name}_control_probe_metrics.csv   (long format)
  plots/{safe_model_name}/{safe_model_name}_probe_vs_controls_by_layer.png

Example:
  python scripts/11_run_controls.py --model_name gpt2-small \
    --input_format prompt_preferences --probe_position prompt_final \
    --controls random_label static_token topic
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.controls import run_all_controls
from src.model_loader import load_adapter
from src.plotting import plot_probe_vs_controls
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run control probes")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--controls", nargs="+", default=["random_label", "static_token", "topic"])
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--need_tokenizer", action="store_true",
                   help="Load the model tokenizer for token-count / first-token features (slower).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    tag = f"{inf}_{pp}" if inf == "prompt_preferences" else pp
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    logger.info("=== Step 11: Control Probes (%s) ===", args.model_name)

    # Load activations + aligned preference split CSVs
    splits, pref_splits = {}, {}
    for split in ("train", "test"):
        try:
            splits[split] = activations_to_numpy(load_activations(args.model_name, split,
                                                                  probe_position=pp, input_format=inf))
        except FileNotFoundError as e:
            logger.error("%s", e); sys.exit(1)
        csv = processed / f"{sn}_prompt_preferences_{split}.csv"
        if not csv.exists():
            logger.error("Missing %s. Run scripts/01b first.", csv); sys.exit(1)
        pref_splits[split] = pd.read_csv(csv)

    # Optional tokenizer (only if user wants token-based surface features)
    tokenizer = None
    if args.need_tokenizer:
        try:
            adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
            tokenizer = adapter.tokenizer
        except Exception as e:
            logger.warning("Could not load tokenizer (%s); using whitespace features.", e)

    control_df = run_all_controls(splits, pref_splits, tokenizer=tokenizer,
                                  controls=args.controls, seed=args.seed)
    control_df["model_name"] = args.model_name

    out_csv = Path(results_dir("tables")) / f"{sn}_control_probe_metrics.csv"
    control_df.to_csv(out_csv, index=False)
    logger.info("Saved control metrics -> %s", out_csv)

    # Real sycophancy probe decodability (from step 03 metrics)
    real_csv = Path(results_dir("tables")) / f"{sn}_{tag}_layer_probe_metrics.csv"
    real_df = pd.read_csv(real_csv) if real_csv.exists() else None

    plot_probe_vs_controls(control_df, real_df, args.model_name)

    # Summary print
    print(f"\n=== Control decodability (peak test metric by control) — {args.model_name} ===")
    if real_df is not None and "test_pearson" in real_df.columns:
        print(f"  sycophancy probe   : peak test Pearson = {real_df['test_pearson'].max():.3f}")
    for cname, g in control_df.groupby("control_name"):
        best = g.loc[g["test_metric"].idxmax()]
        print(f"  {cname:14s} : peak {best['metric_name']} = {best['test_metric']:.3f} "
              f"(feature={best['feature']}, layer={int(best['layer'])})")
    print("\nInterpretation: if a control's peak rivals the sycophancy probe, the probe may be "
          "partly a confound. random_label should be ~0.")

    logger.info("=== Step 11 complete. ===")


if __name__ == "__main__":
    main()
