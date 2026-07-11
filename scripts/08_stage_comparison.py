#!/usr/bin/env python3
"""Step 08: Compare sycophancy across model families / training stages (reviewer feedback #2, #3).

Reads existing per-model artifacts (no recompute) and assembles a cross-model summary:
sycophancy rate, behavior-margin stats, probe decodability, and causal effect per method.

Outputs:
  results/tables/stage_comparison_summary.csv
  plots/comparison/stage_sycophancy_rate.png
  plots/comparison/stage_probe_decodability.png
  plots/comparison/stage_causal_control.png

Example:
  python scripts/08_stage_comparison.py --models gpt2-small EleutherAI/pythia-410m Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_registry import get_model_config
from src.plotting import plot_stage_comparison
from src.utils import get_root, load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)
TAG = "prompt_preferences_prompt_final"


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Stage / family comparison")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--sample_size", type=int, default=300)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _row_for_model(model: str) -> dict:
    sn = safe_model_name(model)
    root = get_root()
    cfg = get_model_config(model)
    row = {"model_name": model, "model_family": cfg.family,
           "training_stage": getattr(cfg, "training_stage", "base")}

    pref = root / "data" / "processed" / f"{sn}_prompt_preferences.csv"
    if pref.exists():
        d = pd.read_csv(pref).dropna(subset=["behavior_margin"])
        row["num_prompts"] = len(d)
        row["sycophancy_rate"] = float((d["behavior_margin"] > 0).mean())
        row["mean_behavior_margin"] = float(d["behavior_margin"].mean())
        row["std_behavior_margin"] = float(d["behavior_margin"].std())

    metrics = root / "results" / "tables" / f"{sn}_{TAG}_layer_probe_metrics.csv"
    if metrics.exists():
        m = pd.read_csv(metrics)
        if "test_pearson" in m.columns:
            b = m.loc[m["test_pearson"].idxmax()]
            row["best_probe_pearson"] = float(b["test_pearson"])
            row["best_probe_spearman"] = float(b.get("test_spearman", float("nan")))
            row["best_probe_layer"] = int(b["layer"])
            row["best_probe_layer_frac"] = float(b["layer"]) / max(len(m) - 1, 1)

    causal = root / "results" / "tables" / f"{sn}_{TAG}_causal_validation.csv"
    if causal.exists():
        c = pd.read_csv(causal).set_index("method")
        for meth in ("probe_gradient", "random", "logit_gradient"):
            if meth in c.index:
                row[f"{meth}_behavior_delta"] = float(c.loc[meth, "behavior_margin_delta"])
                if "answer_flip_rate" in c.columns:
                    row[f"answer_flip_rate_{meth}"] = float(c.loc[meth, "answer_flip_rate"])
                if "accuracy_change" in c.columns:
                    row[f"accuracy_change_{meth}"] = float(c.loc[meth, "accuracy_change"])
    return row


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("=== Step 08: Stage Comparison ===")

    rows = [_row_for_model(m) for m in args.models]
    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / "stage_comparison_summary.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    plot_stage_comparison(df)

    print("\n=== Stage / Family Comparison ===")
    show = [c for c in ["model_name", "training_stage", "num_prompts", "sycophancy_rate",
                        "best_probe_pearson", "probe_gradient_behavior_delta",
                        "random_behavior_delta"] if c in df.columns]
    print(df[show].to_string(index=False, float_format="%.3f"))
    print("\nKey question: does sycophancy_rate and/or causal controllability rise from "
          "base → instruct? (the reviewer #2)")
    logger.info("=== Step 08 complete. ===")


if __name__ == "__main__":
    main()
