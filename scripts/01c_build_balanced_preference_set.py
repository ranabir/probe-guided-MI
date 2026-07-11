#!/usr/bin/env python3
"""Step 01c: Build a balanced preference diagnostic set.

The natural prompt-preference set is heavily imbalanced (e.g. GPT-2: ~7% prompts prefer
sycophancy). A regression/classification signal can be dominated by that base rate. This
script samples an equal number of positive-margin (behavior_margin > 0) and negative-margin
(behavior_margin <= 0) prompts so the diagnostic is balanced.

If positives are too few, it uses ALL positives and an equal number of negatives, and reports
the true imbalance honestly.

Output (one row per prompt, same schema as the natural set):
    data/processed/{safe_model_name}_prompt_preferences_balanced.csv
    data/processed/{safe_model_name}_prompt_preferences_balanced_{train,val,test}.csv

Example:
    python scripts/01c_build_balanced_preference_set.py \
      --model_name gpt2-small \
      --input data/processed/gpt2-small_prompt_preferences.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Build balanced preference diagnostic set")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input", type=str, default=None,
                   help="Defaults to data/processed/{safe_model_name}_prompt_preferences.csv")
    p.add_argument("--train_frac", type=float, default=cfg["data"]["train_frac"])
    p.add_argument("--val_frac", type=float, default=cfg["data"]["val_frac"])
    p.add_argument("--seed", type=int, default=cfg["data"]["random_seed"])
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _split(df, train_frac, val_frac, seed):
    from sklearn.model_selection import train_test_split
    test_frac = 1.0 - train_frac - val_frac
    strat = df["prefers_sycophancy"] if df["prefers_sycophancy"].nunique() > 1 \
        and df["prefers_sycophancy"].value_counts().min() >= 2 else None
    tr, tmp = train_test_split(df, test_size=(val_frac + test_frac), random_state=seed, stratify=strat)
    val_rel = val_frac / (val_frac + test_frac)
    strat2 = tmp["prefers_sycophancy"] if strat is not None and tmp["prefers_sycophancy"].value_counts().min() >= 2 else None
    val, te = train_test_split(tmp, test_size=(1 - val_rel), random_state=seed, stratify=strat2)
    return tr.reset_index(drop=True), val.reset_index(drop=True), te.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    input_path = Path(args.input) if args.input else processed / f"{sn}_prompt_preferences.csv"
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_path
    if not input_path.exists():
        logger.error("Input not found: %s\nRun scripts/01b_build_prompt_preferences.py first.", input_path)
        sys.exit(1)

    logger.info("=== Step 01c: Build Balanced Preference Set ===")
    df = pd.read_csv(input_path).dropna(subset=["behavior_margin"]).reset_index(drop=True)

    from src.data import build_balanced_subset
    balanced, info = build_balanced_subset(df, seed=args.seed)
    pos_n, neg_n, n = info["n_pos_total"], info["n_neg_total"], info["n_each"]
    logger.info("Natural balance: positives=%d negatives=%d (total=%d)", pos_n, neg_n, len(df))
    if n == 0:
        logger.error("One class is empty (pos=%d neg=%d); cannot build a balanced set.", pos_n, neg_n)
        sys.exit(1)

    imbalance_note = ""
    if pos_n < neg_n:
        imbalance_note = (f" (positives were the minority: only {pos_n}/{len(df)} = "
                          f"{info['frac_pos_natural']:.1%} of prompts prefer sycophancy — used all positives)")
    logger.info("Balanced set: %d positives + %d negatives = %d total%s", n, n, len(balanced), imbalance_note)

    out = processed / f"{sn}_prompt_preferences_balanced.csv"
    balanced.to_csv(out, index=False)
    logger.info("Saved -> %s", out)

    tr, val, te = _split(balanced, args.train_frac, args.val_frac, args.seed)
    for name, d in [("train", tr), ("val", val), ("test", te)]:
        d.to_csv(processed / f"{sn}_prompt_preferences_balanced_{name}.csv", index=False)
        logger.info("Saved balanced %s: %d rows", name, len(d))

    print(f"\n=== Balanced Preference Set: {args.model_name} ===")
    print(f"  Natural: {pos_n} pos / {neg_n} neg (total {len(df)})")
    print(f"  Balanced: {n} pos / {n} neg (total {len(balanced)})  | train {len(tr)} / val {len(val)} / test {len(te)}")
    if imbalance_note:
        print(f"  Note:{imbalance_note}")
    print(f"  behavior_margin (balanced): mean={balanced['behavior_margin'].mean():.4f} "
          f"std={balanced['behavior_margin'].std():.4f}")


if __name__ == "__main__":
    main()
