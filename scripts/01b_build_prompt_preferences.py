#!/usr/bin/env python3
"""Step 01b: Build a prompt-level preference dataset (one row per prompt).

Fixes the invalid paired-row prompt_final setup: instead of two rows per prompt with
opposite labels but identical prompt-only activations, we compute a single per-prompt
target from the model's own behavior:

    syc_logprob     = mean-token logprob(sycophantic_response | prompt)
    non_syc_logprob = mean-token logprob(non_sycophantic_response | prompt)
    behavior_margin = syc_logprob - non_syc_logprob
    prefers_sycophancy = 1 if behavior_margin > 0 else 0

Output (one row per prompt):
    data/processed/{safe_model_name}_prompt_preferences.csv
    data/processed/{safe_model_name}_prompt_preferences_{train,val,test}.csv

Example:
    python scripts/01b_build_prompt_preferences.py \
      --model_name gpt2-small \
      --input data/processed/sycophancy_pairs.csv \
      --sample_size 40
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior_metrics import build_prompt_preference_dataset, summarize_behavior_margins
from src.model_loader import load_adapter
from src.utils import load_config, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Build prompt-level preference dataset")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input", type=str, default="data/processed/sycophancy_pairs.csv")
    p.add_argument("--sample_size", type=int, default=None, help="Max number of prompts (pairs)")
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--train_frac", type=float, default=cfg["data"]["train_frac"])
    p.add_argument("--val_frac", type=float, default=cfg["data"]["val_frac"])
    p.add_argument("--seed", type=int, default=cfg["data"]["random_seed"])
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _stratified_or_random_split(df, train_frac, val_frac, seed):
    """Split into train/val/test, stratifying on prefers_sycophancy when feasible."""
    from sklearn.model_selection import train_test_split
    test_frac = 1.0 - train_frac - val_frac

    label = df["prefers_sycophancy"] if "prefers_sycophancy" in df.columns else None
    # Stratify only if every class has >= 2 members
    can_stratify = label is not None and label.value_counts().min() >= 2 and label.nunique() > 1

    strat = label if can_stratify else None
    train_df, temp_df = train_test_split(df, test_size=(val_frac + test_frac),
                                         random_state=seed, stratify=strat)
    val_rel = val_frac / (val_frac + test_frac)
    strat2 = temp_df["prefers_sycophancy"] if can_stratify and temp_df["prefers_sycophancy"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_rel),
                                       random_state=seed, stratify=strat2)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    sn = safe_model_name(args.model_name)
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_path
    if not input_path.exists():
        logger.error("Input pairs file not found: %s\nRun scripts/01_prepare_dataset.py first.", input_path)
        sys.exit(1)

    logger.info("=== Step 01b: Build Prompt Preferences ===")
    logger.info("Model: %s | input: %s", args.model_name, input_path)

    paired_df = pd.read_csv(input_path)
    logger.info("Loaded %d paired rows.", len(paired_df))

    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    logger.info("Model loaded: n_layers=%d, d_model=%d", adapter.n_layers, adapter.d_model)

    out_csv = processed / f"{sn}_prompt_preferences.csv"
    pref_df = build_prompt_preference_dataset(
        adapter, paired_df, output_path=out_csv,
        max_length=args.max_length, normalize=True,
        max_examples=args.sample_size, seed=args.seed,
    )

    # Drop failed rows
    n_before = len(pref_df)
    pref_df = pref_df.dropna(subset=["behavior_margin"]).reset_index(drop=True)
    if len(pref_df) < n_before:
        logger.warning("Dropped %d rows with failed logprob computation.", n_before - len(pref_df))

    # Statistics
    summary = summarize_behavior_margins(pref_df)
    n_pref_syc = int((pref_df["prefers_sycophancy"] == 1).sum())
    n_pref_non = int((pref_df["prefers_sycophancy"] == 0).sum())

    logger.info("Behavior margin summary: %s", summary)
    logger.info("Label balance: prefers_sycophancy=1 -> %d | =0 -> %d", n_pref_syc, n_pref_non)

    # Split
    train_df, val_df, test_df = _stratified_or_random_split(
        pref_df, args.train_frac, args.val_frac, args.seed
    )
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = processed / f"{sn}_prompt_preferences_{name}.csv"
        d.to_csv(path, index=False)
        logger.info("Saved %s: %d rows -> %s", name, len(d), path)

    # Print report
    print(f"\n=== Prompt-Preference Dataset: {args.model_name} ===")
    print(f"  Prompts: {len(pref_df)}  |  train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
    print(f"  behavior_margin: mean={summary.get('mean_margin', float('nan')):.4f} "
          f"median={summary.get('median_margin', float('nan')):.4f} "
          f"std={summary.get('std_margin', float('nan')):.4f}")
    print(f"  prefers_sycophancy: {n_pref_syc} yes / {n_pref_non} no "
          f"(frac={summary.get('frac_prefers_syc', float('nan')):.3f})")
    print("\n  Sample rows:")
    for _, r in pref_df.head(3).iterrows():
        print(f"    margin={r['behavior_margin']:+.4f} prefers_syc={int(r['prefers_sycophancy'])} "
              f"| {str(r['prompt'])[:70]}")

    print(f"\nNext: python scripts/02_cache_activations.py --model_name {args.model_name} "
          f"--input_format prompt_preferences --probe_position prompt_final --sample_size {args.sample_size or len(pref_df)}")


if __name__ == "__main__":
    main()
