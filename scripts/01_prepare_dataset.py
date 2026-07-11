#!/usr/bin/env python3
"""Step 01: Prepare sycophancy dataset.

Dataset hierarchy:
  synthetic           — smoke-test only; always available, no download needed
  anthropic_sycophancy — main demo dataset from Anthropic/model-written-evals (HuggingFace)
  truthfulqa          — generalization dataset (HuggingFace)
  bbq                 — optional future extension (not yet implemented)
  ethics              — optional future extension (not yet implemented)

Output files (flat schema, pipeline-compatible):
  data/processed/sycophancy_pairs.csv   paired schema (prompt + both responses)
  data/processed/train.csv              flat schema  (one row per response)
  data/processed/val.csv
  data/processed/test.csv

Default: --dataset synthetic  (backward-compatible with --synthetic_only flag)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    DATASET_LOADERS,
    load_dataset_by_name,
    load_or_generate_dataset,
    paired_to_flat,
    split_dataset,
    to_paired_schema,
)
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = list(DATASET_LOADERS.keys())


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(
        description="Prepare sycophancy dataset for probe-guided attribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (always works, no internet needed):
  python scripts/01_prepare_dataset.py --dataset synthetic --sample_size 80

  # Main demo dataset:
  python scripts/01_prepare_dataset.py --dataset anthropic_sycophancy --sample_size 300

  # Generalization dataset:
  python scripts/01_prepare_dataset.py --dataset truthfulqa --sample_size 200

  # Legacy flag still works:
  python scripts/01_prepare_dataset.py --synthetic_only --sample_size 80
""",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=AVAILABLE_DATASETS,
        help=(
            "Dataset to load. "
            "'synthetic': smoke-test, no download. "
            "'anthropic_sycophancy': main demo (requires HuggingFace). "
            "'truthfulqa': generalization (requires HuggingFace). "
            f"Available: {AVAILABLE_DATASETS}"
        ),
    )
    p.add_argument("--sample_size", type=int, default=cfg["data"]["sample_size"])
    p.add_argument(
        "--output_format",
        type=str,
        default="paired_rows",
        choices=["paired_rows", "prompt_preferences"],
        help=(
            "paired_rows (default): flat train/val/test for the response-aware pipeline. "
            "prompt_preferences: also produces paired CSV; then run scripts/01b_build_prompt_preferences.py "
            "to compute behavior margins (one row per prompt)."
        ),
    )
    p.add_argument(
        "--synthetic_only",
        action="store_true",
        default=False,
        help="Legacy flag: forces --dataset synthetic regardless of other args.",
    )
    p.add_argument("--seed", type=int, default=cfg["data"]["random_seed"])
    p.add_argument("--train_frac", type=float, default=cfg["data"]["train_frac"])
    p.add_argument("--val_frac", type=float, default=cfg["data"]["val_frac"])
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _build_paired_df(flat_df):
    """Build a paired-schema DataFrame from a flat-schema DataFrame."""
    syc = flat_df[flat_df["label"] == 1][["prompt", "response", "id", "source_dataset", "subset"]].rename(
        columns={"response": "sycophantic_response", "id": "syc_id"}
    )
    non = flat_df[flat_df["label"] == 0][["prompt", "response", "id"]].rename(
        columns={"response": "non_sycophantic_response", "id": "non_id"}
    )
    paired = syc.merge(non, on="prompt", how="inner")
    paired["id"] = paired["syc_id"].str.replace("_syc", "", regex=False)
    paired["label"] = 1
    keep = ["id", "prompt", "sycophantic_response", "non_sycophantic_response",
            "label", "source_dataset", "subset"]
    return paired[[c for c in keep if c in paired.columns]].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --synthetic_only overrides --dataset for backward compatibility
    dataset_name = args.dataset
    if args.synthetic_only:
        logger.info("--synthetic_only flag set; overriding --dataset to 'synthetic'.")
        dataset_name = "synthetic"

    logger.info(
        "Preparing dataset: name=%s sample_size=%d seed=%d",
        dataset_name, args.sample_size, args.seed,
    )

    # Load flat-schema DataFrame
    try:
        flat_df = load_dataset_by_name(dataset_name, sample_size=args.sample_size, seed=args.seed)
    except NotImplementedError as e:
        logger.error("Dataset not yet implemented:\n%s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Dataset loading failed:\n%s", e)
        logger.info("Hint: run with --dataset synthetic to use the always-available smoke-test dataset.")
        sys.exit(1)

    # Ensure required columns exist
    if "response" not in flat_df.columns:
        logger.error("Dataset missing 'response' column. Schema issue — check the loader.")
        sys.exit(1)
    if "label" not in flat_df.columns:
        flat_df["label"] = -1
    if "source_dataset" not in flat_df.columns:
        flat_df["source_dataset"] = dataset_name
    if "subset" not in flat_df.columns:
        flat_df["subset"] = ""

    # Save paired CSV (for behavior metrics and inspection)
    paired_df = _build_paired_df(flat_df)
    pairs_path = processed_dir / "sycophancy_pairs.csv"
    paired_df.to_csv(pairs_path, index=False)
    logger.info("Saved %d pairs -> %s", len(paired_df), pairs_path)

    # Class balance
    balance = flat_df["label"].value_counts().to_dict()
    logger.info("Class balance: %s", balance)

    # Source breakdown
    if "source_dataset" in flat_df.columns:
        src_counts = flat_df["source_dataset"].value_counts().to_dict()
        logger.info("Source breakdown: %s", src_counts)
    if "subset" in flat_df.columns and flat_df["subset"].nunique() > 1:
        sub_counts = flat_df["subset"].value_counts().to_dict()
        logger.info("Subset breakdown: %s", sub_counts)

    # Split into train/val/test (flat schema — pipeline compatible)
    train_df, val_df, test_df = split_dataset(
        flat_df, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = processed_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("Saved %s: %d rows -> %s", split_name, len(split_df), path)

    # Print sample examples
    print(f"\n=== Dataset: {dataset_name} | {len(flat_df)} total rows | {len(paired_df)} pairs ===")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n--- Sample sycophantic examples ---")
    for _, row in flat_df[flat_df["label"] == 1].head(3).iterrows():
        print(f"  [{row.get('subset', row.get('category', ''))}]")
        print(f"  Prompt  : {str(row.get('prompt', ''))[:90]}")
        print(f"  Response: {str(row.get('response', ''))[:90]}")
        print()

    print("--- Sample non-sycophantic examples ---")
    for _, row in flat_df[flat_df["label"] == 0].head(3).iterrows():
        print(f"  [{row.get('subset', row.get('category', ''))}]")
        print(f"  Prompt  : {str(row.get('prompt', ''))[:90]}")
        print(f"  Response: {str(row.get('response', ''))[:90]}")
        print()

    if args.output_format == "prompt_preferences":
        print("\n[prompt_preferences] Paired CSV written. Next, compute behavior margins:")
        print(f"  python scripts/01b_build_prompt_preferences.py --model_name gpt2-small "
              f"--input {pairs_path} --sample_size {args.sample_size // 2}")
    else:
        print(f"\nNext step: python scripts/02_cache_activations.py --model_name gpt2-small --sample_size {args.sample_size}")


if __name__ == "__main__":
    main()
