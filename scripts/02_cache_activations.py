#!/usr/bin/env python3
"""Step 02: Cache activations for each split.

Two input formats:
  paired_rows (default)   — flat train/val/test.csv; one row per response.
                            probe_position controls which token is extracted:
                              response_final: last token of "prompt + response"
                              prompt_final:   last token of "prompt" only
  prompt_preferences      — {sn}_prompt_preferences_{split}.csv; one row per prompt.
                            Always prompt_final. Caches prompt-final hidden state.
                            Stores label = prefers_sycophancy AND target = behavior_margin.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import save_activations, save_metadata
from src.model_loader import load_adapter
from src.utils import load_config, safe_model_name, setup_logging

logger = logging.getLogger(__name__)

PROBE_POSITIONS = ("prompt_final", "response_final")
INPUT_FORMATS = ("paired_rows", "prompt_preferences")


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Cache model activations for probe training")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--backend", type=str, default=cfg["model"]["backend"])
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--batch_size", type=int, default=cfg["model"]["batch_size"])
    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--input_format", type=str, default="paired_rows", choices=INPUT_FORMATS)
    p.add_argument(
        "--probe_position", type=str,
        default=cfg["activations"].get("probe_position", "response_final"),
        choices=PROBE_POSITIONS,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _processed_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed"


def load_split_paired(split, sample_size, seed) -> pd.DataFrame:
    path = _processed_dir() / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}\nRun: python scripts/01_prepare_dataset.py --synthetic_only --sample_size 300"
        )
    df = pd.read_csv(path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def load_split_preferences(model_name, split, sample_size, seed) -> pd.DataFrame:
    sn = safe_model_name(model_name)
    path = _processed_dir() / f"{sn}_prompt_preferences_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt-preference split not found: {path}\n"
            f"Run: python scripts/01b_build_prompt_preferences.py --model_name {model_name} "
            f"--input data/processed/sycophancy_pairs.csv"
        )
    df = pd.read_csv(path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def build_texts(adapter, df, input_format, probe_position) -> List[str]:
    if input_format == "prompt_preferences" or probe_position == "prompt_final":
        return [str(r["prompt"]) for _, r in df.iterrows()]
    return [adapter.format_prompt_response(str(r["prompt"]), str(r["response"]))
            for _, r in df.iterrows()]


def cache_split(adapter, df, split, model_name, input_format, probe_position,
                max_length, batch_size) -> None:
    texts = build_texts(adapter, df, input_format, probe_position)

    if input_format == "prompt_preferences":
        labels = df["prefers_sycophancy"].values.astype(np.int64)
        margins = df["behavior_margin"].values.astype(np.float32)
    else:
        labels = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
        margins = None

    ids = df["id"].tolist() if "id" in df.columns else [str(i) for i in range(len(df))]

    n = len(texts)
    logger.info("Caching %d examples | split=%s | input_format=%s | probe_position=%s",
                n, split, input_format, probe_position)

    all_acts: Dict[str, List[np.ndarray]] = {}
    for start in range(0, n, batch_size):
        batch = texts[start:start + batch_size]
        logger.info("  Batch %d-%d", start, start + len(batch) - 1)
        with torch.no_grad():
            acts = adapter.forward_with_cache(batch, max_length=max_length, token_position="final")
        for k, v in acts.items():
            all_acts.setdefault(k, []).append(v)

    concat = {k: np.concatenate(v, axis=0) for k, v in all_acts.items()}
    logger.info("  Activation shapes: %s", {k: v.shape for k, v in concat.items()})

    save_activations(model_name, split, concat, labels, ids,
                     probe_position=probe_position, input_format=input_format,
                     behavior_margins=margins)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)

    # prompt_preferences is always prompt_final
    probe_position = "prompt_final" if args.input_format == "prompt_preferences" else args.probe_position

    logger.info("=== Step 02: Cache Activations ===")
    logger.info("Model: %s | input_format: %s | probe_position: %s",
                args.model_name, args.input_format, probe_position)

    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype,
                           backend_override=None if args.backend == "auto" else args.backend)
    logger.info("Model loaded: n_layers=%d, d_model=%d", adapter.n_layers, adapter.d_model)

    for split in ("train", "val", "test"):
        try:
            if args.input_format == "prompt_preferences":
                df = load_split_preferences(args.model_name, split, args.sample_size, args.seed)
            else:
                df = load_split_paired(split, args.sample_size, args.seed)
            cache_split(adapter, df, split, args.model_name, args.input_format,
                        probe_position, args.max_length, args.batch_size)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

    meta = {
        "model_name": args.model_name,
        "safe_name": safe_model_name(args.model_name),
        "input_format": args.input_format,
        "probe_position": probe_position,
        "token_position_used": "final",
        "target_columns": (["prefers_sycophancy", "behavior_margin"]
                           if args.input_format == "prompt_preferences" else ["label"]),
        "n_layers": adapter.n_layers,
        "d_model": adapter.d_model,
        "max_length": args.max_length,
        "backend": adapter.model_config.backend,
        "family": adapter.model_config.family,
        "prompt_only_mode": probe_position == "prompt_final",
    }
    save_metadata(args.model_name, meta, probe_position=probe_position, input_format=args.input_format)
    logger.info("=== Step 02 complete. ===")


if __name__ == "__main__":
    main()
