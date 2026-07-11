#!/usr/bin/env python3
"""Step 03: Train linear probes per layer and select an attribution probe.

Classification (paired_rows, response-aware sanity check):
    hidden_state[layer] -> label        metrics: accuracy, AUROC, F1
Regression (prompt_preferences, main experiment):
    hidden_state[layer] -> behavior_margin   metrics: MSE, MAE, Pearson, Spearman, R2
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.probes import (
    save_best_probe,
    save_selection_metadata,
    select_attribution_probe,
    train_all_layer_probes,
)
from src.utils import load_config, results_dir, safe_model_name, setup_logging
from src.visualization import plot_layer_probe_accuracy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Train linear probes per layer")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="paired_rows",
                   choices=["paired_rows", "prompt_preferences"])
    p.add_argument("--probe_position", type=str,
                   default=cfg["activations"].get("probe_position", "response_final"))
    p.add_argument("--probe_type", type=str, default=None,
                   choices=["classification", "regression"],
                   help="Default: regression for prompt_preferences, classification otherwise.")
    p.add_argument("--probe_target", type=str, default=None,
                   help="label | prefers_sycophancy | behavior_margin. Auto-selected by default.")
    p.add_argument("--standardize", type=bool, default=cfg["probe"]["standardize"])
    p.add_argument("--max_iter", type=int, default=cfg["probe"]["max_iter"])
    p.add_argument("--seed", type=int, default=cfg["probe"]["random_seed"])
    p.add_argument("--probe_layer_policy", type=str,
                   default=cfg["probe"].get("probe_layer_policy", "best_late"))
    p.add_argument("--min_probe_layer_frac", type=float,
                   default=cfg["probe"].get("min_probe_layer_frac", 0.65))
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _resolve_target(args):
    """Return (probe_type, probe_target, target_key_in_cache)."""
    if args.input_format == "prompt_preferences":
        ptype = args.probe_type or "regression"
        if ptype == "regression":
            target = args.probe_target or "behavior_margin"
        else:
            target = args.probe_target or "prefers_sycophancy"
    else:
        ptype = args.probe_type or "classification"
        target = args.probe_target or "label"
    return ptype, target


def _extract_target(data, target, probe_type):
    """Pull the target array from a loaded activation cache dict."""
    if target == "behavior_margin":
        if "behavior_margin" not in data:
            raise KeyError("behavior_margin not in cache. Re-run step 02 with --input_format prompt_preferences.")
        return np.asarray(data["behavior_margin"], dtype=np.float32)
    # prefers_sycophancy and label are both stored as 'labels'
    return np.asarray(data["labels"]).astype(np.int64 if probe_type == "classification" else np.float32)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    pp = "prompt_final" if args.input_format == "prompt_preferences" else args.probe_position
    inf = args.input_format
    probe_type, target = _resolve_target(args)

    logger.info("=== Step 03: Train Probes ===")
    logger.info("Model: %s | input_format: %s | probe_position: %s | type: %s | target: %s",
                args.model_name, inf, pp, probe_type, target)

    splits = {}
    for split in ("train", "val", "test"):
        try:
            splits[split] = activations_to_numpy(load_activations(args.model_name, split,
                                                                  probe_position=pp, input_format=inf))
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

    train_X = splits["train"]["hidden_states"]
    val_X = splits["val"]["hidden_states"]
    test_X = splits["test"]["hidden_states"]
    try:
        train_y = _extract_target(splits["train"], target, probe_type)
        val_y = _extract_target(splits["val"], target, probe_type)
        test_y = _extract_target(splits["test"], target, probe_type)
    except KeyError as e:
        logger.error("%s", e)
        sys.exit(1)

    n_layers = train_X.shape[1]
    logger.info("Data shapes: train=%s val=%s test=%s", train_X.shape, val_X.shape, test_X.shape)
    if probe_type == "regression":
        logger.info("Target stats (train): mean=%.4f std=%.4f", float(train_y.mean()), float(train_y.std()))
    else:
        logger.info("Label balance train=%s", dict(zip(*np.unique(train_y, return_counts=True))))

    metrics_list, _ = train_all_layer_probes(
        train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y,
        test_X=test_X, test_y=test_y, model_name=args.model_name,
        probe_position=pp, input_format=inf, task=probe_type,
        standardize=args.standardize, max_iter=args.max_iter, seed=args.seed,
    )

    metrics_df = pd.DataFrame(metrics_list)
    sn = safe_model_name(args.model_name)
    tag = (f"{inf}_{pp}" if inf == "prompt_preferences" else pp)
    tables_dir = Path(results_dir("tables"))
    table_path = tables_dir / f"{sn}_{tag}_layer_probe_metrics.csv"
    metrics_df.to_csv(table_path, index=False)
    logger.info("Saved metrics -> %s", table_path)

    metric_col = "val_pearson" if probe_type == "regression" else "val_auroc"
    attr_probe, sel_meta = select_attribution_probe(
        metrics_list=metrics_list, n_layers=n_layers, model_name=args.model_name,
        probe_position=pp, input_format=inf, policy=args.probe_layer_policy,
        min_probe_layer_frac=args.min_probe_layer_frac, metric_col=metric_col,
    )
    sel_meta["probe_type"] = probe_type
    sel_meta["probe_target"] = target
    save_selection_metadata(sel_meta, args.model_name, probe_position=pp, input_format=inf)

    # Persist the policy-selected probe as the canonical "best" probe so steps 04/05
    # attribute and ablate the SELECTED layer (not merely the highest-metric layer).
    save_best_probe(attr_probe, args.model_name, probe_position=pp, input_format=inf)

    print(f"\n=== Probe Results: {args.model_name} | {inf} | {pp} | {probe_type} ({target}) ===")
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    print(f"\nSelected attribution probe ({args.probe_layer_policy}): layer {attr_probe.layer_idx}")
    print(f"  {metric_col} = {sel_meta['selected_metric']:.4f}  |  reason: {sel_meta['reason']}")
    print(f"  Gradients cover layers 0–{attr_probe.layer_idx} ({attr_probe.layer_idx + 1}/{n_layers})")

    # Reuse the accuracy plot for classification; for regression it plots available columns
    try:
        plot_layer_probe_accuracy(metrics_df, model_name=f"{args.model_name}_{tag}")
    except Exception as e:
        logger.warning("Probe metric plot skipped: %s", e)

    logger.info("=== Step 03 complete. Attribution probe: layer %d ===", attr_probe.layer_idx)


if __name__ == "__main__":
    main()
