#!/usr/bin/env python3
"""Step 17: Interpret the sycophancy subspace — do its directions correspond to the sub-types?

The subspace-ablation result showed rank-5–8 ablation gives clean, strong control while rank-1 is
weak. This script asks WHY: is the subspace a union of the three topic-specific sycophancy
directions (philosophy / NLP / political)? If each sub-type needs its own basis dimension, that
explains why one direction is insufficient and a subspace is required.

Analysis (at the peak-decodable layer, using cached train activations):
  1. captured-energy vs rank — how much of each sub-type's own difference-of-means direction lies
     inside the rank-k subspace (rises toward 1 as k grows if the subspace absorbs that sub-type).
  2. cosine heatmap — sub-type directions vs subspace basis vectors.
  3. singular-value spectrum — is the subspace genuinely multi-dimensional or rank-1 + noise?

Outputs:
  results/tables/{sn}_subspace_interpretation.csv
  results/tables/{sn}_subspace_subtype_cosine.csv
  plots/{sn}/{sn}_subspace_subtype_capture.png
  plots/{sn}/{sn}_subspace_singular_spectrum.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.plotting import plot_subspace_interpretation
from src.sycophancy_subspace import subtype_capture_vs_rank
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Interpret the sycophancy subspace")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    p.add_argument("--layer", type=int, default=None, help="Default: peak-decodable layer")
    p.add_argument("--n_splits", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    tag = f"{inf}_{pp}"
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    logger.info("=== Step 17: Interpret Subspace — %s ===", args.model_name)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    hs, mg = tr["hidden_states"], np.asarray(tr["behavior_margin"])
    ids = tr.get("ids")

    # sub-type labels: align the train activations to their prompt-preference rows via id
    pref = pd.read_csv(processed / f"{sn}_prompt_preferences_train.csv")
    if "subset" not in pref.columns:
        logger.error("No 'subset' column — cannot map sub-types. Rebuild dataset with subset labels.")
        sys.exit(1)
    if ids is not None:
        id2subset = dict(zip(pref["id"].astype(str), pref["subset"].astype(str)))
        subsets = np.array([id2subset.get(str(i), "unknown") for i in ids])
    else:
        # fall back to positional alignment (same order as caching)
        subsets = pref["subset"].astype(str).values[:len(hs)]
    logger.info("Sub-type counts: %s", dict(pd.Series(subsets).value_counts()))

    # peak-decodable layer
    if args.layer is not None:
        layer = args.layer
    else:
        mcsv = Path(results_dir("tables")) / f"{sn}_{tag}_layer_probe_metrics.csv"
        m = pd.read_csv(mcsv)
        layer = int(m.loc[m["test_pearson"].idxmax(), "layer"]) if mcsv.exists() else hs.shape[1] // 2
    logger.info("Analyzing layer %d", layer)

    rows, cos_df, sv = subtype_capture_vs_rank(hs[:, layer, :], mg, subsets, args.ranks,
                                               n_splits=args.n_splits, seed=args.seed)
    cap_df = pd.DataFrame(rows)
    cap_df["model_name"] = args.model_name
    cap_df["layer"] = layer
    cap_path = Path(results_dir("tables")) / f"{sn}_subspace_interpretation.csv"
    cap_df.to_csv(cap_path, index=False)
    cos_df.to_csv(Path(results_dir("tables")) / f"{sn}_subspace_subtype_cosine.csv")
    pd.DataFrame({"dim": range(1, len(sv) + 1), "relative_energy": sv}).to_csv(
        Path(results_dir("tables")) / f"{sn}_subspace_singular_spectrum.csv", index=False)
    logger.info("Saved -> %s", cap_path)

    plot_subspace_interpretation(cap_df, cos_df, sv, args.model_name, layer)

    # summary
    print(f"\n=== Subspace interpretation — {args.model_name} (layer {layer}) ===")
    print("\nCaptured energy of each sub-type direction vs subspace rank:")
    piv = cap_df.pivot_table(index="rank", columns="subtype", values="captured_energy")
    print(piv.to_string(float_format="%.2f"))
    print(f"\nSingular-value spectrum (relative energy per dim): {[round(x,3) for x in sv]}")
    # how many dims to capture 90% energy
    cum = np.cumsum(sv)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"Dims to reach 90% of the difference-of-means energy: {n90}")
    print("\nInterpretation: if each sub-type's capture rises toward 1 only as rank grows, and the")
    print("spectrum is spread across several dims, sycophancy is a multi-directional (per-topic) subspace.")
    logger.info("=== Step 17 complete. ===")


if __name__ == "__main__":
    main()
