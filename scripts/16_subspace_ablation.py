#!/usr/bin/env python3
"""Step 16: Sycophancy-subspace ablation — is control 'clean AND strong' when we remove a
low-rank subspace instead of a single direction?

Tests the hypothesis that sycophancy is not one direction but a small subspace: a single-direction
ablation is clean but weak; ablating a rank-k subspace across a band of layers should flip more
answers while still touching only k of D dims (so the model stays intact).

Sweeps rank k × layer-band, measuring targeted syc→honest flip vs capability side-effect.
Works for TransformerLens AND HuggingFace models.

Outputs:
  results/tables/{sn}_subspace_ablation.csv
  plots/{sn}/{sn}_subspace_ablation_pareto.png
  plots/{sn}/{sn}_subspace_ablation_by_rank.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.metrics import behavioral_intervention_metrics
from src.model_loader import load_adapter
from src.plotting import plot_subspace_ablation
from src.side_effects import compute_side_effect_score, load_basic_prompts
from src.statistics import bootstrap_mean_ci
from src.sycophancy_subspace import build_subspaces_for_layers, subspace_honest_target
from src import residual_interventions as ri
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Sycophancy-subspace ablation sweep")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences")
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    p.add_argument("--layer_bands", nargs="+", default=["peak", "top3", "midband"],
                   help="Named layer bands to try: peak / top3 / midband / all / a,b,c (explicit).")
    p.add_argument("--honest_target", action="store_true",
                   help="Move subspace coords to the honest-mean instead of zero.")
    p.add_argument("--max_examples", type=int, default=25)
    p.add_argument("--side_effect_prompts", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _resolve_band(name, n_layers, peak_layer):
    if "," in name:
        return [int(x) for x in name.split(",") if x.strip()]
    if name == "peak":
        return [peak_layer]
    if name == "top3":
        return sorted({max(0, peak_layer - 1), peak_layer, min(n_layers - 1, peak_layer + 1)})
    if name == "midband":
        lo, hi = int(n_layers * 0.25), int(n_layers * 0.75)
        return list(range(lo, hi + 1))
    if name == "all":
        return list(range(n_layers))
    return [peak_layer]


def main():
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)
    inf, pp = args.input_format, args.probe_position
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"

    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    logger.info("=== Step 16: Subspace Ablation — %s ===", args.model_name)

    tr = activations_to_numpy(load_activations(args.model_name, "train", probe_position=pp, input_format=inf))
    hs, mg = tr["hidden_states"], np.asarray(tr["behavior_margin"])
    n_layers = hs.shape[1]

    # peak-decodable layer from step-03 metrics (fallback: middle)
    tag = f"{inf}_{pp}"
    mcsv = Path(results_dir("tables")) / f"{sn}_{tag}_layer_probe_metrics.csv"
    if mcsv.exists():
        m = pd.read_csv(mcsv)
        peak_layer = int(m.loc[m["test_pearson"].idxmax(), "layer"]) if "test_pearson" in m else n_layers // 2
    else:
        peak_layer = n_layers // 2
    logger.info("Peak-decodable layer: %d (n_layers=%d)", peak_layer, n_layers)

    test_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv").dropna(subset=["behavior_margin"]).reset_index(drop=True)
    se_prompts = load_basic_prompts().head(args.side_effect_prompts)["prompt"].tolist()
    baseline_gen = ri.generate_with_edit(adapter, se_prompts, [], None,
                                         max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    rows = []
    for band_name in args.layer_bands:
        layers = _resolve_band(band_name, n_layers, peak_layer)
        for rank in args.ranks:
            subs = build_subspaces_for_layers(hs, mg, layers, rank=rank, seed=args.seed)
            tgt = subspace_honest_target(hs, mg, subs) if args.honest_target else None
            edit = ri.make_subspace_ablation(subs, honest_target=tgt, norm_preserve=True)

            before, after = ri.run_edit_on_prompts(adapter, test_df, layers, edit,
                                                   max_examples=args.max_examples, seed=args.seed,
                                                   max_length=args.max_length)
            deltas = [a - b for a, b in zip(after, before) if np.isfinite(a) and np.isfinite(b)]
            bm = behavioral_intervention_metrics(before, after)
            ci = bootstrap_mean_ci(deltas, n_boot=args.bootstrap, seed=args.seed) if deltas else {"ci_low": np.nan, "ci_high": np.nan}
            try:
                gen = ri.generate_with_edit(adapter, se_prompts, layers, edit,
                                            max_new_tokens=args.max_new_tokens, max_length=args.max_length)
                se = compute_side_effect_score(baseline_gen, gen)["side_effect_score"]
            except Exception as e:
                logger.warning("side-effect failed: %s", e)
                se = float("nan")
            rows.append({
                "model_name": args.model_name, "layer_band": band_name, "layers": str(layers),
                "n_layers_edited": len(layers), "rank": rank,
                "behavior_margin_delta": float(np.mean(deltas)) if deltas else float("nan"),
                "answer_flip_rate": bm["answer_flip_rate"],
                "targeted_flip_rate": bm["targeted_syc_to_non_syc_flip_rate"],
                "accuracy_change": bm["accuracy_change"], "side_effect_score": se,
                "ci_low": ci.get("ci_low", np.nan), "ci_high": ci.get("ci_high", np.nan),
                "n_examples": len(deltas),
            })
            logger.info("  band=%s rank=%d layers=%s: flip=%.2f side_effect=%.2f",
                        band_name, rank, layers, bm["targeted_syc_to_non_syc_flip_rate"], se)

    df = pd.DataFrame(rows)
    out = Path(results_dir("tables")) / f"{sn}_subspace_ablation.csv"
    df.to_csv(out, index=False)
    logger.info("Saved -> %s", out)
    plot_subspace_ablation(df, args.model_name)

    print(f"\n=== Subspace Ablation — {args.model_name} ===")
    print(df[["layer_band", "rank", "n_layers_edited", "targeted_flip_rate",
              "side_effect_score", "behavior_margin_delta"]].to_string(index=False, float_format="%.3f"))
    clean = df[(df["side_effect_score"] < 0.25) & (df["targeted_flip_rate"] > 0)]
    if not clean.empty:
        best = clean.sort_values("targeted_flip_rate", ascending=False).iloc[0]
        print(f"\nBest CLEAN lever: band={best['layer_band']} rank={int(best['rank'])} → "
              f"flip {best['targeted_flip_rate']:.2f} at side-effect {best['side_effect_score']:.2f}")
    else:
        print("\nNo clean lever (flip>0 and side-effect<0.25) found in this sweep.")
    logger.info("=== Step 16 complete. ===")


if __name__ == "__main__":
    main()
