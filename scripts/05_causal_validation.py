#!/usr/bin/env python3
"""Step 05: Causal validation — probe prediction AND real behavior margin under ablation.

For both input formats we mean-ablate the top-k attributed layers (vs random-k) and report:
  - probe prediction delta (regression: predicted behavior_margin; classification: P(syc))
  - real behavior_margin delta (model logprob preference), via hook-based ablation (TL only)
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.behavior_metrics import compute_behavior_margin_with_ablation
from src.model_loader import load_adapter
from src.patching import (
    run_causal_validation,
    run_preference_causal_validation,
    run_sweep_k_validation,
)
from src.probes import load_best_probe
from src.utils import artifacts_dir, load_config, results_dir, safe_model_name, setup_logging
from src.visualization import plot_causal_validation, plot_sweep_k

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Causal validation")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="paired_rows",
                   choices=["paired_rows", "prompt_preferences"])
    p.add_argument("--probe_position", type=str,
                   default=cfg["activations"].get("probe_position", "response_final"))
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--top_k", type=int, default=cfg["validation"]["top_k"])
    p.add_argument("--random_trials", type=int, default=cfg["validation"]["random_trials"])
    p.add_argument("--max_examples", type=int, default=cfg["validation"]["max_examples"])
    p.add_argument("--intervention", type=str, default="activation_patching",
                   choices=["mean_ablation", "activation_patching"],
                   help="activation_patching (default, TL only) patches the final prompt-token "
                        "residual during the real forward pass; mean_ablation is the blunt fallback.")
    p.add_argument("--metrics", nargs="+", default=["behavior_margin", "answer_flip", "accuracy"],
                   help="Which behavioral metrics to report.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for CIs (0=off).")
    p.add_argument("--method", nargs="+",
                   default=["probe_gradient", "random", "logit_gradient"],
                   help="Methods to compare in causal validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def load_top_layers(model_name, tag):
    attr_dir = Path(artifacts_dir("attribution"))
    sn = safe_model_name(model_name)
    for fname in (f"{sn}_{tag}_top_layers.txt", f"{sn}_top_layers.txt"):
        path = attr_dir / fname
        if path.exists():
            with open(path) as f:
                return [int(x) for x in f.read().strip().split(",") if x.strip()]
    raise FileNotFoundError(f"Top layers file not found for {model_name}/{tag}. Run step 04 first.")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    inf = args.input_format
    pp = "prompt_final" if inf == "prompt_preferences" else args.probe_position
    sn = safe_model_name(args.model_name)
    tag = f"{inf}_{pp}" if inf == "prompt_preferences" else pp

    logger.info("=== Step 05: Causal Validation ===")
    logger.info("Model: %s | input_format: %s | probe_position: %s | top_k=%d",
                args.model_name, inf, pp, args.top_k)

    try:
        probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
        top_layers = load_top_layers(args.model_name, tag)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("Top attributed layers: %s", top_layers)

    # Load caches
    hs_dict, labels_dict = {}, {}
    for split in ("train", "val", "test"):
        try:
            data = activations_to_numpy(load_activations(args.model_name, split, probe_position=pp, input_format=inf))
            hs_dict[split] = data["hidden_states"]
            labels_dict[split] = data["labels"]
        except FileNotFoundError:
            logger.warning("Split %s not found.", split)
    if not hs_dict:
        logger.error("No activation caches found. Run step 02 first.")
        sys.exit(1)

    train_hs = hs_dict.get("train", next(iter(hs_dict.values())))
    mean_acts = train_hs.mean(axis=0)
    test_hs = hs_dict.get("test", next(iter(hs_dict.values())))

    # Load logit-gradient top layers (baseline) if available
    from src.logit_gradient import load_logit_gradient_top_layers
    logit_top = load_logit_gradient_top_layers(args.model_name, probe_position=pp, input_format=inf)
    if logit_top:
        logger.info("Logit-gradient top layers: %s", logit_top)

    # --- Probe-prediction validation ---
    if inf == "prompt_preferences":
        probe_results = run_preference_causal_validation(
            test_hs, probe, top_layers, k=args.top_k,
            random_trials=args.random_trials, seed=args.seed, mean_acts=mean_acts,
        )
        # Add a logit_gradient_topk row: ablate the logit-baseline layers, score the probe
        if logit_top:
            from src.patching import ablate_and_score
            ak = min(args.top_k, len(logit_top), test_hs.shape[1])
            b_lg, a_lg = ablate_and_score(test_hs, probe, logit_top[:ak], mean_acts)
            probe_results.append({"method": "logit_gradient", "k": ak,
                                  "before_score": b_lg, "after_score": a_lg, "delta": a_lg - b_lg,
                                  "layers_ablated": str(logit_top[:ak])})
    else:
        probe_results = run_causal_validation(
            hidden_states_dict=hs_dict, labels_dict=labels_dict, probe=probe,
            top_layers=top_layers, k=args.top_k, random_trials=args.random_trials,
            seed=args.seed, max_examples=args.max_examples,
        )

    # --- Real behavior margin validation (needs the model) ---
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    if inf == "prompt_preferences":
        pairs_df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv")
    else:
        test_df = pd.read_csv(processed / "test.csv")
        syc = test_df[test_df["label"] == 1][["prompt", "response"]].rename(columns={"response": "sycophantic_response"})
        non = test_df[test_df["label"] == 0][["prompt", "response"]].rename(columns={"response": "non_sycophantic_response"})
        pairs_df = syc.merge(non, on="prompt", how="inner").drop_duplicates("prompt")

    logger.info("Loading model adapter for behavior margin computation...")
    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)

    actual_k = min(args.top_k, len(top_layers))
    bm_cap = min(args.max_examples, 15)
    rng = np.random.default_rng(args.seed)
    rand_layers = list(rng.choice(test_hs.shape[1], size=actual_k, replace=False))

    from src.model_adapters import TransformerLensAdapter
    use_patching = (args.intervention == "activation_patching" and isinstance(adapter, TransformerLensAdapter))
    logger.info("Behavior-margin intervention: %s", "activation_patching" if use_patching else "mean_ablation")

    # model_family / training_stage from registry
    from src.model_registry import get_model_config
    mcfg = get_model_config(args.model_name)
    model_family = getattr(mcfg, "family", "unknown")
    training_stage = getattr(mcfg, "training_stage", "base")

    from src.patching import patched_margins_for_layers
    from src.metrics import behavioral_intervention_metrics
    from src.statistics import bootstrap_mean_ci

    bm_by_method = {}  # method -> full metric dict

    def _patch_metrics(layers):
        before, after = patched_margins_for_layers(
            adapter, pairs_df, layers, mean_acts, max_examples=bm_cap, seed=args.seed,
            max_length=args.max_length)
        deltas = [a - b for a, b in zip(after, before) if np.isfinite(a) and np.isfinite(b)]
        ci = bootstrap_mean_ci(deltas, n_boot=args.bootstrap, seed=args.seed) if args.bootstrap else \
            {"mean": float(np.mean(deltas)) if deltas else float("nan"), "ci_low": float("nan"),
             "ci_high": float("nan"), "n": len(deltas)}
        bm = behavioral_intervention_metrics(before, after)
        return {"before_behavior_margin": float(np.nanmean(before)) if before else float("nan"),
                "after_behavior_margin": float(np.nanmean(after)) if after else float("nan"),
                "behavior_margin_delta": ci["mean"], "bootstrap_ci_low": ci["ci_low"],
                "bootstrap_ci_high": ci["ci_high"], "n_examples": ci["n"], **bm}

    method_layers = {"probe_gradient": top_layers[:actual_k],
                     "random": rand_layers}
    if logit_top and "logit_gradient" in args.method:
        method_layers["logit_gradient"] = logit_top[:min(args.top_k, len(logit_top))]
    method_layers = {m: L for m, L in method_layers.items() if m in args.method}

    if use_patching:
        for m, L in method_layers.items():
            logger.info("Patching method=%s layers=%s", m, L)
            bm_by_method[m] = _patch_metrics(L)
    else:
        for m, L in method_layers.items():
            res = compute_behavior_margin_with_ablation(
                adapter, pairs_df, L, mean_acts, max_length=args.max_length,
                max_examples=bm_cap, normalize=True)
            bm_by_method[m] = {**res, "bootstrap_ci_low": float("nan"),
                               "bootstrap_ci_high": float("nan"), "n_examples": bm_cap,
                               "answer_flip_rate": float("nan"),
                               "targeted_syc_to_non_syc_flip_rate": float("nan"),
                               "before_accuracy": float("nan"), "after_accuracy": float("nan"),
                               "accuracy_change": float("nan")}

    before_bm = next(iter(bm_by_method.values())).get("before_behavior_margin", float("nan")) if bm_by_method else float("nan")

    # activation-patching-specific CSV
    if use_patching:
        patch_path = Path(results_dir("tables")) / f"{sn}_{tag}_activation_patching_validation.csv"
        pd.DataFrame([{"model_name": args.model_name, "method": m, "k": actual_k, **{
            k: v for k, v in d.items()}} for m, d in bm_by_method.items()]).to_csv(patch_path, index=False)
        logger.info("Saved activation-patching validation -> %s", patch_path)

    # --- Unified table ---
    rows = []
    for pr in probe_results:
        d = bm_by_method.get(pr["method"], {})
        rows.append({
            "model_name": args.model_name, "model_family": model_family,
            "training_stage": training_stage, "input_format": inf, "probe_position": pp,
            "method": pr["method"], "intervention_type": args.intervention, "k": pr["k"],
            "before_probe_prediction": pr["before_score"],
            "after_probe_prediction": pr["after_score"], "probe_delta": pr["delta"],
            "before_behavior_margin": d.get("before_behavior_margin", float("nan")),
            "after_behavior_margin": d.get("after_behavior_margin", float("nan")),
            "behavior_margin_delta": d.get("behavior_margin_delta", float("nan")),
            "answer_flip_rate": d.get("answer_flip_rate", float("nan")),
            "targeted_syc_to_non_syc_flip_rate": d.get("targeted_syc_to_non_syc_flip_rate", float("nan")),
            "before_accuracy": d.get("before_accuracy", float("nan")),
            "after_accuracy": d.get("after_accuracy", float("nan")),
            "accuracy_change": d.get("accuracy_change", float("nan")),
            "bootstrap_ci_low": d.get("bootstrap_ci_low", float("nan")),
            "bootstrap_ci_high": d.get("bootstrap_ci_high", float("nan")),
            "n_examples": d.get("n_examples", 0),
        })
    val_df = pd.DataFrame(rows)

    tables_dir = Path(results_dir("tables"))
    val_path = tables_dir / f"{sn}_{tag}_causal_validation.csv"
    val_df.to_csv(val_path, index=False)
    logger.info("Saved causal validation -> %s", val_path)

    print(f"\n=== Causal Validation: {args.model_name} | {inf} | {pp} | {args.intervention} ===")
    cols = ["method", "k", "probe_delta", "behavior_margin_delta",
            "bootstrap_ci_low", "bootstrap_ci_high", "answer_flip_rate", "accuracy_change"]
    print(val_df[[c for c in cols if c in val_df.columns]].to_string(index=False, float_format="%.4f"))

    plot_causal_validation(probe_results, model_name=f"{args.model_name}_{tag}")

    # Sweep (probe prediction over k) — reuse generic mean-ablation scorer
    from src.patching import ablate_and_score
    rng2 = np.random.default_rng(args.seed)
    n_layers = test_hs.shape[1]
    k_values = list(range(1, min(21, n_layers + 1)))
    sweep_rows = []
    for k in k_values:
        b, a = ablate_and_score(test_hs, probe, top_layers[:k], mean_acts)
        rand_a = [ablate_and_score(test_hs, probe, list(rng2.choice(n_layers, size=k, replace=False)), mean_acts)[1]
                  for _ in range(args.random_trials)]
        sweep_rows.append({"k": k, "before": b, "probe_after": a,
                           "random_after": float(np.mean(rand_a)),
                           "probe_delta": a - b, "random_delta": float(np.mean(rand_a)) - b})
    pd.DataFrame(sweep_rows).to_csv(tables_dir / f"{sn}_{tag}_causal_sweep.csv", index=False)
    plot_sweep_k(sweep_rows, model_name=f"{args.model_name}_{tag}")

    logger.info("=== Step 05 complete. ===")


if __name__ == "__main__":
    main()
