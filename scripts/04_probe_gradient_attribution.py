#!/usr/bin/env python3
"""Step 04: Probe-gradient attribution.

For classification probes the attribution target is P(positive class).
For regression probes (prompt_preferences) the target is the predicted behavior_margin.
In both cases we backprop the probe output through the hidden states and rank layers
by mean |grad x activation|.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import activations_to_numpy, load_activations
from src.attribution import (
    build_attribution_df,
    compute_component_attribution,
    compute_layer_attribution,
    logit_gradient_baseline,
    top_k_layers,
)
from src.model_loader import load_adapter
from src.probes import load_best_probe, load_selection_metadata
from src.utils import artifacts_dir, load_config, safe_model_name, setup_logging
from src.visualization import plot_component_attribution_heatmap, plot_layer_attribution_barplot

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Compute probe-gradient attribution")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="paired_rows",
                   choices=["paired_rows", "prompt_preferences"])
    p.add_argument("--probe_position", type=str,
                   default=cfg["activations"].get("probe_position", "response_final"))
    p.add_argument("--device", type=str, default=cfg["model"]["device"])
    p.add_argument("--dtype", type=str, default=cfg["model"]["dtype"])
    p.add_argument("--max_length", type=int, default=cfg["model"]["max_length"])
    p.add_argument("--batch_size", type=int, default=cfg["model"]["batch_size"])
    p.add_argument("--max_examples", type=int, default=cfg["attribution"]["max_examples"])
    p.add_argument("--method", type=str, default=cfg["attribution"]["method"])
    p.add_argument("--top_k", type=int, default=cfg["attribution"]["top_k"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def compute_grads_batch(adapter, texts, probe, target_layer, max_length, batch_size):
    all_grads = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        logger.info("  Grad batch %d-%d", start, start + len(batch) - 1)
        try:
            g = adapter.compute_probe_gradients(batch, probe, target_layer,
                                                max_length=max_length, token_position="final")
        except Exception as e:
            logger.warning("Gradient computation failed (%s); zeros.", e)
            g = np.zeros((len(batch), adapter.n_layers, adapter.d_model), dtype=np.float32)
        all_grads.append(g)
    return np.concatenate(all_grads, axis=0)


def _load_test_texts(model_name, input_format, probe_position, adapter, max_examples, seed):
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    if input_format == "prompt_preferences":
        sn = safe_model_name(model_name)
        df = pd.read_csv(processed / f"{sn}_prompt_preferences_test.csv")
        if len(df) > max_examples:
            df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
        return [str(r["prompt"]) for _, r in df.iterrows()]
    df = pd.read_csv(processed / "test.csv")
    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    if probe_position == "prompt_final":
        return [str(r["prompt"]) for _, r in df.iterrows()]
    return [adapter.format_prompt_response(str(r["prompt"]), str(r["response"])) for _, r in df.iterrows()]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)

    inf = args.input_format
    pp = "prompt_final" if inf == "prompt_preferences" else args.probe_position
    sn = safe_model_name(args.model_name)
    tag = f"{inf}_{pp}" if inf == "prompt_preferences" else pp

    logger.info("=== Step 04: Probe-Gradient Attribution ===")
    logger.info("Model: %s | input_format: %s | probe_position: %s", args.model_name, inf, pp)

    try:
        probe = load_best_probe(args.model_name, probe_position=pp, input_format=inf)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    sel_meta = load_selection_metadata(args.model_name, probe_position=pp, input_format=inf)
    target_layer = probe.layer_idx
    logger.info("Attribution probe: layer=%d | type=%s | target=%s",
                target_layer, sel_meta.get("probe_type", getattr(probe, "task", "?")),
                sel_meta.get("probe_target", "?"))

    adapter = load_adapter(args.model_name, device_str=args.device, dtype_str=args.dtype)
    all_texts = _load_test_texts(args.model_name, inf, pp, adapter, args.max_examples, args.seed)

    logger.info("Computing gradients for %d examples (target_layer=%d)...", len(all_texts), target_layer)
    grads = compute_grads_batch(adapter, all_texts, probe, target_layer, args.max_length, args.batch_size)

    try:
        test_cache = activations_to_numpy(load_activations(args.model_name, "test",
                                                          probe_position=pp, input_format=inf))
        acts = test_cache["hidden_states"]
        if len(acts) > args.max_examples:
            acts = acts[:args.max_examples]
    except FileNotFoundError:
        with torch.no_grad():
            acts = adapter.forward_with_cache(all_texts, max_length=args.max_length, token_position="final")["hidden_states"]

    # Align lengths
    m = min(len(grads), len(acts))
    grads, acts = grads[:m], acts[:m]

    layer_scores = compute_layer_attribution(grads, acts, method=args.method)
    logger.info("Layer attribution scores: %s", layer_scores)
    top_layers = top_k_layers(layer_scores, k=args.top_k)
    logger.info("Top-%d layers: %s", args.top_k, top_layers)

    # Component attribution
    component_grads, component_acts = {}, {}
    try:
        tc = activations_to_numpy(load_activations(args.model_name, "test", probe_position=pp, input_format=inf))
        for key in ("attn_out", "mlp_out"):
            if key in tc:
                component_acts[key] = tc[key][:m]
                component_grads[key] = grads
    except Exception:
        pass
    component_df = compute_component_attribution(component_grads, component_acts, method=args.method)
    attr_df = build_attribution_df(layer_scores, component_df if not component_df.empty else None, model_name=args.model_name)

    attr_dir = Path(artifacts_dir("attribution"))
    attr_df[attr_df["component"] == "hidden_states"].to_csv(attr_dir / f"{sn}_{tag}_layer_attribution.csv", index=False)
    attr_df.to_csv(attr_dir / f"{sn}_{tag}_component_attribution.csv", index=False)
    with open(attr_dir / f"{sn}_{tag}_top_layers.txt", "w") as f:
        f.write(",".join(str(l) for l in top_layers))
    logger.info("Saved attribution artifacts with prefix %s_%s_*", sn, tag)

    plot_layer_attribution_barplot(layer_scores, model_name=f"{args.model_name}_{tag}", top_k=args.top_k)
    if not component_df.empty:
        plot_component_attribution_heatmap(component_df, model_name=f"{args.model_name}_{tag}")

    print(f"\n=== Top-{args.top_k} Attributed Layers ({args.model_name}, {tag}) ===")
    for rank, li in enumerate(top_layers):
        print(f"  #{rank+1:2d}  Layer {li:3d}  score={layer_scores[li]:.6f}")
    print(f"\nAttribution probe at layer {target_layer} → gradients non-zero for layers 0–{target_layer}")

    logger.info("Computing logit-gradient baseline ...")
    try:
        from src.logit_gradient import compute_logit_gradient_attribution
        lg = compute_logit_gradient_attribution(
            adapter, all_texts[:min(30, len(all_texts))], model_name=args.model_name,
            probe_position=pp, input_format=inf, max_length=args.max_length, top_k=args.top_k,
        )
        print(f"\n=== Logit-Gradient Baseline Top-{args.top_k} Layers ===\n  Layers: {lg['top_layers']}")
        if lg.get("skipped"):
            print(f"  (skipped untokenizable: {lg['skipped']})")
    except Exception as e:
        logger.warning("Logit-gradient baseline failed: %s", e)

    logger.info("=== Step 04 complete. ===")


if __name__ == "__main__":
    main()
