#!/usr/bin/env python3
"""Step 06: Generate results/report.md with two sections:

  1. Response-aware sanity check (paired_rows + response_final)
     - verifies the probe infrastructure can detect a VISIBLE sycophantic completion
     - limitation: the response text is in the input, so this is not "about to be sycophantic"
  2. Prompt-only preference attribution (prompt_preferences + prompt_final)  [MAIN EXPERIMENT]
     - target is the model's own behavior_margin (regression)
     - asks whether prompt-final activations predict the model's preference
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import load_metadata
from src.probes import load_selection_metadata
from src.utils import load_config, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Generate markdown report")
    p.add_argument("--model_name", type=str, default=cfg["model"]["model_name"])
    p.add_argument("--input_format", type=str, default="prompt_preferences",
                   choices=["paired_rows", "prompt_preferences"])
    p.add_argument("--probe_position", type=str, default="prompt_final")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _table(tag: str, name: str, sn: str) -> str:
    path = Path(results_dir("tables")) / f"{sn}_{tag}_{name}.csv"
    if path.exists():
        try:
            return pd.read_csv(path).to_markdown(index=False, floatfmt=".4f") + "\n"
        except Exception:
            return f"*(could not render {path.name})*\n"
    return f"*(not found: {sn}_{tag}_{name}.csv — run that pipeline variant)*\n"


def _fig(tag: str, name: str, sn: str) -> str:
    path = Path(results_dir("figures")) / f"{sn}_{tag}_{name}.png"
    return f"![{name}]({path})\n" if path.exists() else f"*(figure not found: {name})*\n"


def _plot_links(sn: str) -> str:
    """Embed the v3 gallery plots for this model if they exist."""
    root = Path(__file__).resolve().parent.parent
    pdir = root / "plots" / sn
    names = [
        ("probe_regression_by_layer", "Probe regression by layer"),
        ("behavior_margin_distribution", "Behavior margin distribution"),
        ("probe_gradient_layer_attribution", "Probe-gradient attribution"),
        ("causal_probe_delta", "Causal: probe prediction delta"),
        ("causal_behavior_margin_delta", "Causal: real behavior margin delta"),
        ("topk_sweep_behavior_margin", "Top-k intervention sweep"),
    ]
    out = []
    for key, label in names:
        p = pdir / f"{sn}_{key}.png"
        if p.exists():
            out.append(f"- **{label}** — `{p.relative_to(root)}`\n")
    return "".join(out) if out else "*(run scripts/07_generate_plots.py to produce plots)*\n"


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    sn = safe_model_name(args.model_name)

    logger.info("=== Step 06: Generate Report ===")

    resp_tag = "response_final"
    pref_tag = "prompt_preferences_prompt_final"

    meta_pref = load_metadata(args.model_name, probe_position="prompt_final", input_format="prompt_preferences")
    meta_resp = load_metadata(args.model_name, probe_position="response_final", input_format="paired_rows")
    meta = meta_pref or meta_resp
    sel_resp = load_selection_metadata(args.model_name, probe_position="response_final", input_format="paired_rows")
    sel_pref = load_selection_metadata(args.model_name, probe_position="prompt_final", input_format="prompt_preferences")

    n_layers = meta.get("n_layers", "?")
    d_model = meta.get("d_model", "?")
    backend = meta.get("backend", "?")
    family = meta.get("family", "?")

    # Prompt-preference dataset stats
    pref_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / f"{sn}_prompt_preferences.csv"
    pref_stats = ""
    if pref_csv.exists():
        pdf = pd.read_csv(pref_csv)
        if "behavior_margin" in pdf.columns:
            valid = pdf.dropna(subset=["behavior_margin"])
            frac = float((valid["behavior_margin"] > 0).mean()) if len(valid) else float("nan")
            pref_stats = (
                f"- Prompts: {len(pdf)}\n"
                f"- behavior_margin: mean={valid['behavior_margin'].mean():.4f}, "
                f"median={valid['behavior_margin'].median():.4f}, std={valid['behavior_margin'].std():.4f}\n"
                f"- prefers_sycophancy (margin>0): {int((valid['behavior_margin']>0).sum())}/{len(valid)} "
                f"(frac={frac:.3f})\n"
            )

    lines = [
        f"# Probe-Guided Attribution for Sycophancy\n",
        f"**Model:** `{args.model_name}`\n\n",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "---\n",

        "## 0. Project Summary\n",
        "We replace single-token logit attribution with a **trained behavioral probe** and "
        "backpropagate through the probe to rank model layers, then validate causally by ablation. "
        "This report contains two experiments: a response-aware **sanity check** and the main "
        "prompt-only **preference attribution**.\n",

        f"\n| Property | Value |\n|---|---|\n"
        f"| Model | `{args.model_name}` |\n| Backend | {backend} |\n| Family | {family} |\n"
        f"| Layers | {n_layers} |\n| d_model | {d_model} |\n",

        "\n---\n",
        "## 1. Response-Aware Sanity Check\n",
        "**Setup:** input = `prompt + response`, target = row label (1=sycophantic completion). "
        "Probe position = `response_final`.\n\n",
        "**Purpose:** verify the probe + attribution + ablation infrastructure works when the "
        "sycophantic signal is *visible* in the input.\n\n",
        "**Limitation:** because the response text is in the input, this measures *recognition of a "
        "visible completion*, not whether the model is *about to be* sycophantic. It is a sanity "
        "check, not the scientific claim.\n\n",
    ]

    if sel_resp:
        lines.append(f"Attribution probe layer: **{sel_resp.get('selected_layer','?')}** "
                     f"({sel_resp.get('metric_col','val_auroc')}={sel_resp.get('selected_metric',float('nan')):.4f})\n\n")
    lines += [
        "### Per-layer probe metrics (classification)\n",
        _table(resp_tag, "layer_probe_metrics", sn),
        "\n", _fig(resp_tag, "layer_probe_accuracy", sn),
        "\n### Causal validation\n",
        _table(resp_tag, "causal_validation", sn),
        "\n", _fig(resp_tag, "causal_validation_barplot", sn),

        "\n---\n",
        "## 2. Prompt-Only Preference Attribution  (MAIN EXPERIMENT)\n",
        "**Setup:** input = `prompt` only, target = the model's own `behavior_margin` = "
        "mean-token logprob(sycophantic | prompt) − mean-token logprob(non_sycophantic | prompt). "
        "Probe position = `prompt_final`, probe type = **regression**.\n\n",
        "**Why this is the correct design:** in paired-row format the same prompt yields two rows "
        "with identical prompt-only activations but opposite labels, making label classification "
        "mathematically invalid. Regressing on the per-prompt `behavior_margin` gives one well-posed "
        "target per prompt and asks the real question: *do prompt-final activations encode whether "
        "the model is about to prefer a sycophantic continuation?*\n\n",
        "### Behavior margin distribution\n",
        (pref_stats or "*(prompt-preference dataset not found — run scripts/01b_build_prompt_preferences.py)*\n"),
    ]

    if sel_pref:
        lines.append(f"\nAttribution probe layer: **{sel_pref.get('selected_layer','?')}** "
                     f"({sel_pref.get('metric_col','val_pearson')}={sel_pref.get('selected_metric',float('nan')):.4f}), "
                     f"type={sel_pref.get('probe_type','regression')}, target={sel_pref.get('probe_target','behavior_margin')}\n")
    lines += [
        "\n### Per-layer probe metrics (regression: Pearson / Spearman / R²)\n",
        _table(pref_tag, "layer_probe_metrics", sn),
        "\n", _fig(pref_tag, "layer_probe_accuracy", sn),
        "\n### Layer attribution\n",
        _fig(pref_tag, "layer_attribution_barplot", sn),
        "\n### Causal validation (probe prediction AND real behavior margin)\n",
        _table(pref_tag, "causal_validation", sn),
        "\n**Columns:** `probe_delta` = change in predicted margin after ablation; "
        "`behavior_margin_delta` = change in the model's *real* logprob preference under hook-based "
        "ablation (TransformerLens only; NaN for HF models).\n",
        "\n", _fig(pref_tag, "causal_sweep", sn),

        "\n---\n",
        "## 2b. Presentation Plots\n",
        f"High-DPI gallery in `plots/{sn}/` (see `plots/README.md` for how to read each). "
        "Regenerate with `python scripts/07_generate_plots.py --model_name {model}`.\n\n".replace("{model}", args.model_name),
        _plot_links(sn),
        "\nCross-model comparison: `plots/comparison/` and `plots/comparison/summary_table.md`.\n",

        "\n---\n",
        "## 3. Limitations\n",
        "- Synthetic / small datasets → high-variance correlation estimates.\n",
        "- Length-normalized logprob reduces but does not eliminate length artifacts.\n",
        "- Layer-level mean ablation is coarse (no head-level resolution yet).\n",
        "- Behavior-margin ablation is implemented for TransformerLens; HF (Qwen/Gemma) is pending.\n",
        "- If regression correlations are weak, that is reported honestly — the main win is a "
        "**structurally valid** experiment, not a forced result.\n",

        "\n## 4. Next Steps\n",
        "- Scale prompts (hundreds+) and run Pythia-410M for stronger correlation estimates.\n",
        "- Head-level attribution via `blocks.{i}.attn.hook_result`.\n",
        "- Activation patching with clean/non-syc runs instead of mean ablation.\n",
        "- HF hook-based behavior-margin ablation for Qwen/Gemma.\n",
    ]

    report_path = Path(results_dir("")) / "report.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    logger.info("Report saved -> %s", report_path)
    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()
