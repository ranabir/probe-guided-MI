"""Presentation-ready plotting for the v3 plots/ gallery.

Each per-model plot function:
  - reads a source CSV from results/tables/, artifacts/, or data/processed/
  - writes a high-DPI PNG to plots/{safe_model_name}/
  - returns a dict describing the plot (for plots/README.md)

All paths are built with pathlib relative to the repo root. No absolute paths are hard-coded.
The existing results/figures/ outputs are left untouched (compatibility layer).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_root, results_dir, safe_model_name

logger = logging.getLogger(__name__)

DPI = 200
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfd",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.autolayout": True,
})

METHOD_COLORS = {
    "probe_gradient": "#2c7fb8",
    "random": "#969696",
    "logit_gradient": "#d95f0e",
}


def plots_dir(model_name: Optional[str] = None) -> Path:
    base = get_root() / "plots"
    p = base / (safe_model_name(model_name) if model_name else "")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(get_root()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Per-model plots
# ---------------------------------------------------------------------------

def plot_probe_regression_by_layer(metrics_csv: Path, model_name: str,
                                   n_prompts: Optional[int] = None) -> Optional[Dict]:
    if not metrics_csv.exists():
        logger.warning("Missing %s; skipping probe regression plot.", metrics_csv)
        return None
    df = pd.read_csv(metrics_csv)
    if "test_pearson" not in df.columns:
        logger.warning("%s has no test_pearson (not a regression run); skipping.", metrics_csv.name)
        return None

    sn = safe_model_name(model_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["test_pearson"], marker="o", linewidth=2.2,
            color="#2c7fb8", label="Test Pearson r")
    if "val_pearson" in df.columns:
        ax.plot(df["layer"], df["val_pearson"], marker="s", linewidth=1.6,
                linestyle="--", color="#7fcdbb", label="Val Pearson r")
    if "test_spearman" in df.columns:
        ax.plot(df["layer"], df["test_spearman"], marker="^", linewidth=1.4,
                linestyle="-.", color="#fdae6b", alpha=0.8, label="Test Spearman")
    best = df.loc[df["test_pearson"].idxmax()]
    ax.axvline(best["layer"], color="red", linestyle=":", alpha=0.6,
               label=f"Best test layer ({int(best['layer'])}, r={best['test_pearson']:.2f})")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    cap = f"{model_name} | target = behavior_margin"
    if n_prompts:
        cap += f" | {n_prompts} prompts"
    ax.set_title("Prompt-final probe predicts sycophancy preference by layer")
    ax.text(0.5, -0.16, cap, transform=ax.transAxes, ha="center", fontsize=10, color="#555")
    ax.legend(loc="best", fontsize=9)
    path = plots_dir(model_name) / f"{sn}_probe_regression_by_layer.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "prompt-preference regression",
            "shows": "per-layer test/val Pearson of probe predicting behavior_margin",
            "read": "higher = preference more linearly decodable at that layer; rising trend = upper layers commit",
            "source": _rel(metrics_csv)}


def plot_behavior_margin_distribution(pref_csv: Path, model_name: str) -> Optional[Dict]:
    if not pref_csv.exists():
        logger.warning("Missing %s; skipping margin distribution plot.", pref_csv)
        return None
    df = pd.read_csv(pref_csv).dropna(subset=["behavior_margin"])
    sn = safe_model_name(model_name)
    frac = float((df["behavior_margin"] > 0).mean())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["behavior_margin"], bins=40, color="#2c7fb8", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.6, label="margin = 0")
    ax.set_xlabel("behavior_margin = mean logP(syc) − mean logP(non-syc)")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Distribution of sycophancy preference margins")
    ax.text(0.98, 0.95, f"{frac:.1%} of {len(df)} prompts prefer sycophancy",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="#fff3cc", ec="#e0c060"))
    ax.legend(loc="upper left")
    path = plots_dir(model_name) / f"{sn}_behavior_margin_distribution.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "prompt-preference dataset",
            "shows": "histogram of per-prompt behavior_margin",
            "read": "mass right of 0 = model leans sycophantic; heavy left = resists. Annotated % prefer syc",
            "source": _rel(pref_csv)}


def plot_layer_attribution(attr_csv: Path, model_name: str, top_k: int = 5) -> Optional[Dict]:
    if not attr_csv.exists():
        logger.warning("Missing %s; skipping attribution plot.", attr_csv)
        return None
    df = pd.read_csv(attr_csv)
    df = df[df["component"] == "hidden_states"] if "component" in df.columns else df
    df = df.sort_values("layer")
    sn = safe_model_name(model_name)
    scores = df["attribution_score"].values
    layers = df["layer"].values
    top_idx = set(np.argsort(scores)[::-1][:top_k])
    colors = ["#d95f0e" if i in top_idx else "#2c7fb8" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(int(l)) for l in layers], scores, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attribution score  (mean |grad × activation|)")
    ax.set_title("Probe-gradient attribution by layer")
    ax.text(0.5, -0.16, f"{model_name} | top-{top_k} layers highlighted (orange)",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555")
    path = plots_dir(model_name) / f"{sn}_probe_gradient_layer_attribution.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "probe-gradient attribution",
            "shows": "attribution score per layer, top-k highlighted",
            "read": "taller bar = layer contributes more to the probe's predicted preference",
            "source": _rel(attr_csv)}


def _causal_bars(causal_csv: Path, model_name: str, value_col: str, title: str,
                 ylabel: str, fname: str, experiment: str, read: str) -> Optional[Dict]:
    if not causal_csv.exists():
        logger.warning("Missing %s; skipping %s.", causal_csv, fname)
        return None
    df = pd.read_csv(causal_csv)
    if value_col not in df.columns:
        logger.warning("%s missing column %s; skipping.", causal_csv.name, value_col)
        return None
    sn = safe_model_name(model_name)
    order = [m for m in ("probe_gradient", "random", "logit_gradient") if m in set(df["method"])]
    vals = [float(df[df["method"] == m][value_col].iloc[0]) for m in order]
    colors = [METHOD_COLORS.get(m, "#888") for m in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([m.replace("_", "\n") for m in order], vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.5, -0.16, f"{model_name}", transform=ax.transAxes, ha="center", fontsize=10, color="#555")
    path = plots_dir(model_name) / fname
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": experiment,
            "shows": f"{value_col} by intervention method", "read": read, "source": _rel(causal_csv)}


def plot_causal_probe_delta(causal_csv: Path, model_name: str) -> Optional[Dict]:
    return _causal_bars(causal_csv, model_name, "probe_delta",
                        "Causal validation on probe prediction",
                        "Δ predicted behavior_margin (after − before)",
                        f"{safe_model_name(model_name)}_causal_probe_delta.png",
                        "causal validation (probe prediction)",
                        "more negative for probe_gradient than random = attributed layers matter to the probe")


def plot_causal_behavior_margin_delta(causal_csv: Path, model_name: str) -> Optional[Dict]:
    return _causal_bars(causal_csv, model_name, "behavior_margin_delta",
                        "Causal validation on real behavior preference",
                        "Δ real behavior_margin (after − before)",
                        f"{safe_model_name(model_name)}_causal_behavior_margin_delta.png",
                        "causal validation (real behavior)",
                        "probe_gradient separating from random = attributed layers causally shift behavior")


def plot_topk_sweep(sweep_csv: Path, model_name: str) -> Optional[Dict]:
    if not sweep_csv.exists():
        logger.warning("Missing %s; skipping sweep plot.", sweep_csv)
        return None
    df = pd.read_csv(sweep_csv)
    sn = safe_model_name(model_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    # sweep CSV stores probe prediction after ablation; plot deltas vs k
    if "probe_delta" in df.columns and "random_delta" in df.columns:
        ax.plot(df["k"], df["probe_delta"], marker="o", color=METHOD_COLORS["probe_gradient"],
                linewidth=2, label="probe-gradient top-k")
        ax.plot(df["k"], df["random_delta"], marker="s", color=METHOD_COLORS["random"],
                linewidth=2, linestyle="--", label="random-k")
        ax.set_ylabel("Δ probe prediction (after − before)")
    else:
        ax.plot(df["k"], df["probe_after"], marker="o", label="probe-gradient")
        ax.plot(df["k"], df["random_after"], marker="s", linestyle="--", label="random")
        ax.set_ylabel("probe prediction after ablation")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("k (number of layers intervened)")
    ax.set_title("Top-k intervention sweep")
    ax.text(0.5, -0.16, f"{model_name}", transform=ax.transAxes, ha="center", fontsize=10, color="#555")
    ax.legend(loc="best")
    path = plots_dir(model_name) / f"{sn}_topk_sweep_behavior_margin.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "top-k sweep",
            "shows": "effect size vs number of layers intervened",
            "read": "probe-gradient curve below random = targeted layers do more per-k",
            "source": _rel(sweep_csv)}


# ---------------------------------------------------------------------------
# Causal-control iteration plots
# ---------------------------------------------------------------------------

def _ci_err(df, col_lo="ci_low", col_hi="ci_high", center="behavior_margin_delta"):
    if col_lo in df.columns and col_hi in df.columns and df[col_lo].notna().any():
        lo = (df[center] - df[col_lo]).abs().values
        hi = (df[col_hi] - df[center]).abs().values
        return np.vstack([lo, hi])
    return None


def plot_contrastive_causal_results(df, model_name: str) -> list:
    """Two bars: targeted answer-flip and behavior_margin_delta, grouped by selection×intervention."""
    sn = safe_model_name(model_name)
    d = df.copy()
    d["cond"] = d["layer_selection"] + "\n" + d["intervention"] + \
        d["setting"].map(lambda s: f"\n{s}" if s not in ("mean", "contrastive") else "")
    # keep one row per cond (best setting per intervention already expanded); collapse steering to best |delta|
    d = d.sort_values("behavior_margin_delta", key=lambda s: s.abs(), ascending=False)
    d = d.drop_duplicates(subset=["layer_selection", "intervention"], keep="first")
    sel_color = {"causal_topk": "#d95f0e", "decodable_topk": "#2c7fb8", "random": "#969696"}
    entries = []

    # 1. targeted syc->honest flip rate
    fig, ax = plt.subplots(figsize=(max(8, len(d) * 1.1), 5.5))
    colors = [sel_color.get(s, "#888") for s in d["layer_selection"]]
    bars = ax.bar(d["cond"], d["targeted_syc_to_non_syc_flip_rate"], color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylabel("targeted syc→honest flip rate")
    ax.set_title("Answer-level control: targeted flip by layer-selection × intervention")
    ax.text(0.5, -0.22, f"{model_name} | orange=causal_topk, blue=decodable_topk, grey=random",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    p1 = plots_dir(model_name) / f"{sn}_contrastive_causal_answer_flip.png"
    _save(fig, p1)
    entries.append({"file": _rel(p1), "model": model_name, "experiment": "causal control",
                    "shows": "targeted syc→honest answer-flip by layer selection and intervention",
                    "read": "taller = more answers flipped to honest; does causal_topk (orange) beat decodable/random?",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_contrastive_causal_results.csv")})

    # 2. behavior_margin_delta with CI
    fig, ax = plt.subplots(figsize=(max(8, len(d) * 1.1), 5.5))
    err = _ci_err(d)
    bars = ax.bar(d["cond"], d["behavior_margin_delta"], color=colors,
                  yerr=err, capsize=3, ecolor="#333")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ behavior_margin (neg = less sycophantic)")
    ax.set_title("Behavior-margin change by layer-selection × intervention")
    ax.text(0.5, -0.22, f"{model_name} | negative is the goal; error bars = 95% bootstrap CI",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    p2 = plots_dir(model_name) / f"{sn}_contrastive_causal_behavior_delta.png"
    _save(fig, p2)
    entries.append({"file": _rel(p2), "model": model_name, "experiment": "causal control",
                    "shows": "behavior_margin change per condition with 95% CI",
                    "read": "negative bars clear of 0 = real de-sycophantizing effect",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_contrastive_causal_results.csv")})
    return entries


def plot_causal_vs_decodable_intervention(sweep_df, results_df, model_name: str) -> dict:
    """Decodability + causal effect by layer, marking which layers each selection targeted."""
    sn = safe_model_name(model_name)
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(sweep_df["layer"], sweep_df["test_pearson"], marker="o", color="#2c7fb8",
             linewidth=2, label="decodability (Pearson)")
    ax1.set_xlabel("Layer"); ax1.set_ylabel("decodability (Pearson)", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(sweep_df["layer"], sweep_df["behavior_margin_delta"].abs(), marker="s",
             color="#d95f0e", linewidth=2, linestyle="--", label="|causal effect|")
    ax2.set_ylabel("|causal effect| (|Δmargin|)", color="#d95f0e")
    ax2.tick_params(axis="y", labelcolor="#d95f0e")

    # mark selected layers
    def _mark(selection, color, marker):
        sub = results_df[results_df["layer_selection"] == selection]
        if sub.empty:
            return
        import ast
        layers = ast.literal_eval(sub["layers"].iloc[0])
        for L in layers:
            ax1.axvline(L, color=color, alpha=0.25, linewidth=6, marker=marker)
    _mark("causal_topk", "#d95f0e", "v")
    _mark("decodable_topk", "#2c7fb8", "^")

    ax1.set_title("Causal vs decodable layers (shaded = selected targets)")
    ax1.text(0.5, -0.17, f"{model_name} | orange bands = causal_topk, blue bands = decodable_topk",
             transform=ax1.transAxes, ha="center", fontsize=9, color="#555")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
    path = plots_dir(model_name) / f"{sn}_causal_vs_decodable_layer_intervention.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "causal control",
            "shows": "decodability and causal effect by layer, with selected intervention layers shaded",
            "read": "if orange (causal) and blue (decodable) bands sit at different layers, the two diverge",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv")}


def plot_intervention_search_pareto(ranked, model_name: str) -> dict:
    """Scatter: side_effect_score (x) vs targeted_flip_rate (y); Pareto front highlighted."""
    sn = safe_model_name(model_name)
    df = ranked.dropna(subset=["side_effect_score", "targeted_flip_rate"]).copy()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sel_color = {"causal_topk": "#d95f0e", "decodable_topk": "#2c7fb8"}
    for sel, g in df.groupby("layer_selection_type"):
        ax.scatter(g["side_effect_score"], g["targeted_flip_rate"], s=60, alpha=0.75,
                   color=sel_color.get(sel, "#888"), label=sel, edgecolor="white")
    # Pareto front: max flip for min side-effect
    pareto = []
    for _, r in df.sort_values("side_effect_score").iterrows():
        if not pareto or r["targeted_flip_rate"] > pareto[-1][1]:
            pareto.append((r["side_effect_score"], r["targeted_flip_rate"]))
    if pareto:
        px, py = zip(*pareto)
        ax.plot(px, py, color="#333", linestyle="--", linewidth=1.5, label="Pareto front", zorder=1)
    ax.set_xlabel("side-effect score (lower = better)")
    ax.set_ylabel("targeted syc→honest flip rate (higher = better)")
    ax.set_title("Intervention search: control vs side-effect trade-off")
    ax.text(0.5, -0.13, f"{model_name} | top-left = strong control with little harm (the goal)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    path = plots_dir(model_name) / f"{sn}_intervention_search_pareto.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "intervention search",
            "shows": "control vs side-effect trade-off across intervention configs",
            "read": "top-left points dominate: high flip, low harm; flat/low cloud = no good lever",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_causal_intervention_search.csv")}


def plot_best_interventions_ranked(ranked, model_name: str, top_n: int = 10) -> dict:
    """Horizontal bar chart of top-N configs by objective_score."""
    sn = safe_model_name(model_name)
    df = ranked.head(top_n).iloc[::-1]
    labels = (df["layer_selection_type"] + " | " + df["direction_type"] + " | "
              + df["intervention_type"] + " | " + df["alpha_or_cap"].astype(str))
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.5)))
    colors = ["#d95f0e" if s == "causal_topk" else "#2c7fb8" for s in df["layer_selection_type"]]
    bars = ax.barh(labels, df["objective_score"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("objective = targeted_flip − λ·side_effect")
    ax.set_title(f"Top {top_n} interventions by objective — {model_name}")
    path = plots_dir(model_name) / f"{sn}_best_interventions_ranked.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "intervention search",
            "shows": f"top {top_n} intervention configs by objective score",
            "read": "longer bar = better control-vs-harm trade-off; orange=causal layers",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_causal_intervention_search.csv")}


def plot_clean_control(df, model_name: str) -> list:
    """Clean-control comparison: (1) flip vs side-effect scatter by family; (2) grouped bars."""
    sn = safe_model_name(model_name)
    fam_color = {"additive": "#969696", "additive_normpres": "#9E7BB5",
                 "projection_ablation": "#d95f0e", "mean_shift": "#31a354"}
    entries = []

    # 1. Pareto scatter: side-effect (x) vs targeted flip (y)
    fig, ax = plt.subplots(figsize=(8.6, 6))
    for fam, g in df.groupby("family"):
        ax.scatter(g["side_effect_score"], g["targeted_flip_rate"], s=80, alpha=0.85,
                   color=fam_color.get(fam, "#888"), edgecolor="white", label=fam, zorder=3)
    ax.axvspan(-0.01, 0.25, color="#e8f0e6", alpha=0.6, zorder=0)
    ax.text(0.12, ax.get_ylim()[1] * 0.02, "clean zone\n(low side-effect)", fontsize=9, color="#4d6b48")
    ax.set_xlabel("side-effect score  (lower = model stays intact)")
    ax.set_ylabel("targeted syc→honest flip rate  (higher = more control)")
    ax.set_title("Clean causal control: flip vs. side-effect")
    ax.text(0.5, -0.13, f"{model_name} | top-LEFT is the goal: high control inside the clean zone",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9, loc="upper right")
    p1 = plots_dir(model_name) / f"{sn}_clean_control_pareto.png"
    _save(fig, p1)
    entries.append({"file": _rel(p1), "model": model_name, "experiment": "clean causal control",
                    "shows": "answer-flip vs side-effect for additive vs projection-ablation vs mean-shift",
                    "read": "a point in the shaded clean zone with high flip = capability-preserving control",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_clean_causal_control.csv")})

    # 2. Grouped bars: best flip + its side-effect per family
    best = df.sort_values("targeted_flip_rate", ascending=False).drop_duplicates("family")
    best = best.set_index("family").reindex([f for f in fam_color if f in best.index]).reset_index()
    x = np.arange(len(best)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 5))
    b1 = ax.bar(x - w/2, best["targeted_flip_rate"], w, label="targeted flip (↑ good)", color="#2c7fb8")
    b2 = ax.bar(x + w/2, best["side_effect_score"], w, label="side-effect (↓ good)", color="#d95f0e")
    ax.bar_label(b1, fmt="%.2f", fontsize=9); ax.bar_label(b2, fmt="%.2f", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(best["family"], fontsize=9, rotation=12)
    ax.set_ylabel("rate / score"); ax.set_title("Best flip vs its side-effect, by intervention family")
    ax.text(0.5, -0.18, f"{model_name} | the winning family has a tall blue bar and a short orange bar",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    p2 = plots_dir(model_name) / f"{sn}_clean_control_flip_vs_sideeffect.png"
    _save(fig, p2)
    entries.append({"file": _rel(p2), "model": model_name, "experiment": "clean causal control",
                    "shows": "each family's best flip and the side-effect it incurs",
                    "read": "tall blue + short orange = clean control; tall both = disruptive control",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_clean_causal_control.csv")})
    return entries


def plot_subspace_ablation(df, model_name: str) -> list:
    """(1) flip vs side-effect scatter sized by rank; (2) flip & side-effect vs rank per band."""
    sn = safe_model_name(model_name)
    entries = []
    bands = list(dict.fromkeys(df["layer_band"]))
    band_color = {b: c for b, c in zip(bands, ["#d95f0e", "#2c7fb8", "#31a354", "#9E7BB5", "#969696"])}

    # 1. Pareto scatter (marker size ∝ rank)
    fig, ax = plt.subplots(figsize=(8.6, 6))
    for b, g in df.groupby("layer_band"):
        ax.scatter(g["side_effect_score"], g["targeted_flip_rate"],
                   s=30 + 22 * g["rank"], alpha=0.8, color=band_color.get(b, "#888"),
                   edgecolor="white", label=f"band={b}", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"k{int(r['rank'])}", (r["side_effect_score"], r["targeted_flip_rate"]),
                    fontsize=7, ha="center", va="center", color="#333")
    ax.axvspan(-0.01, 0.25, color="#e8f0e6", alpha=0.6, zorder=0)
    ax.text(0.12, ax.get_ylim()[1] * 0.02, "clean zone", fontsize=9, color="#4d6b48")
    ax.set_xlabel("side-effect score  (lower = model stays intact)")
    ax.set_ylabel("targeted syc→honest flip rate")
    ax.set_title("Subspace ablation: flip vs. side-effect (marker size ∝ rank)")
    ax.text(0.5, -0.13, f"{model_name} | a point in the clean zone with high flip = clean AND strong control",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9, loc="upper right")
    p1 = plots_dir(model_name) / f"{sn}_subspace_ablation_pareto.png"
    _save(fig, p1)
    entries.append({"file": _rel(p1), "model": model_name, "experiment": "subspace ablation",
                    "shows": "flip vs side-effect across subspace rank and layer band",
                    "read": "does a higher-rank subspace reach the clean zone with a high flip?",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_subspace_ablation.csv")})

    # 2. flip & side-effect vs rank (one panel per band would be busy; overlay flip solid, se dashed)
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for b, g in df.groupby("layer_band"):
        g = g.sort_values("rank")
        c = band_color.get(b, "#888")
        ax.plot(g["rank"], g["targeted_flip_rate"], marker="o", color=c, lw=2, label=f"{b}: flip")
        ax.plot(g["rank"], g["side_effect_score"], marker="s", color=c, lw=1.4, ls="--", alpha=0.7,
                label=f"{b}: side-effect")
    ax.set_xlabel("subspace rank k")
    ax.set_ylabel("rate / score")
    ax.set_title("Effect of subspace rank: flip (solid) vs side-effect (dashed)")
    ax.text(0.5, -0.15, f"{model_name} | ideal: solid line rises, dashed line stays low",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=8, ncol=2)
    p2 = plots_dir(model_name) / f"{sn}_subspace_ablation_by_rank.png"
    _save(fig, p2)
    entries.append({"file": _rel(p2), "model": model_name, "experiment": "subspace ablation",
                    "shows": "how flip and side-effect change with subspace rank, per layer band",
                    "read": "flip rising with rank while side-effect stays low = sycophancy is a subspace we can cleanly remove",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_subspace_ablation.csv")})
    return entries


def plot_subspace_interpretation(cap_df, cos_df, singular_values, model_name: str, layer: int) -> list:
    """(1) captured-energy of each sub-type direction vs rank; (2) singular-value spectrum."""
    sn = safe_model_name(model_name)
    entries = []
    sub_colors = ["#d95f0e", "#2c7fb8", "#31a354", "#9E7BB5", "#e6ab02"]

    # 1. captured energy vs rank, one line per sub-type
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, (name, g) in enumerate(cap_df.groupby("subtype")):
        g = g.sort_values("rank")
        short = name.replace("sycophancy_on_", "")
        ax.plot(g["rank"], g["captured_energy"], marker="o", lw=2.2,
                color=sub_colors[i % len(sub_colors)], label=short)
    ax.axhline(1.0, color="#888", ls=":", lw=1)
    ax.set_xlabel("subspace rank k")
    ax.set_ylabel("captured energy of sub-type direction")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Does the subspace absorb each sycophancy sub-type?  (layer {layer})")
    ax.text(0.5, -0.15, f"{model_name} | each line rising to ~1 as k grows = that sub-type gets its own dimension",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9, title="sub-type")
    p1 = plots_dir(model_name) / f"{sn}_subspace_subtype_capture.png"
    _save(fig, p1)
    entries.append({"file": _rel(p1), "model": model_name, "experiment": "subspace interpretation",
                    "shows": "fraction of each sub-type's own direction captured by the rank-k subspace",
                    "read": "sub-types captured only as rank grows = sycophancy is a union of per-topic directions",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_subspace_interpretation.csv")})

    # 2. singular-value spectrum
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sv = list(singular_values)
    bars = ax.bar([str(i + 1) for i in range(len(sv))], sv, color="#2c7fb8")
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    cum = np.cumsum(sv)
    ax.plot(range(len(sv)), cum, marker="o", color="#d95f0e", lw=1.8, label="cumulative")
    ax.axhline(0.9, color="#888", ls=":", lw=1)
    ax.set_xlabel("subspace dimension")
    ax.set_ylabel("relative energy")
    ax.set_title(f"Sycophancy subspace spectrum (layer {layer})")
    ax.text(0.5, -0.17, f"{model_name} | energy spread across several dims = genuinely multi-directional",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    p2 = plots_dir(model_name) / f"{sn}_subspace_singular_spectrum.png"
    _save(fig, p2)
    entries.append({"file": _rel(p2), "model": model_name, "experiment": "subspace interpretation",
                    "shows": "relative energy per subspace dimension (SVD spectrum) + cumulative",
                    "read": "a slow-decaying spectrum = multi-dimensional sycophancy; a spike at dim1 = single direction",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_subspace_interpretation.csv")})
    return entries


def plot_side_effect_summary(metrics: dict, model_name: str) -> dict:
    """Bar chart of side-effect components for the best intervention."""
    sn = safe_model_name(model_name)
    keys = [("side_effect_score", "side-effect score"), ("weirdness_rate", "weirdness rate"),
            ("repetition_increase", "repetition increase"), ("qa_accuracy_drop", "QA accuracy drop")]
    vals = [metrics.get(k, float("nan")) for k, _ in keys]
    labels = [lab for _, lab in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d95f0e" if metrics.get("side_effect_score", 0) > 0.3 else "#31a354"] + ["#2c7fb8"] * 3
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_ylim(0, max(1.0, max([v for v in vals if np.isfinite(v)] + [0.1]) * 1.2))
    ax.set_ylabel("score (0 = clean / no harm)")
    olr = metrics.get("output_length_ratio", float("nan"))
    ax.set_title(f"Side effects of best intervention — {model_name}")
    ax.text(0.5, -0.14, f"output_length_ratio={olr:.2f} (1.0=unchanged) | low bars = capability preserved",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    path = plots_dir(model_name) / f"{sn}_side_effect_summary.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "side effects",
            "shows": "weirdness / repetition / QA-drop and overall side-effect score after intervention",
            "read": "low bars = intervention preserves general capability; high = model disrupted",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_side_effect_eval.csv")}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def plot_probe_vs_controls(control_df, real_df, model_name: str) -> dict:
    """Decodability-by-layer: real sycophancy probe vs random-label / static / topic controls."""
    sn = safe_model_name(model_name)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if real_df is not None and "test_pearson" in real_df.columns:
        ax.plot(real_df["layer"], real_df["test_pearson"], marker="o", linewidth=2.6,
                color="#2c7fb8", label="sycophancy probe (Pearson)", zorder=5)

    # random-label control
    rl = control_df[control_df["control_name"] == "random_label"]
    if not rl.empty:
        ax.plot(rl["layer"], rl["test_metric"], marker="x", linewidth=1.8, linestyle="--",
                color="#969696", label="random-label control")

    # static features: plot the strongest one prominently, others faint
    stat = control_df[control_df["control_name"] == "static_token"]
    if not stat.empty:
        peak_by_feat = stat.groupby("feature")["test_metric"].max().sort_values(ascending=False)
        for i, feat in enumerate(peak_by_feat.index):
            g = stat[stat["feature"] == feat]
            strongest = (i == 0)
            ax.plot(g["layer"], g["test_metric"], marker="." if not strongest else "s",
                    linewidth=2.0 if strongest else 0.8,
                    color="#d95f0e" if strongest else "#fdd0a2",
                    alpha=1.0 if strongest else 0.6,
                    label=f"static: {feat}" + (" (strongest)" if strongest else "") if strongest or i < 1 else None)

    # topic control (accuracy, different scale) — plot faint on same axis with note
    top = control_df[control_df["control_name"] == "topic"]
    if not top.empty:
        ax.plot(top["layer"], top["test_metric"], marker="^", linewidth=1.6, linestyle="-.",
                color="#31a354", alpha=0.8, label="topic control (accuracy)")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test decodability (Pearson; topic = accuracy)")
    ax.set_title("Sycophancy probe vs controls, by layer")
    ax.text(0.5, -0.17, f"{model_name} | random-label should sit near 0; if a surface feature "
            f"rivals the probe, suspect a confound", transform=ax.transAxes, ha="center",
            fontsize=9, color="#555")
    ax.legend(loc="best", fontsize=8, ncol=2)
    path = plots_dir(model_name) / f"{sn}_probe_vs_controls_by_layer.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "confound controls",
            "shows": "sycophancy probe decodability vs random-label / static-feature / topic controls",
            "read": "random-label ~0 = signal is real; a surface feature matching the probe = confound risk",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_control_probe_metrics.csv")}


def plot_decodability_vs_causal(df, model_name: str) -> dict:
    """Dual-axis: decodability (Pearson) and causal effect (|Δbehavior_margin|) per layer."""
    sn = safe_model_name(model_name)
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(df["layer"], df["test_pearson"], marker="o", color="#2c7fb8", linewidth=2.2,
             label="decodability (test Pearson)")
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Decodability (test Pearson)", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.axhline(0, color="gray", linewidth=0.6)

    ax2 = ax1.twinx()
    ax2.plot(df["layer"], df["behavior_margin_delta"].abs(), marker="s", color="#d95f0e",
             linewidth=2.2, linestyle="--", label="|causal Δbehavior_margin|")
    if "bootstrap_ci_low" in df.columns:
        lo = (df["behavior_margin_delta"] - df["bootstrap_ci_low"]).abs()
        hi = (df["bootstrap_ci_high"] - df["behavior_margin_delta"]).abs()
        ax2.errorbar(df["layer"], df["behavior_margin_delta"].abs(), yerr=[lo, hi],
                     fmt="none", ecolor="#d95f0e", alpha=0.4, capsize=2)
    ax2.set_ylabel("|causal effect| (Δbehavior_margin)", color="#d95f0e")
    ax2.tick_params(axis="y", labelcolor="#d95f0e")

    ax1.set_title("Decodability vs causal effect, by layer")
    ax1.text(0.5, -0.17, f"{model_name} | if the two curves don't track, decoding ≠ causation",
             transform=ax1.transAxes, ha="center", fontsize=9, color="#555")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
    path = plots_dir(model_name) / f"{sn}_decodability_vs_causal_effect.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "decodability vs causation",
            "shows": "per-layer decodability and causal effect on two axes",
            "read": "curves tracking = decoding predicts causation; diverging = they're different layers",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv")}


def plot_layerwise_behavior_delta(df, model_name: str) -> dict:
    """Per-layer causal behavior_margin_delta with bootstrap CI band."""
    sn = safe_model_name(model_name)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(df["layer"], df["behavior_margin_delta"], marker="o", color="#d95f0e", linewidth=2)
    if "bootstrap_ci_low" in df.columns:
        ax.fill_between(df["layer"], df["bootstrap_ci_low"], df["bootstrap_ci_high"],
                        color="#d95f0e", alpha=0.2, label="95% CI")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer (patched individually)")
    ax.set_ylabel("Δ behavior_margin (after − before)")
    ax.set_title("Causal effect of patching each layer")
    ax.text(0.5, -0.16, f"{model_name} | CI crossing 0 = effect indistinguishable from none",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    path = plots_dir(model_name) / f"{sn}_layerwise_behavior_delta.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "layerwise causal effect",
            "shows": "behavior_margin change when each layer is patched, with 95% CI",
            "read": "bars/points with CI clear of 0 = a genuinely causal layer",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv")}


def plot_layerwise_answer_flip(df, model_name: str) -> dict:
    """Per-layer A/B answer-flip rate."""
    sn = safe_model_name(model_name)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(df["layer"].astype(int).astype(str), df["answer_flip_rate"], color="#756bb1")
    ax.set_xlabel("Layer (patched individually)")
    ax.set_ylabel("Answer-flip rate")
    ax.set_ylim(0, max(0.05, float(df["answer_flip_rate"].max() if df["answer_flip_rate"].notna().any() else 0.05) * 1.2))
    ax.set_title("A/B answer-flip rate by patched layer")
    ax.text(0.5, -0.16, f"{model_name} | fraction of prompts whose preferred A/B answer changed",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    path = plots_dir(model_name) / f"{sn}_layerwise_answer_flip_rate.png"
    _save(fig, path)
    return {"file": _rel(path), "model": model_name, "experiment": "layerwise answer-flip",
            "shows": "share of prompts that changed preferred answer when each layer is patched",
            "read": "taller bar = patching that layer more often flips the model's A/B choice",
            "source": _rel(Path(results_dir("tables")) / f"{sn}_layerwise_decodability_causal_sweep.csv")}


def plot_direction_comparison(df, model_name: str) -> list:
    """Two plots: behavior_margin_delta vs alpha, and answer_flip_rate vs alpha, per direction."""
    sn = safe_model_name(model_name)
    if "alpha" not in df.columns:
        return []
    dir_colors = {"regression": "#2c7fb8", "logistic": "#756bb1", "diff_of_means": "#d95f0e",
                  "margin_weighted": "#31a354", "random": "#969696"}
    entries = []

    # 1. behavior_margin_delta vs alpha
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d, g in df.groupby("direction"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["behavior_margin_delta"], marker="o", linewidth=2,
                color=dir_colors.get(d, None), label=d)
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Steering strength α"); ax.set_ylabel("Δ behavior_margin vs α=0")
    ax.set_title("Steering: behavior-margin change by direction")
    ax.text(0.5, -0.16, f"{model_name} | steeper slope = stronger causal control", transform=ax.transAxes,
            ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    p1 = plots_dir(model_name) / f"{sn}_direction_comparison_behavior_delta.png"
    _save(fig, p1)
    entries.append({"file": _rel(p1), "model": model_name, "experiment": "direction steering",
                    "shows": "behavior_margin change vs steering strength, per direction method",
                    "read": "steeper line = that direction controls behavior more; compare diff_of_means to regression",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_direction_comparison.csv")})

    # 2. answer_flip_rate vs alpha
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d, g in df.groupby("direction"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["answer_flip_rate"], marker="s", linewidth=2,
                color=dir_colors.get(d, None), label=d)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Steering strength α"); ax.set_ylabel("Answer-flip rate vs α=0")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Steering: A/B answer-flip rate by direction")
    ax.text(0.5, -0.16, f"{model_name} | higher = more prompts changed their preferred answer",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9)
    p2 = plots_dir(model_name) / f"{sn}_direction_comparison_answer_flip.png"
    _save(fig, p2)
    entries.append({"file": _rel(p2), "model": model_name, "experiment": "direction steering",
                    "shows": "answer-flip rate vs steering strength, per direction",
                    "read": "higher flip rate at moderate α = more effective, interpretable control",
                    "source": _rel(Path(results_dir("tables")) / f"{sn}_direction_comparison.csv")})
    return entries


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_stage_comparison(df) -> list:
    """Three comparison plots across model stages/families: sycophancy rate, decodability, causal."""
    comp = get_root() / "plots" / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["label"] = df["model_name"].str.replace("EleutherAI/", "").str.replace("Qwen/", "") \
        + "\n(" + df.get("training_stage", "base").astype(str) + ")"
    entries = []

    def _bar(col, title, ylabel, fname, note):
        if col not in df.columns or df[col].isna().all():
            return None
        fig, ax = plt.subplots(figsize=(max(7, len(df) * 1.4), 5))
        colors = ["#2c7fb8" if s == "base" else "#d95f0e"
                  for s in df.get("training_stage", ["base"] * len(df))]
        bars = ax.bar(df["label"], df[col], color=colors)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", fontsize=9, color="#555")
        path = comp / fname
        _save(fig, path)
        return {"file": _rel(path), "model": "comparison", "experiment": "stage comparison",
                "shows": title.lower(), "read": note, "source": _rel(Path(results_dir("tables")) / "stage_comparison_summary.csv")}

    e = _bar("sycophancy_rate", "Sycophancy rate by model / stage",
             "fraction of prompts preferring sycophancy", "stage_sycophancy_rate.png",
             "base=blue, post-trained=orange; does sycophancy rise after post-training?")
    if e: entries.append(e)
    e = _bar("best_probe_pearson", "Probe decodability by model / stage",
             "best test Pearson", "stage_probe_decodability.png",
             "is sycophancy more decodable in instruct models?")
    if e: entries.append(e)
    e = _bar("probe_gradient_behavior_delta", "Causal control by model / stage",
             "probe-gradient Δ behavior_margin", "stage_causal_control.png",
             "is probe-guided causal control stronger in instruct models?")
    if e: entries.append(e)
    return entries


def plot_model_probe_comparison(model_metrics: Dict[str, Path], out_dir: Path) -> Optional[Dict]:
    series = []
    for model_name, csv in model_metrics.items():
        if csv and csv.exists():
            df = pd.read_csv(csv)
            if "test_pearson" in df.columns and len(df) > 1:
                depth = df["layer"] / (df["layer"].max() or 1)
                series.append((model_name, depth, df["test_pearson"]))
    if not series:
        logger.warning("No regression metrics for comparison; skipping.")
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, depth, pear in series:
        ax.plot(depth, pear, marker="o", linewidth=2, label=model_name)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Relative layer depth (0 = first, 1 = last)")
    ax.set_ylabel("Test Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Probe regression: GPT-2 vs Pythia (depth-normalized)")
    ax.legend(loc="best")
    path = out_dir / "model_probe_regression_comparison.png"
    _save(fig, path)
    return {"file": _rel(path), "model": "comparison", "experiment": "cross-model probe regression",
            "shows": "test Pearson vs normalized depth for each model",
            "read": "curves rising toward deep layers across models = a shared mechanism",
            "source": "; ".join(_rel(c) for c in model_metrics.values() if c and c.exists())}


def plot_model_causal_comparison(model_causal: Dict[str, Path], out_dir: Path) -> Optional[Dict]:
    records = []
    for model_name, csv in model_causal.items():
        if csv and csv.exists():
            df = pd.read_csv(csv)
            for _, r in df.iterrows():
                records.append({"model": model_name, "method": r["method"],
                                "bm_delta": r.get("behavior_margin_delta", np.nan)})
    if not records:
        logger.warning("No causal tables for comparison; skipping.")
        return None
    cdf = pd.DataFrame(records)
    models = list(dict.fromkeys(cdf["model"]))
    methods = [m for m in ("probe_gradient", "random", "logit_gradient") if m in set(cdf["method"])]
    x = np.arange(len(models))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        vals = [cdf[(cdf["model"] == mod) & (cdf["method"] == m)]["bm_delta"].mean() for mod in models]
        ax.bar(x + i * width, vals, width, label=m, color=METHOD_COLORS.get(m, "#888"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Δ behavior_margin (after − before)")
    ax.set_title("Causal effect on behavior by model and method")
    ax.legend(loc="best")
    path = out_dir / "model_causal_behavior_comparison.png"
    _save(fig, path)
    return {"file": _rel(path), "model": "comparison", "experiment": "cross-model causal effect",
            "shows": "behavior_margin_delta grouped by model × method",
            "read": "probe_gradient bar more negative than random within a model = real causal control",
            "source": "; ".join(_rel(c) for c in model_causal.values() if c and c.exists())}


# ---------------------------------------------------------------------------
# plots/README.md
# ---------------------------------------------------------------------------

def update_plots_readme(entries: List[Dict]) -> Path:
    """Write plots/README.md with a table row per plot (idempotent rewrite from `entries`)."""
    root = get_root() / "plots"
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"

    lines = [
        "# Plots Gallery\n",
        "\nPresentation-ready figures for the probe-guided sycophancy attribution project. "
        "Each plot is regenerated by `python scripts/07_generate_plots.py`. "
        "Machine-readable source tables live in `results/tables/` and `artifacts/`.\n",
        "\n| Plot file | Model | Experiment | What it shows | How to read it | Source table |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for e in entries:
        lines.append(
            f"| `{e['file']}` | {e['model']} | {e['experiment']} | {e['shows']} | {e['read']} | `{e['source']}` |\n"
        )
    with open(readme, "w") as f:
        f.writelines(lines)
    logger.info("Wrote %s (%d plots)", readme, len(entries))
    return readme
