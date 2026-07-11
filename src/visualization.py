"""Generate publication-quality figures for probe accuracy, attribution, and causal validation."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import results_dir, safe_model_name

logger = logging.getLogger(__name__)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def plot_layer_probe_accuracy(
    metrics_df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> Path:
    """Line plot of per-layer probe metrics.

    Auto-detects classification (accuracy/AUROC) vs regression (Pearson/R2) columns.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = metrics_df["layer"]

    is_regression = "val_pearson" in metrics_df.columns
    if is_regression:
        ax.plot(layers, metrics_df["val_pearson"], label="Val Pearson r", linewidth=2)
        if "val_spearman" in metrics_df.columns:
            ax.plot(layers, metrics_df["val_spearman"], label="Val Spearman", linewidth=2, linestyle="-.")
        if "val_r2" in metrics_df.columns:
            ax.plot(layers, metrics_df["val_r2"].clip(lower=-1), label="Val R² (clipped)", linewidth=1.5, linestyle="--", alpha=0.7)
        best_row = metrics_df.loc[metrics_df["val_pearson"].idxmax()]
        ax.set_ylabel("Correlation / R²")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.8)
    else:
        ax.plot(layers, metrics_df["train_accuracy"], label="Train Accuracy", linestyle="--", alpha=0.7)
        ax.plot(layers, metrics_df["val_accuracy"], label="Val Accuracy", linewidth=2)
        ax.plot(layers, metrics_df["val_auroc"], label="Val AUROC", linewidth=2, linestyle="-.")
        best_row = metrics_df.loc[metrics_df["val_auroc"].idxmax()]
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)

    ax.axvline(x=best_row["layer"], color="red", linestyle=":", alpha=0.7,
               label=f"Best layer ({int(best_row['layer'])})")
    ax.set_xlabel("Layer")
    ax.set_title(f"Probe Performance by Layer — {model_name}")
    ax.legend(loc="best")

    path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_layer_probe_accuracy.png"
    if save:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
    return path


def plot_layer_attribution_barplot(
    layer_scores: np.ndarray,
    model_name: str,
    top_k: int = 20,
    save: bool = True,
) -> Path:
    """Bar plot of attribution scores per layer."""
    _apply_style()
    n_layers = len(layer_scores)
    top_k = min(top_k, n_layers)

    top_indices = np.argsort(layer_scores)[::-1][:top_k]
    top_scores = layer_scores[top_indices]

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.5), 5))
    colors = ["#e74c3c" if i == top_indices[0] else "#3498db" for i in top_indices]
    bars = ax.bar([str(i) for i in top_indices], top_scores, color=colors)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Attribution Score")
    ax.set_title(f"Top-{top_k} Layer Attribution — {model_name}")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_layer_attribution_barplot.png"
    if save:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
    return path


def plot_component_attribution_heatmap(
    component_df: pd.DataFrame,
    model_name: str,
    save: bool = True,
) -> Optional[Path]:
    """Heatmap: component type x layer, colored by attribution score."""
    if component_df is None or component_df.empty:
        return None

    _apply_style()
    pivot = component_df.pivot_table(
        index="component", columns="layer", values="attribution_score", aggfunc="mean"
    )
    if pivot.empty:
        return None

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.4), max(4, pivot.shape[0] * 0.8)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        annot=pivot.shape[1] <= 30,
        fmt=".2f",
        linewidths=0.3,
    )
    ax.set_title(f"Component Attribution Heatmap — {model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Component")

    path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_component_attribution_heatmap.png"
    if save:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
    return path


def plot_causal_validation(
    results: List[Dict],
    model_name: str,
    save: bool = True,
) -> Path:
    """Bar chart comparing probe-gradient vs random ablation."""
    _apply_style()
    if not results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No validation results", ha="center")
        path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_causal_validation_barplot.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    df = pd.DataFrame(results)
    methods = df["method"].unique().tolist()
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    before_vals = [df[df["method"] == m]["before_score"].mean() for m in methods]
    after_vals = [df[df["method"] == m]["after_score"].mean() for m in methods]

    b1 = ax.bar(x - width / 2, before_vals, width, label="Before Ablation", color="#2ecc71", alpha=0.8)
    b2 = ax.bar(x + width / 2, after_vals, width, label="After Ablation", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in methods])
    ax.set_ylabel("Mean Probe Score P(sycophantic)")
    ax.set_title(f"Causal Validation: Ablation Effect — {model_name}")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)

    path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_causal_validation_barplot.png"
    if save:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
    return path


def plot_sweep_k(
    sweep_results: List[Dict],
    model_name: str,
    save: bool = True,
) -> Path:
    """Line plot for sweep over k ablated layers."""
    _apply_style()
    df = pd.DataFrame(sweep_results)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["k"], df["probe_after"], label="Probe-gradient top-k", linewidth=2, marker="o")
    ax.plot(df["k"], df["random_after"], label="Random-k (avg)", linewidth=2, linestyle="--", marker="s")
    ax.axhline(y=df["before"].mean(), linestyle=":", color="gray", label="Before ablation")

    ax.set_xlabel("k (layers ablated)")
    ax.set_ylabel("Mean Probe Score P(sycophantic)")
    ax.set_title(f"Causal Validation Sweep — {model_name}")
    ax.legend()
    ax.set_ylim(0, 1.05)

    path = Path(results_dir("figures")) / f"{safe_model_name(model_name)}_causal_sweep.png"
    if save:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)
    return path
