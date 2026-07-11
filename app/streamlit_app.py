"""Streamlit interactive demo for probe-guided sycophancy attribution."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_model_registry, safe_model_name

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Probe-Guided Attribution for Sycophancy",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGISTRY = load_model_registry()
MODEL_NAMES = list(REGISTRY.keys())

RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"


def safe(name):
    return safe_model_name(name)


def artifacts_exist(model_name: str) -> dict:
    sn = safe(model_name)
    return {
        "activations": (ARTIFACTS_DIR / "activations" / f"{sn}_test_activations.pt").exists(),
        "probe": (ARTIFACTS_DIR / "probes" / f"{sn}_best_probe.pkl").exists(),
        "attribution": (ARTIFACTS_DIR / "attribution" / f"{sn}_layer_attribution.csv").exists(),
        "validation": (RESULTS_DIR / "tables" / f"{sn}_causal_validation.csv").exists(),
        "probe_metrics": (RESULTS_DIR / "tables" / f"{sn}_layer_probe_metrics.csv").exists(),
    }


def run_command(model_name: str) -> str:
    sn = model_name
    return "\n".join(
        [
            "# Run these commands in order:",
            f"python scripts/01_prepare_dataset.py --synthetic_only --sample_size 300",
            f"python scripts/02_cache_activations.py --model_name {sn}",
            f"python scripts/03_train_probe.py --model_name {sn}",
            f"python scripts/04_probe_gradient_attribution.py --model_name {sn}",
            f"python scripts/05_causal_validation.py --model_name {sn}",
            f"python scripts/06_generate_report.py --model_name {sn}",
        ]
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 Probe-Guided Attribution for Sycophancy")
st.markdown(
    """
**Research Demo** — Instead of attributing sycophantic behavior by taking gradients of a single output token logit,
we train a **behavioral linear probe** on internal activations and backpropagate *through the probe score*
to identify which model layers and components are causally responsible for sycophancy.

> **Sycophancy** = a model agreeing with or validating a user's false belief instead of correcting it.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: Model Selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Model", MODEL_NAMES, index=0)

    reg_info = REGISTRY.get(selected_model, {})
    st.markdown(f"**Backend:** `{reg_info.get('backend', '?')}`")
    st.markdown(f"**Family:** `{reg_info.get('family', '?')}`")
    st.markdown(f"**Chat template:** `{reg_info.get('chat_template', False)}`")

    avail = artifacts_exist(selected_model)
    st.subheader("Artifact Status")
    for key, exists in avail.items():
        icon = "✅" if exists else "❌"
        st.markdown(f"{icon} `{key}`")

    if not all(avail.values()):
        st.warning("Some artifacts are missing. Run the commands below:")
        st.code(run_command(selected_model), language="bash")


# ---------------------------------------------------------------------------
# Section 1: Dataset Viewer
# ---------------------------------------------------------------------------

st.header("1. Dataset Viewer")

pairs_csv = ROOT / "data" / "processed" / "sycophancy_pairs.csv"
if pairs_csv.exists():
    df_pairs = pd.read_csv(pairs_csv)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", len(df_pairs))
    if "label" in df_pairs.columns:
        col2.metric("Sycophantic (1)", int((df_pairs["label"] == 1).sum()))
        col3.metric("Non-sycophantic (0)", int((df_pairs["label"] == 0).sum()))

    # Group by prompt and show side by side
    if "label" in df_pairs.columns and "prompt" in df_pairs.columns:
        sample_prompts = df_pairs["prompt"].unique()[:5]
        for prompt in sample_prompts:
            with st.expander(f"📌 {prompt[:100]}"):
                sub = df_pairs[df_pairs["prompt"] == prompt]
                syc_row = sub[sub["label"] == 1]
                non_row = sub[sub["label"] == 0]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Sycophantic response (label=1)**")
                    if not syc_row.empty:
                        st.info(syc_row.iloc[0]["response"])
                with c2:
                    st.markdown("**Non-sycophantic response (label=0)**")
                    if not non_row.empty:
                        st.success(non_row.iloc[0]["response"])
else:
    st.warning(
        f"Dataset not found at `{pairs_csv}`.\n"
        "Run: `python scripts/01_prepare_dataset.py --synthetic_only --sample_size 300`"
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Probe Accuracy by Layer
# ---------------------------------------------------------------------------

st.header("2. Probe Accuracy by Layer")

metrics_path = RESULTS_DIR / "tables" / f"{safe(selected_model)}_layer_probe_metrics.csv"
probe_fig_path = RESULTS_DIR / "figures" / f"{safe(selected_model)}_layer_probe_accuracy.png"

if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.dataframe(
            metrics_df[["layer", "val_accuracy", "val_auroc", "val_f1", "test_auroc"]].style.highlight_max(
                subset=["val_auroc"], color="#d4edda"
            ),
            height=350,
        )
        best = metrics_df.loc[metrics_df["val_auroc"].idxmax()]
        st.success(
            f"Best layer: **{int(best['layer'])}**  |  Val AUROC: **{best['val_auroc']:.4f}**"
        )

    with col_b:
        if probe_fig_path.exists():
            st.image(str(probe_fig_path), use_container_width=True)
        else:
            st.info("Run step 03 to generate probe accuracy plot.")
else:
    st.info(f"Probe metrics not found for `{selected_model}`. Run steps 02 & 03 first.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Attribution Heatmap / Bar Plot
# ---------------------------------------------------------------------------

st.header("3. Attribution Maps")

layer_attr_path = ARTIFACTS_DIR / "attribution" / f"{safe(selected_model)}_layer_attribution.csv"
attr_fig_path = RESULTS_DIR / "figures" / f"{safe(selected_model)}_layer_attribution_barplot.png"
heatmap_path = RESULTS_DIR / "figures" / f"{safe(selected_model)}_component_attribution_heatmap.png"

if layer_attr_path.exists():
    attr_df = pd.read_csv(layer_attr_path)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Top layers")
        st.dataframe(attr_df.head(15), height=350)
    with col2:
        if attr_fig_path.exists():
            st.image(str(attr_fig_path), use_container_width=True)
        else:
            st.info("Run step 04 to generate attribution bar plot.")

    if heatmap_path.exists():
        st.subheader("Component Attribution Heatmap (attn / mlp)")
        st.image(str(heatmap_path), use_container_width=True)
else:
    st.info(f"Attribution not found for `{selected_model}`. Run steps 02–04 first.")

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Causal Validation Panel
# ---------------------------------------------------------------------------

st.header("4. Causal Validation")

val_csv_path = RESULTS_DIR / "tables" / f"{safe(selected_model)}_causal_validation.csv"
sweep_csv_path = RESULTS_DIR / "tables" / f"{safe(selected_model)}_causal_sweep.csv"
val_fig_path = RESULTS_DIR / "figures" / f"{safe(selected_model)}_causal_validation_barplot.png"
sweep_fig_path = RESULTS_DIR / "figures" / f"{safe(selected_model)}_causal_sweep.png"

if val_csv_path.exists():
    val_df = pd.read_csv(val_csv_path)
    st.subheader("Ablation Results (fixed k from step 05)")
    st.dataframe(
        val_df[["method", "k", "before_score", "after_score", "delta"]].style.format("{:.4f}", subset=["before_score", "after_score", "delta"]),
        use_container_width=True,
    )
    if val_fig_path.exists():
        st.image(str(val_fig_path), use_container_width=True)
else:
    st.info(f"Validation results not found. Run step 05 first.")

if sweep_csv_path.exists():
    sweep_df = pd.read_csv(sweep_csv_path)
    st.subheader("Interactive: Sweep over k (layers ablated)")
    max_k = int(sweep_df["k"].max())
    k_slider = st.slider("k (layers ablated)", min_value=1, max_value=max_k, value=min(5, max_k))
    row = sweep_df[sweep_df["k"] == k_slider]
    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Before ablation", f"{r['before']:.4f}")
        c2.metric("After top-k ablation", f"{r['probe_after']:.4f}", delta=f"{r['probe_delta']:.4f}")
        c3.metric("After random-k ablation", f"{r['random_after']:.4f}", delta=f"{r['random_delta']:.4f}")
    if sweep_fig_path.exists():
        st.image(str(sweep_fig_path), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 5: Single-Example Explorer
# ---------------------------------------------------------------------------

st.header("5. Single-Example Explorer")

if pairs_csv.exists():
    df_ex = pd.read_csv(pairs_csv)

    example_idx = st.number_input(
        "Example index", min_value=0, max_value=len(df_ex) - 1, value=0, step=1
    )
    row = df_ex.iloc[int(example_idx)]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prompt**")
        st.info(row.get("prompt", "N/A"))
        st.markdown("**Response**")
        response_text = row.get("response", "N/A")
        if row.get("label", 0) == 1:
            st.warning(f"⚠️ Sycophantic: {response_text}")
        else:
            st.success(f"✅ Non-sycophantic: {response_text}")

    with col2:
        st.markdown(f"**Label:** `{row.get('label', '?')}`")
        st.markdown(f"**Category:** `{row.get('category', '?')}`")
        st.markdown(f"**Source:** `{row.get('source', '?')}`")

        # Probe score if probe is available
        probe_path_file = ARTIFACTS_DIR / "probes" / f"{safe(selected_model)}_best_probe.pkl"
        activation_path = ARTIFACTS_DIR / "activations" / f"{safe(selected_model)}_test_activations.pt"
        if probe_path_file.exists() and activation_path.exists():
            try:
                import pickle, torch
                with open(probe_path_file, "rb") as f:
                    probe = pickle.load(f)
                data = torch.load(activation_path, map_location="cpu", weights_only=False)
                hs = data["hidden_states"].numpy()
                labels_arr = data["labels"].numpy()
                # Find matching example (approximate — use index in test split)
                test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
                test_idx = min(example_idx % len(hs), len(hs) - 1)
                h = hs[test_idx, probe.layer_idx, :]
                score = float(probe.predict_proba(h.reshape(1, -1))[0])
                st.metric("Probe sycophancy score", f"{score:.4f}")
            except Exception as e:
                st.caption(f"Could not compute probe score: {e}")

        # Top attributed layers
        top_path = ARTIFACTS_DIR / "attribution" / f"{safe(selected_model)}_top_layers.txt"
        if top_path.exists():
            with open(top_path) as f:
                top_layers_str = f.read().strip()
            st.markdown(f"**Top attributed layers:** `{top_layers_str}`")

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
---
**Probe-Guided Attribution for Sycophancy** | Built with TransformerLens + HuggingFace Transformers + Streamlit

Supported models: GPT-2, Pythia, Qwen2.5, Gemma-2
"""
)
