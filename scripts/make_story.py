#!/usr/bin/env python3
"""Generate a self-contained, Anthropic-styled HTML story of the project.

- Re-renders ~9 hero figures from the existing result CSVs in a warm, consistent scientific palette.
- Inlines every figure as base64 so the output is a single portable file: docs/index.html

Run: python scripts/make_story.py
"""
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"

# --- Anthropic-ish palette -------------------------------------------------
INK = "#141413"
MUTED = "#6B6A65"
CARD = "#FAF9F5"
CORAL = "#D97757"
CLOTH = "#BD5D3A"
SLATE = "#5A7D9A"
SAGE = "#7A9B76"
GRAY = "#B0ABA0"
GRID = "#E7E2D6"

plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#C9C3B6", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold", "figure.dpi": 200,
})

FIGS = {}  # name -> base64 png


def _emit(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


def _read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


# --- 1. Cross-model probe regression --------------------------------------
def fig_regression_comparison():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for sn, label, color in [("gpt2-small", "GPT-2 small (12 layers)", CORAL),
                             ("EleutherAI_pythia-410m", "Pythia-410M (24 layers)", SLATE)]:
        d = _read(TABLES / f"{sn}_prompt_preferences_prompt_final_layer_probe_metrics.csv")
        if d is None or "test_pearson" not in d:
            continue
        depth = d["layer"] / max(d["layer"].max(), 1)
        ax.plot(depth, d["test_pearson"], marker="o", ms=5, lw=2.4, color=color, label=label)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
    ax.set_ylabel("probe test correlation (Pearson r)")
    ax.set_ylim(-0.1, 0.7)
    ax.set_title("Sycophancy preference is decodable — and it replicates")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _emit(fig, "regression_comparison")


# --- 2. Behavior margin distribution --------------------------------------
def fig_margin_dist():
    d = _read(PROC / "gpt2-small_prompt_preferences.csv")
    if d is None:
        return
    d = d.dropna(subset=["behavior_margin"])
    frac = float((d["behavior_margin"] > 0).mean())
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hist(d["behavior_margin"], bins=32, color=CORAL, alpha=0.85, edgecolor=CARD)
    ax.axvline(0, color=INK, ls="--", lw=1.4)
    ax.set_xlabel("behavior margin  =  logP(sycophantic) − logP(honest)")
    ax.set_ylabel("number of prompts")
    ax.set_title("How often does GPT-2 prefer the flattering answer?")
    ax.text(0.97, 0.93, f"{frac:.0%} of {len(d)} prompts\nprefer sycophancy",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F2E9DF", ec=CORAL))
    _emit(fig, "margin_dist")


# --- 3. Probe vs controls -------------------------------------------------
def fig_controls():
    c = _read(TABLES / "gpt2-small_control_probe_metrics.csv")
    real = _read(TABLES / "gpt2-small_prompt_preferences_prompt_final_layer_probe_metrics.csv")
    if c is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if real is not None and "test_pearson" in real:
        ax.plot(real["layer"], real["test_pearson"], marker="o", ms=5, lw=2.6, color=CORAL,
                label="sycophancy probe", zorder=5)
    rl = c[c["control_name"] == "random_label"]
    if not rl.empty:
        ax.plot(rl["layer"], rl["test_metric"], marker="x", lw=1.8, ls="--", color=GRAY,
                label="random-label (noise floor)")
    stat = c[c["control_name"] == "static_token"]
    if not stat.empty:
        peak = stat.groupby("feature")["test_metric"].max().sort_values(ascending=False)
        top = peak.index[0]
        g = stat[stat["feature"] == top]
        ax.plot(g["layer"], g["test_metric"], marker="s", lw=2.0, color=SLATE,
                label=f"surface feature: {top}")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("test decodability")
    ax.set_title("Is the probe real, or a surface-feature confound?")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    _emit(fig, "controls")


# --- 4. Stage sycophancy rate ---------------------------------------------
def fig_stage():
    d = _read(TABLES / "stage_comparison_summary.csv")
    if d is None:
        return
    d = d.copy()
    d["short"] = d["model_name"].str.replace("EleutherAI/", "").str.replace("Qwen/", "")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = [CORAL if s == "instruct" else SLATE for s in d.get("training_stage", ["base"] * len(d))]
    bars = ax.bar(d["short"] + "\n(" + d["training_stage"].astype(str) + ")",
                  d["sycophancy_rate"], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f%%", labels=[f"{v*100:.0f}%" for v in d["sycophancy_rate"]],
                 padding=3, fontsize=11)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.2)
    ax.set_ylabel("sycophancy rate")
    ax.set_ylim(0, 0.8)
    ax.set_title("Instruct-tuned models are more sycophantic")
    _emit(fig, "stage")


# --- 5. Decodability vs causal effect -------------------------------------
def fig_decode_vs_causal():
    d = _read(TABLES / "gpt2-small_layerwise_decodability_causal_sweep.csv")
    if d is None:
        return
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax1.plot(d["layer"], d["test_pearson"], marker="o", ms=5, lw=2.4, color=SLATE,
             label="decodability (Pearson)")
    ax1.set_xlabel("layer"); ax1.set_ylabel("decodability (Pearson)", color=SLATE)
    ax1.tick_params(axis="y", labelcolor=SLATE)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(d["layer"], d["behavior_margin_delta"].abs(), marker="s", ms=5, lw=2.4,
             ls="--", color=CORAL, label="|causal effect|")
    ax2.set_ylabel("|causal effect|  (|Δ margin|)", color=CORAL)
    ax2.tick_params(axis="y", labelcolor=CORAL)
    # mark peak-decodable vs peak-causal layer
    peak_dec = int(d.loc[d["test_pearson"].idxmax(), "layer"])
    peak_cau = int(d.loc[d["behavior_margin_delta"].abs().idxmax(), "layer"])
    ax1.axvline(peak_dec, color=SLATE, ls=":", lw=1.4, alpha=0.7)
    ax2.axvline(peak_cau, color=CORAL, ls=":", lw=1.4, alpha=0.7)
    ax1.set_title(f"Decoding ≠ causation: peak-readable (L{peak_dec}) ≠ peak-causal (L{peak_cau})")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, frameon=False, fontsize=9.5, loc="upper left")
    _emit(fig, "decode_vs_causal")


# --- 6. Contrastive causal answer flip ------------------------------------
def fig_answer_flip():
    d = _read(TABLES / "gpt2-small_contrastive_causal_results.csv")
    if d is None or "layer_selection" not in d:
        return
    best = (d.sort_values("targeted_syc_to_non_syc_flip_rate", ascending=False)
            .drop_duplicates("layer_selection"))
    order = ["causal_topk", "decodable_topk", "random"]
    best = best.set_index("layer_selection").reindex([o for o in order if o in best["layer_selection"].values
                                                      or o in best.index]).dropna(how="all").reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colmap = {"causal_topk": CORAL, "decodable_topk": SLATE, "random": GRAY}
    colors = [colmap.get(s, GRAY) for s in best["layer_selection"]]
    bars = ax.bar(best["layer_selection"], best["targeted_syc_to_non_syc_flip_rate"],
                  color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax.set_ylabel("targeted syc→honest flip rate")
    ax.set_ylim(0, max(0.15, best["targeted_syc_to_non_syc_flip_rate"].max() * 1.3))
    ax.set_title("At low strength, no layer choice controls the answer")
    _emit(fig, "answer_flip")


# --- 7. Side effects ------------------------------------------------------
def fig_side_effects():
    d = _read(TABLES / "gpt2-small_side_effect_eval.csv")
    if d is None or "side_effect_score" not in d:
        return
    m = d.iloc[0]
    keys = [("side_effect_score", "side-effect\nscore"), ("weirdness_rate", "weirdness\nrate"),
            ("repetition_increase", "repetition\nincrease"), ("qa_accuracy_drop", "QA accuracy\ndrop")]
    vals = [float(m.get(k, np.nan)) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = [CLOTH, CORAL, CORAL, CORAL]
    bars = ax.bar([lab for _, lab in keys], vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score  (0 = no harm)")
    ax.set_title("…and strong steering makes the model weird")
    _emit(fig, "side_effects")


# --- 8. Intervention search Pareto ----------------------------------------
def fig_pareto():
    d = _read(TABLES / "gpt2-small_causal_intervention_search.csv")
    if d is None or "side_effect_score" not in d:
        return
    d = d.dropna(subset=["side_effect_score", "targeted_flip_rate"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colmap = {"causal_topk": CORAL, "decodable_topk": SLATE}
    for sel, g in d.groupby("layer_selection_type"):
        ax.scatter(g["side_effect_score"], g["targeted_flip_rate"], s=70, alpha=0.8,
                   color=colmap.get(sel, GRAY), edgecolor=CARD, label=sel, zorder=3)
    ax.set_xlabel("side-effect score  (lower = cleaner)")
    ax.set_ylabel("targeted flip rate  (higher = more control)")
    ax.set_title("The trade-off: control vs. keeping the model intact")
    ax.annotate("the goal:\nhigh control, low harm", xy=(0.02, 0.9), xycoords="axes fraction",
                fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    _emit(fig, "pareto")


QWEN = "Qwen_Qwen2.5-0.5B-Instruct"


def fig_clean_control():
    """Qwen: flip vs side-effect for the steering families (additive vs projection/mean-shift)."""
    d = _read(TABLES / f"{QWEN}_clean_causal_control.csv")
    if d is None or "family" not in d:
        return
    fam_color = {"additive": GRAY, "additive_normpres": "#9E7BB5",
                 "projection_ablation": CORAL, "mean_shift": SAGE}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(-0.01, 0.25, color="#EAF0E6", alpha=0.8, zorder=0)
    ax.text(0.11, 0.02, "clean zone", fontsize=9, color="#4d6b48")
    for fam, g in d.groupby("family"):
        ax.scatter(g["side_effect_score"], g["targeted_flip_rate"], s=80, alpha=0.85,
                   color=fam_color.get(fam, GRAY), edgecolor=CARD, label=fam, zorder=3)
    ax.set_xlabel("side-effect score  (lower = model stays intact)")
    ax.set_ylabel("targeted syc→honest flip rate")
    ax.set_title("Steering families: control vs. side-effect (Qwen-Instruct)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _emit(fig, "clean_control")


def fig_subspace_rank():
    """Qwen: flip (solid) and side-effect (dashed) vs subspace rank, per layer band."""
    d = _read(TABLES / f"{QWEN}_subspace_ablation.csv")
    if d is None or "rank" not in d:
        return
    band_color = {"top3": CORAL, "peak": SLATE, "midband": SAGE}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for b, g in d.groupby("layer_band"):
        g = g.sort_values("rank"); c = band_color.get(b, GRAY)
        ax.plot(g["rank"], g["targeted_flip_rate"], marker="o", lw=2.4, color=c, label=f"{b}: flip")
        ax.plot(g["rank"], g["side_effect_score"], marker="s", lw=1.6, ls="--", color=c, alpha=0.7,
                label=f"{b}: side-effect")
    ax.set_xlabel("subspace rank k")
    ax.set_ylabel("rate / score")
    ax.set_title("Subspace ablation: flip rises with rank, side-effect stays low (Qwen)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    _emit(fig, "subspace_rank")


def fig_subtype_capture():
    """Qwen: captured energy of each sub-type direction vs subspace rank."""
    d = _read(TABLES / f"{QWEN}_subspace_interpretation.csv")
    if d is None or "captured_energy" not in d:
        return
    colors = [CORAL, SLATE, SAGE, "#9E7BB5"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, (name, g) in enumerate(d.groupby("subtype")):
        g = g.sort_values("rank")
        short = str(name).replace("sycophancy_on_", "")
        ax.plot(g["rank"], g["captured_energy"], marker="o", lw=2.4,
                color=colors[i % len(colors)], label=short)
    ax.axhline(1.0, color=GRAY, ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("subspace rank k")
    ax.set_ylabel("captured energy of sub-type direction")
    ax.set_title("Each sub-type gets its own dimension (Qwen)")
    ax.legend(frameon=False, fontsize=9, title="sub-type")
    _emit(fig, "subtype_capture")


def build_all():
    fig_regression_comparison(); fig_margin_dist(); fig_controls(); fig_stage()
    fig_decode_vs_causal(); fig_answer_flip(); fig_side_effects(); fig_pareto()
    fig_clean_control(); fig_subspace_rank(); fig_subtype_capture()


# --- HTML -----------------------------------------------------------------
def img(name, caption):
    if name not in FIGS:
        return f'<p class="missing">[figure {name} unavailable]</p>'
    return (f'<figure><img alt="{caption}" src="data:image/png;base64,{FIGS[name]}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def html():
    css = """
    :root{ --bg:#F0EEE6; --card:#FAF9F5; --panel:#FFFFFF; --ink:#141413; --muted:#6B6A65;
           --coral:#D97757; --cloth:#BD5D3A; --slate:#5A7D9A; --sage:#7A9B76; --border:#E5E0D5; }
    *{box-sizing:border-box;}
    html{scroll-behavior:smooth;}
    body{margin:0;background:var(--bg);color:var(--ink);
         font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         line-height:1.65;font-size:17px;-webkit-font-smoothing:antialiased;}
    .wrap{max-width:860px;margin:0 auto;padding:0 24px 120px;}
    h1,h2,h3,h4,.display{font-family:'Fraunces','Georgia',serif;font-weight:600;letter-spacing:-0.01em;line-height:1.15;}
    header{padding:70px 0 26px;}
    .eyebrow{font-family:'Inter',sans-serif;text-transform:uppercase;letter-spacing:0.16em;
             font-size:12px;font-weight:600;color:var(--coral);margin:0 0 18px;}
    h1{font-size:46px;margin:0 0 18px;}
    .lede{font-size:20px;color:var(--muted);max-width:720px;}
    h2{font-size:30px;margin:56px 0 6px;padding-top:6px;}
    h2 .num{color:var(--coral);font-size:13px;display:block;letter-spacing:0.08em;margin-bottom:8px;
            font-family:'Inter',sans-serif;font-weight:700;text-transform:uppercase;}
    h3{font-size:21px;margin:34px 0 6px;}
    h4{font-size:16px;margin:22px 0 4px;color:var(--cloth);}
    p{margin:14px 0;} a{color:var(--cloth);text-decoration:none;} a:hover{text-decoration:underline;}
    .muted{color:var(--muted);} strong{font-weight:650;} em{font-style:italic;}
    /* sticky tab bar */
    .tabbar{position:sticky;top:0;z-index:50;background:rgba(240,238,230,0.92);backdrop-filter:blur(8px);
            border-bottom:1px solid var(--border);}
    .tabbar-inner{max-width:860px;margin:0 auto;display:flex;gap:6px;padding:10px 24px;flex-wrap:wrap;}
    .tab-btn{font-family:'Inter',sans-serif;font-size:14px;font-weight:600;color:var(--muted);
             background:transparent;border:1px solid transparent;border-radius:999px;padding:8px 16px;cursor:pointer;}
    .tab-btn:hover{color:var(--ink);background:#EAE6DB;}
    .tab-btn.active{color:#fff;background:var(--coral);}
    .tab-btn .n{opacity:0.7;font-weight:700;margin-right:6px;}
    .tab-panel{display:none;} .tab-panel.active{display:block;}
    figure{margin:26px 0 8px;background:var(--panel);border:1px solid var(--border);
           border-radius:14px;padding:16px 16px 6px;box-shadow:0 1px 3px rgba(20,20,19,0.04);}
    figure img{width:100%;height:auto;border-radius:8px;display:block;}
    figcaption{font-size:13.5px;color:var(--muted);padding:11px 4px 6px;font-family:'Inter',sans-serif;line-height:1.5;}
    .callout{background:#F5EFE6;border-left:3px solid var(--coral);border-radius:0 10px 10px 0;
             padding:18px 22px;margin:24px 0;font-size:16.5px;}
    .callout .k{font-weight:700;color:var(--cloth);}
    .formula{background:#EFEADF;border-radius:10px;padding:14px 20px;font-family:'SF Mono',ui-monospace,Menlo,monospace;
             font-size:14px;color:var(--ink);margin:16px 0;overflow-x:auto;line-height:1.7;}
    .formula .lbl{color:var(--coral);font-weight:700;}
    .axisbox{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin:16px 0;font-size:15px;}
    .axisbox .t{font-family:'Inter',sans-serif;font-weight:700;font-size:12px;text-transform:uppercase;
                letter-spacing:0.07em;color:var(--muted);margin-bottom:6px;}
    .axisbox b{color:var(--cloth);}
    table{width:100%;border-collapse:collapse;margin:20px 0;font-size:15px;}
    th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);}
    th{font-family:'Inter',sans-serif;font-size:11.5px;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);}
    .index{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:24px 28px;margin:28px 0;}
    .index h4{color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-size:12px;margin:0 0 14px;font-family:'Inter',sans-serif;}
    .index ol{margin:0;padding-left:20px;} .index li{margin:8px 0;font-size:15.5px;}
    .index .tag{display:inline-block;font-size:11px;font-weight:700;color:var(--coral);border:1px solid var(--coral);
                border-radius:999px;padding:1px 8px;margin-left:6px;}
    .flow{list-style:none;padding:0;margin:20px 0;counter-reset:f;}
    .flow li{position:relative;padding:14px 18px 14px 54px;margin:10px 0;background:var(--card);
             border:1px solid var(--border);border-radius:10px;counter-increment:f;}
    .flow li::before{content:counter(f);position:absolute;left:16px;top:14px;width:26px;height:26px;background:var(--coral);
                     color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;
                     font-family:'Fraunces',serif;font-weight:600;font-size:14px;}
    .flow .lead{font-weight:650;}
    .stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0;}
    .stat{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px 20px;}
    .stat .big{font-family:'Fraunces',serif;font-size:30px;font-weight:600;color:var(--coral);}
    .stat .lab{font-size:12.5px;color:var(--muted);margin-top:4px;line-height:1.4;}
    .divider{height:1px;background:var(--border);margin:44px 0;}
    footer{color:var(--muted);font-size:13.5px;padding-top:34px;border-top:1px solid var(--border);margin-top:22px;}
    .pill{display:inline-block;background:#EDE7DB;border-radius:999px;padding:3px 12px;font-size:12.5px;
          font-family:'Inter',sans-serif;color:var(--muted);margin-right:6px;}
    """

    def sec(num, title, anchor=""):
        a = f' id="{anchor}"' if anchor else ""
        n = f'<span class="num">{num}</span>' if num else ""
        return f'<h2{a}>{n}{title}</h2>'

    # ---------------- TAB 1: Overview & research flow ----------------
    tab1 = f"""
  <header>
    <p class="eyebrow">Mechanistic Interpretability · Research Report</p>
    <h1>Reading &amp; controlling sycophancy<br>in language models</h1>
    <p class="lede">Can we locate where a model decides to flatter the user, read that decision from
    its activations, and then switch it off — without breaking the model? A guided tour of the full
    research, ending with the latest result: sycophancy is a <em>multi-dimensional subspace</em>.</p>
    <p style="margin-top:22px"><span class="pill">GPT-2 · Pythia · Qwen-Instruct</span>
       <span class="pill">Anthropic sycophancy evals</span>
       <span class="pill">probes · steering · subspace ablation</span></p>
  </header>

  {sec("Tab 1", "The question &amp; the through-line")}
  <p><strong>Sycophancy</strong> — telling the user what they want to hear rather than what is true —
  is a safety-relevant behavior. We asked two things: <strong>(a)</strong> can we <em>read</em> a
  model's sycophancy preference from its internal activations, and <strong>(b)</strong> once we can,
  can we <em>causally switch it off</em> without degrading the model? The second half is the crux —
  reading a behavior is not the same as controlling it.</p>

  <div class="index">
    <h4>Index</h4>
    <ol>
      <li><a href="#" onclick="showTab('t2');return false;">Probe-guided attribution</a> — the intuition, what we measure, the data, and the first three findings <span class="tag">Tab 2</span></li>
      <li><a href="#" onclick="showTab('t3');return false;">Steering &amp; the sycophancy subspace</a> — intervention types with formulas, side-effects, and the multi-dimensional result <span class="tag">Tab 3</span></li>
    </ol>
    <p class="muted" style="margin:14px 0 0;font-size:14px">Within Tab 2: decodable &amp; replicates ·
    controls · instruct models · decoding ≠ causation. Within Tab 3: five steering methods · the
    side-effect check · subspace ablation · one direction per topic.</p>
  </div>

  {sec("", "The research flow — what led to what", "flow")}
  <ol class="flow">
    <li><span class="lead">Measure the model's own preference.</span> Define <em>behavior_margin</em>
      and train a probe on the prompt-final activation to predict it. <span class="muted">→ is the signal real?</span></li>
    <li><span class="lead">It's decodable and replicates.</span> The probe strengthens with depth in
      both GPT-2 and Pythia. <span class="muted">→ could it be a trivial confound?</span></li>
    <li><span class="lead">Controls say it's real</span> (clears the random-label floor), and
      <span class="lead">instruct models are more sycophantic</span> (66% vs ~50%). <span class="muted">→ where do we intervene?</span></li>
    <li><span class="lead">Decoding ≠ causation.</span> The most <em>readable</em> layer is not the most
      <em>causal</em> one. <span class="muted">→ intervene at causal layers — but does it have side effects?</span></li>
    <li><span class="lead">Probe-direction steering works but makes the model weird.</span> Capping is
      clean but does nothing. <span class="muted">→ maybe sycophancy isn't a single direction.</span></li>
    <li><span class="lead">Sycophancy is a subspace.</span> Removing a rank-2 subspace flips 40% of
      answers at ~0.03 side-effect — clean and moderately strong. <span class="muted">→ what are the extra directions?</span></li>
    <li><span class="lead">One direction per topic.</span> The subspace is a union of topic-specific
      sycophancy directions (political, NLP, philosophy).</li>
  </ol>

  <div class="stat-row">
    <div class="stat"><div class="big">0.52 / 0.61</div><div class="lab">peak probe decodability (GPT-2 / Pythia) — replicated</div></div>
    <div class="stat"><div class="big">66% vs 50%</div><div class="lab">sycophancy: instruct vs base</div></div>
    <div class="stat"><div class="big">40% @ 0.03</div><div class="lab">rank-2 subspace flip / side-effect (Qwen)</div></div>
    <div class="stat"><div class="big">1 dir / topic</div><div class="lab">the subspace is a union of topic directions</div></div>
  </div>
  """

    # ---------------- TAB 2: Probe-guided attribution ----------------
    tab2 = f"""
  {sec("Tab 2", "Probe-guided attribution — intuition, method &amp; data")}
  <h3>The intuition</h3>
  <p>Standard attribution reads a single output-token logit (e.g. the logit for &ldquo;Yes&rdquo;). But
  &ldquo;Yes&rdquo; fires in countless harmless contexts — it is a poor proxy for a <em>behavior</em>.
  Probe-guided attribution instead trains a small linear <strong>probe</strong> on the model's internal
  activations to detect the behavior itself, then attributes and intervenes through that probe. It reads
  the model's <em>state</em>, not one token.</p>

  <h3>What we measure</h3>
  <p>For each prompt (a persona stating a view, then a two-choice question) the target is the model's
  <em>own</em> preference, in log-probabilities:</p>
  <div class="formula"><span class="lbl">behavior_margin</span> = mean-token&nbsp;logP(sycophantic answer | prompt) − mean-token&nbsp;logP(honest answer | prompt)</div>
  <p class="muted" style="font-size:15px">Positive → the model leans sycophantic. We probe the
  activation at the <em>final prompt token</em> — the instant before it answers — like reading someone's
  face the moment they finish hearing a loaded question.</p>

  <h3>The data</h3>
  <p><strong>Sycophancy (main):</strong> Anthropic <code>model-written-evals</code>, three sub-types
  pooled evenly — philosophy (<code>philpapers2020</code>), NLP researchers (<code>nlp_survey</code>),
  and politics (<code>political_typology_quiz</code>). <strong>Side-effects (capability):</strong> 30
  authored general-knowledge prompts (&ldquo;Why is the sky blue?&rdquo;, &ldquo;What is 2+2?&rdquo;),
  unrelated to sycophancy, used to check interventions don't degrade the model.</p>
  {img("margin_dist", "Data characterization: distribution of behavior margins (GPT-2). Sycophancy is common (~50% of prompts) after fixing a truncation bug that had faked a 7% rate.")}

  {sec("", "Finding 1 — decodable, and it replicates")}
  <p>The probe predicts the behavior margin and gets <strong>stronger with depth</strong> — the same
  rising curve in two unrelated model families. Two architectures agreeing points to a real, shared
  mechanism.</p>
  {img("regression_comparison", "Probe test-correlation vs. relative depth. Both GPT-2 and Pythia rise toward the deep layers (peaks ≈ 0.52 and 0.61).")}
  <div class="axisbox"><div class="t">How to read it</div>
    <b>x-axis</b> = relative layer depth (0 = first layer, 1 = last), so the two models line up despite
    different layer counts. <b>y-axis</b> = the probe's held-out test correlation (Pearson r) between
    predicted and true behavior_margin. Higher = the preference is more linearly readable at that depth.</div>

  {sec("", "Finding 2 — it's real, and instruct models are worse")}
  <p>Re-running the identical probe on <strong>control targets</strong>: a random-label probe (shuffled
  target = the noise floor) and surface-feature probes. The sycophancy probe (≈0.52) clears the
  random-label floor (≈0.25) — real signal — but surface features are also decodable, so a confound
  contribution can't be fully excluded.</p>
  {img("controls", "Sycophancy probe vs. the random-label noise floor and a surface-feature control, by layer.")}
  <p>Sycophancy is also amplified by post-training: <strong>Qwen-Instruct prefers the sycophantic
  answer 66% of the time vs. ~50% for base models.</strong></p>
  {img("stage", "Sycophancy rate is markedly higher in the instruction-tuned model.")}

  {sec("", "Finding 3 — decoding ≠ causation (the pivot)")}
  <p>Sweeping every layer independently, the layer where sycophancy is most <em>readable</em> is not the
  layer where intervening most <em>changes behavior</em>. Reading and steering live in different places.</p>
  {img("decode_vs_causal", "Decodability (blue) and causal effect (orange) by layer peak at different depths — the most readable layer is not the most causal one.")}
  <div class="axisbox"><div class="t">How to read it</div>
    <b>x-axis</b> = layer index. <b>Left y-axis</b> (blue) = decodability (probe test Pearson).
    <b>Right y-axis</b> (orange) = causal effect, i.e. |Δ behavior_margin| when that layer's residual is
    patched. The two curves peaking apart is the whole point: attribute where it's causal, not where it reads.</div>
  <div class="callout"><span class="k">This set up the next question.</span> Once we intervene at the
  causal layers to actually change behavior, does the intervention have <em>unintended side effects</em>?
  That is the subject of Tab 3.</div>
  """

    # ---------------- TAB 3: Steering & subspace ----------------
    tab3 = f"""
  {sec("Tab 3", "Steering &amp; the sycophancy subspace — the latest findings")}
  <p>We now use the probe direction to <em>causally</em> intervene, and — crucially — check whether the
  intervention harms the model. Below, <code>h</code> is the residual activation at a layer,
  <code>d̂</code> a unit sycophancy direction, <code>α</code> a steering strength, and <code>‖·‖</code>
  a vector length.</p>

  {sec("", "The five intervention methods (with formulas)")}
  <h4>Additive steering — the blunt baseline</h4>
  <div class="formula"><span class="lbl">h ← h + α · s · d̂</span>&nbsp;&nbsp;&nbsp;(s = mean residual norm at the layer, so α is in &ldquo;residual-norms&rdquo;)</div>
  <p class="muted" style="font-size:15px">Shove the activation along the sycophancy axis. Problem: it
  also inflates ‖h‖, drowning out the rest of the computation → the model goes weird.</p>
  <h4>Norm-preserving additive</h4>
  <div class="formula"><span class="lbl">h ← (h + α·s·d̂) · ( ‖h‖ / ‖h + α·s·d̂‖ )</span></div>
  <p class="muted" style="font-size:15px">Same push, then rescale to the original length — removes the inflation damage.</p>
  <h4>Activation capping</h4>
  <div class="formula"><span class="lbl">h ← h − max(0, (h·d̂) − θ) · d̂</span>&nbsp;&nbsp;&nbsp;(θ = a high quantile of h·d̂ over training data)</div>
  <p class="muted" style="font-size:15px">A limiter: only trim the sycophancy component when it's unusually large. Very clean, but almost no flips.</p>
  <h4>Projection ablation</h4>
  <div class="formula"><span class="lbl">h ← h − (h·d̂) · d̂</span></div>
  <p class="muted" style="font-size:15px">Remove the sycophancy component entirely; leave every orthogonal coordinate untouched. Clean, but weak with a single direction.</p>
  <h4>Mean-shift</h4>
  <div class="formula"><span class="lbl">h ← h + (t − h·d̂) · d̂</span>&nbsp;&nbsp;&nbsp;(t = mean of h·d̂ over honest-preferring training prompts)</div>
  <p class="muted" style="font-size:15px">Set the sycophancy coordinate to a realistic honest value, rather than an extreme.</p>

  {sec("", "The side-effect check (the reviewer's question)")}
  <p>We generated on the basic general-knowledge prompts with and without each intervention, scoring
  weirdness, repetition, length, and basic-QA correctness. The result matched intuition:
  <strong>additive steering flips a large share of answers (up to ~57% on Qwen) but makes the model
  weird</strong> (side-effect 0.42–0.71); <strong>capping stays clean</strong> (~0.04) <strong>but
  produces almost no flips</strong>.</p>
  {img("clean_control", "Qwen: flip vs. side-effect per intervention family. Additive families are high-flip but high-side-effect; projection/mean-shift are in the clean zone but low-flip. The 'golden' top-left corner is still empty here.")}
  <div class="axisbox"><div class="t">How to read it</div>
    <b>x-axis</b> = side-effect score ∈ [0,1] — how much the intervention damages the model (weirdness +
    length deviation + repetition + QA drop); 0 = untouched. <b>y-axis</b> = targeted syc→honest flip
    rate — of prompts that started sycophantic, the fraction that flipped to honest. <b>Goal = top-left</b>
    (high flip, low harm). The shaded band is the &ldquo;clean zone&rdquo; (side-effect &lt; 0.25).</div>
  {img("side_effects", "Capability side-effects (weirdness, repetition, QA drop) of the flip-producing steering intervention on the basic prompts. Tall bars = the model was degraded.")}

  {sec("", "Sycophancy is a subspace, not a single direction")}
  <p>If sycophancy is spread across several directions, removing one leaves the rest — which is why
  single-direction ablation was clean but weak. So we remove a whole rank-<em>k</em> subspace with an
  orthonormal basis <code>V</code> (a k×D matrix):</p>
  <div class="formula"><span class="lbl">h ← h − (h Vᵀ) V</span>&nbsp;&nbsp;&nbsp;(k = 1 recovers single-direction ablation; only k of ~900 dims change)</div>
  <p class="muted" style="font-size:15px">V is built from the top-k singular vectors (SVD) of many
  bootstrap difference-of-means directions — the directions along which activations most consistently
  separate sycophantic from honest.</p>
  <p>On Qwen (scaled eval, 76 test prompts): a <strong>single direction flips only 17.5%</strong>, but a
  <strong>rank-2 subspace flips 40% of answers at a side-effect of just 0.03</strong> — ~2× the control,
  ~15× cleaner than blunt steering.</p>
  {img("subspace_rank", "Subspace ablation on Qwen: flip (solid) rises sharply from rank 1 then plateaus; side-effect (dashed) stays low. Removing a small subspace gives clean, moderately strong control.")}
  <div class="axisbox"><div class="t">How to read it</div>
    <b>x-axis</b> = subspace rank k (how many directions we remove). <b>y-axis</b> = rate/score, with
    <b>solid lines = flip rate</b> (higher = more control) and <b>dashed lines = side-effect</b> (lower =
    cleaner); each color is a layer band. The ideal signature — solid rising, dashed flat and low — is
    exactly what appears.</div>
  <div class="stat-row">
    <div class="stat"><div class="big">0.175</div><div class="lab">flip from a single direction (rank 1)</div></div>
    <div class="stat"><div class="big">0.40</div><div class="lab">flip from a rank-2 subspace</div></div>
    <div class="stat"><div class="big">0.03</div><div class="lab">its side-effect score (near-clean)</div></div>
  </div>

  {sec("", "What the subspace is: one direction per topic")}
  <p>We measured how much of each sub-type's own direction lives inside the rank-k subspace, via
  <em>captured energy</em>:</p>
  <div class="formula"><span class="lbl">captured(d, V)</span> = ‖ V Vᵀ d ‖² / ‖ d ‖²  ∈ [0, 1]&nbsp;&nbsp;&nbsp;(1 = the topic direction lies fully in the subspace)</div>
  <p>The answer is clean: the <strong>dominant direction is the <em>political</em> one</strong> (captured
  at rank 1); the <strong>second dimension is the <em>NLP</em> one</strong> (0 → 0.55 at rank 2) — which
  is why rank-2 doubles the flip; <strong>philosophy stays diffuse</strong> (≤0.22), honestly explaining
  why the flip caps near 40%.</p>
  {img("subtype_capture", "Fraction of each sub-type's direction captured vs. subspace rank (Qwen). Political is captured first, NLP second, philosophy remains diffuse.")}
  <div class="axisbox"><div class="t">How to read it</div>
    <b>x-axis</b> = subspace rank k. <b>y-axis</b> = captured energy ∈ [0,1] — how much of that topic's
    own direction the rank-k subspace absorbs. <b>Each line = one topic.</b> Each added rank tends to
    bring in another topic's direction.</div>

  <div class="divider"></div>
  {sec("", "Where it stands — the golden result")}
  <div class="callout">
    <p style="margin:0"><span class="k">Bottom line.</span> Sycophancy is <strong>decodable</strong> and
    <strong>replicates</strong>; it's <strong>stronger in instruct models</strong>; <strong>decoding ≠
    causation</strong>; probe-direction steering <strong>works but makes the model weird</strong>; and —
    the payoff — sycophancy is a <strong>subspace</strong> (one direction per topic), so removing a
    rank-2 subspace gives <strong>clean, moderately strong control</strong> (40% flip at 0.03
    side-effect). This is the near-&ldquo;golden&rdquo; case: reduce sycophancy without breaking the model.</p>
  </div>
  <p><strong>Honest status &amp; next steps.</strong> Not fully solved — philosophy is under-removed,
  capping the flip near 40%, and eval sets are small. Next: add a philosophy-specific direction to the
  ablation basis; run a proper <strong>MMLU</strong> capability check alongside the basic-prompt eval; and
  repeat on Gemma-2 / larger Qwen and OLMo staged checkpoints (base→SFT→DPO→instruct) to see whether each
  new topic gets its own dimension and whether the clean-control region widens with scale.</p>

  <footer>All figures regenerated from <code>results/tables/</code> via <code>scripts/make_story.py</code>.
  Method &amp; full results in <code>docs/</code>.</footer>
  """

    tabs_js = """
    function showTab(id){
      document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
      document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
      document.getElementById(id).classList.add('active');
      document.querySelector('.tab-btn[data-tab="'+id+'"]').classList.add('active');
      window.scrollTo({top:0,behavior:'instant'});
    }
    """

    body = f"""
  <div class="tabbar"><div class="tabbar-inner">
    <button class="tab-btn active" data-tab="t1" onclick="showTab('t1')"><span class="n">1</span>Overview &amp; flow</button>
    <button class="tab-btn" data-tab="t2" onclick="showTab('t2')"><span class="n">2</span>Probe-guided attribution</button>
    <button class="tab-btn" data-tab="t3" onclick="showTab('t3')"><span class="n">3</span>Steering &amp; subspace</button>
  </div></div>
  <div class="wrap">
    <div id="t1" class="tab-panel active">{tab1}</div>
    <div id="t2" class="tab-panel">{tab2}</div>
    <div id="t3" class="tab-panel">{tab3}</div>
  </div>
  <script>{tabs_js}</script>
  """

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reading &amp; controlling sycophancy — research report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style></head><body>{body}</body></html>"""


def main():
    build_all()
    out = ROOT / "docs" / "index.html"
    out.write_text(html(), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.0f} KB, {len(FIGS)} embedded figures)")


if __name__ == "__main__":
    main()
