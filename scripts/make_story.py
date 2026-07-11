#!/usr/bin/env python3
"""Generate a self-contained, Anthropic-styled HTML story of the project.

- Re-renders ~9 hero figures from the existing result CSVs in a warm, consistent scientific palette.
- Inlines every figure as base64 so the output is a single portable file: docs/PROJECT_STORY.html

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


def build_all():
    fig_regression_comparison(); fig_margin_dist(); fig_controls(); fig_stage()
    fig_decode_vs_causal(); fig_answer_flip(); fig_side_effects(); fig_pareto()


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
    .wrap{max-width:840px;margin:0 auto;padding:0 24px 120px;}
    h1,h2,h3,h4,.display{font-family:'Fraunces','Georgia',serif;font-weight:600;letter-spacing:-0.01em;line-height:1.15;}
    header{padding:88px 0 36px;border-bottom:1px solid var(--border);}
    .eyebrow{font-family:'Inter',sans-serif;text-transform:uppercase;letter-spacing:0.16em;
             font-size:12px;font-weight:600;color:var(--coral);margin:0 0 20px;}
    h1{font-size:50px;margin:0 0 20px;}
    .lede{font-size:20px;color:var(--muted);max-width:700px;}
    h2{font-size:30px;margin:64px 0 6px;padding-top:8px;}
    h2 .num{color:var(--coral);font-size:14px;display:block;letter-spacing:0.08em;margin-bottom:8px;
            font-family:'Inter',sans-serif;font-weight:700;text-transform:uppercase;}
    h3{font-size:21px;margin:34px 0 6px;}
    p{margin:15px 0;}
    a{color:var(--cloth);text-decoration:none;}
    a:hover{text-decoration:underline;}
    .muted{color:var(--muted);}
    strong{font-weight:650;}
    em{font-style:italic;}
    figure{margin:26px 0 8px;background:var(--panel);border:1px solid var(--border);
           border-radius:14px;padding:16px 16px 6px;box-shadow:0 1px 3px rgba(20,20,19,0.04);}
    figure img{width:100%;height:auto;border-radius:8px;display:block;}
    figcaption{font-size:13.5px;color:var(--muted);padding:11px 4px 6px;font-family:'Inter',sans-serif;line-height:1.5;}
    .callout{background:#F5EFE6;border-left:3px solid var(--coral);border-radius:0 10px 10px 0;
             padding:18px 22px;margin:26px 0;font-size:16.5px;}
    .callout .k{font-weight:700;color:var(--cloth);}
    /* research-summary panel */
    .summary{background:var(--panel);border:1px solid var(--border);border-radius:16px;
             padding:28px 30px;margin:30px 0;box-shadow:0 1px 3px rgba(20,20,19,0.04);}
    .summary h4{font-size:13px;text-transform:uppercase;letter-spacing:0.1em;color:var(--coral);
                margin:0 0 6px;font-family:'Inter',sans-serif;font-weight:700;}
    .summary p{margin:0 0 14px;}
    .summary .aim{font-family:'Fraunces',serif;font-size:22px;line-height:1.35;color:var(--ink);margin:0 0 8px;}
    /* question list */
    .qlist{list-style:none;padding:0;margin:18px 0;counter-reset:q;}
    .qlist li{position:relative;padding:14px 16px 14px 52px;margin:10px 0;background:var(--card);
              border:1px solid var(--border);border-radius:10px;counter-increment:q;font-size:16px;}
    .qlist li::before{content:"Q" counter(q);position:absolute;left:14px;top:14px;font-family:'Fraunces',serif;
                      font-weight:600;color:var(--coral);font-size:15px;}
    /* experiment card */
    .exp{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:8px 30px 26px;
         margin:30px 0;box-shadow:0 1px 3px rgba(20,20,19,0.04);}
    .exp-head{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--border);
              padding:20px 0 16px;margin-bottom:6px;}
    .exp-tag{font-family:'Inter',sans-serif;font-weight:700;font-size:12px;letter-spacing:0.08em;
             text-transform:uppercase;color:#fff;background:var(--coral);border-radius:999px;
             padding:5px 12px;white-space:nowrap;}
    .exp-title{font-family:'Fraunces',serif;font-size:22px;font-weight:600;margin:0;}
    .exp .row{display:grid;grid-template-columns:96px 1fr;gap:6px 16px;margin:14px 0;align-items:start;}
    .exp .row .lab{font-family:'Inter',sans-serif;font-size:12px;text-transform:uppercase;letter-spacing:0.07em;
                   color:var(--muted);font-weight:700;padding-top:3px;}
    .exp .obs{background:#F1F4F0;border-left:3px solid var(--sage);border-radius:0 8px 8px 0;
              padding:12px 16px;margin-top:6px;font-size:15.5px;}
    .exp .obs .k{color:#4d6b48;font-weight:700;}
    /* metric dashboard */
    .stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:26px 0;}
    .stat{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--border);
          border-radius:12px;padding:18px 20px;}
    .stat .big{font-family:'Fraunces',serif;font-size:32px;font-weight:600;color:var(--coral);}
    .stat .lab{font-size:12.5px;color:var(--muted);margin-top:4px;line-height:1.4;}
    table{width:100%;border-collapse:collapse;margin:22px 0;font-size:15px;}
    th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);}
    th{font-family:'Inter',sans-serif;font-size:11.5px;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);}
    .formula{background:#EFEADF;border-radius:10px;padding:15px 20px;font-family:'SF Mono',ui-monospace,Menlo,monospace;
             font-size:14px;color:var(--ink);margin:18px 0;overflow-x:auto;}
    .divider{height:1px;background:var(--border);margin:48px 0;}
    footer{color:var(--muted);font-size:13.5px;padding-top:36px;border-top:1px solid var(--border);margin-top:20px;}
    .pill{display:inline-block;background:#EDE7DB;border-radius:999px;padding:3px 12px;font-size:12.5px;
          font-family:'Inter',sans-serif;color:var(--muted);margin-right:6px;}
    /* table of contents */
    .toc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px 26px;margin:30px 0;}
    .toc h4{font-size:12px;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin:0 0 12px;
            font-family:'Inter',sans-serif;font-weight:700;}
    .toc ol{margin:0;padding-left:20px;columns:2;column-gap:32px;}
    .toc li{margin:6px 0;font-size:15px;}
    .nextsteps{list-style:none;padding:0;margin:18px 0;counter-reset:ns;}
    .nextsteps li{position:relative;padding:14px 18px 14px 54px;margin:10px 0;background:var(--card);
                  border:1px solid var(--border);border-radius:10px;counter-increment:ns;}
    .nextsteps li::before{content:counter(ns);position:absolute;left:16px;top:13px;width:26px;height:26px;
                          background:var(--coral);color:#fff;border-radius:50%;display:flex;align-items:center;
                          justify-content:center;font-family:'Fraunces',serif;font-weight:600;font-size:14px;}
    """

    def sec(num, title, anchor=""):
        a = f' id="{anchor}"' if anchor else ""
        n = f'<span class="num">{num}</span>' if num else ""
        return f'<h2{a}>{n}{title}</h2>'

    def experiment(tag, title, aim, method, fig_name, fig_caption, observation):
        return f"""
  <div class="exp">
    <div class="exp-head"><span class="exp-tag">{tag}</span><span class="exp-title">{title}</span></div>
    <div class="row"><div class="lab">Aim</div><div>{aim}</div></div>
    <div class="row"><div class="lab">Method</div><div>{method}</div></div>
    {img(fig_name, fig_caption)}
    <div class="obs"><span class="k">Observation.</span> {observation}</div>
  </div>"""

    body = f"""
<header>
  <p class="eyebrow">Mechanistic Interpretability · Research Report</p>
  <h1>Probe-guided attribution &amp; control of sycophancy in language models</h1>
  <p class="lede">Can we locate where a model decides to flatter the user, read that decision from its
  activations, and then intervene to switch it off — without degrading the model?</p>
  <p style="margin-top:24px"><span class="pill">GPT-2 · Pythia-410M · Qwen2.5-Instruct</span>
     <span class="pill">Anthropic sycophancy evals</span>
     <span class="pill">probes · patching · steering · capping</span></p>
</header>

<div class="wrap">

  <div class="toc">
    <h4>Contents</h4>
    <ol>
      <li><a href="#aim">Research aim</a></li>
      <li><a href="#questions">Research questions</a></li>
      <li><a href="#methods">Data &amp; methods</a></li>
      <li><a href="#experiments">Experiments &amp; results</a></li>
      <li><a href="#observations">Key observations</a></li>
      <li><a href="#conclusion">Conclusion</a></li>
      <li><a href="#next">Next steps</a></li>
    </ol>
  </div>

  {sec("01", "Research aim", "aim")}
  <div class="summary">
    <h4>Aim</h4>
    <p class="aim">To convert a <em>readable</em> internal representation of sycophancy into a
    <em>causal lever</em> — an intervention that reliably changes the model's answer from sycophantic to
    honest, while preserving its general capabilities.</p>
    <p class="muted" style="margin:0">Sycophancy — telling the user what they want to hear rather than
    what is true — is a safety-relevant behavior. If it corresponds to a direction or region in
    activation space, we should be able to both detect it and switch it off. This report tests how far
    that program actually gets.</p>
  </div>

  {sec("02", "Research questions", "questions")}
  <ol class="qlist">
    <li>Is a model's sycophancy preference <strong>linearly decodable</strong> from its activations, and where?</li>
    <li>Is that signal <strong>real</strong>, or a trivial surface-feature / format confound?</li>
    <li>Does sycophancy differ between <strong>base and instruction-tuned</strong> models (pre- vs post-training)?</li>
    <li>Are the most <strong>decodable</strong> layers also the most <strong>causal</strong> ones?</li>
    <li>Can we <strong>intervene</strong> to flip sycophantic answers — and does control come <strong>cleanly</strong>?</li>
  </ol>

  {sec("03", "Data &amp; methods", "methods")}
  <p><strong>Data.</strong> Anthropic's model-written sycophancy evaluations — real, published data —
  pooling three subsets (philosophy, NLP-researcher, and political opinions) into ~300 prompts. Each
  prompt states a persona's view and offers a sycophantic answer (agrees with the persona) and an honest
  answer (the correct/neutral view).</p>

  <p><strong>Target — behavior margin.</strong> Rather than a hand label, we use the model's own preference,
  measured as average per-token log-probability:</p>
  <div class="formula">behavior_margin  =  avg logP(sycophantic answer | prompt)  −  avg logP(honest answer | prompt)</div>
  <p class="muted" style="font-size:15px;margin-top:8px">Positive → the model leans sycophantic; negative → honest.</p>

  <p><strong>Probe.</strong> A linear probe reads the activation at the <em>final prompt token</em> — the
  instant before the model answers — and predicts the behavior margin. If it succeeds, the sycophancy
  decision is encoded in the activations before a single answer-token is produced.</p>

  <p><strong>Interventions.</strong> Contrastive patching (paste an opposite-preference prompt's activation
  in), probe steering (add α·direction), and activation capping (clip the projection onto the direction),
  with mean-ablation as a baseline. <strong>Headline metric:</strong> targeted syc→honest answer-flip rate,
  plus a side-effect score from generation on basic prompts. All estimates carry bootstrap 95% CIs.</p>

  <div class="callout"><span class="k">Correctness note.</span> Building the controls exposed a truncation
  bug — long prompts (~170 tokens) were cut at 128, zeroing 88% of margins and faking a "7% sycophancy rate."
  After fixing (<code>max_length&nbsp;=&nbsp;256</code>), sycophancy is ~50% and all results below are on the
  corrected data.</p>

  {img("margin_dist", "Data characterization: distribution of GPT-2 behavior margins after the fix. Sycophancy is common (~50% of prompts) — not the artifactual 7% — which is what makes the decoding result below meaningful.")}

  {sec("04", "Experiments &amp; results", "experiments")}

  {experiment("Exp 1", "Is sycophancy decodable, and where?",
    "Test whether a linear probe can predict the behavior margin from prompt-final activations, layer by layer, in two model families.",
    "Train per-layer ridge probes on prompt-final activations → behavior_margin; report held-out test Pearson vs relative depth for GPT-2 and Pythia-410M.",
    "regression_comparison",
    "Probe test-correlation vs. relative depth. Both models rise toward the deep layers, peaking near r ≈ 0.5–0.6.",
    "Sycophancy preference is decodable and the signal <strong>strengthens with depth</strong> — with the <strong>same rising shape in two unrelated architectures</strong>, evidence of a real shared mechanism (GPT-2 peaks 0.52, Pythia 0.61).")}

  {experiment("Exp 2", "Is it real, or a surface-feature confound?",
    "Rule out that the probe is just reading prompt format (e.g. the phrase &ldquo;do you agree?&rdquo;) or length.",
    "Re-run the identical probe pipeline on control targets: a random-label probe (shuffled target = noise floor), surface-feature probes, and a topic probe.",
    "controls",
    "The sycophancy probe (coral) clears the random-label noise floor (grey) — but a pure surface feature (blue) is also highly decodable.",
    "There is <strong>real signal above noise</strong> (0.52 vs a 0.25 random-label floor), but surface features are trivially decodable (up to 1.0) — so a confound contribution <strong>cannot be fully excluded</strong> at this sample size.")}

  {experiment("Exp 3", "Base vs. instruction-tuned models",
    "Test whether sycophancy is amplified by post-training (instruction tuning) rather than present only in base models.",
    "Compute sycophancy rate (fraction of prompts with behavior_margin > 0) for base GPT-2, base Pythia-410M, and Qwen2.5-0.5B-Instruct.",
    "stage",
    "Qwen-Instruct is markedly more sycophantic than the base models (though model family is still a confound).",
    "The instruction-tuned model is <strong>more sycophantic (66% vs ~50%)</strong>, consistent with post-training amplifying sycophancy. Caveat: cross-family comparison — the clean test is a single model's base→SFT→DPO→instruct staged checkpoints.")}

  {experiment("Exp 4", "Decodability vs. causal effect, per layer",
    "Test the reviewer's key point: are the most decodable layers also the strongest layers to intervene on?",
    "For every layer, measure decodability (test Pearson) and causal effect (behavior_margin change when that layer's residual is patched), with bootstrap CIs.",
    "decode_vs_causal",
    "Decodability (blue) peaks in the upper-middle layers; causal effect (coral) peaks in the early layers. Dotted lines mark each peak.",
    "The <strong>peak-decodable layer (≈8) is not the peak-causal layer (≈1)</strong> — reading and moving the behavior live in different places. This reframes the whole intervention strategy: target causal layers, not readable ones.")}

  {experiment("Exp 5", "Can we flip the answer at causal layers?",
    "Test whether targeting high-causal layers gives stronger answer-level control than decodable or random layers.",
    "Apply contrastive patching / steering / capping at causal-topk vs decodable-topk vs random layers; measure targeted syc→honest flip rate across an intervention-strength sweep.",
    "answer_flip",
    "At low intervention strength, the targeted answer-flip rate is near zero everywhere — causal layers do no better than random.",
    "At gentle strengths, <strong>no layer choice controls the answer</strong>. But under a stronger sweep, steering the causal layers flips up to <strong>83%</strong> of sycophantic answers, where decodable-layer targeting gives ~0 — so causal-layer targeting <em>is</em> actionable.")}

  {experiment("Exp 6", "Does control break the model?",
    "Check the reviewer's caution: if an intervention reduces sycophancy, does it also harm general capability?",
    "Generate on basic prompts (&ldquo;Why is the sky blue?&rdquo;, &ldquo;What is 2+2?&rdquo;) with and without the flip-producing intervention; score weirdness, repetition, length ratio, and basic-QA drop.",
    "side_effects",
    "The flip-producing intervention degrades general capability — repetitive, longer, weirder output.",
    "The answer-flip comes at a high cost: <strong>side-effect score 0.66</strong> (the model becomes weird and repetitive). Control here reflects <strong>general disruption</strong>, not a clean sycophancy switch.")}

  {experiment("Exp 7", "Is there a clean operating point?",
    "Search for any intervention configuration that achieves meaningful control with low side effects.",
    "Grid over layer-selection × direction × intervention × strength; score each by targeted flip and side-effect; plot the control-vs-harm trade-off (Pareto).",
    "pareto",
    "The control-vs-side-effect trade-off across configs. There is no clean top-left point.",
    "The Pareto front has <strong>no top-left point</strong>: strong control comes with disruption; clean interventions (capping) produce ~0 flips. A capability-preserving lever was <strong>not found</strong> in GPT-2 at these settings.")}

  {sec("05", "Key observations", "observations")}
  <div class="stat-row">
    <div class="stat"><div class="big">0.52 / 0.61</div><div class="lab">peak probe decodability (GPT-2 / Pythia) — replicated across families</div></div>
    <div class="stat"><div class="big">66% vs ~50%</div><div class="lab">sycophancy rate: instruct vs base models</div></div>
    <div class="stat"><div class="big">L8 ≠ L1</div><div class="lab">peak-decodable layer ≠ peak-causal layer</div></div>
    <div class="stat"><div class="big">0.83 / 0.66</div><div class="lab">answer-flip achievable, but at a high side-effect cost</div></div>
  </div>
  <ul>
    <li><strong>Decodable and replicated.</strong> Sycophancy preference is linearly readable, rises with depth, and shows the same shape in GPT-2 and Pythia.</li>
    <li><strong>Signal is real but not clean.</strong> It clears the random-label floor, yet surface features are also highly decodable — confounds aren't fully excluded.</li>
    <li><strong>Post-training matters.</strong> The instruction-tuned model is more sycophantic than base models.</li>
    <li><strong>Decoding ≠ causation.</strong> The most readable layer is not the most causal one — the central methodological finding.</li>
    <li><strong>Control is reachable, not clean.</strong> Steering causal layers flips most answers, but only by disrupting the model; clean interventions flip nothing.</li>
  </ul>

  {sec("06", "Conclusion", "conclusion")}
  <div class="callout">
    <p style="margin:0"><span class="k">Bottom line.</span> We can <strong>read</strong> a model's sycophancy
    preference well, and it strengthens with depth and replicates across models. We can even <strong>flip</strong>
    most sycophantic answers by steering the high-causal layers — but only at strengths that also break the
    model. <strong>Reliable, capability-preserving causal control is not yet achieved.</strong></p>
  </div>
  <p>This is a well-instrumented result rather than an overclaim: decodability is established with controls
  and replication; the causal claim is bounded with answer-flip metrics, side-effect checks, and bootstrap
  CIs. The gap between "readable" and "controllable" is now the sharp, open research question.</p>

  {sec("07", "Next steps", "next")}
  <ol class="nextsteps">
    <li><strong>HuggingFace steering on instruct models.</strong> Implement residual hooks for Qwen/Gemma so the causal question can be tested where sycophancy is a genuine learned behavior.</li>
    <li><strong>OLMo staged checkpoints.</strong> Base → SFT → DPO → instruct of one model — the clean, confound-free test of whether sycophancy is pre- or post-training in origin.</li>
    <li><strong>Sycophancy as a subspace.</strong> A single linear lever may be inherently limited; cap/steer within a low-rank subspace to seek control without the coherence collapse.</li>
    <li><strong>Fine trade-off search.</strong> Sweep capping strength and steering strength densely to confirm whether a capability-preserving operating point exists at all.</li>
  </ol>

  <footer>
    Probe-Guided Attribution for Sycophancy · all figures regenerated from <code>results/tables/</code> via
    <code>scripts/make_story.py</code> · methods &amp; full results in <code>docs/causal_control_iteration_results.md</code>.
  </footer>
</div>
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Probe-guided attribution &amp; control of sycophancy — research report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style></head><body>{body}</body></html>"""


def main():
    build_all()
    out = ROOT / "docs" / "PROJECT_STORY.html"
    out.write_text(html(), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.0f} KB, {len(FIGS)} embedded figures)")


if __name__ == "__main__":
    main()
