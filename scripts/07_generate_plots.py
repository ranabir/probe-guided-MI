#!/usr/bin/env python3
"""Step 07: Generate the presentation-ready plots/ gallery.

Per model:
  python scripts/07_generate_plots.py --model_name gpt2-small
Cross-model:
  python scripts/07_generate_plots.py --comparison

Scans results/tables/, artifacts/, and data/processed/ for the prompt-preference outputs and
renders six per-model plots into plots/{safe_model_name}/ plus comparison plots into
plots/comparison/. Updates plots/README.md and plots/comparison/summary_table.md.
Missing source tables are skipped with an explicit log of the command needed to produce them.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import plotting
from src.utils import get_root, results_dir, safe_model_name, setup_logging

logger = logging.getLogger(__name__)

# Models we know how to compare (extend freely)
KNOWN_MODELS = ["gpt2-small", "EleutherAI/pythia-410m"]
TAG = "prompt_preferences_prompt_final"

ENTRIES_FILE = get_root() / "plots" / ".entries.json"


def _load_entries() -> dict:
    if ENTRIES_FILE.exists():
        with open(ENTRIES_FILE) as f:
            return json.load(f)
    return {}


def _save_entries(entries: dict) -> None:
    ENTRIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTRIES_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def _paths(model_name: str) -> dict:
    sn = safe_model_name(model_name)
    root = get_root()
    return {
        "metrics": root / "results" / "tables" / f"{sn}_{TAG}_layer_probe_metrics.csv",
        "pref": root / "data" / "processed" / f"{sn}_prompt_preferences.csv",
        "attr": root / "artifacts" / "attribution" / f"{sn}_{TAG}_layer_attribution.csv",
        "causal": root / "results" / "tables" / f"{sn}_{TAG}_causal_validation.csv",
        "sweep": root / "results" / "tables" / f"{sn}_{TAG}_causal_sweep.csv",
    }


def generate_for_model(model_name: str) -> list:
    p = _paths(model_name)
    n_prompts = None
    if p["pref"].exists():
        try:
            n_prompts = len(pd.read_csv(p["pref"]))
        except Exception:
            pass

    fns = [
        plotting.plot_probe_regression_by_layer(p["metrics"], model_name, n_prompts),
        plotting.plot_behavior_margin_distribution(p["pref"], model_name),
        plotting.plot_layer_attribution(p["attr"], model_name),
        plotting.plot_causal_probe_delta(p["causal"], model_name),
        plotting.plot_causal_behavior_margin_delta(p["causal"], model_name),
        plotting.plot_topk_sweep(p["sweep"], model_name),
    ]
    entries = [e for e in fns if e]

    # Robustness-iteration plots, regenerated from their CSVs if present.
    root = get_root()
    sn = safe_model_name(model_name)
    control_csv = root / "results" / "tables" / f"{sn}_control_probe_metrics.csv"
    if control_csv.exists():
        real = pd.read_csv(p["metrics"]) if p["metrics"].exists() else None
        e = plotting.plot_probe_vs_controls(pd.read_csv(control_csv), real, model_name)
        if e:
            entries.append(e)
    dir_csv = root / "results" / "tables" / f"{sn}_direction_comparison.csv"
    if dir_csv.exists():
        dc = pd.read_csv(dir_csv)
        if "alpha" in dc.columns:
            entries.extend(plotting.plot_direction_comparison(dc, model_name))
    sweep_csv = root / "results" / "tables" / f"{sn}_layerwise_decodability_causal_sweep.csv"
    if sweep_csv.exists():
        sdf = pd.read_csv(sweep_csv)
        for fn in (plotting.plot_decodability_vs_causal, plotting.plot_layerwise_behavior_delta,
                   plotting.plot_layerwise_answer_flip):
            e = fn(sdf, model_name)
            if e:
                entries.append(e)

    # Causal-control iteration plots (scripts 12/13/14).
    # HuggingFace models write a "pending" marker CSV (TL-only interventions), so guard on the
    # presence of the real results column before plotting.
    cc_csv = root / "results" / "tables" / f"{sn}_contrastive_causal_results.csv"
    if cc_csv.exists():
        cc = pd.read_csv(cc_csv)
        if "layer_selection" in cc.columns and len(cc):
            entries.extend(plotting.plot_contrastive_causal_results(cc, model_name))
            if sweep_csv.exists():
                e = plotting.plot_causal_vs_decodable_intervention(pd.read_csv(sweep_csv), cc, model_name)
                if e:
                    entries.append(e)
    se_csv = root / "results" / "tables" / f"{sn}_side_effect_eval.csv"
    if se_csv.exists():
        se = pd.read_csv(se_csv)
        if "side_effect_score" in se.columns and len(se):
            e = plotting.plot_side_effect_summary(se.iloc[0].to_dict(), model_name)
            if e:
                entries.append(e)
    search_csv = root / "results" / "tables" / f"{sn}_causal_intervention_search.csv"
    if search_csv.exists():
        srch = pd.read_csv(search_csv)
        if "objective_score" in srch.columns and len(srch):
            entries.append(plotting.plot_intervention_search_pareto(srch, model_name))
            entries.append(plotting.plot_best_interventions_ranked(srch, model_name))
    # Clean causal control (script 15) — works for TL and HF models.
    clean_csv = root / "results" / "tables" / f"{sn}_clean_causal_control.csv"
    if clean_csv.exists():
        cdf = pd.read_csv(clean_csv)
        if "family" in cdf.columns and len(cdf):
            entries.extend(plotting.plot_clean_control(cdf, model_name))
    # Subspace ablation (script 16).
    sub_csv = root / "results" / "tables" / f"{sn}_subspace_ablation.csv"
    if sub_csv.exists():
        sdf = pd.read_csv(sub_csv)
        if "rank" in sdf.columns and len(sdf):
            entries.extend(plotting.plot_subspace_ablation(sdf, model_name))
    # Subspace interpretation (script 17).
    interp_csv = root / "results" / "tables" / f"{sn}_subspace_interpretation.csv"
    cos_csv = root / "results" / "tables" / f"{sn}_subspace_subtype_cosine.csv"
    sv_csv = root / "results" / "tables" / f"{sn}_subspace_singular_spectrum.csv"
    if interp_csv.exists() and sv_csv.exists():
        cap = pd.read_csv(interp_csv)
        cos = pd.read_csv(cos_csv, index_col=0) if cos_csv.exists() else pd.DataFrame()
        sv = pd.read_csv(sv_csv)["relative_energy"].tolist()
        layer = int(cap["layer"].iloc[0]) if "layer" in cap.columns and len(cap) else 0
        if "captured_energy" in cap.columns and len(cap):
            entries.extend(plotting.plot_subspace_interpretation(cap, cos, sv, model_name, layer))
    return entries


def generate_comparison() -> list:
    comp_dir = get_root() / "plots" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {m: _paths(m)["metrics"] for m in KNOWN_MODELS}
    causal = {m: _paths(m)["causal"] for m in KNOWN_MODELS}

    entries = []
    e1 = plotting.plot_model_probe_comparison(metrics, comp_dir)
    if e1:
        entries.append(e1)
    e2 = plotting.plot_model_causal_comparison(causal, comp_dir)
    if e2:
        entries.append(e2)

    # summary_table.md
    rows = []
    for m in KNOWN_MODELS:
        p = _paths(m)
        if not p["metrics"].exists():
            continue
        md = pd.read_csv(p["metrics"])
        best = md.loc[md["test_pearson"].idxmax()] if "test_pearson" in md.columns else None
        row = {"model": m, "num_layers": len(md),
               "best_test_pearson": f"{best['test_pearson']:.4f}" if best is not None else "n/a",
               "best_test_spearman": f"{best['test_spearman']:.4f}" if best is not None and "test_spearman" in md.columns else "n/a",
               "best_layer": int(best["layer"]) if best is not None else "n/a"}
        if p["causal"].exists():
            cd = pd.read_csv(p["causal"]).set_index("method")
            for meth in ("probe_gradient", "random", "logit_gradient"):
                if meth in cd.index:
                    row[f"{meth}_probe_delta"] = f"{cd.loc[meth, 'probe_delta']:.4f}"
                    row[f"{meth}_bm_delta"] = f"{cd.loc[meth, 'behavior_margin_delta']:.4f}"
        if p["pref"].exists():
            row["num_prompts"] = len(pd.read_csv(p["pref"]))
        rows.append(row)

    if rows:
        sdf = pd.DataFrame(rows)
        with open(comp_dir / "summary_table.md", "w") as f:
            f.write("# Cross-Model Summary\n\n")
            f.write(sdf.to_markdown(index=False))
            f.write("\n")
        logger.info("Wrote %s", comp_dir / "summary_table.md")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots gallery")
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--comparison", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    registry = _load_entries()

    if args.comparison:
        comp_entries = generate_comparison()
        registry["__comparison__"] = comp_entries
    if args.model_name:
        entries = generate_for_model(args.model_name)
        registry[safe_model_name(args.model_name)] = entries
    if not args.comparison and not args.model_name:
        ap.error("Pass --model_name <name> and/or --comparison")

    _save_entries(registry)

    # Flatten registry into README (per-model first, comparison last)
    all_entries = []
    for key, ents in registry.items():
        if key != "__comparison__":
            all_entries.extend(ents)
    all_entries.extend(registry.get("__comparison__", []))
    plotting.update_plots_readme(all_entries)

    print(f"\nPlots updated. Gallery index: {get_root() / 'plots' / 'README.md'}")


if __name__ == "__main__":
    main()
