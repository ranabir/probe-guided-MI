# Next Experiment Plan v3 — From "Decodable" to "Causal"

## What currently works

- Corrected prompt-preference setup: one row per prompt, target = `behavior_margin`
  (mean-token logprob(syc|prompt) − mean-token logprob(non_syc|prompt)), regression probes.
- GPT-2, 300 Anthropic prompts: test Pearson rises monotonically 0.24 (layer 0) → 0.48 (layer 11);
  late layers 0.38–0.48. Causal probe_delta −0.148 (probe-gradient) vs −0.030 (random) = 5× separation.
- 52 tests pass. Sanity check (response-aware classification) intact.

## Scientific gap

The signal is **decodable** but not yet shown to be **causal beyond baselines on real behavior**:
`behavior_margin_delta` under mean-ablation does NOT separate probe-gradient from random
(−0.065 vs −0.062). Three reasons to address:
1. Mean ablation is blunt (destroys generic info, not the specific direction).
2. Only ~7% of GPT-2 prompts prefer sycophancy → low-variance, imbalanced target.
3. No logit-gradient baseline to contextualize probe-gradient.
4. Only one model (12 layers) — no cross-family evidence.

v3 closes these with: activation patching, a balanced diagnostic set, a logit-gradient baseline,
and a Pythia-410M (24-layer) replication.

## Files modified

| File | Change |
|------|--------|
| `src/patching.py` | Add `patch_selected_layers_from_reference`, `compute_patched_behavior_margin`, `run_patching_behavior_validation` (TL only) |
| `scripts/05_causal_validation.py` | Add `--intervention {mean_ablation,activation_patching}` (default activation_patching for TL); add logit_gradient_topk method; save patching CSV |
| `scripts/03_train_probe.py` | Already defaults to regression for prompt_preferences; add `--dataset_variant {natural,balanced}` |
| `scripts/02_cache_activations.py` | Add `--dataset_variant` to load balanced split files |
| `scripts/06_generate_report.py` | Add v3 sections + plot links |
| `app/streamlit_app.py` | model + dataset_variant + intervention + method selectors; plot gallery |
| `config.yaml` | Add `validation.intervention: activation_patching` |

## New scripts / modules

| File | Purpose |
|------|---------|
| `src/logit_gradient.py` | logit-margin gradient baseline + token-set handling |
| `src/plotting.py` | 8 presentation-ready plot functions + `update_plots_readme` |
| `scripts/01c_build_balanced_preference_set.py` | balanced positive/negative margin diagnostic set |
| `scripts/07_generate_plots.py` | scans tables/artifacts → generates all plots, updates plots/README.md |
| `docs/next_experiment_plan_v3.md` | this file |
| `docs/final_status_v3.md` | written after all runs |

## Output directories (exact)

```
plots/                                  # NEW — presentation-ready gallery
  README.md                             # table describing every plot
  gpt2-small/                           # plots/{safe_model_name}/
  EleutherAI_pythia-410m/
  comparison/                           # cross-model plots + summary_table.md
artifacts/attribution/                  # machine-readable attribution CSVs (existing)
results/tables/                         # machine-readable metric CSVs (existing)
results/figures/                        # report-ready figures (existing — preserved)
logs/                                   # pythia_410m_run.log
```

**Compatibility layer:** existing `results/figures/` plots are preserved and still written by
steps 03–05 via `src/visualization.py`. The new `plots/` gallery is *additive* — `scripts/07`
re-renders higher-quality versions into `plots/` from the same source CSVs. No old path is removed.

## Plot referencing

Every plot in `plots/` is referenced in three places:
1. `plots/README.md` — table (file, model, experiment, what it shows, how to read, source table)
2. `docs/run_log.md` — per-run "plots created" list
3. `results/report.md` and `docs/final_status_v3.md` — embedded with relative paths

## Per-model plots (6)

```
plots/{sn}/{sn}_probe_regression_by_layer.png
plots/{sn}/{sn}_behavior_margin_distribution.png
plots/{sn}/{sn}_probe_gradient_layer_attribution.png
plots/{sn}/{sn}_causal_probe_delta.png
plots/{sn}/{sn}_causal_behavior_margin_delta.png
plots/{sn}/{sn}_topk_sweep_behavior_margin.png
```

## Comparison plots

```
plots/comparison/model_probe_regression_comparison.png   # x=relative depth 0..1, y=test Pearson
plots/comparison/model_causal_behavior_comparison.png    # grouped bars: model × method
plots/comparison/summary_table.md
```

## Source tables consumed by plots

| Plot | Source CSV |
|------|-----------|
| probe_regression_by_layer | `results/tables/{sn}_prompt_preferences_prompt_final_layer_probe_metrics.csv` |
| behavior_margin_distribution | `data/processed/{sn}_prompt_preferences.csv` |
| probe_gradient_layer_attribution | `artifacts/attribution/{sn}_prompt_preferences_prompt_final_layer_attribution.csv` |
| causal_probe_delta | `results/tables/{sn}_prompt_preferences_prompt_final_causal_validation.csv` |
| causal_behavior_margin_delta | same causal_validation CSV (+ activation_patching CSV) |
| topk_sweep_behavior_margin | `results/tables/{sn}_prompt_preferences_prompt_final_causal_sweep.csv` |

## Commands

(See spec execution order — GPT-2 300-prompt path already produced; re-run step 05 with
`--intervention activation_patching`, then 01c balanced, then Pythia, then `07 --comparison`,
then `06` reports, then `pytest`.)

## Success criteria

1. Pythia-410M attempted + documented.
2. probe_regression plots for GPT-2 (+ Pythia if available).
3. behavior_margin distribution plot exists.
4. probe-gradient attribution plot exists.
5. causal plots compare probe-gradient / random / logit-gradient.
6. activation patching implemented for TL or documented.
7. balanced diagnostic set created.
8. plots/README.md explains every plot.
9. run_log.md references plots + source tables.
10. report.md references plots.
11. Streamlit shows plots or exact regen command.
12. pytest passes or failures documented.

## Honest-interpretation rules

- If behavior_margin_delta still ≈ random: "The method identifies decodable preference
  representations, but causal control over behavior remains unproven under this intervention."
- If Pythia separates: "Pythia shows stronger evidence that probe-attributed layers influence
  behavior margin beyond random intervention."
- Mixed → report honestly. Do not overclaim.
