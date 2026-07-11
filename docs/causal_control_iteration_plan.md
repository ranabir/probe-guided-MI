# Causal-Control Iteration Plan

## Main question
Can we get reliable **answer-level** causal control by intervening at **high-causal** layers (not
high-decodable ones), using stronger **contrastive patching** and **activation capping**, while
measuring **side effects**?

## Current repo state (verified)
- Scripts go to `11_run_controls.py` → new scripts are **12, 13, 14**.
- Layerwise sweep CSV `results/tables/{sn}_layerwise_decodability_causal_sweep.csv` has columns:
  `layer, layer_frac, test_pearson, test_spearman, test_r2, behavior_margin_delta,
  bootstrap_ci_low/high, answer_flip_rate, accuracy_change, n_examples`. (Drives layer selection.)
- Corrected data (max_length=256): GPT-2 sycophancy ~50%, best Pearson 0.52 (L8); decodability vs
  causal effect anti-correlated **−0.59**; steering effects small (|Δmargin|≤0.03, flip≤0.05).

## Reuse (no duplication)
| Need | Existing function |
|------|-------------------|
| patch final-token residual at layers with given vectors | `patching.patch_selected_layers_from_reference` |
| behavior margin under patching | `patching.compute_patched_behavior_margin`, `patched_margins_for_layers` |
| additive steering | `directions.steer_logprob / steer_margins_for_prompts` |
| build all directions at a layer | `directions.compute_layer_directions` |
| A/B metrics | `metrics.behavioral_intervention_metrics` (answer_flip, targeted flip, accuracy) |
| bootstrap CI | `statistics.bootstrap_mean_ci` |
| plot dir / save / readme | `plotting.plots_dir/_save/_rel/update_plots_readme`, `utils.results_dir` |
| reference activations (prompt-final residual per layer) | `adapter.forward_with_cache(...)['hidden_states']` |

## New modules
- `src/causal_interventions.py` — `select_layers_from_sweep`, `choose_opposite_preference_reference`,
  `apply_contrastive_patch`, `apply_probe_steering`, `apply_activation_capping`,
  `run_intervention_and_score`.
- `src/side_effects.py` — `load_basic_prompts`, `generate_or_score_outputs`,
  `compute_repetition_score`, `compute_weirdness_flags`, `compute_side_effect_score`,
  `save_side_effect_samples`.
- `src/search.py` — `build_intervention_grid`, `evaluate_intervention_config`, `rank_interventions`.
- `src/plotting.py` (extend) — 6 new plot fns.

## New scripts
- `scripts/12_contrastive_causal.py`
- `scripts/13_side_effect_eval.py`
- `scripts/14_causal_intervention_search.py`

## Modified (additive only)
- `src/plotting.py`, `scripts/07_generate_plots.py` (wire new plots), `README.md`,
  `docs/run_log.md`, `plots/README.md`. Existing scripts/pipeline untouched.

## Layer selection (from sweep CSV)
- `causal_topk`: layers with largest `|behavior_margin_delta|` (fallback `answer_flip_rate`).
- `decodable_topk`: largest `test_pearson` (fallback `test_spearman`).
- `random`: uniform sample (seeded).
- `manual`: comma-separated ids.

## Interventions compared
1. **contrastive_patching** — for each target prompt, pick an opposite-preference reference prompt,
   cache its prompt-final residuals, patch them into the target at selected layers (TL hooks).
2. **probe_steering** — `h += alpha * direction` at selected layers (alpha sweep), norm-scaled.
3. **activation_capping** — project h onto direction; if projection > threshold (train quantile),
   subtract the excess: `h -= cap_strength * max(0, proj − thr) * dir_unit`.
4. **mean_ablation** — baseline (existing mean-patch).

Headline comparison: causal_topk vs decodable_topk vs random, per intervention.

## Side-effect eval (script 13)
- `data/side_effect_eval/basic_prompts.jsonl` (~30 basic prompts, some with `answer` labels).
- For baseline vs best intervention: greedy generate short continuations with/without the
  intervention hooks active at the chosen layers; compute output_length_ratio, repetition_score,
  weirdness flags (empty/repeated/incoherent), basic-QA correctness where labels exist; save
  before/after samples. `side_effect_score` ∈ [0,1], higher = worse.

## Output tables
```
results/tables/{sn}_contrastive_causal_results.csv
results/tables/{sn}_best_causal_intervention.json
results/tables/{sn}_side_effect_eval.csv
results/tables/{sn}_side_effect_samples.csv
results/tables/{sn}_causal_intervention_search.csv
```

## Output plots (plots/{sn}/ + report-ready in results/figures/)
```
{sn}_contrastive_causal_answer_flip.png
{sn}_contrastive_causal_behavior_delta.png
{sn}_causal_vs_decodable_layer_intervention.png
{sn}_side_effect_summary.png
{sn}_intervention_search_pareto.png
{sn}_best_interventions_ranked.png
```

## Success criteria
Pipeline intact; high-causal intervention implemented; causal/decodable/random compared;
contrastive patching (TL) + capping implemented; answer-flip is headline; side-effect eval with
samples; ranked search; plots+docs; tests pass (optional HF skips clean).

## Failure modes (and honest labels)
- margin moves but flip stays low → "margin-level movement, not answer-level control".
- flip rises only with high side effects → "disruption, not clean control".
- capping > steering → "capping cleaner than additive steering".
- nothing works → "useful negative result: representations exposed, not controllable".

## Execution
```
python scripts/12_contrastive_causal.py --model_name gpt2-small --layer_selection causal_topk decodable_topk random --top_k_layers 3 --intervention contrastive_patching probe_steering activation_capping mean_ablation --alphas -5 -3 -1 1 3 5 --max_examples 50 --bootstrap 1000
python scripts/13_side_effect_eval.py --model_name gpt2-small --intervention_config results/tables/gpt2-small_best_causal_intervention.json --num_prompts 30
python scripts/14_causal_intervention_search.py --model_name gpt2-small --max_examples 40 --side_effect_prompts 15 --bootstrap 300 --lambda_side_effect 0.5
python scripts/07_generate_plots.py --model_name gpt2-small
# Qwen: probe_steering + activation_capping only (HF patching skips gracefully)
pytest
```
CLI accepts space-separated lists (argparse nargs="+").
