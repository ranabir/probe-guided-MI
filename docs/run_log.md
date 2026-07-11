# Run Log — Probe-Guided Attribution v2

## Dataset Preparation

---

### Step 01 — Synthetic dataset (smoke-test)

**Command:**
```bash
python scripts/01_prepare_dataset.py --dataset synthetic --synthetic_only --sample_size 80
```

**Inputs:** None (generated in-memory)

**Outputs:**
- `data/processed/sycophancy_pairs.csv` — 40 paired rows
- `data/processed/train.csv` — 55 flat rows
- `data/processed/val.csv` — 12 flat rows
- `data/processed/test.csv` — 13 flat rows

**Result:** ✅ Success  
Class balance: {0: 40, 1: 40}  
Source: synthetic  
Subsets: science (22), geography (16), history (16), opinion (8), conspiracy (8), health (6), preference (4)

---

### Step 01 — Anthropic sycophancy dataset

**Command:**
```bash
python scripts/01_prepare_dataset.py --dataset anthropic_sycophancy --sample_size 100
```

**Inputs:** HuggingFace `Anthropic/model-written-evals` (JSONL files)

**Outputs:**
- `data/processed/sycophancy_pairs.csv` — 50 paired rows
- `data/processed/train.csv` — 69 flat rows
- `data/processed/val.csv` — 15 flat rows
- `data/processed/test.csv` — 16 flat rows

**Result:** ✅ Success  
Total available: 30,168 pairs across 3 subsets  
Subsets loaded:
- sycophancy_on_philpapers2020: 9,984 examples
- sycophancy_on_nlp_survey: 9,984 examples
- sycophancy_on_political_typology_quiz: 10,200 examples

Sampled: 50 pairs (100 flat rows)  
Note: Response format is "(A)" or "(B)" — single-character answer letters from multiple-choice questions.

**Known quirk:** The response field contains just `(A)` or `(B)`. For logprob computation this is clean; for text display it's minimal. Future improvement: reconstruct full response text from question options.

---

### Step 01 — TruthfulQA (pending)

**Command:**
```bash
python scripts/01_prepare_dataset.py --dataset truthfulqa --sample_size 200
```

**Status:** Not yet run. Implementation complete in `src/data.py`. Expected to work but untested.

---

## Pipeline Steps (pending Anthropic data)

Steps 02–06 with Anthropic data and `--probe_position prompt_final` are pending.
Commands are documented in `docs/implementation_plan_v2.md`.

---

## What's Ready to Run Next

```bash
# With Anthropic dataset (real sycophancy evals):
python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 100 --probe_position prompt_final
python scripts/03_train_probe.py --model_name gpt2-small --probe_position prompt_final
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --probe_position prompt_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --probe_position prompt_final --top_k 3 --max_examples 20
python scripts/06_generate_report.py --model_name gpt2-small --probe_position prompt_final
```

Note: Re-run step 01 with `--dataset anthropic_sycophancy` before step 02 if you want to use the Anthropic data.

---

# Prompt-Preference Fix — Run Log (gpt2-small)

## Sanity check (paired_rows + response_final)

### Step 02 — cache response_final
Command: `python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 80 --input_format paired_rows --probe_position response_final`
Inputs: data/processed/{train,val,test}.csv
Outputs: artifacts/activations/gpt2-small_response_final_{train,val,test}_activations.pt
Result: ✅ shapes (55/12/13, 12, 768)

### Step 03 — classification probe
Command: `... --input_format paired_rows --probe_position response_final --probe_type classification --probe_target label`
Outputs: artifacts/probes/gpt2-small_response_final_best_probe.pkl, results/tables/gpt2-small_response_final_layer_probe_metrics.csv
Result: ✅ best_late selected layer 7 (val_auroc=1.0); selected probe persisted as best probe

### Step 04 — attribution
Result: ✅ Attribution probe at layer 7 → gradients non-zero for layers 0–7

### Step 05 — causal validation
Result: ✅
```
method          k  before_probe  after_probe  probe_delta  before_bm  after_bm  bm_delta
probe_gradient  3  0.9271        0.2871       -0.6400      0.6988     0.6706    -0.0282
random          3  0.9271        0.6711       -0.2560      0.6988     0.7375    +0.0387
```
Probe-gradient ablation drops probe prediction far more than random AND reduces the real
behavior margin, while random ablation increases it. Sanity check passes.

## Main experiment (prompt_preferences + prompt_final)

### Step 01b — build prompt preferences
Command: `python scripts/01b_build_prompt_preferences.py --model_name gpt2-small --input data/processed/sycophancy_pairs.csv --sample_size 40`
Inputs: data/processed/sycophancy_pairs.csv (synthetic, real text responses)
Outputs: data/processed/gpt2-small_prompt_preferences{,_train,_val,_test}.csv
Result: ✅ 40 prompts | behavior_margin mean=0.3495 std=0.6263 | prefers_sycophancy 27/13 (0.675)

### Step 02 — cache prompt_final activations
Outputs: artifacts/activations/gpt2-small_prompt_preferences_prompt_final_{split}_activations.pt
Result: ✅ stores labels=prefers_sycophancy AND behavior_margin targets

### Step 03 — regression probe
Command: `... --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin`
Result: ✅ valid regression (one target per prompt). best_late selected layer 11 (val_pearson=0.29).
Honest: train overfits (R²=1.0, 27 ex × 768 dims); val/test Pearson weak/noisy (test layer 10 r=0.53).

### Step 04 — regression attribution
Result: ✅ predicted-margin gradients; layers ranked; non-zero for layers 0–11

### Step 05 — causal validation (probe prediction + real behavior margin)
Result: ✅
```
method          k  before_probe  after_probe  probe_delta  before_bm  after_bm  bm_delta
probe_gradient  3  0.2942        0.3382       +0.0440      0.4064     1.2197    +0.8133
random          3  0.2942        0.3030       +0.0088      0.4064     1.4521    +1.0457
```
Honest: effects are weak/noisy at this scale. The win is structural validity — the table now
reports a real behavior_margin_delta from hook-based forward-pass ablation, not just probe score.

### Step 06 — report
Result: ✅ two-section report.md (response-aware sanity check + prompt-only preference)

### pytest
Result: ✅ 52 passed

---

# v3 Run Log — Causal Upgrade (2026-06-14)

## GPT-2 small (300 Anthropic prompts)

### Step 04 — attribution + logit-gradient baseline
Command: `python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 50`
Outputs: artifacts/attribution/gpt2-small_prompt_preferences_prompt_final_{layer_attribution,logit_gradient_layer_attribution,logit_gradient_tokens,logit_gradient_top_layers}.*
Result: ✅ probe-gradient top layers + logit-gradient baseline (syc_ids/non_ids saved; "Agree","Incorrect" multi-token → first token used)

### Step 05 — causal validation (activation patching)
Command: `... --top_k 5 --intervention activation_patching`
Outputs: results/tables/gpt2-small_prompt_preferences_prompt_final_{causal_validation,activation_patching_validation,causal_sweep}.csv
Result: ✅
```
method          probe_delta  behavior_margin_delta
probe_gradient  -0.1478      +0.0263
random          -0.0887      +0.0496
logit_gradient   0.0000      +0.0788
```
Probe-prediction: probe_gradient ≫ random ≫ logit (clean). Real behavior: NOT separated (probe_gradient moves behavior least). Honest mixed result.

### Step 01c — balanced diagnostic
Command: `python scripts/01c_build_balanced_preference_set.py --model_name gpt2-small --input data/processed/gpt2-small_prompt_preferences.csv`
Result: ✅ 21 pos / 21 neg = 42 (natural imbalance 7.0% documented)

### Step 07 — plots
Outputs: plots/gpt2-small/*.png (6), plots/README.md
Source tables: layer_probe_metrics.csv, prompt_preferences.csv, layer_attribution.csv, causal_validation.csv, causal_sweep.csv

## Pythia-410M (300 Anthropic prompts) — logs/pythia_410m_run.log
Full chain 01b→07 ran exit 0.
Result: ✅ test Pearson rises 0.26 (L0) → peak 0.442 (L8), stays 0.36–0.44 through upper layers (24 layers). Causal: probe_gradient bm_delta +0.0003 vs random +0.0413 — not separated. probe_delta 0.0000 (probe's own layer not in ablated set; cached-activation ablation artifact — real metric is behavior_margin_delta via patching).
Plots: plots/EleutherAI_pythia-410m/*.png (6)

## Comparison
Command: `python scripts/07_generate_plots.py --comparison`
Outputs: plots/comparison/{model_probe_regression_comparison,model_causal_behavior_comparison}.png, plots/comparison/summary_table.md
Both models: best_test_pearson 0.44–0.48 (decoding replicates). Causal control unproven in both.

## Tests
`pytest` → 71 passed (added test_balanced_preference_set, test_logit_gradient, test_activation_patching, test_plot_generation).

---

# reviewer feedback Iteration — Run Log (2026-06-22)

## CRITICAL FIX: truncation bug
`max_length 128 → 256`. Anthropic prompts ~170 tok median (max 236); 128 truncated answers →
88% of behavior margins were degenerate 0.0. Post-fix: sycophancy ~50% (was artifactual 7%),
0% zero margins. All models rebuilt. Defensive bounds guards added to patching/steering/logprob.

## GPT-2 small (rebuild_gpt2.sh, logs/gpt2_rebuild_maxlen256.log)
01b→05, 01c, 11, 10, 09, 07. Outputs:
- best test Pearson 0.518 @ L8 | sycophancy rate 0.505
- causal (activation_patching): probe_gradient Δ−0.042 CI[−0.104,+0.016], random Δ−0.076 | flip 0
- controls: random_label 0.246 | static contains_do_you_agree 1.00 | topic 0.67
- layerwise: corr(decodability,|causal|) = −0.59
- directions: logistic −0.58 / regression −0.52 / diff_of_means −0.05 / random ~0
Plots: plots/gpt2-small/*.png (12)

## Pythia-410M (rebuild_pythia.sh, logs/pythia_rebuild_maxlen256.log)
- best test Pearson 0.611 @ L16 | sycophancy rate 0.508
Plots: plots/EleutherAI_pythia-410m/*.png

## Qwen2.5-0.5B-Instruct (run_qwen.sh, logs/instruct_model_runs.log)
HF adapter; patching TL-only so step 05 used mean_ablation fallback (causal Δ = n/a).
- best test Pearson 0.421 @ L5 | **sycophancy rate 0.66** (instruct > base)

## Stage comparison (scripts/08)
results/tables/stage_comparison_summary.csv + plots/comparison/stage_*.png
Instruct sycophancy 0.66 > base ~0.50.

## Plots + tests
07 --comparison → plots/README.md (20 plots). pytest → 102 passed.

## Pending
OLMo staged run (no verified IDs); HF steering/patching; Gemma / Qwen-1.5B.

---

# Causal-Control Iteration — Run Log (2026-07-01)

## Step 12 — contrastive_causal (GPT-2)
Command: `python scripts/12_contrastive_causal.py --model_name gpt2-small --layer_selection causal_topk decodable_topk random --top_k_layers 3 --intervention contrastive_patching probe_steering activation_capping mean_ablation --alphas -5 -3 -1 1 3 5 --max_examples 25 --bootstrap 300`
Inputs: layerwise sweep CSV, prompt_preferences test split, train activations.
Outputs: results/tables/gpt2-small_contrastive_causal_results.csv, _best_causal_intervention.json; 3 plots.
Result: ✅ low-strength flips ≤0.08 for all selections (causal≈random≈decodable); no clean control.

## Step 13 — side_effect_eval (GPT-2)
Command: `python scripts/13_side_effect_eval.py --model_name gpt2-small --num_prompts 20 --max_new_tokens 25`
Outputs: results/tables/gpt2-small_side_effect_eval.csv, _side_effect_samples.csv; summary plot; docs/side_effect_eval_notes.md.
Result: ✅ side_effect_score 0.661, weirdness 1.0, length 1.46×, qa_drop 0.125 — the steered model degrades.

## Step 14 — causal_intervention_search (GPT-2)
Command: `python scripts/14_causal_intervention_search.py --model_name gpt2-small --directions regression diff_of_means --interventions probe_steering activation_capping --alphas -3 -1 1 3 --cap_quantiles 0.5 0.75 0.9 --max_examples 25 --side_effect_prompts 10 --max_new_tokens 15 --bootstrap 200 --lambda_side_effect 0.5`
Outputs: results/tables/gpt2-small_causal_intervention_search.csv; pareto + ranked plots.
Result: ✅ best = causal_topk diff_of_means steering α=−3: targeted flip 0.83 but side-effect 0.66. Capping clean (~0.04) but flip 0.

## Qwen2.5-0.5B-Instruct
TL-only interventions → graceful pending markers in contrastive_causal_results.csv (HF residual hooks not implemented).

## Plots + tests
07 regenerated → plots/README.md (39 plots). pytest → 131 passed.

## Interpretation
Targeting high-causal layers with strong steering DOES flip answers (0.83) where decodable-layer targeting gives ~0 — but with heavy side effects. Reliable, capability-preserving causal control NOT achieved in GPT-2. Useful, honest trade-off result.

---

# Clean Causal Control + HF Interventions — Run Log (2026-07-11)

## New module: src/residual_interventions.py (backend-agnostic TL + HF)
Unified residual edit for both TransformerLens and HuggingFace: additive, additive_normpres,
projection_ablation, mean_shift, cap. Verified steering works on Qwen (instruct) — Problem 1 solved.

## Step 15 — clean_causal_control
GPT-2 (layers 1-3): `python scripts/15_clean_causal_control.py --model_name gpt2-small --layer_selection causal_topk --top_k_layers 3 --alphas -6 -4 -2 2 4 6 --max_examples 25 --side_effect_prompts 12`
  → additive flip 0.25 @ side-effect 0.68; projection_ablation/mean_shift flip 0.0 @ side-effect 0.05 (13x cleaner, but no flip).
Qwen (layers 4-6): `python scripts/15_clean_causal_control.py --model_name Qwen/Qwen2.5-0.5B-Instruct --layer_selection manual --manual_layers 4,5,6 --alphas -6 -4 -2 2 4 6 --max_examples 20 --side_effect_prompts 10`
  → additive flip 0.57 @ side-effect 0.42-0.71; additive_normpres 0.57 @ 0.44 (Pareto improvement); projection_ablation flip 0.07 @ side-effect 0.016 (first clean-zone control point).

Outputs: results/tables/{sn}_clean_causal_control.csv; plots/{sn}/{sn}_clean_control_{pareto,flip_vs_sideeffect}.png
Tests: +9 (test_residual_interventions.py). pytest → 140 passed.

Interpretation: instruct models now steerable; norm-inflation diagnosis confirmed (clean methods 13-30x lower side-effect); strong-AND-clean control still unreached (top-left of Pareto empty).
