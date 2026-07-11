#!/bin/bash
# =============================================================================
# Canonical end-to-end pipeline for one TransformerLens model (default: gpt2-small).
# Runs every numbered stage in order and regenerates plots + the HTML report.
#
# Usage:
#   bash run/run_full_pipeline.sh                  # gpt2-small, 300 prompts
#   bash run/run_full_pipeline.sh gpt2-small 300
#
# Heavy HuggingFace / larger models: see run/run_qwen.sh, run/run_pythia.sh.
# =============================================================================
set -e
cd "$(dirname "$0")/.."

MODEL="${1:-gpt2-small}"
N="${2:-300}"
LOG="logs/full_pipeline_$(echo "$MODEL" | tr '/' '_').log"
mkdir -p logs
echo "=== Full pipeline: $MODEL ($N prompts) — $(date) ===" | tee "$LOG"

step () { echo -e "\n>>> $*" | tee -a "$LOG"; }

# --- Data ---------------------------------------------------------------------
step "01  prepare dataset (Anthropic sycophancy, paired form)"
python3 scripts/01_prepare_dataset.py --dataset anthropic_sycophancy --sample_size $((N*2)) --output_format prompt_preferences >> "$LOG" 2>&1

step "01b build prompt-preference dataset (behavior margins)"
python3 scripts/01b_build_prompt_preferences.py --model_name "$MODEL" --input data/processed/sycophancy_pairs.csv --sample_size "$N" >> "$LOG" 2>&1

step "01c build balanced diagnostic set"
python3 scripts/01c_build_balanced_preference_set.py --model_name "$MODEL" --input "data/processed/$(echo "$MODEL" | tr '/' '_')_prompt_preferences.csv" >> "$LOG" 2>&1

# --- Probe pipeline -----------------------------------------------------------
step "02  cache prompt-final activations"
python3 scripts/02_cache_activations.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --sample_size "$N" >> "$LOG" 2>&1

step "03  train per-layer regression probes"
python3 scripts/03_train_probe.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin >> "$LOG" 2>&1

step "04  probe-gradient attribution + logit-gradient baseline"
python3 scripts/04_probe_gradient_attribution.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --max_examples 50 >> "$LOG" 2>&1

step "05  causal validation (activation patching, bootstrap CIs)"
python3 scripts/05_causal_validation.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching --bootstrap 1000 >> "$LOG" 2>&1

# --- Robustness (controls, directions, layer sweep) ---------------------------
step "11  control probes (random-label / surface-feature / topic)"
python3 scripts/11_run_controls.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic >> "$LOG" 2>&1

step "10  layerwise decodability vs causal sweep"
python3 scripts/10_layerwise_causal_sweep.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --max_examples 30 --bootstrap 1000 >> "$LOG" 2>&1

step "09  direction comparison (regression vs diff-of-means vs random)"
python3 scripts/09_direction_comparison.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final --max_examples 20 >> "$LOG" 2>&1

# --- Causal-control iteration -------------------------------------------------
step "12  contrastive causal (causal vs decodable vs random layers)"
python3 scripts/12_contrastive_causal.py --model_name "$MODEL" --layer_selection causal_topk decodable_topk random --top_k_layers 3 --intervention contrastive_patching probe_steering activation_capping mean_ablation --alphas -5 -3 -1 1 3 5 --max_examples 25 --bootstrap 300 >> "$LOG" 2>&1

step "13  side-effect evaluation of the best intervention"
python3 scripts/13_side_effect_eval.py --model_name "$MODEL" --num_prompts 20 --max_new_tokens 25 >> "$LOG" 2>&1

step "14  causal intervention search (control vs side-effect trade-off)"
python3 scripts/14_causal_intervention_search.py --model_name "$MODEL" --directions regression diff_of_means --interventions probe_steering activation_capping --alphas -3 -1 1 3 --cap_quantiles 0.5 0.75 0.9 --max_examples 25 --side_effect_prompts 10 --max_new_tokens 15 --bootstrap 200 --lambda_side_effect 0.5 >> "$LOG" 2>&1

# --- Reporting ----------------------------------------------------------------
step "06  generate markdown report"
python3 scripts/06_generate_report.py --model_name "$MODEL" --input_format prompt_preferences --probe_position prompt_final >> "$LOG" 2>&1

step "07  generate plot gallery"
python3 scripts/07_generate_plots.py --model_name "$MODEL" >> "$LOG" 2>&1

echo -e "\n=== COMPLETE: $MODEL — $(date) ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
