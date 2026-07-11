#!/bin/bash
# =============================================================================
# V2.0 full regeneration of all three models + cross-model comparison + story.
#   1. GPT-2 small        — full pipeline 01→14 (TransformerLens)
#   2. Pythia-410M        — probe pipeline 01b→07 (TransformerLens)
#   3. Qwen2.5-0.5B-Instr — probe pipeline 01b→07 (HuggingFace; TL-only steps skip)
#   4. Stage comparison + comparison plots + illustrated report
# =============================================================================
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
V2LOG="logs/v2_all_$(date +%Y%m%d_%H%M).log"
echo "=== V2.0 ALL-MODEL RUN — $(date) ===" | tee "$V2LOG"

# --- 1. GPT-2 small: full pipeline including causal-control stages ------------
echo -e "\n########## 1/3  GPT-2 small (full 01-14) ##########" | tee -a "$V2LOG"
bash run/run_full_pipeline.sh gpt2-small 300 2>&1 | tee -a "$V2LOG"

# --- 2. Pythia-410M: probe pipeline ------------------------------------------
echo -e "\n########## 2/3  Pythia-410M (probe pipeline) ##########" | tee -a "$V2LOG"
M="EleutherAI/pythia-410m"
python3 scripts/01b_build_prompt_preferences.py --model_name "$M" --input data/processed/sycophancy_pairs.csv --sample_size 300 2>&1 | tee -a "$V2LOG"
python3 scripts/02_cache_activations.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --sample_size 300 2>&1 | tee -a "$V2LOG"
python3 scripts/03_train_probe.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin 2>&1 | tee -a "$V2LOG"
python3 scripts/04_probe_gradient_attribution.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 50 2>&1 | tee -a "$V2LOG"
python3 scripts/05_causal_validation.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching --bootstrap 1000 2>&1 | tee -a "$V2LOG"
python3 scripts/11_run_controls.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic 2>&1 | tee -a "$V2LOG"
python3 scripts/10_layerwise_causal_sweep.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 30 --bootstrap 1000 2>&1 | tee -a "$V2LOG"
python3 scripts/06_generate_report.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final 2>&1 | tee -a "$V2LOG"
python3 scripts/07_generate_plots.py --model_name "$M" 2>&1 | tee -a "$V2LOG"

# --- 3. Qwen2.5-0.5B-Instruct: probe pipeline (HF) ---------------------------
echo -e "\n########## 3/3  Qwen2.5-0.5B-Instruct (probe pipeline, HF) ##########" | tee -a "$V2LOG"
Q="Qwen/Qwen2.5-0.5B-Instruct"
python3 scripts/01b_build_prompt_preferences.py --model_name "$Q" --input data/processed/sycophancy_pairs.csv --sample_size 200 2>&1 | tee -a "$V2LOG"
python3 scripts/02_cache_activations.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --sample_size 200 2>&1 | tee -a "$V2LOG"
python3 scripts/03_train_probe.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin 2>&1 | tee -a "$V2LOG"
python3 scripts/04_probe_gradient_attribution.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --max_examples 30 2>&1 | tee -a "$V2LOG"
python3 scripts/11_run_controls.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic 2>&1 | tee -a "$V2LOG"
python3 scripts/07_generate_plots.py --model_name "$Q" 2>&1 | tee -a "$V2LOG"

# --- 4. Cross-model comparison + story ---------------------------------------
echo -e "\n########## 4  Comparison + story ##########" | tee -a "$V2LOG"
python3 scripts/08_stage_comparison.py --models gpt2-small EleutherAI/pythia-410m Qwen/Qwen2.5-0.5B-Instruct 2>&1 | tee -a "$V2LOG"
python3 scripts/07_generate_plots.py --comparison 2>&1 | tee -a "$V2LOG"
python3 scripts/make_story.py 2>&1 | tee -a "$V2LOG"

echo -e "\n=== V2.0 ALL-MODEL RUN COMPLETE — $(date) ===" | tee -a "$V2LOG"
echo "V2 log: $V2LOG" | tee -a "$V2LOG"
