#!/bin/bash
# v3 Pythia-410M main experiment chain. Logs to logs/pythia_410m_run.log.
set -e
cd "$(dirname "$0")"
M="EleutherAI/pythia-410m"
LOG="logs/pythia_410m_run.log"
mkdir -p logs
echo "=== Pythia-410M run started $(date) ===" | tee "$LOG"

echo ">>> 01b build prompt preferences" | tee -a "$LOG"
python3 scripts/01b_build_prompt_preferences.py --model_name "$M" --input data/processed/sycophancy_pairs.csv --sample_size 300 >> "$LOG" 2>&1

echo ">>> 02 cache activations" | tee -a "$LOG"
python3 scripts/02_cache_activations.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --sample_size 300 >> "$LOG" 2>&1

echo ">>> 03 train regression probe" | tee -a "$LOG"
python3 scripts/03_train_probe.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin >> "$LOG" 2>&1

echo ">>> 04 attribution" | tee -a "$LOG"
python3 scripts/04_probe_gradient_attribution.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 50 >> "$LOG" 2>&1

echo ">>> 05 causal validation (activation patching)" | tee -a "$LOG"
python3 scripts/05_causal_validation.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching >> "$LOG" 2>&1

echo ">>> 07 plots" | tee -a "$LOG"
python3 scripts/07_generate_plots.py --model_name "$M" >> "$LOG" 2>&1

echo "=== Pythia-410M run COMPLETE $(date) ===" | tee -a "$LOG"
