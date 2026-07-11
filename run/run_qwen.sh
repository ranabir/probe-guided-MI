#!/bin/bash
# Qwen2.5-0.5B-Instruct (instruct model, HuggingFace adapter). Uzay feedback #1.
# Patching/steering are TL-only, so step 05 uses mean_ablation fallback and step 09 marks pending.
set -e
cd "$(dirname "$0")"
M="Qwen/Qwen2.5-0.5B-Instruct"
LOG="logs/instruct_model_runs.log"
mkdir -p logs
echo "=== Qwen2.5-0.5B-Instruct run $(date) ===" | tee "$LOG"

run() { echo ">>> $1" | tee -a "$LOG"; shift; "$@" >> "$LOG" 2>&1; }

run "01b prompt preferences" python3 scripts/01b_build_prompt_preferences.py --model_name "$M" --input data/processed/sycophancy_pairs.csv --sample_size 200
run "02 cache activations" python3 scripts/02_cache_activations.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --sample_size 200
run "03 train probe" python3 scripts/03_train_probe.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin
run "04 attribution" python3 scripts/04_probe_gradient_attribution.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 30
run "05 causal (mean_ablation fallback for HF)" python3 scripts/05_causal_validation.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 30 --intervention mean_ablation --bootstrap 500
run "11 controls" python3 scripts/11_run_controls.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic
run "07 plots" python3 scripts/07_generate_plots.py --model_name "$M"

echo "=== Qwen run COMPLETE $(date) ===" | tee -a "$LOG"
python3 -c "
import pandas as pd
m = pd.read_csv('results/tables/Qwen_Qwen2.5-0.5B-Instruct_prompt_preferences_prompt_final_layer_probe_metrics.csv')
print('best test Pearson:', round(m['test_pearson'].max(),3), 'at layer', int(m.loc[m['test_pearson'].idxmax(),'layer']))
p = pd.read_csv('data/processed/Qwen_Qwen2.5-0.5B-Instruct_prompt_preferences.csv')
print('sycophancy rate:', round((p['behavior_margin']>0).mean(),3))
" 2>&1 | tee -a "$LOG"
