#!/bin/bash
# Rebuild GPT-2 prompt-preference pipeline with corrected max_length=256 (fixes truncated margins).
set -e
cd "$(dirname "$0")"
M="gpt2-small"
LOG="logs/gpt2_rebuild_maxlen256.log"
mkdir -p logs
echo "=== GPT-2 rebuild (max_length=256) $(date) ===" | tee "$LOG"

python3 scripts/01b_build_prompt_preferences.py --model_name "$M" --input data/processed/sycophancy_pairs.csv --sample_size 300 >> "$LOG" 2>&1
echo ">>> 01b done" | tee -a "$LOG"
python3 scripts/02_cache_activations.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --sample_size 300 >> "$LOG" 2>&1
echo ">>> 02 done" | tee -a "$LOG"
python3 scripts/03_train_probe.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin >> "$LOG" 2>&1
echo ">>> 03 done" | tee -a "$LOG"
python3 scripts/04_probe_gradient_attribution.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 50 >> "$LOG" 2>&1
echo ">>> 04 done" | tee -a "$LOG"
python3 scripts/05_causal_validation.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching --bootstrap 1000 >> "$LOG" 2>&1
echo ">>> 05 done" | tee -a "$LOG"
python3 scripts/01c_build_balanced_preference_set.py --model_name "$M" --input data/processed/${M}_prompt_preferences.csv >> "$LOG" 2>&1
echo ">>> 01c done" | tee -a "$LOG"
python3 scripts/11_run_controls.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic >> "$LOG" 2>&1
echo ">>> 11 done" | tee -a "$LOG"
python3 scripts/10_layerwise_causal_sweep.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 30 --bootstrap 1000 >> "$LOG" 2>&1
echo ">>> 10 done" | tee -a "$LOG"
python3 scripts/09_direction_comparison.py --model_name "$M" --input_format prompt_preferences --probe_position prompt_final --max_examples 20 >> "$LOG" 2>&1
echo ">>> 09 done" | tee -a "$LOG"
python3 scripts/07_generate_plots.py --model_name "$M" >> "$LOG" 2>&1
echo "=== GPT-2 rebuild COMPLETE $(date) ===" | tee -a "$LOG"

echo "--- key results ---" | tee -a "$LOG"
python3 -c "
import pandas as pd
m = pd.read_csv('results/tables/${M}_prompt_preferences_prompt_final_layer_probe_metrics.csv')
print('best test Pearson:', round(m['test_pearson'].max(),3), 'at layer', int(m.loc[m['test_pearson'].idxmax(),'layer']))
p = pd.read_csv('data/processed/${M}_prompt_preferences.csv')
print('sycophancy rate:', round((p['behavior_margin']>0).mean(),3), '| exact-zero margins:', round((p['behavior_margin']==0).mean(),3))
" | tee -a "$LOG"
