#!/bin/bash
# Scale the Qwen eval: rebuild at 500 sycophancy prompts (~75 test), re-cache activations,
# retrain probe, then re-run subspace ablation with tighter settings.
set -e
cd "$(dirname "$0")/.."
Q="Qwen/Qwen2.5-0.5B-Instruct"
LOG="logs/scale_qwen_$(date +%Y%m%d_%H%M).log"
mkdir -p logs
echo "=== Scale Qwen eval — $(date) ===" | tee "$LOG"

echo ">>> prepare 500-pair Anthropic sycophancy dataset" | tee -a "$LOG"
python3 scripts/01_prepare_dataset.py --dataset anthropic_sycophancy --sample_size 1000 --output_format prompt_preferences >> "$LOG" 2>&1

echo ">>> 01b build Qwen prompt-preferences (500 -> ~75 test)" | tee -a "$LOG"
python3 scripts/01b_build_prompt_preferences.py --model_name "$Q" --input data/processed/sycophancy_pairs.csv --sample_size 500 >> "$LOG" 2>&1

echo ">>> 02 cache activations" | tee -a "$LOG"
python3 scripts/02_cache_activations.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --sample_size 500 >> "$LOG" 2>&1

echo ">>> 03 retrain probe" | tee -a "$LOG"
python3 scripts/03_train_probe.py --model_name "$Q" --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin >> "$LOG" 2>&1

echo ">>> 16 subspace ablation (scaled: top3 band, max_examples 60, side_effect 25, bootstrap 1000)" | tee -a "$LOG"
python3 scripts/16_subspace_ablation.py --model_name "$Q" --ranks 1 2 3 5 8 12 --layer_bands top3 peak --max_examples 60 --side_effect_prompts 25 --max_new_tokens 16 --bootstrap 1000 >> "$LOG" 2>&1

echo ">>> 07 plots" | tee -a "$LOG"
python3 scripts/07_generate_plots.py --model_name "$Q" >> "$LOG" 2>&1

echo "=== Scale Qwen eval COMPLETE — $(date) ===" | tee -a "$LOG"
python3 -c "
import pandas as pd
d = pd.read_csv('data/processed/Qwen_Qwen2.5-0.5B-Instruct_prompt_preferences_test.csv')
print('test prompts:', len(d))
s = pd.read_csv('results/tables/Qwen_Qwen2.5-0.5B-Instruct_subspace_ablation.csv')
top = s[s.layer_band=='top3'].sort_values('rank')
print(top[['rank','targeted_flip_rate','side_effect_score','ci_low','ci_high','n_examples']].to_string(index=False))
" 2>&1 | tee -a "$LOG"
