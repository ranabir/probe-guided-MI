# Prompt-Preference Fix Plan

## 1. Why paired-row `prompt_final` probing is invalid

In `paired_rows` format each prompt produces two rows:

| row | input | label |
|-----|-------|-------|
| 1 | prompt + sycophantic_response | 1 |
| 2 | prompt + non_sycophantic_response | 0 |

In `prompt_final` mode, we extract the hidden state at the **last token of the prompt only**. The response text is never fed to the model. Therefore both rows for a given prompt produce the **identical** activation vector `h(prompt)`, but carry **opposite labels** (1 and 0).

A classifier trained on `{(h(prompt), 1), (h(prompt), 0)}` is mathematically ill-posed: the same feature vector maps to both classes with equal frequency. The Bayes-optimal predictor is the base rate, so expected AUROC → 0.5. The low AUROC (~0.22) we observed is not a bug — it is the honest consequence of an invalid target. (It dipped below 0.5 only due to small-sample noise.)

**Conclusion:** the *label* is a property of the (prompt, response) pair, but the *features* are a property of the prompt alone. The target must be redefined as a property of the prompt.

## 2. The fix: prompt-level preference dataset

Collapse to **one row per prompt**. The target becomes a property of the model's behavior on that prompt:

```
syc_logprob     = mean-token logprob( sycophantic_response | prompt )
non_syc_logprob = mean-token logprob( non_sycophantic_response | prompt )
behavior_margin = syc_logprob - non_syc_logprob
prefers_sycophancy = 1 if behavior_margin > 0 else 0
```

Now `h(prompt) → behavior_margin` is a well-posed **regression** (one unique target per unique prompt), and `h(prompt) → prefers_sycophancy` is a well-posed **classification**. The scientific question becomes: *do the model's prompt-final activations encode whether it is about to prefer a sycophantic continuation?*

### New dataset schema (one row per prompt)
```
id, prompt, sycophantic_response, non_sycophantic_response,
syc_logprob, non_syc_logprob, behavior_margin, prefers_sycophancy,
source_dataset, subset
```

Length normalization (mean token logprob, not sum) is used to avoid the bias from syc/non responses having different lengths.

## 3. Scripts / modules modified or added

| File | Change |
|------|--------|
| `src/behavior_metrics.py` | Add `compute_continuation_logprob(adapter, prompt, continuation, normalize=True)`; refactor `compute_pair_logprob_margin` to use it; add `build_prompt_preference_dataset()` |
| `src/probes.py` | Add regression support to `LinearProbe` (`task="regression"` via Ridge); regression metrics (MSE/MAE/Pearson/Spearman/R2); `train_all_layer_probes(task=...)`; `select_attribution_probe` selects by val Pearson for regression |
| `src/model_adapters.py` | `_probe_forward` branches on `probe.task`: regression → linear output, classification → sigmoid |
| `src/activation_cache.py` | Path builders accept optional `input_format`; `prompt_preferences` inserts tag into filename; save/load `behavior_margin` array alongside labels |
| `scripts/01_prepare_dataset.py` | Add `--output_format {paired_rows, prompt_preferences}` (default `paired_rows`). For `prompt_preferences`, point user to 01b |
| **`scripts/01b_build_prompt_preferences.py`** (NEW) | Load paired CSV + model, compute margins, save one-row-per-prompt CSVs + splits |
| `scripts/02_cache_activations.py` | Add `--input_format`; for `prompt_preferences` run on prompt only, cache prompt-final state, store `prefers_sycophancy` label + `behavior_margin` target |
| `scripts/03_train_probe.py` | Add `--input_format`, `--probe_type {classification,regression}`, `--probe_target`; regression default for prompt_preferences |
| `scripts/04_probe_gradient_attribution.py` | Add `--input_format`; regression probes attribute predicted margin |
| `scripts/05_causal_validation.py` | Add `--input_format`; for prompt_preferences report probe-prediction delta AND real behavior_margin delta under hook ablation |
| `scripts/06_generate_report.py` | Add `--input_format`; two-section report (response-aware sanity check + prompt-only preference) |
| `app/streamlit_app.py` | Add input_format selector + prompt-preference panel |
| `README.md` | New section: "Correct experiment design after prompt_final diagnosis" |
| `docs/run_log.md` | Append before/after each run |

### Artifact naming
- `paired_rows` (default, unchanged): `{sn}_{probe_position}_{split}_activations.pt`
- `prompt_preferences` (new): `{sn}_prompt_preferences_{probe_position}_{split}_activations.pt`

Metadata for prompt_preferences includes: `input_format: prompt_preferences`, `probe_position: prompt_final`, `target_columns: [prefers_sycophancy, behavior_margin]`.

## 4. Intermediate artifacts

```
data/processed/{sn}_prompt_preferences.csv
data/processed/{sn}_prompt_preferences_train.csv
data/processed/{sn}_prompt_preferences_val.csv
data/processed/{sn}_prompt_preferences_test.csv
artifacts/activations/{sn}_prompt_preferences_prompt_final_{split}_activations.pt
artifacts/probes/{sn}_prompt_preferences_prompt_final_best_probe.pkl
artifacts/probes/{sn}_prompt_preferences_prompt_final_selected_probe_metadata.json
artifacts/attribution/{sn}_prompt_preferences_prompt_final_layer_attribution.csv
results/tables/{sn}_prompt_preferences_prompt_final_layer_probe_metrics.csv
results/tables/{sn}_prompt_preferences_prompt_final_causal_validation.csv
results/figures/{sn}_prompt_preferences_prompt_final_*.png
```

## 5. Commands

### Sanity check (response-aware, must still work)
```
python scripts/01_prepare_dataset.py --dataset synthetic --synthetic_only --sample_size 80 --output_format paired_rows
python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 80 --input_format paired_rows --probe_position response_final
python scripts/03_train_probe.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --probe_type classification --probe_target label
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --top_k 3 --max_examples 20
```

### Main experiment (prompt preference, the correct design)
```
python scripts/01_prepare_dataset.py --dataset synthetic --synthetic_only --sample_size 80 --output_format paired_rows
python scripts/01b_build_prompt_preferences.py --model_name gpt2-small --input data/processed/sycophancy_pairs.csv --sample_size 40
python scripts/02_cache_activations.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --sample_size 40
python scripts/03_train_probe.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --top_k 3 --max_examples 20
python scripts/06_generate_report.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final
pytest
```

### Then Pythia-410M (same flow, sample_size 100)

## 6. Success criteria

1. Existing `paired_rows` smoke test still works (response-aware classification).
2. `prompt_preferences` dataset has exactly one row per prompt.
3. `behavior_margin` computed from full-continuation **mean** logprob.
4. Regression probe trains without invalid duplicate-opposite labels.
5. Report includes regression metrics (Pearson/Spearman/R2), not only AUROC.
6. Causal validation reports `behavior_margin_delta`, not only probe delta.
7. `docs/run_log.md` and `results/report.md` updated.
8. `pytest` passes.

## 7. Scientific honesty note

With only ~40 synthetic prompts and a 12-layer GPT-2, regression correlations may be weak. That is acceptable and will be reported as-is. The deliverable is a **structurally valid** experiment: the probe now predicts a real model-behavior target instead of an impossible duplicate-label target. Correlations are expected to be stronger on Pythia-410M and with more prompts.
