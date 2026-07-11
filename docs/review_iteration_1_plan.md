# reviewer feedback — Implementation Plan

## 1. Existing repo state (what works today)

- 9 scripts (`01`, `01b`, `01c`, `02`–`07`) + full `src/` (16 modules) + Streamlit app + 71 passing tests.
- **Main experiment (valid):** `--input_format prompt_preferences --probe_position prompt_final`,
  regression probe on `behavior_margin = mean-token logP(syc|prompt) − logP(non_syc|prompt)`.
- **Results on 300 Anthropic prompts (3 subsets pooled ~evenly):**
  - GPT-2 (12L): test Pearson 0.24 → **0.48** (L11). Pythia-410M (24L): 0.26 → **0.44** (L8).
  - Causal (activation patching): probe-gradient does **not** beat random on real `behavior_margin_delta`
    (GPT-2 +0.026 vs +0.050; Pythia +0.000 vs +0.041). Decodable but causally unproven.
  - Sycophancy rare: only **7%** of GPT-2 prompts prefer syc.
- Artifact naming: `{safe_model_name}_{tag}_...` where `tag = artifact_tag(input_format, probe_position)`
  (e.g. `prompt_preferences_prompt_final`). Plots in `plots/{safe_model_name}/` + `plots/comparison/`.
- HF cache has Qwen2.5-0.5B/1.5B-Instruct and gemma-2-2b-it. No OLMo locally.

## 2. Feedback → change mapping

| # | reviewer feedback | Change |
|---|---------------|--------|
| 1 | Test instruct/post-trained models | Run Qwen2.5-0.5B-Instruct (in cache) through the pipeline; HF adapter already exists. Document Gemma/1.5B attempts. |
| 2,3 | Pretrain vs post-train; OLMo stages | `scripts/08_stage_comparison.py` + registry placeholders for OLMo base/SFT/DPO/instruct (TODO IDs). |
| 4,5 | Surface-feature confounds; controls | `src/controls.py` + `scripts/11_run_controls.py`: random-label, static-feature, topic probes. |
| 6 | Difference-of-means vs probe directions | `src/directions.py` + `scripts/09_direction_comparison.py`. |
| 7,8 | Causal claim; sweep all layers | `scripts/10_layerwise_causal_sweep.py` — decodability AND causal effect per layer. |
| 9 | Answer-flip + accuracy in A/B terms | `answer_flip_rate`, `targeted_syc_to_non_syc_flip_rate`, `accuracy_change` in `src/metrics.py`/`behavior_metrics.py`; wired into `05`, `09`, `10`. |
| 10 | Error bars / bootstrap CI | `src/statistics.py` (`bootstrap_mean_ci`, `bootstrap_metric_ci`); CIs on key plots. |

## 3. New scripts
- `scripts/08_stage_comparison.py` — cross-stage/family summary + 3 comparison plots.
- `scripts/09_direction_comparison.py` — regression vs logistic vs diff-of-means vs random directions, alpha sweep.
- `scripts/10_layerwise_causal_sweep.py` — per-layer decodability + causal effect + bootstrap CI.
- `scripts/11_run_controls.py` — random-label / static-feature / topic controls.

## 4. New modules
- `src/controls.py` — control-probe construction + training.
- `src/directions.py` — probe / logistic / diff-of-means / margin-weighted directions + patching.
- `src/statistics.py` — bootstrap CIs.
- (extend) `src/metrics.py`, `src/behavior_metrics.py` — answer-flip + accuracy.
- (extend) `src/plotting.py` — control comparison, decodability-vs-causal, layerwise flip, direction comparison, stage plots, error bars.

## 5. Existing scripts modified
- `scripts/05_causal_validation.py` — `--metrics`, `--bootstrap`, `--method`, new output columns (answer_flip, accuracy_change, CIs, model_family, training_stage).
- `scripts/07_generate_plots.py` — new required plots + control/direction/layerwise/stage plots.
- `scripts/06_generate_report.py` — the reviewer-iteration sections.
- `model_registry.yaml` — `model_family` + `training_stage` fields; OLMo placeholders.
- `src/model_loader.py`, `src/model_adapters.py` — graceful HF fallback messaging (already mostly present).
- `app/streamlit_app.py` — controls / directions / layer-sweep / stage panels.

## 6. New result tables
```
results/tables/{sn}_control_probe_metrics.csv
results/tables/{sn}_direction_comparison.csv
results/tables/{sn}_layerwise_decodability_causal_sweep.csv
results/tables/stage_comparison_summary.csv
```
Plus expanded `results/tables/{sn}_{tag}_causal_validation.csv` with flip/accuracy/CI columns.

## 7. New plots
```
plots/{sn}/{sn}_probe_vs_controls_by_layer.png
plots/{sn}/{sn}_decodability_vs_causal_effect.png
plots/{sn}/{sn}_layerwise_answer_flip_rate.png
plots/{sn}/{sn}_direction_comparison.png  (+ _behavior_delta, _answer_flip)
plots/comparison/stage_sycophancy_rate.png
plots/comparison/stage_probe_decodability.png
plots/comparison/stage_causal_control.png
plots/comparison/method_comparison_with_error_bars.png
```

## 8. Docs
- `docs/review_iteration_1_plan.md` (this file).
- `docs/review_iteration_1_changes.md` — feedback → code/result/plot map + limitations.
- `docs/review_iteration_1_final_status.md` — readiness, key plots, honest claims, next step.
- Update `README.md` ("reviewer feedback iteration"), `docs/run_log.md`, `plots/README.md`.

## 9. Execution order
1. Verify base pipeline (gpt2 artifacts exist → reuse).
2. `src/statistics.py`, metrics extensions, `src/controls.py`, `src/directions.py` (+ tests).
3. `scripts/11_run_controls.py` on gpt2.
4. `scripts/10_layerwise_causal_sweep.py` on gpt2.
5. `scripts/09_direction_comparison.py` on gpt2.
6. Update `05` (flip/accuracy/bootstrap) + re-run gpt2.
7. Instruct model: Qwen2.5-0.5B-Instruct (01b→03 + controls). Document Gemma/1.5B.
8. OLMo registry scaffolding; `scripts/08_stage_comparison.py` across available models.
9. Plotting + `07`; comparison plots.
10. Docs + `06`; `pytest`.

## 10. Success criteria (= ready to send back to the reviewer)
1. Base pipeline still works; 2. ≥1 instruct model run + documented; 3. random-label control;
4. static-feature control; 5. diff-of-means direction compared to probe; 6. layerwise decodability
+ causal sweep; 7. answer-flip + accuracy reported; 8. bootstrap CIs on main plots;
9. `plots/README.md` documents all plots; 10. `docs/review_iteration_1_changes.md` maps each comment;
11. `docs/review_iteration_1_final_status.md` states readiness; 12. pytest green or failures documented.

## 11. Scope realism (honest)
- **Run now:** gpt2 controls/directions/layer-sweep/flip-metrics/bootstrap; Qwen-0.5B-Instruct pipeline + controls; stage comparison across gpt2/pythia/qwen.
- **Scaffold + document pending:** OLMo (no local IDs), HF activation-patching steering (TL-only for now), Gemma/1.5B if slow/OOM on MPS.
- Honesty rules from spec section O are adopted verbatim in the final status doc.
