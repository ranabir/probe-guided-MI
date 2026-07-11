# Implementation Plan v2 — Probe-Guided Attribution Upgrade

## What Currently Works

- Full 6-step pipeline end-to-end on gpt2-small with 80 synthetic examples
- Artifact paths: `{safe_model_name}_{split}_activations.pt` (no probe_position)
- Best probe selected by best val AUROC across all layers (picks layer 3 for GPT-2)
- Causal validation: mean ablation on cached hidden states → probe score drop
- Streamlit app loads gpt2-small artifacts
- 18 unit tests passing

Key result: probe-gradient ablation drops sycophancy score 0.922 → 0.307; random ablation has zero effect.

## Scientific Weaknesses

1. **Probe sees response text.** The current `final` token position is the last token of the full "prompt + response" string. The probe may learn stylistic cues ("Yes", "Absolutely") rather than behavioral intent.

2. **Causal validation only measures probe score, not actual model output.** Dropping the probe score doesn't prove the model generates different text — it only proves we broke the probe's input.

3. **Probe is at layer 3 of 12.** Gradients at layers 4–11 are exactly zero (mathematically expected). For a 12-layer model this is OK, but for a demo it looks weak. Using a later-layer probe gives richer attribution coverage.

4. **No Pythia-410M results.** Only gpt2-small is run.

5. **No behavior-level validation.** The logprob margin (P(sycophantic response | prompt) vs P(honest response | prompt)) is not measured before or after intervention.

## Files Modified (never deleted)

| File | Change |
|------|--------|
| `config.yaml` | Add `probe_position`, `probe_layer_policy`, `min_probe_layer_frac` |
| `src/activation_cache.py` | Add `probe_position` param to all path/save/load functions; backward-compat fallback |
| `src/probes.py` | Add `probe_position` to paths; add `select_attribution_probe()` for policy |
| `src/model_adapters.py` | Add `compute_logprob()` to base and both adapters |
| `scripts/02_cache_activations.py` | Add `--probe_position`; for `prompt_final`, pass only prompt text |
| `scripts/03_train_probe.py` | Add `--probe_position`, `--probe_layer_policy`; save selection metadata |
| `scripts/04_probe_gradient_attribution.py` | Add `--probe_position`; use policy-selected probe; update artifact names |
| `scripts/05_causal_validation.py` | Add `--probe_position`; add behavior margin computation and reporting |
| `scripts/06_generate_report.py` | Major upgrade: add behavior margin section, probe position, limitations |
| `app/streamlit_app.py` | Add probe_position selector; behavior margin panel; handle legacy artifacts |
| `tests/test_smoke_pipeline.py` | Add `prompt_final` path check |

## New Files Added

| File | Purpose |
|------|---------|
| `src/behavior_metrics.py` | compute_continuation_logprob, compute_pair_logprob_margin, compute_behavior_margins_for_dataset |
| `tests/test_behavior_metrics.py` | Unit tests for behavior metrics |
| `tests/test_probe_position.py` | Verify prompt_final vs response_final extraction |
| `tests/test_probe_selection.py` | Verify best_late / best_any / final_layer policies |
| `docs/implementation_plan_v2.md` | This file |
| `docs/run_log.md` | Step-by-step execution log |
| `docs/final_status_v2.md` | Written after all runs complete |

## Artifact Naming Convention

New artifacts include `probe_position` in the name:

```
artifacts/activations/{safe_model_name}_{probe_position}_{split}_activations.pt
artifacts/activations/{safe_model_name}_{probe_position}_metadata.json
artifacts/probes/{safe_model_name}_{probe_position}_probe_layer_{layer}.pkl
artifacts/probes/{safe_model_name}_{probe_position}_best_probe.pkl
artifacts/probes/{safe_model_name}_{probe_position}_selected_probe_metadata.json
artifacts/attribution/{safe_model_name}_{probe_position}_layer_attribution.csv
artifacts/attribution/{safe_model_name}_{probe_position}_component_attribution.csv
artifacts/attribution/{safe_model_name}_{probe_position}_top_layers.txt
artifacts/attribution/{safe_model_name}_{probe_position}_logit_baseline_layers.txt
results/tables/{safe_model_name}_{probe_position}_layer_probe_metrics.csv
results/tables/{safe_model_name}_{probe_position}_causal_validation.csv
results/tables/{safe_model_name}_{probe_position}_causal_sweep.csv
results/tables/{safe_model_name}_{probe_position}_behavior_margins.csv
results/figures/{safe_model_name}_{probe_position}_*.png
```

Legacy (no probe_position): all existing gpt2-small artifacts remain untouched and loadable via fallback.

## probe_position Design

| Mode | Input to model | Token extracted | Label source |
|------|---------------|-----------------|-------------|
| `response_final` | prompt + response | last token of full text | dataset label |
| `prompt_final` | prompt only | last token of prompt | dataset label |

For `prompt_final`: we tokenize only the prompt, run the model, extract the final token's hidden state. The label comes from the original dataset row (1 = sycophantic pair, 0 = honest pair). Across different prompts, the model's internal state at the final prompt token carries varying signals about what it's "set up" to generate. This mode rules out the confound of response-style features.

## Probe Layer Policy

| Policy | Selection rule |
|--------|---------------|
| `best_any` | Highest val AUROC across all layers |
| `final_layer` | Always use the last layer |
| `best_late` | Highest val AUROC among layers >= `min_probe_layer_frac * n_layers` |

Default: `best_late` with `min_probe_layer_frac = 0.65`

For GPT-2 (12 layers): best_late considers layers 8–11. Gradients then flow through layers 0–7 (67% of the network covered).

## Behavior Metrics Design

```
logprob(syc_response | prompt) = sum of log P(token_i | prompt + syc_response[:i])
logprob(non_syc_response | prompt) = sum of log P(token_i | prompt + non[:i])
behavior_margin = logprob(syc) - logprob(non_syc)

If behavior_margin > 0: model prefers sycophantic completion
If behavior_margin < 0: model prefers honest completion
```

Behavior-level ablation (TL models only for MVP):
- Register mean-ablation hooks in the actual forward pass
- Re-compute logprob margins with ablated model
- Compare: does ablating top-k attribution layers reduce the model's preference for sycophantic completions?

## Causal Validation Output (new columns)

```
model_name, probe_position, method, k,
before_probe_score, after_probe_score, probe_delta,
before_behavior_margin, after_behavior_margin, behavior_margin_delta
```

## Commands to Run

### Smoke test (gpt2-small + prompt_final):
```bash
python scripts/01_prepare_dataset.py --synthetic_only --sample_size 80
python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 80 --probe_position prompt_final
python scripts/03_train_probe.py --model_name gpt2-small --probe_position prompt_final
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --probe_position prompt_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --probe_position prompt_final --top_k 3 --max_examples 20
python scripts/06_generate_report.py --model_name gpt2-small --probe_position prompt_final
pytest
```

### Main demo (Pythia-410M + prompt_final):
```bash
python scripts/02_cache_activations.py --model_name EleutherAI/pythia-410m --sample_size 100 --probe_position prompt_final
python scripts/03_train_probe.py --model_name EleutherAI/pythia-410m --probe_position prompt_final
python scripts/04_probe_gradient_attribution.py --model_name EleutherAI/pythia-410m --probe_position prompt_final --max_examples 30
python scripts/05_causal_validation.py --model_name EleutherAI/pythia-410m --probe_position prompt_final --top_k 5 --max_examples 30
python scripts/06_generate_report.py --model_name EleutherAI/pythia-410m --probe_position prompt_final
```

## Success Criteria

- [ ] pytest: all 18 old tests pass + all new tests pass
- [ ] `02_cache_activations.py --probe_position prompt_final` produces `gpt2-small_prompt_final_train_activations.pt`
- [ ] `03_train_probe.py --probe_position prompt_final` saves `gpt2-small_prompt_final_best_probe.pkl` + selection metadata
- [ ] `04_probe_gradient_attribution.py --probe_position prompt_final` shows non-zero attribution for layers before probe layer
- [ ] `05_causal_validation.py --probe_position prompt_final` produces CSV with both probe score AND behavior margin columns
- [ ] `06_generate_report.py` produces upgraded report.md
- [ ] Old gpt2-small artifacts (no probe_position) still loadable by streamlit app
- [ ] Streamlit app shows probe_position selector
- [ ] Pythia-410M run attempted and result documented in run_log.md

---

## Dataset Strategy Update (v2.1)

### Hierarchy
1. **Synthetic** — smoke-test only (always available)
2. **Anthropic model-written-evals** — main demo, ~30k examples across 3 subsets
3. **TruthfulQA** — generalization dataset
4. **BBQ** — optional future (not yet implemented)
5. **ETHICS** — optional future (not yet implemented)

### Schema
- `sycophancy_pairs.csv` — paired format for behavior metrics
- `train/val/test.csv` — flat format for pipeline (backward-compatible)

### Files Changed
- `src/data.py` — full restructure with `load_synthetic_sycophancy()`, `load_anthropic_sycophancy()`, `load_truthfulqa()`, stubs for bbq/ethics, `paired_to_flat()`, `to_paired_schema()`, `load_dataset_by_name()`
- `scripts/01_prepare_dataset.py` — added `--dataset {synthetic,anthropic_sycophancy,truthfulqa,bbq,ethics}`, preserved `--synthetic_only` as legacy alias
- `README.md` — added Dataset Strategy section

### Verified Working
- `--dataset synthetic --sample_size 80` ✅
- `--dataset anthropic_sycophancy --sample_size 100` ✅ (loads from HuggingFace JSONL files)
- `--synthetic_only` legacy flag still works ✅
