# reviewer feedback — Changes Map

Each reviewer comment → what changed in code → which result/plot addresses it → limitations.

> **Headline correctness fix (discovered during this iteration):** the Anthropic prompts are long
> (~170 tokens median, max 236). The pipeline used `max_length=128`, which **truncated the answer
> off the end** and made **88% of behavior margins degenerate zeros**. The earlier "7% sycophancy
> rate (rare)" was an artifact. After fixing to `max_length=256`, sycophancy is ~50% (balanced) and
> all margins are real. Every number below is post-fix.

---

## 1. Test instruction-tuned / post-trained models

- **Code:** `model_registry.yaml` (Qwen/Gemma already present + `training_stage`), HuggingFace adapter
  path exercised end-to-end; `run_qwen.sh`.
- **Result:** Qwen2.5-0.5B-Instruct run (200 prompts). Best test Pearson **0.42 (layer 5)**.
  **Sycophancy rate 66%** vs ~50% for base GPT-2 / Pythia.
- **Plot:** `plots/comparison/stage_sycophancy_rate.png`.
- **Limitation:** patching/steering are TransformerLens-only, so Qwen's causal columns are NaN
  (mean-ablation fallback). Gemma / Qwen-1.5B not yet run (time/memory); pattern is in place.

## 2. Pretraining vs post-training origin of sycophancy

- **Code:** `scripts/08_stage_comparison.py` + `training_stage` in registry/`ModelConfig`.
- **Result:** `results/tables/stage_comparison_summary.csv`. Instruct (Qwen, 66%) > base (GPT-2 50.5%,
  Pythia 50.8%) on **sycophancy rate** — consistent with sycophancy being amplified by post-training.
- **Plot:** `plots/comparison/stage_sycophancy_rate.png`, `stage_probe_decodability.png`,
  `stage_causal_control.png`.
- **Limitation:** one instruct model so far; needs a true within-family base→instruct pair (OLMo) to
  isolate the post-training effect rather than confounding it with model family.

## 3. OLMo staged checkpoints

- **Code:** registry placeholders `olmo-{base,sft,dpo,instruct}-PLACEHOLDER` with `training_stage`
  and TODO `hf_name`s.
- **Status:** **Pending** — no OLMo locally; exact repo IDs must be verified before running. The
  stage-comparison machinery will accept them as-is once IDs are filled in.

## 4 & 5. Surface-feature confounds + controls

- **Code:** `src/controls.py` (`build_static_feature_targets`, `shuffle_target`, `topic_codes`,
  `run_all_controls`) + `scripts/11_run_controls.py`.
- **Result (GPT-2):** sycophancy probe peak **0.518**; controls peak — random-label **0.246**,
  `contains_do_you_agree` **1.00**, topic **0.67**.
  - random-label floor (0.25) is *non-trivial* with only 46 test prompts → the real signal sits
    ~0.27 above noise, not 0.52 above.
  - surface features are trivially decodable (expected) → we cannot claim the probe is *only*
    sycophancy.
- **Plot:** `plots/{model}/{model}_probe_vs_controls_by_layer.png`.
- **Limitation:** controls show what is *also* decodable, not a clean causal disentanglement.

## 6. Difference-of-means vs probe directions

- **Code:** `src/directions.py` (regression / logistic / diff-of-means / margin-weighted / random)
  + `scripts/09_direction_comparison.py` (alpha sweep, norm-scaled steering, TL only).
- **Result (GPT-2, layer 8, α∈[-3,3]):** logistic **−0.58** and regression **−0.52** max |Δmargin|
  vs diff-of-means **−0.05**, random ~0. **Learned probe directions steer better here** — the
  opposite of the confound worry.
- **Plot:** `plots/{model}/{model}_direction_comparison_behavior_delta.png`, `_answer_flip.png`.
- **Limitation:** effects are modest; answer-flip ≤0.05; HF models not steerable yet.

## 7 & 8. Causal claim + sweep ALL layers

- **Code:** `scripts/10_layerwise_causal_sweep.py` (per-layer decodability + per-layer patching with
  bootstrap CI).
- **Result (GPT-2):** **correlation(decodability, |causal effect|) across layers = −0.59.** The
  most decodable layers (8–11) have the *smallest* causal effect; earlier layers move behavior more.
- **Plot:** `plots/{model}/{model}_decodability_vs_causal_effect.png`,
  `_layerwise_behavior_delta.png`, `_layerwise_answer_flip_rate.png`.
- **Takeaway:** decoding ≠ causation, demonstrated directly. This is the strongest methodological
  result of the iteration.

## 9. Answer-flip rate + accuracy change (A/B)

- **Code:** `src/metrics.py` (`answer_flip_rate`, `targeted_syc_to_non_syc_flip_rate`,
  `accuracy_change`, `behavioral_intervention_metrics`); wired into `05`, `09`, `10`.
- **Result:** reported in every causal table. On GPT-2, single-layer/patching answer-flip ≈ 0 —
  honest signal that these interventions rarely change the model's actual A/B choice.
- **Limitation:** flips are rare at these intervention strengths; larger/steering interventions
  (step 09) produce small but non-zero flips.

## 10. Bootstrap error bars

- **Code:** `src/statistics.py` (`bootstrap_mean_ci`, `bootstrap_metric_ci`, `pearson_on_pairs`).
- **Result:** 95% CIs on causal `behavior_margin_delta` (e.g. GPT-2 probe-gradient
  [−0.104, +0.016] — crosses 0) and on the layerwise sweep band.
- **Plot:** CI bands in `*_decodability_vs_causal_effect.png` and `*_layerwise_behavior_delta.png`.

---

## Tests
102 passing (added `test_bootstrap`, `test_answer_flip_metrics`, `test_controls`,
`test_directions`, `test_layerwise_causal_sweep`).
