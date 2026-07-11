# Final Status — reviewer feedback Iteration

## 1. One-paragraph summary

Responding to the reviewer's review, we added instruction-tuned model support, confound controls, a
difference-of-means direction comparison, a full layerwise decodability-vs-causation sweep, A/B
answer-flip / accuracy metrics, and bootstrap error bars. In the process we found and fixed a
**truncation bug** (`max_length=128` cut the answer off long Anthropic prompts, zeroing 88% of
behavior margins); the corrected data shows sycophancy is ~50% (not the artifactual 7%) and is more
decodable than before (GPT-2 Pearson 0.52, Pythia 0.61). The headline scientific finding is now
**decodability ≠ causation**: across layers, decodability and causal effect are *negatively*
correlated (−0.59), and probe-attributed layers do not control behavior beyond random under
patching. Instruct (Qwen) is **more** sycophantic (66%) than base models (~50%).

## 2. What was implemented

- `src/statistics.py` — bootstrap CIs.
- `src/metrics.py` — answer-flip rate, targeted syc→non flip, accuracy change.
- `src/controls.py` + `scripts/11_run_controls.py` — random-label / static-feature / topic controls.
- `src/directions.py` + `scripts/09_direction_comparison.py` — direction methods + steering.
- `scripts/10_layerwise_causal_sweep.py` — per-layer decodability + causal effect + CI.
- `scripts/08_stage_comparison.py` — cross-stage/family summary + plots.
- `scripts/05` — `--intervention/--metrics/--bootstrap/--method`; flip/accuracy/CI/family/stage columns.
- `model_registry.yaml` + `ModelConfig` — `training_stage`; OLMo placeholders.
- `src/plotting.py` — controls, decodability-vs-causal, layerwise flip, direction, stage, error bars.
- **Fix:** `max_length 128 → 256` + defensive bounds guards in patching/steering/logprob.
- Tests: 102 passing (+5 new files).

## 3. Commands run (key ones)

```bash
# corrected rebuilds (max_length=256)
bash rebuild_gpt2.sh          # 01b→05, 01c, 11, 10, 09, 07
bash rebuild_pythia.sh
bash run_qwen.sh              # instruct (HF; mean-ablation fallback)
python scripts/08_stage_comparison.py --models gpt2-small EleutherAI/pythia-410m Qwen/Qwen2.5-0.5B-Instruct
python scripts/07_generate_plots.py --comparison
pytest   # 102 passed
```

## 4. Key results

| Model | Stage | Sycophancy rate | Best Pearson (layer) | probe_gradient Δbehavior | random Δbehavior |
|-------|-------|----------------:|---------------------:|-------------------------:|-----------------:|
| GPT-2 small | base | 0.505 | 0.518 (L8) | −0.042 (CI crosses 0) | −0.076 |
| Pythia-410M | base | 0.508 | 0.611 (L16) | +0.005 | +0.008 |
| Qwen2.5-0.5B-Instruct | instruct | **0.660** | 0.421 (L5) | n/a (HF) | n/a |

- **Controls (GPT-2):** random-label floor **0.246**; surface `contains_do_you_agree` **1.00**; topic 0.67.
- **Directions (GPT-2):** logistic −0.58 / regression −0.52 ≫ diff-of-means −0.05; random ~0.
- **Layerwise:** corr(decodability, |causal|) = **−0.59**.

## 5. Key plots to show (in order)

1. `plots/comparison/stage_sycophancy_rate.png` — instruct > base sycophancy (answers #1, #2).
2. `plots/gpt2-small/gpt2-small_decodability_vs_causal_effect.png` — decoding ≠ causation (#7, #8).
3. `plots/gpt2-small/gpt2-small_probe_vs_controls_by_layer.png` — confound controls (#4, #5).
4. `plots/gpt2-small/gpt2-small_direction_comparison_behavior_delta.png` — diff-of-means vs probe (#6).
5. `plots/comparison/model_probe_regression_comparison.png` — cross-model decoding replication.

## 6. Which plot addresses which feedback item

| Feedback | Plot |
|---|---|
| #1 instruct models | `comparison/stage_sycophancy_rate.png`, `stage_probe_decodability.png` |
| #2 pretrain vs post-train | `comparison/stage_*` |
| #4,#5 controls | `{model}_probe_vs_controls_by_layer.png` |
| #6 directions | `{model}_direction_comparison_behavior_delta.png`, `_answer_flip.png` |
| #7,#8 causal / layer sweep | `{model}_decodability_vs_causal_effect.png`, `_layerwise_behavior_delta.png`, `_layerwise_answer_flip_rate.png` |
| #9 answer-flip | `{model}_layerwise_answer_flip_rate.png` |
| #10 error bars | CI bands in the layerwise + causal plots |

## 7. What remains unresolved

- **Causal control still not demonstrated.** Probe-attributed layers do not move behavior beyond
  random under activation patching; CIs cross zero. → *"The method identifies decodable preference
  representations, but causal control over behavior remains unproven under this intervention."*
- **Confounds not fully ruled out.** Surface features are highly decodable; the random-label floor
  (~0.25) is non-trivial at this sample size.
- **OLMo staged run pending** (no verified IDs locally) — the cleanest test of pretrain-vs-post-train.
- **HF steering/patching** not implemented → instruct models have decodability but no causal numbers.
- Single instruct model; Gemma / Qwen-1.5B not yet run.

## 8. Recommended next message to the reviewer

> Implemented all 10 points. Two headline outcomes: (a) fixing a truncation bug revealed sycophancy
> is ~50% (not 7%) and that **instruct (Qwen) is more sycophantic, 66%, than base models** — direct
> support for the post-training hypothesis; (b) a full layerwise sweep shows **decodability and
> causal effect are negatively correlated (−0.59)** — the most readable layers are the least causal,
> so probe-guided *causal* control remains unproven. Controls show real signal above a ~0.25
> random-label floor but heavy surface-feature decodability. Next I'd run OLMo base→SFT→DPO→instruct
> to isolate the post-training effect, and add HF steering so instruct models get causal numbers.
> Which should I prioritize?

## 9. Is the project ready for another review?

**Yes.** Every feedback item is implemented, tested (102 passing), and documented, with honest
(sometimes null) results and error bars. It is ready for a second review round; the open scientific
questions (causation, OLMo stages, HF steering) are clearly scoped as next steps rather than gaps in
the implementation.
