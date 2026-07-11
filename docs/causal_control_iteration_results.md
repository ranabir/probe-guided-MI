# Causal-Control Iteration — Results

## 1. Motivation
Prior iteration: sycophancy preference is **decodable** but probe-attributed layers gave no causal
control, and decodability vs causal effect were **anti-correlated** across layers.

> **Note on the correlation magnitude (honesty).** The robustness-iteration sweep reported
> r ≈ −0.59; the causal-control iteration re-ran the layerwise sweep with fewer examples (n=30 for
> speed) and gets a weaker, noisier r ≈ −0.07. The linear correlation is clearly **sample-size
> sensitive**, so we do not lean on its exact value. What is robust across both runs is the
> qualitative claim used throughout: the **peak-decodable layer (≈8) is not the peak-causal layer
> (≈1)** — reading and moving the behavior live in different places. Reviewer
guidance: convert readable representations into interventions that *causally* change behavior, try
intervening at high-causal (not high-decodable) layers and stronger interventions (contrastive
patching, activation capping), and check **side effects**.

## 2. Hypothesis
Intervening at **high-causal** layers (from the layerwise sweep) with **stronger interventions**
gives better **answer-level** control (targeted syc→honest flips) than intervening at
high-decodable or random layers — and we must verify any control doesn't just break the model.

## 3. Methods
- `select_layers_from_sweep`: causal_topk (|behavior_margin_delta|), decodable_topk (Pearson), random.
- Interventions: **contrastive_patching** (patch the final prompt-token residual with an
  opposite-preference reference prompt's residual), **probe_steering** (h += α·dir), **activation
  capping** (clip projection onto the probe direction above a train-quantile threshold),
  **mean_ablation** (baseline). All TransformerLens; HF skipped with a pending marker.
- Headline metric: **targeted syc→honest answer-flip rate** + behavior_margin_delta with 95% bootstrap CI.
- Side-effect eval: greedy generation on 30 basic prompts, with/without intervention; score from
  weirdness, length ratio, repetition increase, basic-QA drop.
- Intervention search: grid over layer-selection × direction × intervention × strength, ranked by
  `objective = targeted_flip − λ·side_effect_score`.

## 4. Commands run
```
python scripts/12_contrastive_causal.py --model_name gpt2-small --layer_selection causal_topk decodable_topk random --top_k_layers 3 --intervention contrastive_patching probe_steering activation_capping mean_ablation --alphas -5 -3 -1 1 3 5 --max_examples 25 --bootstrap 300
python scripts/13_side_effect_eval.py --model_name gpt2-small --num_prompts 20 --max_new_tokens 25
python scripts/14_causal_intervention_search.py --model_name gpt2-small --directions regression diff_of_means --interventions probe_steering activation_capping --alphas -3 -1 1 3 --cap_quantiles 0.5 0.75 0.9 --max_examples 25 --side_effect_prompts 10 --max_new_tokens 15 --bootstrap 200 --lambda_side_effect 0.5
python scripts/07_generate_plots.py --model_name gpt2-small
# Qwen (HF): TL-only interventions → graceful pending marker
```

## 5. Results (GPT-2 small)

### Contrastive-causal (step 12), best targeted flip per selection
| Layer selection | best targeted flip | via | Δmargin |
|---|---:|---|---:|
| causal_topk | 0.083 | probe_steering α=+1 | +0.009 |
| decodable_topk | 0.000 | contrastive_patching | −0.024 |
| random | 0.083 | probe_steering α=+1 | +0.007 |

At low strength, **no selection gives meaningful answer-level control**; flips are ≤0.08 and
indistinguishable between causal_topk and random. Contrastive patching and capping barely move flips.

### Intervention search (step 14), top config
| selection | direction | intervention | strength | targeted flip | side-effect | objective |
|---|---|---|---:|---:|---:|---:|
| **causal_topk** | diff_of_means | probe_steering | α=−3 | **0.833** | **0.665** | 0.501 |

At **higher steering strength on causal layers**, targeted flips jump to 0.83 — but side-effect
score is 0.66 (the model becomes weird/repetitive). Activation capping is clean (side-effect ≈0.04)
but flips ≈0.

### Side-effect eval (step 13) of the step-12 "best" (α=+1 causal_topk)
side_effect_score **0.661**, weirdness_rate **1.0**, output_length_ratio **1.46**, qa_accuracy_drop **0.125**.
(GPT-2 base already generates oddly; strong steering amplifies it.)

## 6. Plots and how to read them
- `gpt2-small_contrastive_causal_answer_flip.png` — targeted flip by selection×intervention (causal vs decodable vs random).
- `gpt2-small_contrastive_causal_behavior_delta.png` — Δmargin with 95% CI.
- `gpt2-small_causal_vs_decodable_layer_intervention.png` — which layers each selection targets vs the decodability/causal curves.
- `gpt2-small_side_effect_summary.png` — capability harm of the chosen intervention.
- `gpt2-small_intervention_search_pareto.png` — control vs side-effect trade-off (top-left = good).
- `gpt2-small_best_interventions_ranked.png` — top configs by objective.

## 7. Did causal_topk beat decodable_topk?
**Partially.** At low strength, no. At higher steering strength (search), the single best config is
**causal_topk + diff_of_means steering (flip 0.83)**, and the best decodable_topk configs gave ~0
flips — so causal-layer targeting *can* produce strong answer-level effects where decodable-layer
targeting does not. But it only does so at strengths that incur large side effects.

## 8. Did answer-flip improve?
Yes at high strength (0.0 → 0.83 targeted), but **not cleanly** — the flips come with side-effect
score 0.66.

## 9. Did side effects appear?
**Yes.** The flip-producing intervention makes GPT-2 weird (weirdness_rate 1.0, length 1.46×).
Activation capping avoids side effects but produces no flips.

## 10. Is reliable causal control demonstrated?
**No — useful negative-leaning result.** *"Intervention produces answer-level movement only at
strengths that also disrupt the model; behavior changes may reflect general disruption, not clean
sycophancy control. Activation capping is cleaner but ineffective at these settings."*

## 11. What remains unresolved
- A clean, capability-preserving lever (high flip, low side-effect) was not found in GPT-2 at these
  settings. The Pareto front is poor (no top-left point).
- HF (Qwen/Gemma) steering/patching not implemented → instruct models have no causal numbers.
- Sycophancy may be multi-directional; single-direction steering/capping may be inherently limited.
- Larger instruct models (where sycophancy is a real post-trained behavior) may respond more cleanly
  and are the most promising next target.
