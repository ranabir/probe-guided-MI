# Final Status v3 — Probe-Guided Attribution for Sycophancy

**Date:** 2026-06-14

## What changed in v3

| Area | Change |
|------|--------|
| Activation patching | `src/patching.py`: `patch_selected_layers_from_reference`, `compute_patched_behavior_margin`, `run_patching_behavior_validation` — surgically overwrite the final prompt-token residual at selected layers during the **real forward pass** (TL only). |
| Logit-gradient baseline | New `src/logit_gradient.py` — logit-margin gradient attribution with careful token handling (single-token preferred, multi-token logged, token IDs saved). |
| Causal validation | `scripts/05` gained `--intervention {mean_ablation,activation_patching}` (default patching for TL) and a `logit_gradient` method, so probe-gradient / random / logit-gradient are compared on both probe prediction and real behavior. |
| Balanced diagnostic | New `scripts/01c_build_balanced_preference_set.py` + `src/data.build_balanced_subset`. |
| Plots gallery | New `src/plotting.py` + `scripts/07_generate_plots.py` → `plots/{model}/` (6 plots each) + `plots/comparison/` (+ `summary_table.md`) + auto `plots/README.md`. Old `results/figures/` preserved (compatibility layer). |
| Pythia-410M | Full prompt-preference chain run (24 layers). |
| Tests | +4 files (balanced set, logit gradient, activation patching, plot generation). **71 passing.** |

## Exact commands run

```bash
# GPT-2 (300 Anthropic prompts) — 01/01b/02/03 already done; v3 re-ran:
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 50
python scripts/05_causal_validation.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching
python scripts/01c_build_balanced_preference_set.py --model_name gpt2-small --input data/processed/gpt2-small_prompt_preferences.csv
python scripts/07_generate_plots.py --model_name gpt2-small

# Pythia-410M (bash run_pythia.sh → logs/pythia_410m_run.log)
python scripts/01b_build_prompt_preferences.py --model_name EleutherAI/pythia-410m --input data/processed/sycophancy_pairs.csv --sample_size 300
python scripts/02_cache_activations.py --model_name EleutherAI/pythia-410m --input_format prompt_preferences --probe_position prompt_final --sample_size 300
python scripts/03_train_probe.py --model_name EleutherAI/pythia-410m --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin
python scripts/04_probe_gradient_attribution.py --model_name EleutherAI/pythia-410m --input_format prompt_preferences --probe_position prompt_final --max_examples 50
python scripts/05_causal_validation.py --model_name EleutherAI/pythia-410m --input_format prompt_preferences --probe_position prompt_final --top_k 5 --max_examples 50 --intervention activation_patching
python scripts/07_generate_plots.py --model_name EleutherAI/pythia-410m

python scripts/07_generate_plots.py --comparison
python scripts/06_generate_report.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final
pytest   # 71 passed
```

## What worked

- **Cross-model decoding replication (the strong result).** A linear probe on the final
  prompt-token residual predicts the model's `behavior_margin` with test Pearson that **rises
  into the upper layers in both models**:
  - GPT-2 (12 layers): 0.24 (L0) → **0.48 (L11)**
  - Pythia-410M (24 layers): 0.26 (L0) → peak **0.44 (L8)**, sustained 0.36–0.44 through upper layers
  This shape replicating across two independent model families is the headline finding.
- Activation patching, logit-gradient baseline, balanced set, plots, Pythia run — all implemented and run.
- On **probe prediction**, probe-gradient layers dominate (GPT-2: −0.148 vs random −0.089 vs logit 0.000).

## What did not work / remains unproven

- **Causal control over real behavior is not established.** Under activation patching,
  `behavior_margin_delta` for probe-gradient layers does **not** exceed random in either model
  (GPT-2: +0.026 vs random +0.050; Pythia: +0.0003 vs random +0.041). 
- `probe_delta` is a weak causal proxy: mean-ablating *cached* activations only moves the probe
  when its own layer is in the ablated set (hence Pythia's 0.0000). The honest causal metric is
  `behavior_margin_delta` from forward-pass patching.
- Strong class imbalance: only ~7% of prompts prefer sycophancy (GPT-2), low margin variance.

## Scientific interpretation (no overclaim)

> The method identifies **decodable** sycophancy-preference representations whose linear
> readability increases with depth, and this pattern **replicates across GPT-2 and Pythia-410M**.
> However, **causal control over behavior remains unproven under this intervention**: ablating or
> patching the probe-attributed layers does not shift the model's real logprob preference more
> than random layers do.

## Key plots to show first in the demo

1. `plots/comparison/model_probe_regression_comparison.png` — the cross-model decoding story (lead with this).
2. `plots/gpt2-small/gpt2-small_probe_regression_by_layer.png` — the clean per-layer rise.
3. `plots/gpt2-small/gpt2-small_behavior_margin_distribution.png` — honest data context (imbalance).
4. `plots/gpt2-small/gpt2-small_causal_behavior_margin_delta.png` — the honest null on causation.

## Is this ready to send back to the reviewer?

**Yes, as an honest interim result.** The infrastructure is complete, correct, and cross-model;
the decoding finding is real and replicated. It should be framed as *"we can read sycophancy
preference from prompt-final activations across models; establishing causation is the open
problem"* — not as a solved causal story.

## Remaining limitations / next steps

- Causation: try directional (contrastive opposite-class) patching, denoising patching, and
  head-level patching rather than mean-reference patching of one position.
- Data: Anthropic responses are short ("(A) Agree"); richer continuations + more prompts.
- Balanced-variant downstream runs (`--dataset_variant balanced`) are **not yet threaded** through
  02/03/04/05 — the balanced set is built and inspectable but not yet used end-to-end (documented gap).
- Streamlit gallery + selectors (model / variant / intervention / method).
