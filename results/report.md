# Probe-Guided Attribution for Sycophancy
**Model:** `EleutherAI/pythia-410m`

**Generated:** 2026-07-10 21:30:53

---
## 0. Project Summary
We replace single-token logit attribution with a **trained behavioral probe** and backpropagate through the probe to rank model layers, then validate causally by ablation. This report contains two experiments: a response-aware **sanity check** and the main prompt-only **preference attribution**.

| Property | Value |
|---|---|
| Model | `EleutherAI/pythia-410m` |
| Backend | transformer_lens |
| Family | pythia |
| Layers | 24 |
| d_model | 1024 |

---
## 1. Response-Aware Sanity Check
**Setup:** input = `prompt + response`, target = row label (1=sycophantic completion). Probe position = `response_final`.

**Purpose:** verify the probe + attribution + ablation infrastructure works when the sycophantic signal is *visible* in the input.

**Limitation:** because the response text is in the input, this measures *recognition of a visible completion*, not whether the model is *about to be* sycophantic. It is a sanity check, not the scientific claim.

### Per-layer probe metrics (classification)
*(not found: EleutherAI_pythia-410m_response_final_layer_probe_metrics.csv — run that pipeline variant)*

*(figure not found: layer_probe_accuracy)*

### Causal validation
*(not found: EleutherAI_pythia-410m_response_final_causal_validation.csv — run that pipeline variant)*

*(figure not found: causal_validation_barplot)*

---
## 2. Prompt-Only Preference Attribution  (MAIN EXPERIMENT)
**Setup:** input = `prompt` only, target = the model's own `behavior_margin` = mean-token logprob(sycophantic | prompt) − mean-token logprob(non_sycophantic | prompt). Probe position = `prompt_final`, probe type = **regression**.

**Why this is the correct design:** in paired-row format the same prompt yields two rows with identical prompt-only activations but opposite labels, making label classification mathematically invalid. Regressing on the per-prompt `behavior_margin` gives one well-posed target per prompt and asks the real question: *do prompt-final activations encode whether the model is about to prefer a sycophantic continuation?*

### Behavior margin distribution
- Prompts: 299
- behavior_margin: mean=-0.0028, median=0.0625, std=0.7575
- prefers_sycophancy (margin>0): 152/299 (frac=0.508)

Attribution probe layer: **15** (val_pearson=0.5930), type=regression, target=behavior_margin

### Per-layer probe metrics (regression: Pearson / Spearman / R²)
|   layer |   train_mse |   train_mae |   train_r2 |   train_pearson |   train_spearman |   val_mse |   val_mae |   val_r2 |   val_pearson |   val_spearman |   test_mse |   test_mae |   test_r2 |   test_pearson |   test_spearman |
|--------:|------------:|------------:|-----------:|----------------:|-----------------:|----------:|----------:|---------:|--------------:|---------------:|-----------:|-----------:|----------:|---------------:|----------------:|
|  0.0000 |      0.0167 |      0.0986 |     0.9689 |          0.9886 |           0.9698 |    1.3280 |    0.9504 |  -1.0093 |        0.3139 |         0.3787 |     0.8330 |     0.7706 |   -0.3006 |         0.3741 |          0.3459 |
|  1.0000 |      0.0033 |      0.0458 |     0.9938 |          0.9977 |           0.9855 |    0.7235 |    0.6435 |  -0.0948 |        0.5640 |         0.5531 |     0.7381 |     0.6955 |   -0.1525 |         0.3742 |          0.3437 |
|  2.0000 |      0.0023 |      0.0376 |     0.9957 |          0.9984 |           0.9872 |    0.7786 |    0.7139 |  -0.1780 |        0.5034 |         0.4582 |     0.7118 |     0.6906 |   -0.1115 |         0.4547 |          0.3590 |
|  3.0000 |      0.0018 |      0.0315 |     0.9967 |          0.9987 |           0.9879 |    0.8847 |    0.7376 |  -0.3385 |        0.4525 |         0.4529 |     0.6473 |     0.6303 |   -0.0108 |         0.4782 |          0.3764 |
|  4.0000 |      0.0012 |      0.0258 |     0.9979 |          0.9992 |           0.9904 |    1.2783 |    0.8635 |  -0.9342 |        0.1268 |         0.1402 |     0.8130 |     0.7274 |   -0.2695 |         0.3652 |          0.2949 |
|  5.0000 |      0.0013 |      0.0267 |     0.9976 |          0.9991 |           0.9900 |    0.7774 |    0.7063 |  -0.1762 |        0.4041 |         0.3335 |     0.7682 |     0.6908 |   -0.1995 |         0.4268 |          0.2687 |
|  6.0000 |      0.0014 |      0.0289 |     0.9974 |          0.9990 |           0.9891 |    1.1727 |    0.7893 |  -0.7744 |        0.2769 |         0.2729 |     0.7734 |     0.7180 |   -0.2076 |         0.3170 |          0.2423 |
|  7.0000 |      0.0014 |      0.0281 |     0.9974 |          0.9990 |           0.9890 |    1.0871 |    0.7946 |  -0.6449 |        0.2780 |         0.2646 |     0.6632 |     0.6104 |   -0.0356 |         0.4393 |          0.4424 |
|  8.0000 |      0.0007 |      0.0209 |     0.9986 |          0.9995 |           0.9944 |    0.8083 |    0.7095 |  -0.2229 |        0.4036 |         0.3882 |     0.5748 |     0.5672 |    0.1025 |         0.5682 |          0.4480 |
|  9.0000 |      0.0008 |      0.0212 |     0.9986 |          0.9995 |           0.9940 |    0.8509 |    0.7232 |  -0.2875 |        0.3576 |         0.3545 |     0.4549 |     0.5042 |    0.2898 |         0.5852 |          0.5655 |
| 10.0000 |      0.0005 |      0.0179 |     0.9990 |          0.9996 |           0.9958 |    0.8204 |    0.7098 |  -0.2413 |        0.3252 |         0.3384 |     0.4770 |     0.5278 |    0.2552 |         0.5816 |          0.5264 |
| 11.0000 |      0.0006 |      0.0197 |     0.9988 |          0.9995 |           0.9937 |    0.7245 |    0.6482 |  -0.0962 |        0.4951 |         0.3921 |     0.5121 |     0.5793 |    0.2003 |         0.5608 |          0.5129 |
| 12.0000 |      0.0005 |      0.0166 |     0.9990 |          0.9996 |           0.9949 |    0.8278 |    0.6881 |  -0.2526 |        0.4547 |         0.4320 |     0.5926 |     0.6236 |    0.0747 |         0.5267 |          0.5193 |
| 13.0000 |      0.0007 |      0.0191 |     0.9988 |          0.9995 |           0.9943 |    0.6808 |    0.5924 |  -0.0302 |        0.5983 |         0.6068 |     0.5401 |     0.6040 |    0.1567 |         0.5207 |          0.5229 |
| 14.0000 |      0.0006 |      0.0186 |     0.9989 |          0.9996 |           0.9940 |    0.6194 |    0.5529 |   0.0628 |        0.5865 |         0.5460 |     0.4756 |     0.5580 |    0.2574 |         0.5838 |          0.5857 |
| 15.0000 |      0.0006 |      0.0179 |     0.9989 |          0.9996 |           0.9938 |    0.5979 |    0.5634 |   0.0954 |        0.5930 |         0.5539 |     0.5309 |     0.5694 |    0.1710 |         0.5117 |          0.4906 |
| 16.0000 |      0.0009 |      0.0213 |     0.9984 |          0.9994 |           0.9922 |    0.5441 |    0.5641 |   0.1767 |        0.5637 |         0.5388 |     0.4207 |     0.5048 |    0.3432 |         0.6109 |          0.5603 |
| 17.0000 |      0.0007 |      0.0192 |     0.9987 |          0.9995 |           0.9938 |    0.5535 |    0.5884 |   0.1624 |        0.5371 |         0.5085 |     0.4377 |     0.5132 |    0.3166 |         0.5970 |          0.5080 |
| 18.0000 |      0.0007 |      0.0192 |     0.9988 |          0.9995 |           0.9939 |    0.6453 |    0.6161 |   0.0236 |        0.4916 |         0.4124 |     0.4609 |     0.5359 |    0.2803 |         0.5660 |          0.4823 |
| 19.0000 |      0.0007 |      0.0194 |     0.9987 |          0.9995 |           0.9933 |    0.7888 |    0.6839 |  -0.1934 |        0.4038 |         0.3502 |     0.4984 |     0.5734 |    0.2218 |         0.5439 |          0.5001 |
| 20.0000 |      0.0005 |      0.0165 |     0.9991 |          0.9997 |           0.9955 |    0.8749 |    0.7403 |  -0.3238 |        0.3215 |         0.2677 |     0.5209 |     0.5741 |    0.1866 |         0.5446 |          0.4441 |
| 21.0000 |      0.0004 |      0.0143 |     0.9993 |          0.9998 |           0.9962 |    0.7956 |    0.6856 |  -0.2038 |        0.3847 |         0.3631 |     0.5347 |     0.5876 |    0.1651 |         0.5329 |          0.4646 |
| 22.0000 |      0.0003 |      0.0124 |     0.9995 |          0.9998 |           0.9976 |    0.7745 |    0.6915 |  -0.1719 |        0.3899 |         0.3851 |     0.4673 |     0.5591 |    0.2703 |         0.5607 |          0.5146 |
| 23.0000 |      0.0002 |      0.0096 |     0.9997 |          0.9999 |           0.9986 |    0.7905 |    0.6853 |  -0.1961 |        0.3776 |         0.4576 |     0.5054 |     0.5672 |    0.2108 |         0.5228 |          0.4657 |

![layer_probe_accuracy](/Users/rono/Projects/probe-guided sycophancy attrib/results/figures/EleutherAI_pythia-410m_prompt_preferences_prompt_final_layer_probe_accuracy.png)

### Layer attribution
![layer_attribution_barplot](/Users/rono/Projects/probe-guided sycophancy attrib/results/figures/EleutherAI_pythia-410m_prompt_preferences_prompt_final_layer_attribution_barplot.png)

### Causal validation (probe prediction AND real behavior margin)
| model_name             | model_family   | training_stage   | input_format       | probe_position   | method         | intervention_type   |   k |   before_probe_prediction |   after_probe_prediction |   probe_delta |   before_behavior_margin |   after_behavior_margin |   behavior_margin_delta |   answer_flip_rate |   targeted_syc_to_non_syc_flip_rate |   before_accuracy |   after_accuracy |   accuracy_change |   bootstrap_ci_low |   bootstrap_ci_high |   n_examples |
|:-----------------------|:---------------|:-----------------|:-------------------|:-----------------|:---------------|:--------------------|----:|--------------------------:|-------------------------:|--------------:|-------------------------:|------------------------:|------------------------:|-------------------:|------------------------------------:|------------------:|-----------------:|------------------:|-------------------:|--------------------:|-------------:|
| EleutherAI/pythia-410m | pythia         | base             | prompt_preferences | prompt_final     | probe_gradient | activation_patching |   5 |                   -0.1252 |                   0.0155 |        0.1407 |                   0.1443 |                  0.1497 |                  0.0053 |             0.0000 |                              0.0000 |            0.3333 |           0.3333 |            0.0000 |            -0.0317 |              0.0388 |           15 |
| EleutherAI/pythia-410m | pythia         | base             | prompt_preferences | prompt_final     | random         | activation_patching |   5 |                   -0.1252 |                  -0.1252 |        0.0000 |                   0.1443 |                  0.1522 |                  0.0079 |             0.0000 |                              0.0000 |            0.3333 |           0.3333 |            0.0000 |            -0.0131 |              0.0285 |           15 |
| EleutherAI/pythia-410m | pythia         | base             | prompt_preferences | prompt_final     | logit_gradient | activation_patching |   5 |                   -0.1252 |                   0.0155 |        0.1407 |                   0.1443 |                  0.1497 |                  0.0053 |             0.0000 |                              0.0000 |            0.3333 |           0.3333 |            0.0000 |            -0.0317 |              0.0388 |           15 |

**Columns:** `probe_delta` = change in predicted margin after ablation; `behavior_margin_delta` = change in the model's *real* logprob preference under hook-based ablation (TransformerLens only; NaN for HF models).

![causal_sweep](/Users/rono/Projects/probe-guided sycophancy attrib/results/figures/EleutherAI_pythia-410m_prompt_preferences_prompt_final_causal_sweep.png)

---
## 2b. Presentation Plots
High-DPI gallery in `plots/EleutherAI_pythia-410m/` (see `plots/README.md` for how to read each). Regenerate with `python scripts/07_generate_plots.py --model_name EleutherAI/pythia-410m`.

- **Probe regression by layer** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_probe_regression_by_layer.png`
- **Behavior margin distribution** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_behavior_margin_distribution.png`
- **Probe-gradient attribution** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_probe_gradient_layer_attribution.png`
- **Causal: probe prediction delta** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_causal_probe_delta.png`
- **Causal: real behavior margin delta** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_causal_behavior_margin_delta.png`
- **Top-k intervention sweep** — `plots/EleutherAI_pythia-410m/EleutherAI_pythia-410m_topk_sweep_behavior_margin.png`

Cross-model comparison: `plots/comparison/` and `plots/comparison/summary_table.md`.

---
## 3. Limitations
- Synthetic / small datasets → high-variance correlation estimates.
- Length-normalized logprob reduces but does not eliminate length artifacts.
- Layer-level mean ablation is coarse (no head-level resolution yet).
- Behavior-margin ablation is implemented for TransformerLens; HF (Qwen/Gemma) is pending.
- If regression correlations are weak, that is reported honestly — the main win is a **structurally valid** experiment, not a forced result.

## 4. Next Steps
- Scale prompts (hundreds+) and run Pythia-410M for stronger correlation estimates.
- Head-level attribution via `blocks.{i}.attn.hook_result`.
- Activation patching with clean/non-syc runs instead of mean ablation.
- HF hook-based behavior-margin ablation for Qwen/Gemma.
