# Probe-Guided Attribution for Sycophancy in Language Models

> **TL;DR** — We train linear probes to read a language model's *sycophancy preference* from its
> activations, attribute it to layers, and then try to *causally control* it. Sycophancy is
> decodable (and the signal replicates across GPT-2 and Pythia); instruct-tuned models are more
> sycophantic; but the most *decodable* layers are not the most *causal* ones, and reliable,
> capability-preserving control is **not yet achieved**. Full illustrated write-up:
> [`docs/PROJECT_STORY.html`](docs/PROJECT_STORY.html).

## Repository structure

```
scripts/        numbered pipeline stages 01 → 14 (+ make_story.py)   ← run in order
src/            library modules (adapters, probes, patching, controls, directions, …)
run/            shell runners; run/run_full_pipeline.sh is the canonical end-to-end run
tests/          pytest suite (unit + no-model smoke tests)
config.yaml     experiment defaults        model_registry.yaml   per-model backend/dtype/stage
data/           side_effect_eval prompts committed; processed CSVs are regenerated
artifacts/      attribution CSVs committed; activations (.pt) & probes (.pkl) are regenerated
results/        tables (CSV) + report-ready figures
plots/          presentation gallery (per-model + comparison) with plots/README.md
docs/           plans, results, run logs, and the illustrated report (see docs/README.md)
```

Quickstart: `pip install -r requirements.txt` then `bash run/run_full_pipeline.sh gpt2-small 300`.
Run order and every stage's I/O are documented in [`run/README.md`](run/README.md).

## Motivation

Standard mechanistic interpretability techniques attribute model behavior by taking gradients of a single output token logit — for example, the logit for the word "Yes." But for complex behaviors like sycophancy, the signal that matters is not a single token but a behavioral pattern: does the model agree with false claims rather than correcting them? This project replaces the logit target with a **trained behavioral probe** and backpropagates through the probe score to identify which model components are causally responsible for sycophantic behavior. The result is a more behaviorally grounded attribution method that generalizes across model families.

## Research Question

> Which layers and components of a language model encode the decision to be sycophantic, and can probe-guided gradient attribution find them more reliably than random or single-logit baselines?

## Method

```
Prompt + Sycophantic / Non-Sycophantic Response pair
                    │
         ┌──────────▼──────────┐
         │  Model Forward Pass  │  (GPT-2 / Pythia / Qwen / Gemma)
         │  hidden states h_l   │
         └──────────┬──────────┘
                    │  [N × n_layers × d_model]
         ┌──────────▼──────────┐
         │   Linear Probe      │  trained on h_l → sycophancy label
         │   P(sycophantic)    │
         └──────────┬──────────┘
                    │  backprop
         ┌──────────▼──────────┐
         │  Attribution Score  │  |∂P/∂h_l × h_l| per layer
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │ Causal Validation   │  ablate top-k layers → probe score drops?
         │  vs. random-k       │  probe-gradient >> random → causal evidence
         └─────────────────────┘
```

## Supported Model Families

| Family | Example Models | Backend |
|--------|---------------|---------|
| GPT-2 | `gpt2-small` | TransformerLens |
| Pythia | `EleutherAI/pythia-410m`, `pythia-1b` | TransformerLens |
| Qwen | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct` | HuggingFace |
| Gemma | `google/gemma-2-2b-it` | HuggingFace |

## Dataset Strategy

The project uses a tiered dataset hierarchy. Each tier serves a specific purpose:

| Tier | Dataset | Purpose | Status |
|------|---------|---------|--------|
| 1 | **Synthetic** | Smoke-test — verifies the pipeline runs end-to-end. No download needed. | ✅ Always available |
| 2 | **Anthropic model-written-evals** | Main demo — real sycophancy examples written by Claude. Three subsets covering philosophy, NLP, and politics. | ✅ Available via HuggingFace |
| 3 | **TruthfulQA** | Generalization — tests if attribution works for misconception-following vs correction | ✅ Available via HuggingFace |
| 4 | **BBQ** | Safety/bias extension | 🔜 Future iteration |
| 5 | **ETHICS / Social-Chem-101** | Morality/norm-sensitive extension | 🔜 Future iteration |

### Dataset Schema

All datasets are saved in two forms:

**Paired schema** (`sycophancy_pairs.csv`) — one row per prompt, used for behavior metrics:
```
id, prompt, sycophantic_response, non_sycophantic_response, label, source_dataset, subset
```

**Flat schema** (`train/val/test.csv`) — one row per response, used by the activation-caching pipeline:
```
id, prompt, response, label, category, source_dataset, subset
```

### Dataset Commands

```bash
# Smoke test (always works, no internet):
python scripts/01_prepare_dataset.py --dataset synthetic --sample_size 80

# Main demo (Anthropic sycophancy evals):
python scripts/01_prepare_dataset.py --dataset anthropic_sycophancy --sample_size 300

# Generalization (TruthfulQA):
python scripts/01_prepare_dataset.py --dataset truthfulqa --sample_size 200
```

### Anthropic Model-Written Sycophancy Evals

Source: `Anthropic/model-written-evals` on HuggingFace  
Subsets used:
- `sycophancy_on_philpapers2020` — philosophy survey agreement traps
- `sycophancy_on_nlp_survey` — NLP researcher opinion agreement traps  
- `sycophancy_on_political_typology_quiz` — political opinion agreement traps

Each example gives a prompt where a user states their view, and two responses:
- `answer_matching_behavior`: the sycophantic answer (agrees with the user)
- `answer_not_matching_behavior`: the honest answer (gives the correct view)

This is a much stronger dataset than synthetic data for a research pitch because:
1. It was designed by Anthropic specifically to elicit sycophancy
2. It covers realistic scenarios (expert opinion, political views)
3. It has thousands of examples per subset (~10k each)
4. It has been used in published sycophancy research

### Why This Hierarchy Matters for the Pitch

The synthetic dataset proves the pipeline works. The Anthropic dataset proves it works on real, published sycophancy benchmarks. TruthfulQA shows the method generalizes beyond curated sycophancy pairs to naturally occurring misconceptions. Together they make a credible cross-dataset story.

## Correct Experiment Design (after the `prompt_final` diagnosis)

An early experiment exposed a subtle but fatal flaw, now fixed.

### The bug: paired-row `prompt_final` classification is invalid

In paired-row format each prompt appears twice:

| row | input | label |
|-----|-------|-------|
| 1 | prompt + sycophantic_response | 1 |
| 2 | prompt + non_sycophantic_response | 0 |

In `prompt_final` mode the probe only sees the **prompt**, extracting the hidden state at the
last prompt token. The response is never fed in. So **both rows produce the identical activation
vector but carry opposite labels**. A classifier on `{(h, 1), (h, 0)}` is mathematically ill-posed —
expected AUROC is 0.5. The low AUROC we observed (~0.22) was the honest consequence of an invalid
target, not a code bug.

### The fix: prompt-level preference regression

Collapse to **one row per prompt** and make the target a property of the *model's behavior*:

```
syc_logprob     = mean-token logprob(sycophantic_response | prompt)
non_syc_logprob = mean-token logprob(non_sycophantic_response | prompt)
behavior_margin = syc_logprob - non_syc_logprob
prefers_sycophancy = 1 if behavior_margin > 0 else 0
```

Now `h(prompt) → behavior_margin` is a well-posed **regression** (one unique target per prompt). The
scientific question becomes the right one: *do the model's prompt-final activations encode whether it
is about to prefer a sycophantic continuation?*

### Two modes

| Mode | Input | Target | Role |
|------|-------|--------|------|
| **Response-aware** (`--input_format paired_rows --probe_position response_final`) | prompt + response | row label (classification) | **Sanity check** — verifies probe/attribution/ablation infra on a *visible* completion. Not "about to be sycophantic." |
| **Prompt-preference** (`--input_format prompt_preferences --probe_position prompt_final`) | prompt only | `behavior_margin` (regression) | **Main experiment** — the structurally valid claim. |

### Running the corrected pipeline

```bash
# Sanity check (response-aware classification)
python scripts/01_prepare_dataset.py --dataset synthetic --synthetic_only --sample_size 80 --output_format paired_rows
python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 80 --input_format paired_rows --probe_position response_final
python scripts/03_train_probe.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --probe_type classification --probe_target label
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --input_format paired_rows --probe_position response_final --top_k 3 --max_examples 20

# Main experiment (prompt-preference regression)
python scripts/01b_build_prompt_preferences.py --model_name gpt2-small --input data/processed/sycophancy_pairs.csv --sample_size 40
python scripts/02_cache_activations.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --sample_size 40
python scripts/03_train_probe.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --probe_type regression --probe_target behavior_margin
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --top_k 3 --max_examples 20
python scripts/06_generate_report.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final
```

> **Honesty note:** with only ~40 synthetic prompts on a 12-layer GPT-2, regression correlations are
> weak and noisy. That is reported as-is. The deliverable is a **structurally valid** experiment, not
> a forced result; correlations are expected to strengthen on Pythia-410M and with more prompts.

## Robustness Iteration: Controls, Baselines, and Causal Sweeps

A central concern in probing work is that *decodable ≠ causal*, and that base models may not be where
sycophancy lives. This iteration adds controls, baselines, instruct models, a full layerwise causal
sweep, A/B metrics, and error bars. See `docs/review_iteration_1_changes.md` and
`docs/review_iteration_1_final_status.md` for the full write-up.

**Why each addition matters:**
- **Instruct models** (`08_stage_comparison.py`): sycophancy is plausibly a *post-training* trait.
  We find Qwen2.5-0.5B-Instruct is more sycophantic (66%) than base GPT-2/Pythia (~50%).
- **Controls** (`11_run_controls.py`, `src/controls.py`): a random-label probe (noise floor),
  surface-feature probes, and a topic probe test whether the sycophancy probe is a confound.
- **Difference-of-means direction** (`09_direction_comparison.py`, `src/directions.py`): a simple
  contrastive direction is a strong baseline for learned probe directions under steering.
- **Layerwise decodability vs causal sweep** (`10_layerwise_causal_sweep.py`): every layer, not just
  the best-decoding one — revealing decodability and causal effect are *negatively* correlated.
- **Answer-flip rate + accuracy change** (`src/metrics.py`): easier to interpret than raw
  behavior_margin — what fraction of A/B answers actually changed.
- **Bootstrap error bars** (`src/statistics.py`): 95% CIs on every causal estimate.

> **Important correctness fix found during this iteration:** Anthropic prompts are long (~170
> tokens); the old `max_length=128` truncated answers and zeroed 88% of behavior margins. Fixed to
> `max_length=256`; sycophancy is ~50% (not the artifactual 7%) and decodability is higher
> (GPT-2 0.52, Pythia 0.61).

```bash
# controls / directions / layerwise sweep / stage comparison
python scripts/11_run_controls.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --controls random_label static_token topic
python scripts/09_direction_comparison.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 20
python scripts/10_layerwise_causal_sweep.py --model_name gpt2-small --input_format prompt_preferences --probe_position prompt_final --max_examples 30 --bootstrap 1000
python scripts/08_stage_comparison.py --models gpt2-small EleutherAI/pythia-410m Qwen/Qwen2.5-0.5B-Instruct
```

## Causal-Control Iteration

The prior iteration showed sycophancy is **decodable** but that decodability and causal effect are
**anti-correlated across layers** — the most readable layers are not the most causal. This iteration
tries to convert readable representations into interventions that **causally change the answer**, and
checks whether any such intervention also harms general capabilities.

- **Intervene at high-causal layers** (`scripts/12_contrastive_causal.py`): layers are selected from
  the layerwise sweep by *causal effect* (not decodability), compared against decodable-topk and
  random.
- **Stronger interventions:** contrastive patching (patch the final prompt-token residual with an
  *opposite-preference* reference prompt's residual), additive **probe steering**, and **activation
  capping** (clip the residual's projection onto the probe direction), with **mean ablation** as a
  baseline.
- **Answer-flip rate** is the headline metric (did the model's A/B choice actually change?), not just
  behavior_margin.
- **Side-effect evaluation** (`scripts/13_side_effect_eval.py`): generate on basic prompts with/without
  the intervention and measure weirdness, repetition, length, and basic-QA drop — so we can tell clean
  control from "the model just broke."
- **Intervention search** (`scripts/14_causal_intervention_search.py`): grid over layer × direction ×
  intervention × strength, ranked by `objective = targeted_flip − λ·side_effect`, with a Pareto plot.

> **Honest finding:** targeting high-causal layers with strong steering *can* flip up to ~83% of
> sycophantic answers (where decodable-layer targeting flips ~0), but only at strengths that make the
> model weird (side-effect score ~0.66); activation capping stays clean but flips nothing. Reliable,
> capability-preserving causal control is **not yet achieved** in GPT-2. See
> `docs/causal_control_iteration_results.md`.

```bash
python scripts/12_contrastive_causal.py --model_name gpt2-small --layer_selection causal_topk decodable_topk random --top_k_layers 3 --intervention contrastive_patching probe_steering activation_capping mean_ablation --alphas -5 -3 -1 1 3 5 --max_examples 25 --bootstrap 300
python scripts/13_side_effect_eval.py --model_name gpt2-small --num_prompts 30
python scripts/14_causal_intervention_search.py --model_name gpt2-small --max_examples 25 --side_effect_prompts 10 --lambda_side_effect 0.5
python scripts/07_generate_plots.py --model_name gpt2-small
```

## Setup

```bash
# Clone the repo and enter the directory
cd probe-guided\ sycophancy\ attrib

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Optional: install TransformerLens for GPT-2 / Pythia**
```bash
pip install transformer_lens
```

## Step-by-Step Execution

### Smoke test (GPT-2 small, synthetic data, ~5 min)

```bash
python scripts/01_prepare_dataset.py --synthetic_only --sample_size 80
python scripts/02_cache_activations.py --model_name gpt2-small --sample_size 80
python scripts/03_train_probe.py --model_name gpt2-small
python scripts/04_probe_gradient_attribution.py --model_name gpt2-small --max_examples 20
python scripts/05_causal_validation.py --model_name gpt2-small --top_k 3 --max_examples 20
python scripts/06_generate_report.py --model_name gpt2-small
```

### Main demo (Pythia-410M)

```bash
python scripts/01_prepare_dataset.py --synthetic_only --sample_size 300
python scripts/02_cache_activations.py --model_name EleutherAI/pythia-410m --sample_size 300
python scripts/03_train_probe.py --model_name EleutherAI/pythia-410m
python scripts/04_probe_gradient_attribution.py --model_name EleutherAI/pythia-410m --max_examples 50
python scripts/05_causal_validation.py --model_name EleutherAI/pythia-410m --top_k 5 --max_examples 50
python scripts/06_generate_report.py --model_name EleutherAI/pythia-410m
```

### Qwen extension

```bash
python scripts/02_cache_activations.py --model_name Qwen/Qwen2.5-0.5B-Instruct --sample_size 200
python scripts/03_train_probe.py --model_name Qwen/Qwen2.5-0.5B-Instruct
python scripts/04_probe_gradient_attribution.py --model_name Qwen/Qwen2.5-0.5B-Instruct --max_examples 30
python scripts/05_causal_validation.py --model_name Qwen/Qwen2.5-0.5B-Instruct --top_k 5 --max_examples 30
python scripts/06_generate_report.py --model_name Qwen/Qwen2.5-0.5B-Instruct
```

### Gemma extension

```bash
python scripts/02_cache_activations.py --model_name google/gemma-2-2b-it --sample_size 200
python scripts/03_train_probe.py --model_name google/gemma-2-2b-it
python scripts/04_probe_gradient_attribution.py --model_name google/gemma-2-2b-it --max_examples 30
python scripts/05_causal_validation.py --model_name google/gemma-2-2b-it --top_k 5 --max_examples 30
python scripts/06_generate_report.py --model_name google/gemma-2-2b-it
```

### Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Repository Structure

```
probe-guided-attribution/
├── README.md
├── requirements.txt
├── config.yaml              # Default experiment config
├── model_registry.yaml      # Per-model backend/dtype/template settings
├── scripts/
│   ├── 01_prepare_dataset.py
│   ├── 02_cache_activations.py
│   ├── 03_train_probe.py
│   ├── 04_probe_gradient_attribution.py
│   ├── 05_causal_validation.py
│   └── 06_generate_report.py
├── src/
│   ├── data.py              # Dataset loading + synthetic generation
│   ├── model_registry.py    # Registry lookup with fallback heuristics
│   ├── model_loader.py      # Factory: pick TransformerLens or HF adapter
│   ├── model_adapters.py    # BaseModelAdapter, TransformerLensAdapter, HFAdapter
│   ├── activation_cache.py  # Save/load .pt activation caches
│   ├── hooks.py             # TL hook helpers
│   ├── probes.py            # LinearProbe training + evaluation
│   ├── attribution.py       # Grad-norm, grad×act, pair-diff attribution
│   ├── patching.py          # Mean ablation + causal validation
│   ├── metrics.py           # Accuracy / AUROC / F1
│   ├── visualization.py     # Matplotlib figures
│   └── utils.py             # Config loading, path helpers, device selection
├── app/
│   └── streamlit_app.py
├── data/processed/          # train.csv / val.csv / test.csv
├── artifacts/               # activations/*.pt  probes/*.pkl  attribution/*.csv
├── results/figures/         # PNG plots
├── results/tables/          # CSV metric tables
├── results/report.md        # Auto-generated report
└── tests/
    ├── test_data.py
    ├── test_model_loading.py
    ├── test_probe.py
    └── test_smoke_pipeline.py
```

## Config

Edit `config.yaml` to change defaults, or pass CLI flags per script. Key flags:

| Flag | Description |
|------|-------------|
| `--model_name` | Model to use (see `model_registry.yaml` for valid names) |
| `--sample_size` | Number of examples to use |
| `--device` | `auto` / `cpu` / `cuda` / `mps` |
| `--dtype` | `auto` / `float32` / `float16` / `bfloat16` |
| `--max_examples` | Cap examples for gradient / validation steps |
| `--top_k` | Number of top layers to ablate in validation |

## Citation

If you use this code, please cite:

```
@misc{probe-guided-sycophancy-2026,
  title  = {Probe-Guided Attribution for Sycophancy in Language Models},
  year   = {2026},
  note   = {Research demo. https://github.com/your-repo}
}
```
