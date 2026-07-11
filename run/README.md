# Pipeline runners

Convenience scripts that chain the numbered `scripts/NN_*.py` stages in order.
All write a log to `logs/`.

| Script | What it does |
|--------|--------------|
| `run_full_pipeline.sh [model] [n_prompts]` | **Canonical end-to-end run** for one TransformerLens model (default `gpt2-small`, 300 prompts): stages 01→14 + report + plots. |
| `run_pythia.sh` | Full prompt-preference chain for `EleutherAI/pythia-410m`. |
| `run_qwen.sh` | Prompt-preference chain for `Qwen/Qwen2.5-0.5B-Instruct` (HuggingFace; TL-only interventions skip gracefully). |
| `rebuild_gpt2.sh`, `rebuild_pythia.sh` | Re-cache + retrain a model with the corrected `max_length=256`. |

## Canonical sequence (what `run_full_pipeline.sh` executes)

```
01  prepare_dataset                 → data/processed/*.csv
01b build_prompt_preferences        → {model}_prompt_preferences*.csv   (behavior margins)
01c build_balanced_preference_set   → balanced diagnostic split
02  cache_activations               → artifacts/activations/*.pt
03  train_probe                     → artifacts/probes/*.pkl + layer metrics
04  probe_gradient_attribution      → attribution CSVs + logit-gradient baseline
05  causal_validation               → causal_validation.csv (patching, bootstrap CIs)
11  run_controls                    → control_probe_metrics.csv
10  layerwise_causal_sweep          → decodability vs causal effect per layer
09  direction_comparison            → steering directions comparison
12  contrastive_causal              → causal vs decodable vs random interventions
13  side_effect_eval                → capability side-effects of best intervention
14  causal_intervention_search      → control/side-effect Pareto search
06  generate_report                 → results/report.md
07  generate_plots                  → plots/{model}/*.png + plots/README.md
```

After a model run, build the illustrated report:

```
python scripts/make_story.py        → docs/PROJECT_STORY.html
```
