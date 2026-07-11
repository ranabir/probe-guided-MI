# Documentation index

Read in this order for the full research narrative.

## Start here
| Doc | What it is |
|-----|-----------|
| [`PROJECT_STORY.html`](PROJECT_STORY.html) | **Self-contained illustrated research report** (open in a browser). The best single overview. |
| [`PROJECT_PITCH.md`](PROJECT_PITCH.md) | One-page summary with the key plots and claims. |
| [`../README.md`](../README.md) | Repository README: setup, dataset strategy, full run sequence. |

## Method & design (chronological)
| Doc | What it is |
|-----|-----------|
| [`implementation_plan_v2.md`](implementation_plan_v2.md) | Original build plan (probe pipeline, adapters). |
| [`prompt_preference_fix_plan.md`](prompt_preference_fix_plan.md) | The `prompt_final` diagnosis and the regression-target fix. |
| [`next_experiment_plan_v3.md`](next_experiment_plan_v3.md) | Plan for activation patching, baselines, plots. |
| [`final_status_v3.md`](final_status_v3.md) | Status after the cross-model decoding result. |

## Review iteration 1 — controls, baselines, instruct models
| Doc | What it is |
|-----|-----------|
| [`review_iteration_1_plan.md`](review_iteration_1_plan.md) | Plan responding to reviewer feedback. |
| [`review_iteration_1_changes.md`](review_iteration_1_changes.md) | Each reviewer comment → code/result/plot map. |
| [`review_iteration_1_final_status.md`](review_iteration_1_final_status.md) | Readiness, key plots, honest claims. |

## Causal-control iteration — the current frontier
| Doc | What it is |
|-----|-----------|
| [`causal_control_iteration_plan.md`](causal_control_iteration_plan.md) | Plan: intervene at high-causal layers; contrastive patching; capping; side-effect eval. |
| [`causal_control_iteration_results.md`](causal_control_iteration_results.md) | Full results, plots, and honest interpretation. |
| [`final_status_causal_control_iteration.md`](final_status_causal_control_iteration.md) | What worked, what failed, next experiment. |
| [`side_effect_eval_notes.md`](side_effect_eval_notes.md) | Raw before/after generation samples. |

## Logs
| Doc | What it is |
|-----|-----------|
| [`run_log.md`](run_log.md) | Cumulative run log across all iterations. |
| [`RUN_LOG_V2.md`](RUN_LOG_V2.md) | Clean, structured V2.0 end-to-end run (single reproducible pass). |
