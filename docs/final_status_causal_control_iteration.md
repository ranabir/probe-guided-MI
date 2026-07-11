# Final Status — Causal-Control Iteration

## What changed
- **New modules:** `src/causal_interventions.py`, `src/side_effects.py`, `src/search.py`.
- **New scripts:** `scripts/12_contrastive_causal.py`, `scripts/13_side_effect_eval.py`,
  `scripts/14_causal_intervention_search.py`.
- **New data:** `data/side_effect_eval/basic_prompts.jsonl` (30 basic prompts).
- **Extended:** `src/plotting.py` (+6 plot fns), `scripts/07_generate_plots.py` (wires new plots),
  `README.md`, `docs/run_log.md`, `plots/README.md`.
- **Tests:** +4 files (`test_causal_interventions`, `test_side_effects`, `test_intervention_search`,
  `test_plots_causal_control`). **131 passing.**
- Existing pipeline untouched; HF models skip TL-only interventions with clean pending markers.

## Commands run
See `docs/causal_control_iteration_results.md` §4. GPT-2 full; Qwen → HF pending markers.

## Key metrics (GPT-2 small)
- Low-strength interventions: targeted flip ≤ 0.08 for all selections (causal ≈ random; no control).
- Best search config: **causal_topk + diff_of_means steering, α=−3 → targeted flip 0.83**, but
  **side-effect 0.66** (weird/repetitive). Activation capping: side-effect ~0.04 but flip ~0.

## Key plots
1. `plots/gpt2-small/gpt2-small_intervention_search_pareto.png` — control vs side-effect trade-off (no clean top-left point).
2. `plots/gpt2-small/gpt2-small_contrastive_causal_answer_flip.png` — causal vs decodable vs random flips.
3. `plots/gpt2-small/gpt2-small_side_effect_summary.png` — the disruption cost.
4. `plots/gpt2-small/gpt2-small_causal_vs_decodable_layer_intervention.png` — selected layers vs the curves.

## What worked
- Targeting **high-causal layers** with stronger steering *does* produce large answer-flips (0.83),
  which high-decodable targeting does not — confirming the layerwise-sweep insight is actionable.
- The full instrumentation (contrastive patching, capping, side-effect eval, search, Pareto) works
  and gives honest, quantified trade-offs.

## What failed
- No **clean** lever: the flip-producing intervention disrupts the model; the clean intervention
  (capping) doesn't flip answers. Reliable, capability-preserving causal control is **not achieved**
  in GPT-2 at these settings.

## Is this ready to update the reviewer?
**Yes** — it directly answers the causal question with an honest, well-instrumented negative-leaning
result, plus the side-effect check he asked for.

## Suggested concise message to the reviewer
> Implemented the causal-control push. Targeting **high-causal** layers (from the layerwise sweep)
> with stronger steering *does* produce large answer-flips (targeted syc→honest up to **0.83**) where
> high-decodable-layer targeting gives ~0 — so the "intervene where it's causal, not where it's
> readable" idea is actionable. **But** those flips come with heavy side effects (side-effect score
> 0.66: the model gets repetitive/weird), while **activation capping** stays clean (≈0.04) but flips
> nothing. So: answer-level control is reachable but not *cleanly* — current single-direction
> interventions trade control against coherence. Added a side-effect eval and a Pareto search to make
> this trade-off explicit. Next: test on a larger instruct model (where sycophancy is a genuine
> post-trained behavior and may be a cleaner lever), and consider multi-direction / subspace
> interventions since sycophancy may not be one direction. Worth a look?

## Next experiment
- Implement **HF residual hooks** so instruct models (Qwen/Gemma) get causal numbers.
- **Multi-direction / subspace** capping (sycophancy may not be a single direction — the crux the reviewer flagged).
- Tune capping strength upward and steering strength to find any capability-preserving operating point.
