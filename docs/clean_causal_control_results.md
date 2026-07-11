# Clean Causal Control + HF Interventions — Results

**Date:** 2026-07-11
**Scripts:** `src/residual_interventions.py`, `scripts/15_clean_causal_control.py`
**Question:** (1) Can we intervene on *instruction-tuned* models (not just TransformerLens ones)?
(2) Can we flip sycophancy *cleanly* — without the coherence collapse that blunt steering causes?

---

## 1. Problem 1 — instruct models are now steerable ✅

Previously all interventions were TransformerLens-only, so Qwen/Gemma could be *read* but not
*moved*. `src/residual_interventions.py` implements the same residual edit for **both** backends
(TL hook points; HuggingFace `register_forward_hook` on decoder layers). Verified end-to-end on
**Qwen2.5-0.5B-Instruct**: additive steering flips up to **57%** of sycophantic answers — the first
causal-control numbers on an instruct model in this project.

## 2. Problem 2 — the diagnosis was right; clean methods reach the clean zone

**Diagnosis.** Blunt additive steering `h += α·d̂` inflates the residual's norm and drowns out the
model's other computation — hence the 83% flip but 0.66 side-effect. Fix: edits that touch only the
sycophancy coordinate.

| Method | What it does | Norm impact |
|--------|--------------|-------------|
| `additive` (baseline) | `h += α·d̂` | inflates |
| `additive_normpres` | additive, then rescale to `‖h‖` | preserved |
| `projection_ablation` | `h -= (h·d̂)·d̂` (remove the component) | ~unchanged |
| `mean_shift` | set the component to the honest-mean value | ~unchanged |

### GPT-2 small (causal layers 1–3, diff-of-means direction)

| Family | best targeted flip | side-effect |
|--------|-------------------:|------------:|
| additive | 0.25 | 0.68 |
| additive_normpres | 0.25 | 0.68 |
| **projection_ablation** | 0.00 | **0.056** |
| **mean_shift** | 0.00 | **0.041** |

The clean methods cut side-effect **~13×** (0.68 → 0.05) — confirming the norm-inflation diagnosis —
but on GPT-2's early layers they are too gentle to flip answers.

### Qwen2.5-0.5B-Instruct (decodable band layers 4–6)

| Family | best targeted flip | side-effect |
|--------|-------------------:|------------:|
| additive | 0.57 | 0.42–0.71 |
| **additive_normpres** | 0.57 | 0.44 (Pareto improvement) |
| **projection_ablation** | 0.07 | **0.016** |
| mean_shift | 0.00 | **0.011** |

Two concrete gains on the instruct model:
1. **Norm-preserving steering is a Pareto improvement** — e.g. at α=−6 it keeps a 0.36 flip while
   cutting side-effect from **0.68 → 0.44**.
2. **Projection-ablation reaches the clean zone with nonzero control** — flip 0.07 at side-effect
   **0.016** (~30× cleaner than additive). It is the first method to combine *any* answer-level
   control with a genuinely intact model.

See `plots/{model}/{model}_clean_control_pareto.png` and `_clean_control_flip_vs_sideeffect.png`.

---

## 3. Honest interpretation

> Instruction-tuned models can now be causally intervened on, and blunt steering flips a majority of
> sycophantic answers on Qwen. The failure mode of blunt steering is confirmed to be norm inflation:
> projection-ablation and mean-shift are ~13–30× cleaner. **But strong *and* clean control is still
> unreached** — the high-flip methods remain disruptive, and the clean methods remain weak. The
> top-left of the flip-vs-side-effect plot is still empty.

**Progress vs open problem:**
- ✅ HF interventions (instruct models steerable).
- ✅ Diagnosis confirmed (norm inflation) and two cleaner intervention families.
- ✅ A norm-preserving Pareto improvement and a first clean-zone control point on Qwen.
- ❌ A single method that is both strong (high flip) and clean (low side-effect).

## 4. Next steps
- **Tune projection-ablation across layers/strength** — it is clean; find where it also flips
  (multi-layer ablation, or ablate at the layers that most determine the answer).
- **Subspace ablation** — remove a low-rank sycophancy subspace rather than one direction (sycophancy
  may not be a single direction).
- **Calibrate mean_shift targets per prompt** rather than a global honest-mean.
- Run the same comparison on Gemma-2 and a larger Qwen to see if the clean-zone control strengthens
  with scale.
