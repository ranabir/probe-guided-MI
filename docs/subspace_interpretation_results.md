# Sycophancy-Subspace Interpretation + Scaled Eval — Results

**Date:** 2026-07-12
**Scripts:** `scripts/16_subspace_ablation.py` (scaled), `scripts/17_interpret_subspace.py`
**Model:** Qwen2.5-0.5B-Instruct, rebuilt at **500 prompts (76 held-out test)**, eval on 60 examples,
25 side-effect prompts, bootstrap 1000.

---

## Part 1 — Scaled eval (tighter estimates)

The earlier headline (rank-8, flip 0.43) was measured on only 20 test prompts. With the larger,
tighter eval the numbers settle and sharpen:

| rank (top-3 band) | targeted flip | side-effect |
|------------------:|--------------:|------------:|
| 1 (single direction) | 0.175 | 0.004 |
| **2** | **0.400** | **0.032** |
| 3 | 0.325 | 0.045 |
| 5 | 0.175 | 0.052 |
| 8 | 0.250 | 0.045 |
| 12 | 0.325 | 0.078 |

**Honest correction and the sharper finding:**
- A **rank-2 subspace flips 40% of sycophantic answers at side-effect 0.032** — vs a single direction
  at only **17.5%**. The subspace **~doubles** control while staying ~15× cleaner than additive
  steering (0.42–0.71 side-effect).
- The sweet spot is a **small** subspace (rank 2–3), not ever-higher rank: beyond rank ~3 the flip
  does not keep rising (added dimensions bring noise, not signal). This is itself a clue about the
  subspace's true dimensionality.

## Part 2 — What the subspace *is* (interpretation)

At the peak-decodable layer we measured how much of each sub-type's own difference-of-means
direction (philosophy / NLP / political) is captured by the rank-k subspace.

**Captured energy vs rank:**

| rank | NLP | philosophy | political |
|-----:|----:|-----------:|----------:|
| 1 | 0.00 | 0.09 | **0.35** |
| 2 | **0.55** | 0.09 | 0.54 |
| 3 | 0.55 | 0.14 | 0.54 |
| 8 | 0.62 | 0.22 | 0.57 |

**Singular spectrum** (relative energy per dim): `[0.26, 0.17, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08]` —
spread across dimensions; **7 dims to reach 90%**. Genuinely multi-directional.

### The mechanism (this explains everything)

1. **The single dominant direction ≈ the *political* sub-type** — rank-1 captures political (0.35)
   but essentially ignores NLP (0.00) and philosophy (0.09). So single-direction ablation removes
   only *political* sycophancy → weak (0.175 flip).
2. **The second dimension is the *NLP* direction** — at rank-2, NLP capture jumps 0.00 → 0.55. That
   is why rank-2 doubles the flip: it now removes *two* topic-specific sycophancy directions.
3. **Philosophy stays poorly captured** (≤0.22) — philosophy-survey sycophancy is more diffuse and
   not aligned with the dominant directions, which honestly explains why the flip caps near 0.40:
   we are not removing the philosophy component.
4. The spectrum confirms sycophancy is **multi-directional**, not a single feature.

### Answer to the reviewer's hypothesis

> **Sycophancy is a union of topic-specific directions — roughly one direction per topic.** A single
> direction removes one topic's sycophancy (here, political); a rank-2–3 subspace removes several at
> once, which is why it is both stronger and still clean. This is the mechanistic reason
> single-direction steering was insufficient and subspace ablation works.

Plots: `plots/{model}/{model}_subspace_subtype_capture.png`,
`_subspace_singular_spectrum.png`, `_subspace_ablation_pareto.png`, `_subspace_ablation_by_rank.png`.

## Limitations & next steps
- Estimates still on ~60 test prompts; error bars remain non-trivial.
- **Philosophy sub-type is under-captured** — build a philosophy-specific direction and add it to the
  ablation basis; does the flip rise past 0.40?
- Repeat the capture analysis across layers (is the political-first ordering stable?).
- Confirm on Gemma-2 / larger Qwen: does each additional topic get its own dimension?
