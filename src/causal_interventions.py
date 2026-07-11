"""Causal interventions targeting high-causal layers (not just high-decodable ones).

The layerwise sweep showed decodability and causal effect are anti-correlated, so this module
selects intervention layers by *causal* effect and compares four interventions:
  - contrastive_patching : patch the final prompt-token residual with an opposite-preference
                           reference prompt's residual (stronger than mean replacement).
  - probe_steering       : add alpha * probe-direction to the residual (reuses src.directions).
  - activation_capping   : clip the residual's projection onto the probe direction above a threshold.
  - mean_ablation        : baseline (reuses src.patching mean reference).

All four reuse the existing TransformerLens patching/steering infrastructure. HuggingFace models
are skipped gracefully (hook-based residual edits are TL-only in this repo).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def select_layers_from_sweep(sweep_df: pd.DataFrame, selection: str, k: int,
                             manual: Optional[List[int]] = None, seed: int = 42) -> List[int]:
    """Pick k layers from the layerwise sweep table by selection strategy.

    causal_topk    -> largest |behavior_margin_delta| (fallback answer_flip_rate)
    decodable_topk -> largest test_pearson (fallback test_spearman)
    random         -> uniform sample
    manual         -> the provided list (truncated to k if longer)
    """
    layers = sorted(sweep_df["layer"].astype(int).unique().tolist())
    if selection == "manual":
        return (manual or [])[:k] if manual else layers[:k]
    if selection == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(layers, size=min(k, len(layers)), replace=False).tolist())
    if selection == "causal_topk":
        col = "behavior_margin_delta" if "behavior_margin_delta" in sweep_df else "answer_flip_rate"
        ranked = sweep_df.assign(_score=sweep_df[col].abs()).sort_values("_score", ascending=False)
    elif selection == "decodable_topk":
        col = "test_pearson" if "test_pearson" in sweep_df else "test_spearman"
        ranked = sweep_df.assign(_score=sweep_df[col].abs()).sort_values("_score", ascending=False)
    else:
        raise ValueError(f"Unknown selection: {selection}")
    return sorted(ranked["layer"].astype(int).head(k).tolist())


# ---------------------------------------------------------------------------
# Contrastive reference selection
# ---------------------------------------------------------------------------

def choose_opposite_preference_reference(pref_df: pd.DataFrame, target_idx: int,
                                         seed: int = 42) -> Optional[int]:
    """Return the row index of a reference prompt with the OPPOSITE preference to the target.

    If target prefers sycophancy (margin>0), reference should be honest-preferring (margin<=0),
    and vice versa. Returns None if no opposite-preference prompt exists.
    """
    target_margin = pref_df.iloc[target_idx]["behavior_margin"]
    if not np.isfinite(target_margin):
        return None
    if target_margin > 0:
        pool = pref_df.index[pref_df["behavior_margin"] <= 0].tolist()
    else:
        pool = pref_df.index[pref_df["behavior_margin"] > 0].tolist()
    pool = [i for i in pool if i != target_idx]
    if not pool:
        return None
    rng = np.random.default_rng(seed + target_idx)
    return int(rng.choice(pool))


# ---------------------------------------------------------------------------
# Reference activation caching (prompt-final residual per layer)
# ---------------------------------------------------------------------------

def cache_prompt_final_residuals(adapter, prompts: List[str], max_length: int = 256) -> np.ndarray:
    """Return [N, L, D] prompt-final hidden states (residual stream) for the given prompts."""
    acts = adapter.forward_with_cache(prompts, max_length=max_length, token_position="final")
    return acts["hidden_states"]


# ---------------------------------------------------------------------------
# Interventions — each returns (before_margins, after_margins) per prompt
# ---------------------------------------------------------------------------

def apply_contrastive_patch(adapter, pairs_df, ref_residuals: np.ndarray, ref_indices: List[int],
                            layers: List[int], max_examples: int = 50, seed: int = 42,
                            max_length: int = 256) -> Tuple[list, list]:
    """Patch each target's selected-layer residuals with an opposite-preference reference's
    residuals, then measure behavior_margin before/after. TL only."""
    from src.patching import compute_patched_behavior_margin
    from src.model_adapters import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        raise NotImplementedError("Contrastive patching is TransformerLens-only.")

    df = pairs_df.reset_index(drop=True)
    idx = list(range(len(df)))
    if len(idx) > max_examples:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(idx, size=max_examples, replace=False).tolist())

    syc_col = "sycophantic_response"
    non_col = "non_sycophantic_response"
    before, after = [], []
    for i in idx:
        ref_i = ref_indices[i]
        if ref_i is None:
            continue
        r = df.iloc[i]
        patch_vectors = {l: ref_residuals[ref_i, l, :] for l in layers}
        try:
            b = compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                [], {}, max_length=max_length)
            a = compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                layers, patch_vectors, max_length=max_length)
            before.append(b); after.append(a)
        except Exception as e:
            logger.warning("contrastive patch failed: %s", e)
    return before, after


def apply_probe_steering(adapter, pairs_df, direction: np.ndarray, layer: int, alpha: float,
                         resid_scale: float, max_examples: int = 50, seed: int = 42,
                         max_length: int = 256) -> Tuple[list, list]:
    """Additive steering h += alpha*resid_scale*direction at one layer (reuses src.directions)."""
    from src.directions import steer_margins_for_prompts
    base = steer_margins_for_prompts(adapter, pairs_df, layer, direction * resid_scale, 0.0,
                                     max_examples=max_examples, seed=seed, max_length=max_length)
    after = steer_margins_for_prompts(adapter, pairs_df, layer, direction * resid_scale, float(alpha),
                                      max_examples=max_examples, seed=seed, max_length=max_length)
    return base, after


def apply_activation_capping(adapter, pairs_df, direction: np.ndarray, layer: int,
                             threshold: float, cap_strength: float = 1.0,
                             max_examples: int = 50, seed: int = 42,
                             max_length: int = 256) -> Tuple[list, list]:
    """Cap the residual's projection onto `direction` above `threshold` at one layer (TL only).

    h_new = h - cap_strength * max(0, proj - threshold) * dir_unit, applied at the final prompt token.
    """
    import torch
    from src.model_adapters import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        raise NotImplementedError("Activation capping is TransformerLens-only.")

    dir_unit = direction / (np.linalg.norm(direction) + 1e-8)
    dvec = torch.tensor(dir_unit, dtype=torch.float32)

    syc_col, non_col = "sycophantic_response", "non_sycophantic_response"
    df = pairs_df.reset_index(drop=True)
    idx = list(range(len(df)))
    if len(idx) > max_examples:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(idx, size=max_examples, replace=False).tolist())

    def _logprob(prompt, continuation, capped):
        full_ids = adapter.tokenizer.encode(prompt + " " + continuation, return_tensors="pt",
                                            truncation=True, max_length=max_length).to(adapter.device)
        seq_len = full_ids.shape[1]
        prompt_len = len(adapter.tokenizer.encode(prompt, add_special_tokens=True))
        if prompt_len >= seq_len:
            return 0.0
        patch_pos = max(min(prompt_len - 1, seq_len - 1), 0)

        def hook_fn(val, hook):
            if not capped:
                return val
            out = val.clone()
            d = dvec.to(val.device).to(val.dtype)
            proj = (out[:, patch_pos, :] * d).sum(-1, keepdim=True)  # [1,1]
            excess = torch.clamp(proj - threshold, min=0.0)
            out[:, patch_pos, :] = out[:, patch_pos, :] - cap_strength * excess * d
            return out

        with torch.no_grad():
            with adapter.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
                logits = adapter.model(full_ids)
        lp = torch.log_softmax(logits[0].float().cpu(), dim=-1)
        cont = full_ids[0, prompt_len:].cpu()
        n = cont.numel()
        if n == 0:
            return 0.0
        pos = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
        return float(lp[pos, cont].sum().item()) / n

    before, after = [], []
    for i in idx:
        r = df.iloc[i]
        try:
            b = _logprob(r["prompt"], r[syc_col], False) - _logprob(r["prompt"], r[non_col], False)
            a = _logprob(r["prompt"], r[syc_col], True) - _logprob(r["prompt"], r[non_col], True)
            before.append(b); after.append(a)
        except Exception as e:
            logger.warning("capping failed: %s", e)
    return before, after


def estimate_cap_threshold(train_X_layer: np.ndarray, direction: np.ndarray,
                           quantile: float = 0.75) -> float:
    """Threshold = given quantile of projections of training activations onto the unit direction."""
    dir_unit = direction / (np.linalg.norm(direction) + 1e-8)
    proj = train_X_layer @ dir_unit
    return float(np.quantile(proj, quantile))


# ---------------------------------------------------------------------------
# Scoring wrapper
# ---------------------------------------------------------------------------

def score_before_after(before: list, after: list, bootstrap: int = 0, seed: int = 42) -> Dict:
    """Compute the standard metric bundle + bootstrap CI on behavior_margin_delta."""
    from src.metrics import behavioral_intervention_metrics
    from src.statistics import bootstrap_mean_ci
    b = [x for x in before if np.isfinite(x)]
    a = [after[i] for i in range(len(after)) if i < len(before) and np.isfinite(before[i]) and np.isfinite(after[i])]
    deltas = [after[i] - before[i] for i in range(min(len(before), len(after)))
              if np.isfinite(before[i]) and np.isfinite(after[i])]
    out = {
        "before_behavior_margin": float(np.mean(b)) if b else float("nan"),
        "after_behavior_margin": float(np.mean([x for x in after if np.isfinite(x)])) if after else float("nan"),
        "behavior_margin_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "n_examples": len(deltas),
    }
    out.update(behavioral_intervention_metrics(before, after))
    if bootstrap and deltas:
        ci = bootstrap_mean_ci(deltas, n_boot=bootstrap, seed=seed)
        out["ci_low"], out["ci_high"] = ci["ci_low"], ci["ci_high"]
    else:
        out["ci_low"] = out["ci_high"] = float("nan")
    return out
