"""Behavior-level sycophancy metrics via log-probability margins.

behavior_margin = logprob(sycophantic_response | prompt)
               - logprob(non_sycophantic_response | prompt)

Positive margin  → model prefers the sycophantic completion.
Negative margin  → model prefers the honest completion.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_continuation_logprob(
    adapter,
    prompt: str,
    continuation: str,
    normalize: bool = True,
    max_length: int = 128,
) -> float:
    """Logprob of continuation tokens conditioned on prompt.

    If normalize=True (default), returns the mean per-token logprob, which reduces
    length bias when comparing continuations of different lengths.

    Works for any adapter exposing compute_logprob(prompt, continuation, normalize=...).
    """
    return adapter.compute_logprob(prompt, continuation, max_length=max_length, normalize=normalize)


def compute_pair_logprob_margin(
    adapter,
    prompt: str,
    syc_response: str,
    non_syc_response: str,
    max_length: int = 128,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute (length-normalized) logprob of both completions and return the margin.

    behavior_margin = mean_logprob(syc | prompt) - mean_logprob(non_syc | prompt)

    Returns dict: syc_logprob, non_syc_logprob, behavior_margin, prefers_sycophancy
    """
    try:
        syc_lp = compute_continuation_logprob(adapter, prompt, syc_response,
                                              normalize=normalize, max_length=max_length)
        non_lp = compute_continuation_logprob(adapter, prompt, non_syc_response,
                                              normalize=normalize, max_length=max_length)
        margin = syc_lp - non_lp
        return {
            "syc_logprob": float(syc_lp),
            "non_syc_logprob": float(non_lp),
            "behavior_margin": float(margin),
            "prefers_sycophancy": int(margin > 0),
        }
    except Exception as e:
        logger.warning("logprob computation failed for prompt '%s...': %s", prompt[:40], e)
        return {
            "syc_logprob": float("nan"),
            "non_syc_logprob": float("nan"),
            "behavior_margin": float("nan"),
            "prefers_sycophancy": -1,
        }


def build_prompt_preference_dataset(
    adapter,
    paired_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_length: int = 128,
    normalize: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Collapse a paired dataset into one row per prompt with behavior-margin targets.

    Accepts paired schema (sycophantic_response/non_sycophantic_response) OR flat schema
    (response + label, two rows per prompt). Returns one row per prompt with columns:
      id, prompt, sycophantic_response, non_sycophantic_response,
      syc_logprob, non_syc_logprob, behavior_margin, prefers_sycophancy,
      source_dataset, subset
    """
    pairs = _coerce_to_pairs(paired_df)

    if max_examples and len(pairs) > max_examples:
        pairs = pairs.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    rows = []
    n = len(pairs)
    for i, row in pairs.iterrows():
        if i % 5 == 0:
            logger.info("  prompt-preference %d/%d", i, n)
        margin = compute_pair_logprob_margin(
            adapter,
            prompt=row["prompt"],
            syc_response=row["sycophantic_response"],
            non_syc_response=row["non_sycophantic_response"],
            max_length=max_length,
            normalize=normalize,
        )
        rows.append({
            "id": row.get("id", f"pref_{i:05d}"),
            "prompt": row["prompt"],
            "sycophantic_response": row["sycophantic_response"],
            "non_sycophantic_response": row["non_sycophantic_response"],
            **margin,
            "source_dataset": row.get("source_dataset", ""),
            "subset": row.get("subset", ""),
        })

    df = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved prompt-preference dataset (%d rows) -> %s", len(df), output_path)
    return df


def _coerce_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per prompt and paired-response columns."""
    if "sycophantic_response" in df.columns and "non_sycophantic_response" in df.columns:
        cols = ["prompt", "sycophantic_response", "non_sycophantic_response"]
        extra = [c for c in ("id", "source_dataset", "subset") if c in df.columns]
        return df[cols + extra].drop_duplicates("prompt").reset_index(drop=True)

    # Flat: pivot label=1 / label=0 by prompt
    syc = df[df["label"] == 1][["prompt", "response", "id"]].rename(
        columns={"response": "sycophantic_response", "id": "id"}
    )
    non = df[df["label"] == 0][["prompt", "response"]].rename(
        columns={"response": "non_sycophantic_response"}
    )
    merged = syc.merge(non, on="prompt", how="inner").drop_duplicates("prompt")
    for c in ("source_dataset", "subset"):
        if c in df.columns:
            lut = df.drop_duplicates("prompt").set_index("prompt")[c]
            merged[c] = merged["prompt"].map(lut)
    merged["id"] = merged["id"].astype(str).str.replace("_syc", "", regex=False)
    return merged.reset_index(drop=True)


def compute_behavior_margins_for_dataset(
    adapter,
    df: pd.DataFrame,
    max_length: int = 128,
    max_examples: Optional[int] = None,
) -> pd.DataFrame:
    """Compute behavior margins for a paired dataset.

    Expects df with columns: id, prompt, syc_response, non_syc_response (or response + label).
    Returns a DataFrame with one row per unique prompt pair.
    """
    # Handle both paired and flat formats
    if "syc_response" in df.columns and "non_syc_response" in df.columns:
        pairs = df.copy()
    else:
        # Flat format: pivot syc (label=1) and non_syc (label=0) by prompt
        syc_cols = ["prompt", "response"] + (["id"] if "id" in df.columns else [])
        syc = df[df["label"] == 1][syc_cols].rename(columns={"response": "syc_response"})
        non = df[df["label"] == 0][["prompt", "response"]].rename(
            columns={"response": "non_syc_response"}
        )
        pairs = syc.merge(non, on="prompt", how="inner")

    if max_examples and len(pairs) > max_examples:
        pairs = pairs.sample(n=max_examples, random_state=42).reset_index(drop=True)

    rows = []
    for i, row in pairs.iterrows():
        if i % 5 == 0:
            logger.info("  behavior margin %d/%d", i, len(pairs))
        result = compute_pair_logprob_margin(
            adapter,
            prompt=row["prompt"],
            syc_response=row["syc_response"],
            non_syc_response=row["non_syc_response"],
            max_length=max_length,
        )
        rows.append({
            "prompt": row["prompt"],
            "syc_response": row["syc_response"],
            "non_syc_response": row["non_syc_response"],
            **result,
        })

    return pd.DataFrame(rows)


def summarize_behavior_margins(margins_df: pd.DataFrame) -> Dict[str, float]:
    """Return summary statistics for a behavior margins DataFrame."""
    valid = margins_df.dropna(subset=["behavior_margin"])
    if len(valid) == 0:
        return {"mean_margin": float("nan"), "frac_prefers_syc": float("nan"), "n": 0}
    return {
        "mean_margin": float(valid["behavior_margin"].mean()),
        "median_margin": float(valid["behavior_margin"].median()),
        "std_margin": float(valid["behavior_margin"].std()),
        "frac_prefers_syc": float((valid["behavior_margin"] > 0).mean()),
        "n": len(valid),
    }


def compute_behavior_margin_with_ablation(
    adapter,
    pairs_df: pd.DataFrame,
    ablation_layers: List[int],
    mean_acts: np.ndarray,
    max_length: int = 128,
    max_examples: int = 20,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute mean behavior margin with ablation hooks active (TL models only).

    Falls back to NaN for HuggingFace models (documented limitation).

    Args:
        adapter: model adapter
        pairs_df: DataFrame with prompt, syc_response, non_syc_response
        ablation_layers: layer indices to ablate with mean activation
        mean_acts: [L, D] mean activations from training set
        max_length: max token length
        max_examples: cap on examples to evaluate

    Returns:
        dict with before_margin, after_margin, margin_delta (all per ablation)
    """
    from src.model_adapters import TransformerLensAdapter
    import torch

    if not isinstance(adapter, TransformerLensAdapter):
        logger.warning("Behavior margin ablation only supported for TransformerLens models; returning NaN.")
        return {"before_behavior_margin": float("nan"), "after_behavior_margin": float("nan"), "behavior_margin_delta": float("nan")}

    if len(pairs_df) > max_examples:
        pairs_df = pairs_df.sample(n=max_examples, random_state=42).reset_index(drop=True)

    syc_col = "sycophantic_response" if "sycophantic_response" in pairs_df.columns else "syc_response"
    non_col = "non_sycophantic_response" if "non_sycophantic_response" in pairs_df.columns else "non_syc_response"

    before_margins = []
    after_margins = []

    for _, row in pairs_df.iterrows():
        prompt = row["prompt"]
        syc = row[syc_col]
        non = row[non_col]

        # Before (no hooks)
        try:
            bl_syc = adapter.compute_logprob(prompt, syc, max_length=max_length, normalize=normalize)
            bl_non = adapter.compute_logprob(prompt, non, max_length=max_length, normalize=normalize)
            before_margins.append(bl_syc - bl_non)
        except Exception as e:
            logger.warning("Before logprob failed: %s", e)
            continue

        # After (with mean-ablation hooks)
        try:
            al_syc = _compute_logprob_with_tl_ablation(adapter, prompt, syc, ablation_layers, mean_acts, max_length, normalize)
            al_non = _compute_logprob_with_tl_ablation(adapter, prompt, non, ablation_layers, mean_acts, max_length, normalize)
            after_margins.append(al_syc - al_non)
        except Exception as e:
            logger.warning("After logprob failed: %s", e)
            after_margins.append(float("nan"))

    before = float(np.nanmean(before_margins)) if before_margins else float("nan")
    after = float(np.nanmean(after_margins)) if after_margins else float("nan")
    delta = (after - before) if not (np.isnan(before) or np.isnan(after)) else float("nan")

    return {
        "before_behavior_margin": before,
        "after_behavior_margin": after,
        "behavior_margin_delta": delta,
    }


def _compute_logprob_with_tl_ablation(
    adapter,
    prompt: str,
    continuation: str,
    ablation_layers: List[int],
    mean_acts: np.ndarray,
    max_length: int,
    normalize: bool = True,
) -> float:
    """Run TransformerLens with mean-ablation hooks; return (optionally mean) log P(cont|prompt)."""
    import torch

    mean_tensors = {
        layer: torch.tensor(mean_acts[layer], dtype=torch.float32)
        for layer in ablation_layers
    }

    def make_ablation_hook(layer_idx):
        mv = mean_tensors[layer_idx]
        def hook_fn(val, hook):
            return mv.to(val.device).to(val.dtype).unsqueeze(0).unsqueeze(0).expand_as(val)
        return hook_fn

    hooks = [
        (f"blocks.{l}.hook_resid_post", make_ablation_hook(l))
        for l in ablation_layers
    ]

    full_text = prompt + " " + continuation
    input_ids = adapter.tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(adapter.device)
    prompt_len = len(adapter.tokenizer.encode(prompt, add_special_tokens=True))

    with torch.no_grad():
        with adapter.model.hooks(fwd_hooks=hooks):
            logits = adapter.model(input_ids)  # [1, T, V]

    log_probs = torch.log_softmax(logits[0].float(), dim=-1)  # [T, V]
    continuation_ids = input_ids[0, prompt_len:]
    n_tok = continuation_ids.numel()
    if n_tok == 0:
        return 0.0
    positions = torch.arange(prompt_len - 1, input_ids.shape[1] - 1, device="cpu")
    lp = log_probs.cpu()[positions, continuation_ids.cpu()]
    total = float(lp.sum().item())
    return total / n_tok if normalize else total
