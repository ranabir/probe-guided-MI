"""Direction methods for sycophancy steering, and a comparison to probe directions.

reviewer feedback #6: compare learned probe directions (regression / logistic) against the simple
difference-of-means direction. We compute a candidate "sycophancy direction" per layer by several
methods, then steer along it (TransformerLens) and measure the behavioral effect.

A direction is a unit vector in residual space at one layer. Steering adds `alpha * direction` to
the final prompt-token residual during the forward pass and re-measures the behavior margin.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direction constructors (each returns a 1-D vector of length d_model)
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def regression_direction(probe) -> np.ndarray:
    """Direction from a fitted regression probe (its weight vector), folding in the scaler."""
    coef = np.asarray(probe.coef_, dtype=np.float64).reshape(-1)
    if getattr(probe, "scaler", None) is not None:
        # standardized weights -> raw space: w_raw = w / scale
        coef = coef / probe.scaler.scale_
    return _unit(coef)


def logistic_direction(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Fit a logistic probe for prefers_sycophancy at one layer; return its direction."""
    from src.probes import LinearProbe
    probe = LinearProbe(task="classification", seed=seed)
    probe.fit(X, labels)
    coef = np.asarray(probe.coef_, dtype=np.float64).reshape(-1)
    if probe.scaler is not None:
        coef = coef / probe.scaler.scale_
    return _unit(coef)


def diff_of_means_direction(X: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """mean(hidden | margin>0) − mean(hidden | margin<=0) at one layer. X is [N, D]."""
    m = np.asarray(margins)
    pos, neg = X[m > 0], X[m <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return _unit(np.zeros(X.shape[1], dtype=np.float32))
    return _unit(pos.mean(axis=0) - neg.mean(axis=0))


def margin_weighted_direction(X: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Weighted-mean direction: each hidden state weighted by its (centered) behavior_margin."""
    m = np.asarray(margins, dtype=np.float64)
    w = m - m.mean()
    if np.allclose(w, 0):
        return _unit(np.zeros(X.shape[1], dtype=np.float32))
    return _unit((X * w[:, None]).sum(axis=0))


def random_direction(d_model: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=d_model))


def compute_layer_directions(layer_X: np.ndarray, margins: np.ndarray,
                             regression_probe=None, seed: int = 42) -> Dict[str, np.ndarray]:
    """All candidate directions for one layer's activations [N, D]."""
    labels = (np.asarray(margins) > 0).astype(int)
    dirs = {
        "diff_of_means": diff_of_means_direction(layer_X, margins),
        "margin_weighted": margin_weighted_direction(layer_X, margins),
        "random": random_direction(layer_X.shape[1], seed=seed),
    }
    if len(np.unique(labels)) > 1:
        dirs["logistic"] = logistic_direction(layer_X, labels, seed=seed)
    if regression_probe is not None:
        dirs["regression"] = regression_direction(regression_probe)
    return dirs


# ---------------------------------------------------------------------------
# Steering (TransformerLens only)
# ---------------------------------------------------------------------------

def steer_logprob(adapter, prompt: str, continuation: str, layer: int,
                  direction: np.ndarray, alpha: float, max_length: int = 128,
                  normalize: bool = True) -> float:
    """log P(continuation | prompt) after adding alpha*direction to the final prompt-token
    residual at `layer` during the forward pass (TransformerLens)."""
    import torch
    from src.model_adapters import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        raise NotImplementedError("Steering is implemented for TransformerLens models only.")

    full_ids = adapter.tokenizer.encode(prompt + " " + continuation, return_tensors="pt",
                                        truncation=True, max_length=max_length).to(adapter.device)
    seq_len = full_ids.shape[1]
    prompt_len = len(adapter.tokenizer.encode(prompt, add_special_tokens=True))
    if prompt_len >= seq_len:
        return 0.0
    patch_pos = max(min(prompt_len - 1, seq_len - 1), 0)
    vec = torch.tensor(direction, dtype=torch.float32)

    def hook_fn(val, hook):  # val [1, T, D]
        out = val.clone()
        out[:, patch_pos, :] = out[:, patch_pos, :] + alpha * vec.to(val.device).to(val.dtype)
        return out

    with torch.no_grad():
        with adapter.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
            logits = adapter.model(full_ids)
    log_probs = torch.log_softmax(logits[0].float().cpu(), dim=-1)
    cont_ids = full_ids[0, prompt_len:].cpu()
    n_tok = cont_ids.numel()
    if n_tok == 0:
        return 0.0
    positions = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
    total = float(log_probs[positions, cont_ids].sum().item())
    return total / n_tok if normalize else total


def steer_behavior_margin(adapter, prompt: str, syc: str, non: str, layer: int,
                          direction: np.ndarray, alpha: float, max_length: int = 128,
                          normalize: bool = True) -> float:
    s = steer_logprob(adapter, prompt, syc, layer, direction, alpha, max_length, normalize)
    n = steer_logprob(adapter, prompt, non, layer, direction, alpha, max_length, normalize)
    return s - n


def steer_margins_for_prompts(adapter, pairs_df, layer: int, direction: np.ndarray,
                              alpha: float, max_examples: int = 20, seed: int = 42,
                              max_length: int = 128) -> List[float]:
    """Per-prompt behavior margin after steering by alpha along `direction` at `layer`."""
    syc_col = "sycophantic_response" if "sycophantic_response" in pairs_df.columns else "syc_response"
    non_col = "non_sycophantic_response" if "non_sycophantic_response" in pairs_df.columns else "non_syc_response"
    df = pairs_df
    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        try:
            out.append(steer_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                             layer, direction, alpha, max_length, normalize=True))
        except Exception as e:
            logger.warning("steer failed: %s", e)
            out.append(float("nan"))
    return out
