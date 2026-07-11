"""Probe-gradient attribution: rank layers and components by causal importance."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribution score functions
# ---------------------------------------------------------------------------


def grad_norm(grads: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """Mean absolute gradient. grads/acts shape: [N, L, D] -> [L]."""
    return np.abs(grads).mean(axis=(0, 2))


def grad_times_activation(grads: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """Mean |grad * activation|. -> [L]."""
    return np.abs(grads * acts).mean(axis=(0, 2))


def pair_diff_grad(
    grads_syc: np.ndarray,
    acts_syc: np.ndarray,
    acts_non: np.ndarray,
) -> np.ndarray:
    """Mean |(act_syc - act_non) * grad_syc|. -> [L]."""
    diff = acts_syc - acts_non
    return np.abs(diff * grads_syc).mean(axis=(0, 2))


ATTRIBUTION_METHODS = {
    "grad_norm": grad_norm,
    "grad_times_activation": grad_times_activation,
}


def compute_layer_attribution(
    grads: np.ndarray,
    acts: np.ndarray,
    method: str = "grad_times_activation",
    grads_syc: Optional[np.ndarray] = None,
    acts_syc: Optional[np.ndarray] = None,
    acts_non: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return attribution scores per layer, shape [L]."""
    if method == "pair_diff_grad" and grads_syc is not None and acts_syc is not None and acts_non is not None:
        return pair_diff_grad(grads_syc, acts_syc, acts_non)
    fn = ATTRIBUTION_METHODS.get(method, grad_times_activation)
    return fn(grads, acts)


def compute_component_attribution(
    component_grads: Dict[str, np.ndarray],
    component_acts: Dict[str, np.ndarray],
    method: str = "grad_times_activation",
) -> pd.DataFrame:
    """Compute attribution for each component type across layers.

    Args:
        component_grads: dict of key -> [N, L, D] gradients
        component_acts: dict of key -> [N, L, D] activations

    Returns:
        DataFrame with columns [component, layer, attribution_score]
    """
    rows = []
    for comp_name, grads in component_grads.items():
        acts = component_acts.get(comp_name, grads)
        scores = compute_layer_attribution(grads, acts, method=method)
        n_layers = len(scores)
        for layer in range(n_layers):
            rows.append(
                {
                    "component": comp_name,
                    "layer": layer,
                    "attribution_score": float(scores[layer]),
                }
            )
    return pd.DataFrame(rows)


def logit_gradient_baseline(
    adapter,
    texts: List[str],
    max_length: int = 128,
    token_position: str = "final",
    syc_tokens: Optional[List[str]] = None,
    non_syc_tokens: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute gradients of the sycophancy logit margin w.r.t. hidden states.

    logit_margin = mean(logits[syc_tokens]) - mean(logits[non_syc_tokens])

    Returns array [N, L, D].
    """
    import torch
    from src.model_adapters import TransformerLensAdapter, _extract_position

    if syc_tokens is None:
        syc_tokens = ["Yes", "Absolutely", "Correct", "Indeed", "Sure"]
    if non_syc_tokens is None:
        non_syc_tokens = ["No", "Actually", "Incorrect", "However", "Wrong"]

    tok = adapter.tokenizer

    def get_ids(words):
        ids = []
        for w in words:
            enc_ids = tok.encode(w, add_special_tokens=False)
            if enc_ids:
                ids.append(enc_ids[0])
        return list(set(ids))

    syc_ids = get_ids(syc_tokens)
    non_syc_ids = get_ids(non_syc_tokens)

    if not syc_ids or not non_syc_ids:
        logger.warning("Could not find token IDs for logit baseline; returning zeros.")
        return np.zeros((len(texts), adapter.n_layers, adapter.d_model), dtype=np.float32)

    enc = adapter.tokenize(texts, max_length=max_length)
    attention_mask = enc.get("attention_mask")
    n = len(texts)

    if isinstance(adapter, TransformerLensAdapter):
        return _tl_logit_baseline(adapter, enc, syc_ids, non_syc_ids, token_position, n)

    # ---- HuggingFace path ----
    hidden_cache: List[torch.Tensor] = []
    hooks = []
    layers = adapter._get_layers()

    def make_hook(store):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_req = h.float().requires_grad_(True)
            store.append(h_req)
            if isinstance(out, tuple):
                return (h_req,) + out[1:]
            return h_req
        return fn

    for layer in layers:
        hooks.append(layer.register_forward_hook(make_hook(hidden_cache)))

    try:
        for p in adapter.model.parameters():
            p.requires_grad_(False)
        with torch.enable_grad():
            outputs = adapter.model(**enc)
        logits = outputs.logits
    finally:
        for h in hooks:
            h.remove()

    if attention_mask is not None:
        lengths = attention_mask.sum(dim=1) - 1
        final_logits = logits[torch.arange(n), lengths.cpu()]
    else:
        final_logits = logits[:, -1, :]

    syc_ids_t = torch.tensor(syc_ids, dtype=torch.long)
    non_ids_t = torch.tensor(non_syc_ids, dtype=torch.long)
    margin = final_logits[:, syc_ids_t].mean(dim=1) - final_logits[:, non_ids_t].mean(dim=1)
    margin.sum().backward()

    if not hidden_cache:
        return np.zeros((n, adapter.n_layers, adapter.d_model), dtype=np.float32)

    grad_layers = []
    for h in hidden_cache:
        if h.grad is not None:
            g = _extract_position(h.grad, token_position, attention_mask)
            grad_layers.append(g.float().detach().cpu().numpy())
        else:
            pos = _extract_position(h.detach().cpu(), token_position, attention_mask)
            grad_layers.append(np.zeros_like(pos.numpy(), dtype=np.float32))

    return np.stack(grad_layers, axis=1)


def _tl_logit_baseline(adapter, enc, syc_ids, non_syc_ids, token_position, n):
    """TransformerLens-specific logit gradient baseline using TL hook system."""
    import torch
    from src.model_adapters import _extract_position

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    all_hook_names = [f"blocks.{i}.hook_resid_post" for i in range(adapter.n_layers)]
    stored: Dict[str, torch.Tensor] = {}

    def embed_hook(val, hook):
        val_req = val.detach().float().requires_grad_(True)
        stored["__embed__"] = val_req
        return val_req

    def layer_hook(val, hook):
        val_f = val.float()
        val_f.retain_grad()
        stored[hook.name] = val_f
        return val_f

    fwd_hooks = [("hook_embed", embed_hook)] + [(n_, layer_hook) for n_ in all_hook_names]

    with torch.enable_grad():
        logits = adapter.model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

    if attention_mask is not None:
        lengths = attention_mask.sum(dim=1) - 1
        final_logits = logits[torch.arange(n), lengths.cpu()]
    else:
        final_logits = logits[:, -1, :]

    syc_ids_t = torch.tensor(syc_ids, dtype=torch.long)
    non_ids_t = torch.tensor(non_syc_ids, dtype=torch.long)
    margin = final_logits[:, syc_ids_t].float().mean(dim=1) - final_logits[:, non_ids_t].float().mean(dim=1)
    margin.sum().backward()

    grad_layers = []
    for name in all_hook_names:
        act = stored.get(name)
        if act is not None and act.grad is not None:
            g = _extract_position(act.grad, token_position, attention_mask)
            grad_layers.append(g.float().detach().cpu().numpy())
        else:
            grad_layers.append(np.zeros((n, adapter.d_model), dtype=np.float32))

    return np.stack(grad_layers, axis=1)


def top_k_layers(scores: np.ndarray, k: int) -> List[int]:
    """Return indices of top-k layers by attribution score."""
    return list(np.argsort(scores)[::-1][:k])


def build_attribution_df(
    layer_scores: np.ndarray,
    component_df: Optional[pd.DataFrame] = None,
    model_name: str = "",
) -> pd.DataFrame:
    """Build a summary DataFrame of layer-level attribution."""
    rows = []
    for i, score in enumerate(layer_scores):
        rows.append(
            {
                "model_name": model_name,
                "component": "hidden_states",
                "layer": i,
                "head": None,
                "attribution_score": float(score),
            }
        )

    df = pd.DataFrame(rows)
    if component_df is not None and not component_df.empty:
        component_df = component_df.copy()
        component_df["model_name"] = model_name
        component_df["head"] = None
        df = pd.concat([df, component_df], ignore_index=True)

    return df.sort_values("attribution_score", ascending=False).reset_index(drop=True)
