"""Backend-agnostic residual-stream interventions (TransformerLens AND HuggingFace).

Solves two problems:
  1. HF steering/patching — the same residual edit works on Qwen/Gemma via PyTorch forward hooks,
     not just TransformerLens hook points. Instruct models can finally be intervened on.
  2. Clean causal control — beyond blunt additive steering (which inflates the residual norm and
     breaks the model), we add gentler edits that touch only the sycophancy coordinate:
       - projection_ablation : remove the direction component        h -= (h·d̂) d̂
       - mean_shift          : set the component to the honest value  h += (target − h·d̂) d̂
       - additive            : the old baseline                       h += α d̂        (norm inflates)
     Any edit can be norm-preserving (rescale back to the original ‖h‖).

An "edit" is a function edit_fn(vec[B,D], layer) -> vec[B,D] applied to the residual at the final
prompt-token position (for logprob) or the last position each step (for generation).
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edit builders (return edit_fn(vec, layer)); directions are unit vectors per layer
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _renorm(orig, edited):
    on = orig.norm(dim=-1, keepdim=True)
    en = edited.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return edited * (on / en)


def make_additive(unit_dirs: Dict[int, np.ndarray], alpha: float,
                  resid_scale: Dict[int, float], norm_preserve: bool = False) -> Callable:
    """h += alpha * resid_scale * d̂  (the blunt baseline; norm grows unless norm_preserve)."""
    torch = _torch()
    d = {L: torch.tensor(v, dtype=torch.float32) for L, v in unit_dirs.items()}

    def edit(vec, layer):
        dv = d[layer].to(vec.device).to(vec.dtype)
        out = vec + alpha * resid_scale[layer] * dv
        return _renorm(vec, out) if norm_preserve else out
    return edit


def make_projection_ablation(unit_dirs: Dict[int, np.ndarray], target: float = 0.0,
                             norm_preserve: bool = True) -> Callable:
    """Set the component along d̂ to `target` (default 0 = fully remove it).
    Touches only the sycophancy coordinate; everything orthogonal is untouched."""
    torch = _torch()
    d = {L: torch.tensor(v, dtype=torch.float32) for L, v in unit_dirs.items()}

    def edit(vec, layer):
        dv = d[layer].to(vec.device).to(vec.dtype)
        proj = (vec * dv).sum(-1, keepdim=True)          # [B,1]
        out = vec - (proj - target) * dv                 # move component to `target`
        return _renorm(vec, out) if norm_preserve else out
    return edit


def make_mean_shift(unit_dirs: Dict[int, np.ndarray], honest_proj: Dict[int, float],
                    norm_preserve: bool = True) -> Callable:
    """Set the component along d̂ to the honest-preferring mean projection (in-distribution)."""
    torch = _torch()
    d = {L: torch.tensor(v, dtype=torch.float32) for L, v in unit_dirs.items()}

    def edit(vec, layer):
        dv = d[layer].to(vec.device).to(vec.dtype)
        proj = (vec * dv).sum(-1, keepdim=True)
        out = vec + (honest_proj[layer] - proj) * dv
        return _renorm(vec, out) if norm_preserve else out
    return edit


def make_cap(unit_dirs: Dict[int, np.ndarray], threshold: Dict[int, float],
             strength: float = 1.0, norm_preserve: bool = False) -> Callable:
    """Clip the component along d̂ only when it exceeds `threshold`."""
    torch = _torch()
    d = {L: torch.tensor(v, dtype=torch.float32) for L, v in unit_dirs.items()}

    def edit(vec, layer):
        dv = d[layer].to(vec.device).to(vec.dtype)
        proj = (vec * dv).sum(-1, keepdim=True)
        excess = (proj - threshold[layer]).clamp_min(0.0)
        out = vec - strength * excess * dv
        return _renorm(vec, out) if norm_preserve else out
    return edit


# ---------------------------------------------------------------------------
# Unified residual-edit logprob (TL + HF)
# ---------------------------------------------------------------------------

def _prompt_len_and_ids(adapter, prompt, continuation, max_length):
    full_ids = adapter.tokenizer.encode(prompt + " " + continuation, return_tensors="pt",
                                        truncation=True, max_length=max_length).to(adapter.device)
    prompt_len = len(adapter.tokenizer.encode(prompt, add_special_tokens=True))
    return full_ids, prompt_len


def _score_logits(logits, full_ids, prompt_len, normalize):
    torch = _torch()
    lp = torch.log_softmax(logits[0].float().cpu(), dim=-1)
    cont = full_ids[0, prompt_len:].cpu()
    n = cont.numel()
    if n == 0:
        return 0.0
    pos = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
    total = float(lp[pos, cont].sum().item())
    return total / n if normalize else total


def resid_edit_logprob(adapter, prompt: str, continuation: str, layers: List[int],
                       edit_fn: Optional[Callable], max_length: int = 256,
                       normalize: bool = True) -> float:
    """log P(continuation | prompt) with edit_fn applied to the final prompt-token residual at
    each layer in `layers`. edit_fn=None → clean baseline. Works for TL and HF adapters."""
    torch = _torch()
    from src.model_adapters import TransformerLensAdapter

    full_ids, prompt_len = _prompt_len_and_ids(adapter, prompt, continuation, max_length)
    seq_len = full_ids.shape[1]
    if prompt_len >= seq_len:
        return 0.0
    pos = max(min(prompt_len - 1, seq_len - 1), 0)
    is_tl = isinstance(adapter, TransformerLensAdapter)

    if edit_fn is None or not layers:
        with torch.no_grad():
            logits = adapter.model(full_ids) if is_tl else adapter.model(full_ids).logits
        return _score_logits(logits, full_ids, prompt_len, normalize)

    if is_tl:
        def mk(L):
            def hook(val, hook):
                out = val.clone()
                out[:, pos, :] = edit_fn(out[:, pos, :], L)
                return out
            return hook
        with torch.no_grad():
            with adapter.model.hooks(fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(L)) for L in layers]):
                logits = adapter.model(full_ids)
        return _score_logits(logits, full_ids, prompt_len, normalize)

    # HuggingFace: forward hooks on decoder layers
    mods = adapter._get_layers()
    handles = []

    def mk(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, pos, :] = edit_fn(h[:, pos, :], L)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook
    try:
        for L in layers:
            handles.append(mods[L].register_forward_hook(mk(L)))
        with torch.no_grad():
            logits = adapter.model(full_ids).logits
    finally:
        for hh in handles:
            hh.remove()
    return _score_logits(logits, full_ids, prompt_len, normalize)


def resid_edit_margin(adapter, prompt, syc, non, layers, edit_fn, max_length=256, normalize=True):
    s = resid_edit_logprob(adapter, prompt, syc, layers, edit_fn, max_length, normalize)
    n = resid_edit_logprob(adapter, prompt, non, layers, edit_fn, max_length, normalize)
    return s - n


def run_edit_on_prompts(adapter, pairs_df, layers: List[int], edit_fn: Callable,
                        max_examples: int = 30, seed: int = 42, max_length: int = 256
                        ) -> Tuple[list, list]:
    """Per-prompt (before, after) behavior margins for one edit. Works for TL and HF."""
    syc_col = "sycophantic_response" if "sycophantic_response" in pairs_df.columns else "syc_response"
    non_col = "non_sycophantic_response" if "non_sycophantic_response" in pairs_df.columns else "non_syc_response"
    df = pairs_df.reset_index(drop=True)
    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    before, after = [], []
    for _, r in df.iterrows():
        try:
            b = resid_edit_margin(adapter, r["prompt"], r[syc_col], r[non_col], layers, None, max_length)
            a = resid_edit_margin(adapter, r["prompt"], r[syc_col], r[non_col], layers, edit_fn, max_length)
            before.append(b); after.append(a)
        except Exception as e:
            logger.warning("edit eval failed: %s", e)
    return before, after


# ---------------------------------------------------------------------------
# Generation with edit (side-effect measurement) — TL + HF
# ---------------------------------------------------------------------------

def generate_with_edit(adapter, prompts: List[str], layers: List[int],
                       edit_fn: Optional[Callable], max_new_tokens: int = 25,
                       max_length: int = 256) -> List[str]:
    """Greedy-generate a short continuation per prompt, applying edit_fn at the last position each
    step. edit_fn=None → baseline. Works for TL and HF."""
    torch = _torch()
    from src.model_adapters import TransformerLensAdapter
    is_tl = isinstance(adapter, TransformerLensAdapter)
    mods = None if is_tl else adapter._get_layers()

    def hooks_ctx():
        class _Ctx:
            def __enter__(self_):
                self_.handles = []
                if edit_fn is None or not layers:
                    return self_
                if is_tl:
                    def mk(L):
                        def hook(val, hook):
                            out = val.clone()
                            out[:, -1, :] = edit_fn(out[:, -1, :], L)
                            return out
                        return hook
                    self_.tl = adapter.model.hooks(fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(L)) for L in layers])
                    self_.tl.__enter__()
                else:
                    def mk(L):
                        def hook(module, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            h = h.clone()
                            h[:, -1, :] = edit_fn(h[:, -1, :], L)
                            return (h,) + out[1:] if isinstance(out, tuple) else h
                        return hook
                    for L in layers:
                        self_.handles.append(mods[L].register_forward_hook(mk(L)))
                return self_

            def __exit__(self_, *a):
                for hh in getattr(self_, "handles", []):
                    hh.remove()
                if hasattr(self_, "tl"):
                    self_.tl.__exit__(*a)
        return _Ctx()

    outputs = []
    for prompt in prompts:
        ids = adapter.tokenizer.encode(prompt, return_tensors="pt").to(adapter.device)
        start = ids.shape[1]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                with hooks_ctx():
                    logits = adapter.model(ids) if is_tl else adapter.model(ids).logits
            nxt = int(logits[0, -1].argmax().item())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            if ids.shape[1] >= max_length:
                break
        outputs.append(adapter.tokenizer.decode(ids[0, start:]).strip())
    return outputs


# ---------------------------------------------------------------------------
# Helpers to build per-layer directions / targets from training activations
# ---------------------------------------------------------------------------

def unit_directions_for_layers(train_hs: np.ndarray, margins: np.ndarray, layers: List[int],
                               method: str = "diff_of_means", regression_probe=None,
                               seed: int = 42) -> Dict[int, np.ndarray]:
    """Return {layer: unit direction} using the requested method (reuses src.directions)."""
    from src.directions import compute_layer_directions
    out = {}
    for L in layers:
        dirs = compute_layer_directions(train_hs[:, L, :], margins, regression_probe=regression_probe, seed=seed)
        v = dirs.get(method, dirs.get("diff_of_means"))
        out[L] = v / (np.linalg.norm(v) + 1e-8)
    return out


def honest_mean_projection(train_hs: np.ndarray, margins: np.ndarray,
                           unit_dirs: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Mean projection of honest-preferring (margin<=0) training activations onto each direction."""
    honest = np.asarray(margins) <= 0
    out = {}
    for L, d in unit_dirs.items():
        X = train_hs[honest, L, :] if honest.any() else train_hs[:, L, :]
        out[L] = float((X @ d).mean())
    return out


def resid_scale_for_layers(train_hs: np.ndarray, layers: List[int]) -> Dict[int, float]:
    return {L: float(np.linalg.norm(train_hs[:, L, :], axis=1).mean()) for L in layers}
