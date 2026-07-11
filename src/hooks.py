"""Hook helpers for TransformerLens — thin wrappers kept separate for clarity."""
from typing import Callable, Dict, List, Tuple

import torch


def make_cache_hook(cache: Dict, key: str) -> Callable:
    """Return a TL-style hook function that saves val into cache[key]."""
    def hook_fn(val: torch.Tensor, hook) -> None:
        cache[key] = val.detach().cpu()
    return hook_fn


def make_ablation_hook(replacement: torch.Tensor) -> Callable:
    """Return a hook that replaces the activation with a fixed tensor."""
    def hook_fn(val: torch.Tensor, hook) -> torch.Tensor:
        return replacement.to(val.device).to(val.dtype)
    return hook_fn


def make_mean_ablation_hook(mean_vec: torch.Tensor) -> Callable:
    """Return a hook that replaces every token position with mean_vec [D]."""
    def hook_fn(val: torch.Tensor, hook) -> torch.Tensor:
        # val: [B, T, D]
        out = val.clone()
        out[:] = mean_vec.to(val.device).to(val.dtype)
        return out
    return hook_fn


def build_tl_hook_list(
    hook_names: List[str],
    cache: Dict,
) -> List[Tuple[str, Callable]]:
    """Build a list of (hook_name, hook_fn) for TransformerLens."""
    return [(name, make_cache_hook(cache, name)) for name in hook_names]
