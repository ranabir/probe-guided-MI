"""Causal validation via activation ablation/patching."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def mean_ablate_layers(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    probe,
    target_layers: List[int],
    mean_acts: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Replace specified layers with their mean activation and re-score the probe.

    Args:
        hidden_states: [N, L, D]
        labels: [N]
        probe: fitted LinearProbe
        target_layers: list of layer indices to ablate
        mean_acts: [L, D] dataset mean; if None computed from hidden_states

    Returns:
        (before_score, after_score) average P(sycophantic) on sycophantic examples.
    """
    syc_mask = labels == 1
    if syc_mask.sum() == 0:
        logger.warning("No sycophantic examples; returning dummy scores.")
        return 0.5, 0.5

    if mean_acts is None:
        mean_acts = hidden_states.mean(axis=0)  # [L, D]

    # Before
    before_scores = probe.predict_proba(hidden_states[syc_mask, probe.layer_idx, :])
    before = float(before_scores.mean())

    # Ablate
    ablated = hidden_states.copy()
    for layer in target_layers:
        ablated[:, layer, :] = mean_acts[layer, :]

    after_scores = probe.predict_proba(ablated[syc_mask, probe.layer_idx, :])
    after = float(after_scores.mean())

    return before, after


def ablate_and_score(
    hidden_states: np.ndarray,
    probe,
    target_layers: List[int],
    mean_acts: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Mean-ablate target_layers and return (before_pred, after_pred).

    Works for any probe via predict_score (regression -> value, classification -> P(pos)).
    Averages over ALL examples (no syc-only mask), suitable for prompt-preference probes.
    """
    if mean_acts is None:
        mean_acts = hidden_states.mean(axis=0)
    li = probe.layer_idx
    before = float(probe.predict_score(hidden_states[:, li, :]).mean())
    ablated = hidden_states.copy()
    for layer in target_layers:
        ablated[:, layer, :] = mean_acts[layer, :]
    after = float(probe.predict_score(ablated[:, li, :]).mean())
    return before, after


def run_preference_causal_validation(
    hidden_states: np.ndarray,
    probe,
    top_layers: List[int],
    k: int = 5,
    random_trials: int = 5,
    seed: int = 42,
    mean_acts: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Probe-prediction causal validation for regression/preference probes.

    Returns list of dicts with method, k, before_score, after_score, delta.
    """
    rng = np.random.default_rng(seed)
    n_layers = hidden_states.shape[1]
    if mean_acts is None:
        mean_acts = hidden_states.mean(axis=0)

    actual_k = min(k, len(top_layers), n_layers)
    results = []

    b, a = ablate_and_score(hidden_states, probe, top_layers[:actual_k], mean_acts)
    results.append({"method": "probe_gradient", "k": actual_k,
                    "before_score": b, "after_score": a, "delta": a - b,
                    "layers_ablated": str(top_layers[:actual_k])})
    logger.info("Probe-gradient k=%d: before=%.4f after=%.4f delta=%.4f", actual_k, b, a, a - b)

    rb, ra = [], []
    for _ in range(random_trials):
        rl = list(rng.choice(n_layers, size=actual_k, replace=False))
        b2, a2 = ablate_and_score(hidden_states, probe, rl, mean_acts)
        rb.append(b2); ra.append(a2)
    results.append({"method": "random", "k": actual_k,
                    "before_score": float(np.mean(rb)), "after_score": float(np.mean(ra)),
                    "delta": float(np.mean(ra)) - float(np.mean(rb)), "layers_ablated": "random"})
    logger.info("Random k=%d: before=%.4f after=%.4f delta=%.4f",
                actual_k, np.mean(rb), np.mean(ra), np.mean(ra) - np.mean(rb))
    return results


def run_causal_validation(
    hidden_states_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    probe,
    top_layers: List[int],
    k: int = 5,
    random_trials: int = 5,
    seed: int = 42,
    max_examples: int = 50,
) -> List[Dict]:
    """Run ablation experiments for probe-gradient top-k vs random-k.

    Args:
        hidden_states_dict: {'test': [N, L, D], ...}
        labels_dict: {'test': [N], ...}
        probe: fitted LinearProbe
        top_layers: layers ranked by probe-gradient attribution
        k: number of layers to ablate
        random_trials: how many random-k ablations to average
        seed: random seed

    Returns:
        List of result dicts.
    """
    rng = np.random.default_rng(seed)
    split = "test"
    hs = hidden_states_dict.get(split, next(iter(hidden_states_dict.values())))
    labels = labels_dict.get(split, next(iter(labels_dict.values())))

    # Subsample
    if len(hs) > max_examples:
        idx = rng.choice(len(hs), size=max_examples, replace=False)
        hs = hs[idx]
        labels = labels[idx]

    n_layers = hs.shape[1]
    mean_acts = hs.mean(axis=0)
    results = []

    # 1. Probe-gradient top-k
    actual_k = min(k, len(top_layers), n_layers)
    top_k = top_layers[:actual_k]
    before, after = mean_ablate_layers(hs, labels, probe, top_k, mean_acts)
    results.append(
        {
            "method": "probe_gradient",
            "k": actual_k,
            "before_score": before,
            "after_score": after,
            "delta": after - before,
            "layers_ablated": str(top_k),
        }
    )
    logger.info("Probe-gradient  k=%d: before=%.3f after=%.3f delta=%.3f", actual_k, before, after, after - before)

    # 2. Random-k ablations
    random_deltas = []
    random_befores = []
    random_afters = []
    for trial in range(random_trials):
        rand_layers = list(rng.choice(n_layers, size=actual_k, replace=False))
        b, a = mean_ablate_layers(hs, labels, probe, rand_layers, mean_acts)
        random_deltas.append(a - b)
        random_befores.append(b)
        random_afters.append(a)

    rand_before = float(np.mean(random_befores))
    rand_after = float(np.mean(random_afters))
    rand_delta = float(np.mean(random_deltas))
    results.append(
        {
            "method": "random",
            "k": actual_k,
            "before_score": rand_before,
            "after_score": rand_after,
            "delta": rand_delta,
            "layers_ablated": "random",
        }
    )
    logger.info("Random          k=%d: before=%.3f after=%.3f delta=%.3f", actual_k, rand_before, rand_after, rand_delta)

    return results


def patch_selected_layers_from_reference(
    adapter,
    prompt: str,
    continuation: str,
    selected_layers: List[int],
    patch_vectors: Dict[int, np.ndarray],
    max_length: int = 128,
    normalize: bool = True,
) -> float:
    """Surgical activation patching (TransformerLens).

    During the real forward pass over `prompt + continuation`, overwrite the residual stream
    at the FINAL prompt-token position with a reference vector, at each selected layer. Returns
    the (length-normalized) logprob of the continuation under that intervention.

    This is more targeted than mean ablation: it replaces only the position the probe reads
    (the last prompt token) with a contrastive/reference activation, and measures the REAL
    downstream effect on the continuation the model would generate.
    """
    from src.model_adapters import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        raise NotImplementedError("Activation patching is implemented for TransformerLens models only.")

    full_ids = adapter.tokenizer.encode(prompt + " " + continuation, return_tensors="pt",
                                        truncation=True, max_length=max_length).to(adapter.device)
    seq_len = full_ids.shape[1]
    prompt_len = len(adapter.tokenizer.encode(prompt, add_special_tokens=True))
    if prompt_len >= seq_len:
        return 0.0  # prompt filled the window; continuation truncated away
    patch_pos = max(min(prompt_len - 1, seq_len - 1), 0)

    pv = {l: torch.tensor(patch_vectors[l], dtype=torch.float32) for l in selected_layers}

    def make_hook(layer_idx):
        vec = pv[layer_idx]
        def hook_fn(val, hook):  # val: [1, T, D]
            out = val.clone()
            out[:, patch_pos, :] = vec.to(val.device).to(val.dtype)
            return out
        return hook_fn

    hooks = [(f"blocks.{l}.hook_resid_post", make_hook(l)) for l in selected_layers]

    with torch.no_grad():
        with adapter.model.hooks(fwd_hooks=hooks):
            logits = adapter.model(full_ids)

    log_probs = torch.log_softmax(logits[0].float().cpu(), dim=-1)
    continuation_ids = full_ids[0, prompt_len:].cpu()
    n_tok = continuation_ids.numel()
    if n_tok == 0:
        return 0.0
    positions = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
    total = float(log_probs[positions, continuation_ids].sum().item())
    return total / n_tok if normalize else total


def compute_patched_behavior_margin(
    adapter,
    prompt: str,
    syc_response: str,
    non_syc_response: str,
    selected_layers: List[int],
    patch_vectors: Dict[int, np.ndarray],
    max_length: int = 128,
    normalize: bool = True,
) -> float:
    """behavior_margin under activation patching of the final prompt-token residual."""
    syc_lp = patch_selected_layers_from_reference(
        adapter, prompt, syc_response, selected_layers, patch_vectors, max_length, normalize)
    non_lp = patch_selected_layers_from_reference(
        adapter, prompt, non_syc_response, selected_layers, patch_vectors, max_length, normalize)
    return syc_lp - non_lp


def patched_margins_for_layers(
    adapter,
    pairs_df,
    layers: List[int],
    mean_acts: np.ndarray,
    max_examples: int = 15,
    seed: int = 42,
    max_length: int = 128,
) -> Tuple[list, list]:
    """Return (before_margins, after_margins) per prompt under patching `layers` (TL only).

    `layers=[]` returns the unpatched margin for both (useful for a clean baseline).
    """
    import numpy as _np
    syc_col = "sycophantic_response" if "sycophantic_response" in pairs_df.columns else "syc_response"
    non_col = "non_sycophantic_response" if "non_sycophantic_response" in pairs_df.columns else "non_syc_response"
    df = pairs_df
    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    patch_vectors = {l: mean_acts[l] for l in layers}
    before, after = [], []
    for _, r in df.iterrows():
        try:
            b = compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                [], {}, max_length=max_length)
            a = compute_patched_behavior_margin(adapter, r["prompt"], r[syc_col], r[non_col],
                                                layers, patch_vectors, max_length=max_length)
            before.append(b); after.append(a)
        except Exception as e:
            logger.warning("patching eval failed: %s", e)
    return before, after


def run_patching_behavior_validation(
    adapter,
    pairs_df,
    top_layers: List[int],
    mean_acts: np.ndarray,
    k: int = 5,
    random_trials: int = 3,
    max_examples: int = 15,
    seed: int = 42,
    max_length: int = 128,
) -> Dict[str, dict]:
    """Activation-patching causal validation (TL only).

    Patches the final-prompt-token residual at selected layers with the per-layer mean
    prompt-final activation (a neutral reference), via the real forward pass, then measures
    the change in the model's behavior_margin. Compares probe-gradient top-k vs random-k.

    Returns {'probe_gradient': {...}, 'random': {...}} each with before/after/delta.
    """
    from src.model_adapters import TransformerLensAdapter
    import numpy as _np

    if not isinstance(adapter, TransformerLensAdapter):
        nan = float("nan")
        empty = {"before_behavior_margin": nan, "after_behavior_margin": nan, "behavior_margin_delta": nan}
        return {"probe_gradient": dict(empty), "random": dict(empty)}

    rng = _np.random.default_rng(seed)
    syc_col = "sycophantic_response" if "sycophantic_response" in pairs_df.columns else "syc_response"
    non_col = "non_sycophantic_response" if "non_sycophantic_response" in pairs_df.columns else "non_syc_response"

    df = pairs_df
    if len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    n_layers = mean_acts.shape[0]
    actual_k = min(k, len(top_layers), n_layers)
    top_k = top_layers[:actual_k]
    rand_k = list(rng.choice(n_layers, size=actual_k, replace=False))

    def _eval(layers):
        before, after = patched_margins_for_layers(
            adapter, df, layers, mean_acts, max_examples=max_examples, seed=seed, max_length=max_length)
        bm = float(_np.nanmean(before)) if before else float("nan")
        am = float(_np.nanmean(after)) if after else float("nan")
        out = {"before_behavior_margin": bm, "after_behavior_margin": am,
               "behavior_margin_delta": am - bm}
        # A/B behavioral metrics (answer-flip, accuracy change)
        try:
            from src.metrics import behavioral_intervention_metrics
            out.update(behavioral_intervention_metrics(before, after))
        except Exception:
            pass
        out["_before_margins"] = before
        out["_after_margins"] = after
        return out

    logger.info("Activation patching: probe-gradient layers %s", top_k)
    pg = _eval(top_k)
    logger.info("Activation patching: random layers %s", rand_k)
    rnd = _eval(rand_k)
    return {"probe_gradient": pg, "random": rnd}


def run_sweep_k_validation(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    probe,
    top_layers: List[int],
    k_values: Optional[List[int]] = None,
    random_trials: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """Sweep over k values for the Streamlit causal panel."""
    rng = np.random.default_rng(seed)
    n_layers = hidden_states.shape[1]
    mean_acts = hidden_states.mean(axis=0)

    if k_values is None:
        k_values = list(range(1, min(21, n_layers + 1)))

    results = []
    for k in k_values:
        tk = top_layers[:k]
        b, a = mean_ablate_layers(hidden_states, labels, probe, tk, mean_acts)
        rand_afters = []
        for _ in range(random_trials):
            rl = list(rng.choice(n_layers, size=k, replace=False))
            _, ra = mean_ablate_layers(hidden_states, labels, probe, rl, mean_acts)
            rand_afters.append(ra)
        results.append(
            {
                "k": k,
                "before": b,
                "probe_after": a,
                "random_after": float(np.mean(rand_afters)),
                "probe_delta": a - b,
                "random_delta": float(np.mean(rand_afters)) - b,
            }
        )

    return results
