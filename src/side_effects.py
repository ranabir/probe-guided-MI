"""Side-effect evaluation: does an intervention harm general capabilities or make the model weird?

If an intervention reduces sycophancy, we must check it doesn't just break the model. We generate
short greedy continuations on basic prompts with and without the intervention active (at the chosen
layers), and compare output length, repetition, coherence/weirdness, and basic-QA correctness.

Generation + intervention hooks are TransformerLens-only; HF models are scored where feasible and
flagged otherwise.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_basic_prompts(path: Optional[Path] = None) -> pd.DataFrame:
    from src.utils import get_root
    path = path or get_root() / "data" / "side_effect_eval" / "basic_prompts.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generation (TransformerLens greedy, with optional intervention hook)
# ---------------------------------------------------------------------------

def _build_intervention_hook(adapter, intervention: str, layers: List[int],
                             direction: Optional[np.ndarray], alpha: float,
                             threshold: float, cap_strength: float, mean_acts=None):
    """Return a list of (hook_name, fn) applying the intervention to the LAST position each step."""
    import torch
    hooks = []
    if intervention == "probe_steering" and direction is not None:
        dvec = torch.tensor(direction, dtype=torch.float32)
        for L in layers:
            def make(L):
                def fn(val, hook):
                    out = val.clone()
                    out[:, -1, :] = out[:, -1, :] + alpha * dvec.to(val.device).to(val.dtype)
                    return out
                return fn
            hooks.append((f"blocks.{L}.hook_resid_post", make(L)))
    elif intervention == "activation_capping" and direction is not None:
        dunit = direction / (np.linalg.norm(direction) + 1e-8)
        dvec = torch.tensor(dunit, dtype=torch.float32)
        for L in layers:
            def make(L):
                def fn(val, hook):
                    out = val.clone()
                    d = dvec.to(val.device).to(val.dtype)
                    proj = (out[:, -1, :] * d).sum(-1, keepdim=True)
                    excess = torch.clamp(proj - threshold, min=0.0)
                    out[:, -1, :] = out[:, -1, :] - cap_strength * excess * d
                    return out
                return fn
            hooks.append((f"blocks.{L}.hook_resid_post", make(L)))
    elif intervention == "mean_ablation" and mean_acts is not None:
        for L in layers:
            mv = torch.tensor(mean_acts[L], dtype=torch.float32)
            def make(mv):
                def fn(val, hook):
                    out = val.clone()
                    out[:, -1, :] = mv.to(val.device).to(val.dtype)
                    return out
                return fn
            hooks.append((f"blocks.{L}.hook_resid_post", make(mv)))
    return hooks


def generate_or_score_outputs(adapter, prompts: List[str], intervention: Optional[dict] = None,
                              max_new_tokens: int = 30, max_length: int = 256) -> List[str]:
    """Greedy-generate a short continuation per prompt (TL). intervention=None -> baseline.

    intervention dict keys: intervention, layers, direction, alpha, threshold, cap_strength, mean_acts.
    """
    import torch
    from src.model_adapters import TransformerLensAdapter
    if not isinstance(adapter, TransformerLensAdapter):
        raise NotImplementedError("Side-effect generation is TransformerLens-only.")

    hooks = []
    if intervention:
        hooks = _build_intervention_hook(
            adapter, intervention.get("intervention", ""), intervention.get("layers", []),
            intervention.get("direction"), intervention.get("alpha", 0.0),
            intervention.get("threshold", 0.0), intervention.get("cap_strength", 1.0),
            intervention.get("mean_acts"))

    outputs = []
    for prompt in prompts:
        ids = adapter.tokenizer.encode(prompt, return_tensors="pt").to(adapter.device)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                if hooks:
                    with adapter.model.hooks(fwd_hooks=hooks):
                        logits = adapter.model(ids)
                else:
                    logits = adapter.model(ids)
            nxt = int(logits[0, -1].argmax().item())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            if ids.shape[1] >= max_length:
                break
        gen_ids = ids[0, len(adapter.tokenizer.encode(prompt)):]
        outputs.append(adapter.tokenizer.decode(gen_ids).strip())
    return outputs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_repetition_score(text: str) -> float:
    """Fraction of tokens that are immediate repeats / low unique-ratio. 0=diverse, 1=fully repeated."""
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    unique_ratio = len(set(toks)) / len(toks)
    return float(1.0 - unique_ratio)


def compute_weirdness_flags(text: str) -> Dict[str, int]:
    toks = text.split()
    flags = {
        "empty": int(len(text.strip()) == 0),
        "repeated": int(len(toks) >= 4 and len(set(toks)) / len(toks) < 0.4),
        "incoherent": int(bool(re.search(r"(.)\1{6,}", text))),  # long char runs
        "very_short": int(0 < len(toks) < 2),
    }
    return flags


def _qa_correct(output: str, answer: Optional[str]) -> Optional[int]:
    if not answer:
        return None
    return int(answer.lower() in output.lower())


def compute_side_effect_score(baseline_outputs: List[str], intervened_outputs: List[str],
                              answers: Optional[List[Optional[str]]] = None) -> Dict[str, float]:
    """Aggregate side-effect score in [0,1] (higher = worse) plus component metrics."""
    n = min(len(baseline_outputs), len(intervened_outputs))
    if n == 0:
        return {"side_effect_score": float("nan")}
    len_ratios, rep_deltas, weird_rate, acc_drop = [], [], [], []
    weird_total = 0
    for i in range(n):
        b, a = baseline_outputs[i], intervened_outputs[i]
        bl = max(len(b.split()), 1)
        len_ratios.append(len(a.split()) / bl)
        rep_deltas.append(max(0.0, compute_repetition_score(a) - compute_repetition_score(b)))
        wf = compute_weirdness_flags(a)
        weird_total += int(any(wf.values()))
        if answers and i < len(answers):
            cb = _qa_correct(b, answers[i]); ca = _qa_correct(a, answers[i])
            if cb is not None and ca is not None:
                acc_drop.append(max(0, cb - ca))
    weird_rate_val = weird_total / n
    # length deviation from 1.0 (both shorter and much longer are suspicious)
    len_dev = float(np.mean([abs(r - 1.0) for r in len_ratios]))
    rep_component = float(np.mean(rep_deltas))
    acc_component = float(np.mean(acc_drop)) if acc_drop else 0.0
    # weighted side-effect score, clipped to [0,1]
    score = float(np.clip(0.4 * weird_rate_val + 0.3 * min(len_dev, 1.0)
                          + 0.2 * rep_component + 0.1 * acc_component, 0, 1))
    return {
        "side_effect_score": score,
        "weirdness_rate": weird_rate_val,
        "output_length_ratio": float(np.mean(len_ratios)),
        "repetition_increase": rep_component,
        "qa_accuracy_drop": acc_component,
        "n": n,
    }


def save_side_effect_samples(prompts: List[str], baseline: List[str], intervened: List[str],
                             path: Path) -> None:
    rows = [{"prompt": p, "baseline_output": b, "intervened_output": i}
            for p, b, i in zip(prompts, baseline, intervened)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved %d side-effect samples -> %s", len(rows), path)
