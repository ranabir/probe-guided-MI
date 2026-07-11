"""Logit-gradient baseline for sycophancy attribution.

Instead of a trained probe, this baseline defines sycophancy via a fixed token contrast:

    logit_margin = mean(logits[syc_tokens]) - mean(logits[non_syc_tokens])

and attributes that margin to layers by gradient. It is the "naive" comparison point that
probe-gradient attribution should beat (or at least differ from) if the probe is adding value.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.attribution import compute_layer_attribution, logit_gradient_baseline, top_k_layers
from src.utils import artifacts_dir, results_dir, safe_model_name

logger = logging.getLogger(__name__)

SYC_TOKENS = ["Yes", "Absolutely", "Correct", "Indeed", "Agree", "Right"]
NON_SYC_TOKENS = ["No", "Actually", "Incorrect", "However", "Wrong"]


def resolve_token_ids(tokenizer, words: List[str]) -> Tuple[List[int], List[str]]:
    """Map words to single token IDs where possible.

    Tries both the bare word and a leading-space variant (common in BPE tokenizers).
    Returns (unique_ids, skipped_words). Multi-token words are still used via their first
    token but logged as approximations.
    """
    ids: List[int] = []
    skipped: List[str] = []
    for w in words:
        candidates = []
        for variant in (w, " " + w):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                enc = []
            if enc:
                candidates.append((variant, enc))
        if not candidates:
            skipped.append(w)
            continue
        # Prefer a single-token encoding if any variant gives one
        single = [enc[0] for variant, enc in candidates if len(enc) == 1]
        if single:
            ids.append(single[0])
        else:
            # fall back to first token of the first candidate; log as multi-token
            ids.append(candidates[0][1][0])
            logger.info("Token %r is multi-token %s; using first token id %d",
                        w, candidates[0][1], candidates[0][1][0])
    return list(dict.fromkeys(ids)), skipped


def compute_logit_gradient_attribution(
    adapter,
    texts: List[str],
    model_name: str,
    probe_position: str = "prompt_final",
    input_format: str = "prompt_preferences",
    max_length: int = 128,
    top_k: int = 20,
    syc_tokens: Optional[List[str]] = None,
    non_syc_tokens: Optional[List[str]] = None,
) -> Dict:
    """Compute layer attribution for the logit margin and persist it.

    Returns dict with: layer_scores (np.ndarray), top_layers (list), syc_ids, non_syc_ids, skipped.
    """
    syc_tokens = syc_tokens or SYC_TOKENS
    non_syc_tokens = non_syc_tokens or NON_SYC_TOKENS

    syc_ids, syc_skip = resolve_token_ids(adapter.tokenizer, syc_tokens)
    non_ids, non_skip = resolve_token_ids(adapter.tokenizer, non_syc_tokens)
    logger.info("Logit baseline tokens: syc_ids=%s non_ids=%s", syc_ids, non_ids)
    if syc_skip or non_skip:
        logger.warning("Skipped tokens (no encoding): syc=%s non=%s", syc_skip, non_skip)

    grads = logit_gradient_baseline(
        adapter, texts, max_length=max_length, token_position="final",
        syc_tokens=syc_tokens, non_syc_tokens=non_syc_tokens,
    )  # [N, L, D]

    # Use grad-norm attribution (no probe activations to multiply against here)
    layer_scores = compute_layer_attribution(grads, grads, method="grad_norm")
    top_layers = [int(x) for x in top_k_layers(layer_scores, k=top_k)]

    sn = safe_model_name(model_name)
    tag = f"{input_format}_{probe_position}" if input_format == "prompt_preferences" else probe_position

    # Save attribution table (artifacts + results/tables)
    rows = [{"model_name": model_name, "component": "hidden_states", "layer": i,
             "attribution_score": float(layer_scores[i])} for i in range(len(layer_scores))]
    df = pd.DataFrame(rows).sort_values("attribution_score", ascending=False)
    art_path = Path(artifacts_dir("attribution")) / f"{sn}_{tag}_logit_gradient_layer_attribution.csv"
    tab_path = Path(results_dir("tables")) / f"{sn}_{tag}_logit_gradient_layer_attribution.csv"
    df.to_csv(art_path, index=False)
    df.to_csv(tab_path, index=False)

    # Save the token ids actually used
    meta = {"syc_tokens": syc_tokens, "non_syc_tokens": non_syc_tokens,
            "syc_ids": syc_ids, "non_syc_ids": non_ids,
            "skipped_syc": syc_skip, "skipped_non": non_skip,
            "top_layers": top_layers}
    with open(Path(artifacts_dir("attribution")) / f"{sn}_{tag}_logit_gradient_tokens.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save top layers for the causal-validation step to consume
    with open(Path(artifacts_dir("attribution")) / f"{sn}_{tag}_logit_gradient_top_layers.txt", "w") as f:
        f.write(",".join(str(l) for l in top_layers))

    logger.info("Logit-gradient top-%d layers: %s", top_k, top_layers)
    return {"layer_scores": layer_scores, "top_layers": top_layers,
            "syc_ids": syc_ids, "non_syc_ids": non_ids, "skipped": syc_skip + non_skip}


def load_logit_gradient_top_layers(model_name: str, probe_position: str = "prompt_final",
                                   input_format: str = "prompt_preferences") -> Optional[List[int]]:
    sn = safe_model_name(model_name)
    tag = f"{input_format}_{probe_position}" if input_format == "prompt_preferences" else probe_position
    path = Path(artifacts_dir("attribution")) / f"{sn}_{tag}_logit_gradient_top_layers.txt"
    if not path.exists():
        return None
    with open(path) as f:
        return [int(x) for x in f.read().strip().split(",") if x.strip()]
