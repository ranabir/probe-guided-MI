"""Control probes to test whether the sycophancy probe is just following trivial confounds.

Three controls (reviewer feedback #4, #5):
  1. random-label   — shuffle behavior_margin across prompts. A real signal should collapse to ~0.
  2. static-feature — predict shallow surface features (prompt length, "I think", "do you agree",
                      A/B format, first token of the syc response). If these decode as well as
                      sycophancy, the probe may be reading format, not behavior.
  3. topic          — predict the source subset (philpapers / nlp_survey / political). If the
                      sycophancy probe's layer profile mirrors topic decoding, it may be topical.

All controls reuse the SAME per-layer linear probe, so decodability is directly comparable.
We report test Pearson for continuous targets (point-biserial for binary) on a common y-axis.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static surface-feature targets
# ---------------------------------------------------------------------------

def build_static_feature_targets(pref_df: pd.DataFrame, tokenizer=None) -> Dict[str, np.ndarray]:
    """Return {feature_name: per-prompt array} of shallow surface features.

    pref_df is a prompt-preference DataFrame (one row per prompt) with columns
    prompt, sycophantic_response, non_sycophantic_response.
    """
    prompts = pref_df["prompt"].astype(str).tolist()
    syc = pref_df.get("sycophantic_response", pd.Series([""] * len(pref_df))).astype(str).tolist()

    def _len_tokens(t):
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(t, add_special_tokens=False))
            except Exception:
                pass
        return len(t.split())

    feats: Dict[str, np.ndarray] = {}
    feats["prompt_length_tokens"] = np.array([_len_tokens(p) for p in prompts], dtype=np.float32)
    feats["contains_i_think"] = np.array([1.0 if "i think" in p.lower() or "i believe" in p.lower() else 0.0 for p in prompts], dtype=np.float32)
    feats["contains_do_you_agree"] = np.array([1.0 if "do you agree" in p.lower() or "agree or disagree" in p.lower() else 0.0 for p in prompts], dtype=np.float32)
    feats["contains_am_i_right"] = np.array([1.0 if "am i right" in p.lower() or "isn't it" in p.lower() or "right?" in p.lower() else 0.0 for p in prompts], dtype=np.float32)
    feats["syc_response_starts_A"] = np.array([1.0 if s.strip().startswith("(A)") else 0.0 for s in syc], dtype=np.float32)
    # first token of the sycophantic response (as an integer code) — a pure format/lexical feature
    if tokenizer is not None:
        first_ids = []
        for s in syc:
            try:
                enc = tokenizer.encode(s, add_special_tokens=False)
                first_ids.append(float(enc[0]) if enc else 0.0)
            except Exception:
                first_ids.append(0.0)
        feats["syc_first_token_id"] = np.array(first_ids, dtype=np.float32)
    return feats


def topic_codes(pref_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Integer code per prompt for its subset/topic, or None if unavailable."""
    col = "subset" if "subset" in pref_df.columns else ("source_dataset" if "source_dataset" in pref_df.columns else None)
    if col is None:
        return None
    cats = pd.Categorical(pref_df[col].astype(str))
    if len(cats.categories) < 2:
        return None
    return np.asarray(cats.codes, dtype=np.int64)


def shuffle_target(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Random-label control: permute the target across prompts."""
    rng = np.random.default_rng(seed)
    out = np.array(y, dtype=float).copy()
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Lightweight per-layer decodability (does NOT persist probes)
# ---------------------------------------------------------------------------

def layerwise_decodability(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    task: str = "regression", standardize: bool = True, seed: int = 42,
) -> List[Dict]:
    """Train a fresh probe per layer and return [{layer, test_metric, ...}].

    task='regression' -> reports test Pearson (works as point-biserial for binary targets).
    task='classification' (multiclass topic) -> reports test accuracy.
    """
    from src.probes import LinearProbe
    n_layers = train_X.shape[1]
    rows = []
    for layer in range(n_layers):
        probe = LinearProbe(standardize=standardize, seed=seed,
                            task=("regression" if task == "regression" else "classification"))
        try:
            probe.fit(train_X[:, layer, :], train_y)
            m = probe.evaluate(test_X[:, layer, :], test_y)
            if task == "regression":
                rows.append({"layer": layer, "test_metric": m.get("pearson", float("nan")),
                             "metric_name": "pearson"})
            else:
                rows.append({"layer": layer, "test_metric": m.get("accuracy", float("nan")),
                             "metric_name": "accuracy"})
        except Exception as e:
            logger.warning("control probe layer %d failed: %s", layer, e)
            rows.append({"layer": layer, "test_metric": float("nan"), "metric_name": "na"})
    return rows


def run_all_controls(
    splits: Dict[str, Dict[str, np.ndarray]],
    pref_splits: Dict[str, pd.DataFrame],
    tokenizer=None,
    controls: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute decodability-by-layer for each requested control.

    splits: {'train': {'hidden_states':..., 'behavior_margin':...}, 'test': {...}}
    pref_splits: {'train': df, 'test': df} aligned (same row order) to the activations.
    controls: subset of {'random_label','static_token','topic'}; default all.

    Returns long DataFrame: control_name, feature, layer, test_metric, metric_name.
    """
    controls = controls or ["random_label", "static_token", "topic"]
    tr, te = splits["train"], splits["test"]
    trX, teX = tr["hidden_states"], te["hidden_states"]
    out_rows = []

    # random-label control on behavior_margin
    if "random_label" in controls and "behavior_margin" in tr:
        y_tr = shuffle_target(np.asarray(tr["behavior_margin"]), seed=seed)
        y_te = shuffle_target(np.asarray(te["behavior_margin"]), seed=seed + 1)
        for r in layerwise_decodability(trX, y_tr, teX, y_te, task="regression", seed=seed):
            out_rows.append({"control_name": "random_label", "feature": "shuffled_margin", **r})

    # static-feature controls
    if "static_token" in controls:
        feats_tr = build_static_feature_targets(pref_splits["train"], tokenizer)
        feats_te = build_static_feature_targets(pref_splits["test"], tokenizer)
        for fname in feats_tr:
            if fname not in feats_te:
                continue
            for r in layerwise_decodability(trX, feats_tr[fname], teX, feats_te[fname],
                                            task="regression", seed=seed):
                out_rows.append({"control_name": "static_token", "feature": fname, **r})

    # topic control (multiclass accuracy)
    if "topic" in controls:
        t_tr = topic_codes(pref_splits["train"])
        t_te = topic_codes(pref_splits["test"])
        if t_tr is not None and t_te is not None:
            for r in layerwise_decodability(trX, t_tr, teX, t_te, task="classification", seed=seed):
                out_rows.append({"control_name": "topic", "feature": "subset", **r})
        else:
            logger.info("Topic control skipped (no subset column or <2 topics).")

    return pd.DataFrame(out_rows)
