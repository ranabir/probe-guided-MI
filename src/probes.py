"""Linear probe training, evaluation, and attribution-probe selection.

Supports two probe tasks:
  - classification: hidden_state -> binary label (LogisticRegression)
  - regression:     hidden_state -> continuous target, e.g. behavior_margin (Ridge)
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.utils import artifact_tag, artifacts_dir, safe_model_name

logger = logging.getLogger(__name__)


class LinearProbe:
    """Linear probe wrapping sklearn LogisticRegression (classification) or Ridge (regression)."""

    def __init__(self, standardize: bool = True, max_iter: int = 1000, seed: int = 42,
                 task: str = "classification"):
        self.standardize = standardize
        self.max_iter = max_iter
        self.seed = seed
        self.task = task  # "classification" | "regression"
        self.scaler: Optional[StandardScaler] = StandardScaler() if standardize else None
        self.clf = None
        self.layer_idx: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.task == "regression":
            self.clf = Ridge(alpha=1.0, random_state=self.seed)
        else:
            self.clf = LogisticRegression(
                max_iter=self.max_iter, random_state=self.seed, C=1.0, solver="lbfgs"
            )
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Classification only: P(positive class)."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Unified scalar output: regression -> predicted value; classification -> P(positive)."""
        if self.task == "regression":
            return self.predict(X)
        return self.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.task == "regression":
            from scipy.stats import pearsonr, spearmanr
            preds = self.predict(X)
            has_var = len(np.unique(y)) > 1 and len(np.unique(preds)) > 1
            return {
                "mse": float(mean_squared_error(y, preds)),
                "mae": float(mean_absolute_error(y, preds)),
                "r2": float(r2_score(y, preds)) if len(y) > 1 else 0.0,
                "pearson": float(pearsonr(y, preds)[0]) if has_var else 0.0,
                "spearman": float(spearmanr(y, preds)[0]) if has_var else 0.0,
            }
        preds = self.predict(X)
        n_classes = len(np.unique(y))
        if n_classes > 2:
            # Multiclass (e.g. topic control): accuracy + macro-F1, no binary AUROC
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "f1": float(f1_score(y, preds, average="macro", zero_division=0)),
                "auroc": float("nan"),
            }
        proba = self.predict_proba(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auroc": float(roc_auc_score(y, proba)) if n_classes > 1 else 0.5,
        }

    @property
    def coef_(self):
        return self.clf.coef_

    @property
    def intercept_(self):
        return np.atleast_1d(self.clf.intercept_)


# ---------------------------------------------------------------------------
# Path helpers — accept optional probe_position + input_format
# ---------------------------------------------------------------------------

def _ptag(probe_position: Optional[str], input_format: Optional[str]) -> str:
    tag = artifact_tag(input_format, probe_position)
    return f"_{tag}" if tag else ""


def probe_path(model_name: str, layer: int,
               probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    base = artifacts_dir("probes")
    return base / f"{safe_model_name(model_name)}{_ptag(probe_position, input_format)}_probe_layer_{layer}.pkl"


def best_probe_path(model_name: str,
                    probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    base = artifacts_dir("probes")
    return base / f"{safe_model_name(model_name)}{_ptag(probe_position, input_format)}_best_probe.pkl"


def selection_metadata_path(model_name: str,
                            probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    base = artifacts_dir("probes")
    return base / f"{safe_model_name(model_name)}{_ptag(probe_position, input_format)}_selected_probe_metadata.json"


def save_probe(probe: LinearProbe, model_name: str, layer: int,
               probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    path = probe_path(model_name, layer, probe_position, input_format)
    with open(path, "wb") as f:
        pickle.dump(probe, f)
    return path


def load_probe(model_name: str, layer: int,
               probe_position: Optional[str] = None, input_format: Optional[str] = None) -> LinearProbe:
    path = probe_path(model_name, layer, probe_position, input_format)
    if not path.exists():
        legacy = probe_path(model_name, layer, probe_position, None)
        if legacy.exists():
            logger.warning("Loading fallback probe: %s", legacy)
            with open(legacy, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Probe not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_best_probe(probe: LinearProbe, model_name: str,
                    probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    path = best_probe_path(model_name, probe_position, input_format)
    with open(path, "wb") as f:
        pickle.dump(probe, f)
    logger.info("Saved best probe (layer=%s) -> %s", probe.layer_idx, path)
    return path


def load_best_probe(model_name: str,
                    probe_position: Optional[str] = None, input_format: Optional[str] = None) -> LinearProbe:
    path = best_probe_path(model_name, probe_position, input_format)
    if not path.exists():
        for legacy in (
            best_probe_path(model_name, probe_position, None),
            best_probe_path(model_name, None, None),
        ):
            if legacy.exists():
                logger.warning("Loading fallback best probe: %s", legacy)
                with open(legacy, "rb") as f:
                    return pickle.load(f)
        cmd = f"python scripts/03_train_probe.py --model_name {model_name}"
        if input_format:
            cmd += f" --input_format {input_format}"
        if probe_position:
            cmd += f" --probe_position {probe_position}"
        raise FileNotFoundError(f"Best probe not found: {path}\nRun: {cmd}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Attribution probe selection policy
# ---------------------------------------------------------------------------

def select_attribution_probe(
    metrics_list: List[Dict],
    n_layers: int,
    model_name: str,
    probe_position: Optional[str] = None,
    input_format: Optional[str] = None,
    policy: str = "best_late",
    min_probe_layer_frac: float = 0.65,
    metric_col: str = "val_auroc",
) -> Tuple[LinearProbe, Dict]:
    """Select which probe layer to use for attribution gradients.

    metric_col is the validation metric to maximize. Higher is better for both
    'val_auroc' (classification) and 'val_pearson' (regression).
    """
    if policy not in ("best_any", "best_late", "final_layer"):
        raise ValueError(f"Unknown policy: {policy!r}. Use best_any | best_late | final_layer")

    import pandas as pd
    df = pd.DataFrame(metrics_list)
    if df.empty:
        raise ValueError("metrics_list is empty; cannot select an attribution probe.")
    if metric_col not in df.columns:
        metric_col = "val_auroc" if "val_auroc" in df.columns else df.columns[1]

    best_any_row = df.loc[df[metric_col].idxmax()]

    if policy == "best_any":
        selected_row = best_any_row
        reason = f"best {metric_col} across all layers"
    elif policy == "final_layer":
        selected_row = df.iloc[-1]
        reason = "final layer (policy=final_layer)"
    elif policy == "best_late":
        min_layer = int(min_probe_layer_frac * n_layers)
        late_df = df[df["layer"] >= min_layer]
        if late_df.empty:
            late_df = df
            reason = f"best_late fallback (no layers >= {min_layer})"
        else:
            reason = f"best {metric_col} among layers >= {min_layer} ({policy})"
        selected_row = late_df.loc[late_df[metric_col].idxmax()]
    else:
        raise ValueError(f"Unknown policy: {policy!r}. Use best_any | best_late | final_layer")

    selected_layer = int(selected_row["layer"])
    probe = load_probe(model_name, selected_layer, probe_position, input_format)
    probe.layer_idx = selected_layer

    meta = {
        "metric_col": metric_col,
        "best_any_layer": int(best_any_row["layer"]),
        "best_any_metric": float(best_any_row[metric_col]),
        "selected_policy": policy,
        "selected_layer": selected_layer,
        "selected_metric": float(selected_row[metric_col]),
        "min_probe_layer_frac": min_probe_layer_frac,
        "reason": reason,
    }
    logger.info(
        "Attribution probe: policy=%s → layer %d (%s=%.4f) | best_any=layer %d (%.4f)",
        policy, selected_layer, metric_col, meta["selected_metric"],
        meta["best_any_layer"], meta["best_any_metric"],
    )
    return probe, meta


def save_selection_metadata(meta: dict, model_name: str,
                            probe_position: Optional[str] = None, input_format: Optional[str] = None) -> Path:
    path = selection_metadata_path(model_name, probe_position, input_format)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved probe selection metadata -> %s", path)
    return path


def load_selection_metadata(model_name: str,
                            probe_position: Optional[str] = None, input_format: Optional[str] = None) -> dict:
    path = selection_metadata_path(model_name, probe_position, input_format)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Train all layers
# ---------------------------------------------------------------------------

def train_all_layer_probes(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    model_name: str,
    probe_position: Optional[str] = None,
    input_format: Optional[str] = None,
    task: str = "classification",
    standardize: bool = True,
    max_iter: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict], LinearProbe]:
    """Train one probe per layer. Save all. Return metrics list and best probe.

    For classification, best = highest val AUROC.
    For regression, best = highest val Pearson correlation.
    """
    n_layers = train_X.shape[1]
    metrics_list = []
    best_metric = -np.inf
    best_probe = None
    best_key = "val_pearson" if task == "regression" else "val_auroc"

    for layer in range(n_layers):
        probe = LinearProbe(standardize=standardize, max_iter=max_iter, seed=seed, task=task)
        probe.fit(train_X[:, layer, :], train_y)
        probe.layer_idx = layer

        tr = probe.evaluate(train_X[:, layer, :], train_y)
        vl = probe.evaluate(val_X[:, layer, :], val_y)
        te = probe.evaluate(test_X[:, layer, :], test_y)

        row = {"layer": layer}
        for split_name, m in (("train", tr), ("val", vl), ("test", te)):
            for k, v in m.items():
                row[f"{split_name}_{k}"] = v
        metrics_list.append(row)
        save_probe(probe, model_name, layer, probe_position, input_format)

        cur = vl.get(best_key.replace("val_", ""), -np.inf)
        if cur > best_metric:
            best_metric = cur
            best_probe = probe

        if task == "regression":
            logger.info("Layer %2d | train_pearson=%.3f val_pearson=%.3f val_r2=%.3f",
                        layer, tr.get("pearson", 0), vl.get("pearson", 0), vl.get("r2", 0))
        else:
            logger.info("Layer %2d | train_acc=%.3f val_acc=%.3f val_auroc=%.3f",
                        layer, tr["accuracy"], vl["accuracy"], vl["auroc"])

    if best_probe:
        save_best_probe(best_probe, model_name, probe_position, input_format)

    return metrics_list, best_probe
