"""Bootstrap confidence intervals over prompts/examples.

Used to put 95% error bars on probe correlations, behavior_margin deltas, answer-flip rates,
accuracy changes, and sycophancy rates. Bootstrapping resamples examples (not model runs), so it
captures sampling variability of the small evaluation sets we use.
"""
import logging
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000, ci: float = 0.95,
                      seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI for the mean of `values`.

    Returns dict: mean, ci_low, ci_high, se, n.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "se": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(arr[0]), "ci_low": float(arr[0]), "ci_high": float(arr[0]),
                "se": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi,
            "se": float(boot_means.std()), "n": n}


def bootstrap_metric_ci(metric_fn: Callable[[np.ndarray], float], data: Sequence,
                        n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI for an arbitrary metric computed over resampled `data`.

    `metric_fn` takes a resampled array/list and returns a scalar (e.g. a Pearson correlation
    computed on resampled (x, y) pairs). `data` is indexable by position.

    Returns dict: value, ci_low, ci_high, se, n.
    """
    data = list(data)
    n = len(data)
    if n == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "se": float("nan"), "n": 0}
    try:
        point = float(metric_fn(data))
    except Exception:
        point = float("nan")
    if n == 1:
        return {"value": point, "ci_low": point, "ci_high": point, "se": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [data[i] for i in idx]
        try:
            v = metric_fn(resampled)
            if np.isfinite(v):
                boots.append(float(v))
        except Exception:
            continue
    if not boots:
        return {"value": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "se": float("nan"), "n": n}
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return {"value": point, "ci_low": lo, "ci_high": hi, "se": float(np.std(boots)), "n": n}


def pearson_on_pairs(pairs: Sequence[Tuple[float, float]]) -> float:
    """Pearson correlation on a list of (x, y) pairs — convenient for bootstrap_metric_ci."""
    from scipy.stats import pearsonr
    arr = np.asarray(pairs, dtype=float)
    if len(arr) < 2 or np.unique(arr[:, 0]).size < 2 or np.unique(arr[:, 1]).size < 2:
        return float("nan")
    return float(pearsonr(arr[:, 0], arr[:, 1])[0])
