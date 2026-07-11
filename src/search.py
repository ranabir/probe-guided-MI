"""Intervention search: find (layer-selection × direction × intervention × strength) configs that
maximize targeted answer-flip while minimizing side effects.

Objective: score = targeted_syc_to_honest_flip_rate − lambda_side_effect * side_effect_score.

Reuses src.causal_interventions for the interventions and src.side_effects for the side-effect
penalty (estimated once per config from a small basic-prompt set).
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_intervention_grid(layers_by_selection: Dict[str, List[int]],
                            directions: List[str], interventions: List[str],
                            alphas: List[float], cap_quantiles: List[float]) -> List[dict]:
    """Cartesian grid of intervention configs. Steering uses alphas; capping uses cap_quantiles."""
    grid = []
    for selection, layers in layers_by_selection.items():
        for direction in directions:
            for intervention in interventions:
                if intervention == "probe_steering":
                    for a in alphas:
                        grid.append({"layer_selection": selection, "layers": layers,
                                     "direction": direction, "intervention": intervention,
                                     "alpha_or_cap": a})
                elif intervention == "activation_capping":
                    for q in cap_quantiles:
                        grid.append({"layer_selection": selection, "layers": layers,
                                     "direction": direction, "intervention": intervention,
                                     "alpha_or_cap": q})
                elif intervention == "contrastive_patching":
                    grid.append({"layer_selection": selection, "layers": layers,
                                 "direction": direction, "intervention": intervention,
                                 "alpha_or_cap": None})
    return grid


def compute_objective(targeted_flip: float, side_effect: float, lambda_side_effect: float) -> float:
    tf = 0.0 if (targeted_flip is None or not np.isfinite(targeted_flip)) else targeted_flip
    se = 0.0 if (side_effect is None or not np.isfinite(side_effect)) else side_effect
    return float(tf - lambda_side_effect * se)


def rank_interventions(results: List[dict], lambda_side_effect: float = 0.5) -> pd.DataFrame:
    """Attach objective_score and rank descending."""
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["objective_score"] = [
        compute_objective(r.get("targeted_flip_rate"), r.get("side_effect_score", 0.0), lambda_side_effect)
        for _, r in df.iterrows()
    ]
    df = df.sort_values("objective_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
