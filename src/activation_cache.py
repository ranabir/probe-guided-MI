"""Save and load activation caches (pt files + metadata).

Naming convention (v3):
  {safe_model_name}_{tag}_{split}_activations.pt
where tag = artifact_tag(input_format, probe_position), e.g.:
  - "response_final"                      (paired_rows + response_final)
  - "prompt_preferences_prompt_final"     (prompt_preferences + prompt_final)
  - ""                                    (legacy v1, no tag)

All load functions fall back to the legacy / less-tagged path if the requested path is missing.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils import artifact_tag, artifacts_dir, safe_model_name

logger = logging.getLogger(__name__)


def _tagged(model_name: str, split: str, suffix: str,
            input_format: Optional[str], probe_position: Optional[str]) -> Path:
    base = artifacts_dir("activations")
    sn = safe_model_name(model_name)
    tag = artifact_tag(input_format, probe_position)
    name = f"{sn}_{tag}_{split}_{suffix}" if tag else f"{sn}_{split}_{suffix}"
    return base / name


def cache_path(model_name: str, split: str,
               probe_position: Optional[str] = None,
               input_format: Optional[str] = None) -> Path:
    return _tagged(model_name, split, "activations.pt", input_format, probe_position)


def metadata_path(model_name: str,
                  probe_position: Optional[str] = None,
                  input_format: Optional[str] = None) -> Path:
    base = artifacts_dir("activations")
    sn = safe_model_name(model_name)
    tag = artifact_tag(input_format, probe_position)
    name = f"{sn}_{tag}_metadata.json" if tag else f"{sn}_metadata.json"
    return base / name


def save_activations(
    model_name: str,
    split: str,
    activations: Dict[str, np.ndarray],
    labels: np.ndarray,
    ids: Optional[List[str]] = None,
    probe_position: Optional[str] = None,
    input_format: Optional[str] = None,
    behavior_margins: Optional[np.ndarray] = None,
) -> Path:
    path = cache_path(model_name, split, probe_position, input_format)
    payload = {key: torch.from_numpy(val) for key, val in activations.items()}
    payload["labels"] = torch.from_numpy(labels.astype(np.int64))
    if ids is not None:
        payload["ids"] = ids
    if behavior_margins is not None:
        payload["behavior_margin"] = torch.from_numpy(np.asarray(behavior_margins, dtype=np.float32))
    torch.save(payload, path)
    logger.info("Saved %s activations -> %s", split, path)
    return path


def load_activations(
    model_name: str,
    split: str,
    probe_position: Optional[str] = None,
    input_format: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Load activations. Falls back to less-tagged paths if the exact path is missing."""
    primary = cache_path(model_name, split, probe_position, input_format)
    if primary.exists():
        logger.info("Loaded %s activations from %s", split, primary)
        return torch.load(primary, map_location="cpu", weights_only=False)

    # Fallbacks: drop input_format, then drop probe_position
    candidates = []
    if input_format:
        candidates.append(cache_path(model_name, split, probe_position, None))
    if probe_position:
        candidates.append(cache_path(model_name, split, None, None))
    for legacy in candidates:
        if legacy.exists():
            logger.warning("Requested cache missing; loading fallback: %s", legacy)
            return torch.load(legacy, map_location="cpu", weights_only=False)

    cmd = f"python scripts/02_cache_activations.py --model_name {model_name}"
    if input_format:
        cmd += f" --input_format {input_format}"
    if probe_position:
        cmd += f" --probe_position {probe_position}"
    raise FileNotFoundError(f"Activation cache not found: {primary}\nRun: {cmd}")


def save_metadata(model_name: str, meta: dict,
                  probe_position: Optional[str] = None,
                  input_format: Optional[str] = None) -> Path:
    path = metadata_path(model_name, probe_position, input_format)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata -> %s", path)
    return path


def load_metadata(model_name: str,
                  probe_position: Optional[str] = None,
                  input_format: Optional[str] = None) -> dict:
    for path in (
        metadata_path(model_name, probe_position, input_format),
        metadata_path(model_name, probe_position, None),
        metadata_path(model_name, None, None),
    ):
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


def activations_to_numpy(data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }
