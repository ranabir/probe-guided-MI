"""Shared utilities: config loading, path helpers, safe name conversion, device selection."""
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent


def get_root() -> Path:
    return ROOT


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    path = registry_path or ROOT / "model_registry.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def safe_model_name(model_name: str) -> str:
    """Convert a model name like 'EleutherAI/pythia-410m' to 'EleutherAI_pythia-410m'."""
    return re.sub(r"[/\\]", "_", model_name)


def artifact_tag(input_format: Optional[str] = None, probe_position: Optional[str] = None) -> str:
    """Build the filename tag from (input_format, probe_position).

    Backward-compatible rules:
      - input_format is omitted when None or 'paired_rows' (the default/legacy format)
      - probe_position is included when present

    Examples:
      (None, None)                         -> ""                              (legacy)
      ('paired_rows', 'response_final')    -> "response_final"                (legacy v2)
      ('prompt_preferences','prompt_final')-> "prompt_preferences_prompt_final"
    """
    parts = []
    if input_format and input_format != "paired_rows":
        parts.append(input_format)
    if probe_position:
        parts.append(probe_position)
    return "_".join(parts)


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str, default: str = "float32") -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": None,
    }
    return mapping.get(dtype_str or default, torch.float32)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def artifacts_dir(sub: str) -> Path:
    p = ROOT / "artifacts" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(sub: str) -> Path:
    p = ROOT / "results" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
