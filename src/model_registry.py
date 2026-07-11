"""Model registry: lookup backend, family, and config for a given model name."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import load_model_registry

logger = logging.getLogger(__name__)


class ModelConfig:
    """Typed wrapper around a model registry entry."""

    def __init__(self, model_name: str, entry: Dict[str, Any]):
        self.model_name = model_name
        self.backend: str = entry.get("backend", "huggingface")
        self.family: str = entry.get("family", "unknown")
        self.hf_name: str = entry.get("hf_name", model_name)
        self.display_name: str = entry.get("display_name", model_name)
        self.default_dtype: str = entry.get("default_dtype", "float32")
        self.trust_remote_code: bool = entry.get("trust_remote_code", False)
        self.chat_template: bool = entry.get("chat_template", False)
        self.layer_attr: str = entry.get("layer_attr", "model.layers")
        self.hidden_attr: str = entry.get("hidden_attr", "hidden_states")
        self.n_layers: Optional[int] = entry.get("n_layers", None)
        self.d_model: Optional[int] = entry.get("d_model", None)
        # Post-training stage: base | sft | dpo | rlhf | instruct  (reviewer feedback #2,#3)
        self.training_stage: str = entry.get("training_stage", "base")

    def __repr__(self) -> str:
        return (
            f"ModelConfig(name={self.model_name!r}, backend={self.backend!r}, "
            f"family={self.family!r}, dtype={self.default_dtype!r})"
        )


def get_model_config(model_name: str, registry_path: Optional[Path] = None) -> ModelConfig:
    """Look up model_name in registry; return a best-guess config if not found."""
    registry = load_model_registry(registry_path)

    if model_name in registry:
        return ModelConfig(model_name, registry[model_name])

    # Partial match (e.g. user gave 'pythia-410m' instead of full path)
    for key, entry in registry.items():
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            logger.info("Model %r matched registry key %r", model_name, key)
            return ModelConfig(model_name, {**entry, "hf_name": model_name})

    # Heuristic fallback
    logger.warning(
        "Model %r not found in registry. Inferring backend from name.", model_name
    )
    backend = _infer_backend(model_name)
    family = _infer_family(model_name)
    dtype = "float32" if family in ("gpt2",) else "float16"
    return ModelConfig(
        model_name,
        {
            "backend": backend,
            "family": family,
            "hf_name": model_name,
            "display_name": model_name,
            "default_dtype": dtype,
            "trust_remote_code": "qwen" in model_name.lower(),
            "chat_template": any(k in model_name.lower() for k in ("instruct", "-it", "chat")),
        },
    )


def _infer_backend(model_name: str) -> str:
    tl_models = ("gpt2", "pythia", "gpt-neo", "gpt-j", "opt", "bloom")
    name_lower = model_name.lower()
    if any(k in name_lower for k in tl_models):
        return "transformer_lens"
    return "huggingface"


def _infer_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "gpt2" in name_lower or "gpt-2" in name_lower:
        return "gpt2"
    if "pythia" in name_lower:
        return "pythia"
    if "qwen" in name_lower:
        return "qwen"
    if "gemma" in name_lower:
        return "gemma"
    if "llama" in name_lower:
        return "llama"
    if "mistral" in name_lower:
        return "mistral"
    return "unknown"


def list_registered_models(registry_path: Optional[Path] = None) -> Dict[str, ModelConfig]:
    registry = load_model_registry(registry_path)
    return {name: ModelConfig(name, entry) for name, entry in registry.items()}
