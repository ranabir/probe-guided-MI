"""Factory: loads the right adapter for a given model name."""
import logging
from typing import Optional

import torch

from src.model_adapters import BaseModelAdapter, HuggingFaceAdapter, TransformerLensAdapter
from src.model_registry import ModelConfig, get_model_config
from src.utils import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


def load_adapter(
    model_name: str,
    device_str: str = "auto",
    dtype_str: str = "auto",
    backend_override: Optional[str] = None,
) -> BaseModelAdapter:
    """Instantiate and load the correct adapter for model_name.

    Args:
        model_name: Short name ('gpt2-small') or HuggingFace path.
        device_str: 'auto' | 'cpu' | 'cuda' | 'mps'.
        dtype_str: 'auto' | 'float32' | 'float16' | 'bfloat16'.
        backend_override: Force a specific backend regardless of registry.

    Returns:
        A loaded BaseModelAdapter with model/tokenizer ready.
    """
    config = get_model_config(model_name)
    device = resolve_device(device_str)
    dtype = _resolve_dtype(dtype_str, config, device)

    backend = backend_override or config.backend
    logger.info(
        "Loading model=%s  backend=%s  device=%s  dtype=%s",
        model_name, backend, device, dtype,
    )

    if backend == "transformer_lens":
        adapter = TransformerLensAdapter(config, device, dtype)
    else:
        adapter = _make_hf_adapter(config, device, dtype)

    adapter.load_model_and_tokenizer()
    return adapter


def _resolve_dtype(dtype_str: str, config: ModelConfig, device: torch.device) -> torch.dtype:
    if dtype_str != "auto":
        return resolve_dtype(dtype_str)

    # MPS does not support float16 reliably; use float32
    if device.type == "mps":
        return torch.float32

    raw = config.default_dtype
    return resolve_dtype(raw, default="float32")


def _make_hf_adapter(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> HuggingFaceAdapter:
    family = config.family
    if family == "qwen":
        try:
            from src.model_adapters import HuggingFaceAdapter as QwenAdapter
            return QwenAdapter(config, device, dtype)
        except ImportError:
            pass
    if family == "gemma":
        try:
            from src.model_adapters import HuggingFaceAdapter as GemmaAdapter
            return GemmaAdapter(config, device, dtype)
        except ImportError:
            pass
    return HuggingFaceAdapter(config, device, dtype)
