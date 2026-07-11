"""Tests for model registry and loader (no actual model loading)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_registry import get_model_config, list_registered_models


def test_registry_loads():
    models = list_registered_models()
    assert len(models) > 0


def test_gpt2_small_config():
    cfg = get_model_config("gpt2-small")
    assert cfg.backend == "transformer_lens"
    assert cfg.family == "gpt2"
    assert cfg.default_dtype == "float32"
    assert cfg.chat_template is False


def test_pythia_config():
    cfg = get_model_config("EleutherAI/pythia-410m")
    assert cfg.backend == "transformer_lens"
    assert cfg.family == "pythia"


def test_qwen_config():
    cfg = get_model_config("Qwen/Qwen2.5-0.5B-Instruct")
    assert cfg.backend == "huggingface"
    assert cfg.family == "qwen"
    assert cfg.chat_template is True
    assert cfg.trust_remote_code is True


def test_gemma_config():
    cfg = get_model_config("google/gemma-2-2b-it")
    assert cfg.backend == "huggingface"
    assert cfg.family == "gemma"
    assert cfg.chat_template is True


def test_unknown_model_fallback():
    cfg = get_model_config("some-unknown-model/v1")
    assert cfg.backend in ("huggingface", "transformer_lens")


def test_safe_model_name():
    from src.utils import safe_model_name
    assert safe_model_name("EleutherAI/pythia-410m") == "EleutherAI_pythia-410m"
    assert safe_model_name("gpt2-small") == "gpt2-small"
    assert safe_model_name("google/gemma-2-2b-it") == "google_gemma-2-2b-it"
