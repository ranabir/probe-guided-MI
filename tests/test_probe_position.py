"""Tests verifying prompt_final vs response_final activation naming and paths."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_cache import cache_path, metadata_path, save_activations, load_activations
from src.probes import probe_path, best_probe_path
from src.utils import safe_model_name


def test_cache_path_no_probe_position():
    p = cache_path("gpt2-small", "train", probe_position=None)
    assert "gpt2-small_train_activations" in p.name
    assert "prompt" not in p.name


def test_cache_path_with_probe_position():
    p = cache_path("gpt2-small", "train", probe_position="prompt_final")
    assert "prompt_final" in p.name
    assert "gpt2-small_prompt_final_train_activations" in p.name


def test_cache_path_response_final():
    p = cache_path("gpt2-small", "val", probe_position="response_final")
    assert "response_final" in p.name


def test_probe_path_with_position():
    p = probe_path("gpt2-small", 7, probe_position="prompt_final")
    assert "prompt_final" in p.name
    assert "probe_layer_7" in p.name


def test_best_probe_path_legacy():
    p = best_probe_path("gpt2-small", probe_position=None)
    assert "prompt_final" not in p.name
    assert "response_final" not in p.name


def test_metadata_path_with_position():
    p = metadata_path("EleutherAI/pythia-410m", probe_position="prompt_final")
    assert "prompt_final" in p.name
    assert "EleutherAI_pythia-410m" in p.name


def test_save_load_roundtrip(tmp_path):
    """Save activations with probe_position and reload them."""
    import torch
    from unittest.mock import patch

    acts = {"hidden_states": np.random.randn(10, 4, 8).astype(np.float32)}
    labels = np.zeros(10, dtype=np.int64)

    # Patch artifacts_dir to use tmp_path
    with patch("src.activation_cache.artifacts_dir", return_value=tmp_path):
        save_activations("mymodel", "train", acts, labels, probe_position="prompt_final")
        data = load_activations("mymodel", "train", probe_position="prompt_final")

    assert "hidden_states" in data
    assert data["hidden_states"].shape == (10, 4, 8)


def test_legacy_fallback(tmp_path):
    """If probe_position path is missing, load_activations falls back to legacy."""
    import torch
    from unittest.mock import patch

    acts = {"hidden_states": np.random.randn(5, 3, 4).astype(np.float32)}
    labels = np.zeros(5, dtype=np.int64)

    with patch("src.activation_cache.artifacts_dir", return_value=tmp_path):
        # Save WITHOUT probe_position (legacy)
        save_activations("mymodel2", "test", acts, labels, probe_position=None)
        # Load WITH probe_position — should fall back to legacy
        data = load_activations("mymodel2", "test", probe_position="prompt_final")

    assert "hidden_states" in data
