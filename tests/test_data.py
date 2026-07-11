"""Tests for dataset generation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import generate_synthetic_dataset, split_dataset


def test_generate_synthetic_dataset_size():
    df = generate_synthetic_dataset(sample_size=100, seed=0)
    assert len(df) == 100


def test_generate_synthetic_dataset_columns():
    df = generate_synthetic_dataset(sample_size=20, seed=0)
    for col in ("id", "prompt", "response", "label", "category", "source_dataset"):
        assert col in df.columns, f"Missing column: {col}"


def test_generate_synthetic_labels_balanced():
    df = generate_synthetic_dataset(sample_size=80, seed=0)
    counts = df["label"].value_counts()
    assert counts.get(0, 0) > 0
    assert counts.get(1, 0) > 0
    # Should be roughly balanced (within 20%)
    ratio = counts.get(0, 0) / counts.get(1, 1)
    assert 0.7 < ratio < 1.4


def test_split_dataset():
    df = generate_synthetic_dataset(sample_size=100, seed=0)
    train, val, test = split_dataset(df, train_frac=0.7, val_frac=0.15, seed=0)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_no_duplicates_across_splits():
    df = generate_synthetic_dataset(sample_size=100, seed=0)
    train, val, test = split_dataset(df, train_frac=0.7, val_frac=0.15, seed=0)
    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
