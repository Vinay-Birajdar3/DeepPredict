"""Unit tests for feature selection, scaling, and sequence generation."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    identify_constant_features,
    split_train_val_engines,
    CMapssPreprocessor,
    create_sliding_sequences,
)
from src.config import PreprocessingConfig


def test_identify_constant_features():
    df = pd.DataFrame({
        "dynamic_sensor": [1.0, 2.0, 3.0, 4.0, 5.0],
        "constant_sensor": [518.67, 518.67, 518.67, 518.67, 518.67],
    })
    dropped = identify_constant_features(df, ["dynamic_sensor", "constant_sensor"])
    assert "constant_sensor" in dropped
    assert "dynamic_sensor" not in dropped


def test_split_train_val_engines_no_overlap():
    """Confirms train and validation splits have mutually exclusive engine IDs."""
    data = []
    for eng in range(1, 21):
        for cycle in range(1, 11):
            data.append({"unit_number": eng, "time_in_cycles": cycle, "s_2": cycle * 0.5})
    df = pd.DataFrame(data)

    tr_df, val_df, tr_ids, val_ids = split_train_val_engines(df, val_ratio=0.25, random_seed=42)

    assert len(tr_ids) == 15
    assert len(val_ids) == 5
    # Strict disjointness
    assert len(set(tr_ids).intersection(set(val_ids))) == 0
    assert len(tr_df) + len(val_df) == len(df)


def test_sliding_sequences_shape():
    # 1 engine with 50 cycles, 4 features
    cycles = 50
    w_len = 30
    df = pd.DataFrame({
        "unit_number": [1] * cycles,
        "time_in_cycles": list(range(1, cycles + 1)),
        "s_2": np.random.randn(cycles),
        "s_3": np.random.randn(cycles),
        "s_4": np.random.randn(cycles),
        "RUL": list(range(cycles - 1, -1, -1)),
    })

    X, y, eng_ids = create_sliding_sequences(df, ["s_2", "s_3", "s_4"], "RUL", window_length=w_len)

    expected_windows = cycles - w_len + 1  # 50 - 30 + 1 = 21
    assert X.shape == (expected_windows, w_len, 3)
    assert len(y) == expected_windows
    assert len(eng_ids) == expected_windows
    # Final target matches cycle 30's RUL = 50 - 30 = 20
    assert y[0] == 20.0

