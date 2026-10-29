"""Unit tests for C-MAPSS data loader and RUL label calculations."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import DatasetConfig, ALL_RAW_COLUMNS, TARGET_COLUMN, TARGET_CAPPED_COLUMN
from src.data_loader import (
    load_raw_dataset_file,
    validate_dataset_integrity,
    compute_training_rul,
    compute_test_rul,
)


def test_compute_training_rul_logic():
    # Synthetic dataset with 2 engines
    data = {
        "unit_number": [1, 1, 1, 2, 2],
        "time_in_cycles": [1, 2, 3, 1, 2],
    }
    for c in ALL_RAW_COLUMNS[2:]:
        data[c] = np.random.randn(5)

    df = pd.DataFrame(data)
    df_rul = compute_training_rul(df, rul_cap=2)

    # Engine 1 max cycle = 3.
    # Expected RUL: [2, 1, 0]
    expected_e1 = [2, 1, 0]
    actual_e1 = df_rul[df_rul["unit_number"] == 1][TARGET_COLUMN].tolist()
    assert actual_e1 == expected_e1

    # Capped RUL with cap = 2: [2, 1, 0]
    actual_e1_capped = df_rul[df_rul["unit_number"] == 1][TARGET_CAPPED_COLUMN].tolist()
    assert actual_e1_capped == [2, 1, 0]


def test_compute_test_rul_logic():
    # Engine 1 has 2 cycles, Engine 2 has 3 cycles
    data = {
        "unit_number": [1, 1, 2, 2, 2],
        "time_in_cycles": [1, 2, 1, 2, 3],
    }
    for c in ALL_RAW_COLUMNS[2:]:
        data[c] = np.random.randn(5)

    test_df = pd.DataFrame(data)
    # Ground truth final RUL: Engine 1 = 50, Engine 2 = 25
    truth = np.array([50.0, 25.0])

    test_res = compute_test_rul(test_df, truth, rul_cap=125)

    # Engine 1: cycle 2 is last -> RUL=50. cycle 1 -> 50 + (2 - 1) = 51.
    e1_rul = test_res[test_res["unit_number"] == 1][TARGET_COLUMN].tolist()
    assert e1_rul == [51.0, 50.0]

    # Engine 2: cycle 3 is last -> RUL=25. cycle 2 -> 26. cycle 1 -> 27.
    e2_rul = test_res[test_res["unit_number"] == 2][TARGET_COLUMN].tolist()
    assert e2_rul == [27.0, 26.0, 25.0]


def test_integrity_validation_detects_errors():
    # DataFrame with missing values
    df_bad = pd.DataFrame({
        "unit_number": [1, 1],
        "time_in_cycles": [1, 2],
        "s_2": [10.0, np.nan],
    })
    with pytest.raises(ValueError, match="null values"):
        validate_dataset_integrity(df_bad)

