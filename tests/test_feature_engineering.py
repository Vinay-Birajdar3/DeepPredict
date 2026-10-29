"""Tests for converting sensor windows into tabular features."""

import numpy as np
import pytest

from src.feature_engineering import extract_tabular_window_features


def test_extract_tabular_window_features_values_and_columns():
    sequences = np.array([[[1.0, 4.0], [3.0, 8.0]]])

    result = extract_tabular_window_features(sequences, ["s_1", "s_2"])

    assert result.columns.tolist() == [
        "s_1_last", "s_1_mean", "s_1_std", "s_1_min", "s_1_max", "s_1_delta",
        "s_2_last", "s_2_mean", "s_2_std", "s_2_min", "s_2_max", "s_2_delta",
    ]
    assert result.loc[0, "s_1_last"] == 3.0
    assert result.loc[0, "s_1_mean"] == 2.0
    assert result.loc[0, "s_1_delta"] == 2.0
    assert result.loc[0, "s_2_std"] == 2.0


@pytest.mark.parametrize(
    ("sequences", "feature_names", "message"),
    [
        (np.zeros((2, 3)), ["s_1"], "3D array"),
        (np.zeros((0, 3, 1)), ["s_1"], "contain samples"),
        (np.zeros((2, 3, 2)), ["s_1"], "feature names"),
        (np.zeros((2, 3, 2)), ["s_1", "s_1"], "unique"),
        (np.full((2, 3, 1), np.nan), ["s_1"], "finite values"),
        (np.full((2, 3, 1), "bad"), ["s_1"], "numeric values"),
    ],
)
def test_extract_tabular_window_features_rejects_invalid_input(
    sequences, feature_names, message
):
    with pytest.raises(ValueError, match=message):
        extract_tabular_window_features(sequences, feature_names)
