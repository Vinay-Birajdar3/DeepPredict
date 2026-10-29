"""Feature engineering module for tabular baseline models.

Extracts statistical window aggregations (mean, std, min, max, delta/slope)
from multivariate sensor sequences for tree-based regressors (Random Forest, XGBoost).
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def extract_tabular_window_features(
    X_sequences: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Transforms 3D temporal sequences (N, L, F) into 2D tabular features (N, F_engineered).

    Calculates:
    - Last reading (current state)
    - Window Mean
    - Window Standard Deviation
    - Window Minimum
    - Window Maximum
    - Window Delta (Last reading - First reading, indicating degradation velocity)

    Args:
        X_sequences: Array of shape (num_windows, window_length, num_features).
        feature_names: Names of the base sensor channels.

    Returns:
        pd.DataFrame: Tabular features with descriptive column names.
    """
    sequences = np.asarray(X_sequences)
    if sequences.ndim != 3:
        raise ValueError(
            "X_sequences must be a 3D array with shape "
            "(num_samples, window_length, num_features)"
        )

    num_samples, window_len, num_features = sequences.shape
    if num_samples == 0 or window_len == 0 or num_features == 0:
        raise ValueError("X_sequences must contain samples, timesteps, and features.")
    if len(feature_names) != num_features:
        raise ValueError(
            f"Received {len(feature_names)} feature names for {num_features} input features."
        )
    if any(not isinstance(name, str) or not name.strip() for name in feature_names):
        raise ValueError("Feature names must be non-empty strings.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique.")

    try:
        sequences = sequences.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("X_sequences must contain numeric values.") from exc
    if not np.isfinite(sequences).all():
        raise ValueError("X_sequences must contain only finite values.")

    features_dict = {}

    for idx, name in enumerate(feature_names):
        sensor_data = sequences[:, :, idx]

        features_dict[f"{name}_last"] = sensor_data[:, -1]
        features_dict[f"{name}_mean"] = np.mean(sensor_data, axis=1)
        features_dict[f"{name}_std"] = np.std(sensor_data, axis=1)
        features_dict[f"{name}_min"] = np.min(sensor_data, axis=1)
        features_dict[f"{name}_max"] = np.max(sensor_data, axis=1)
        features_dict[f"{name}_delta"] = sensor_data[:, -1] - sensor_data[:, 0]

    return pd.DataFrame(features_dict)
