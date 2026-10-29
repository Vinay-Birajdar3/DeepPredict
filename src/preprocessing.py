"""Data preprocessing, feature selection, leakage-free scaling, and sequence windowing.

Ensures zero data leakage:
- Train/Validation splitting occurs strictly at the engine level.
- Scaler is fitted ONLY on the training split engines.
- Test data and validation data are strictly transformed using the training scaler.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from src.config import (
    INDEX_COLUMNS,
    SETTING_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    TARGET_CAPPED_COLUMN,
    PreprocessingConfig,
    MODELS_DIR,
)
from src.logging_config import setup_logger

logger = setup_logger("Preprocessing")


def identify_constant_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 1e-4,
) -> List[str]:
    """Identifies features with variance below threshold.

    Args:
        df: DataFrame (training split only).
        feature_cols: Columns to inspect.
        variance_threshold: Minimum variance required to keep feature.

    Returns:
        List[str]: List of low-variance/constant column names to drop.
    """
    variances = df[feature_cols].var()
    low_var_cols = variances[variances <= variance_threshold].index.tolist()
    logger.info(
        f"Identified {len(low_var_cols)} constant/low-variance features: {low_var_cols}"
    )
    return low_var_cols


def split_train_val_engines(
    train_df: pd.DataFrame,
    val_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Splits engines into train and validation sets to prevent cross-engine window leakage.

    Args:
        train_df: Complete training DataFrame.
        val_ratio: Fraction of engines for validation (default 0.20).
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (train_split_df, val_split_df, train_engine_ids, val_engine_ids)
    """
    rng = np.random.RandomState(random_seed)
    unique_engines = np.sort(train_df["unit_number"].unique())
    shuffled_engines = rng.permutation(unique_engines)

    n_val = int(len(unique_engines) * val_ratio)
    val_engines = sorted(shuffled_engines[:n_val].tolist())
    train_engines = sorted(shuffled_engines[n_val:].tolist())

    train_split = train_df[train_df["unit_number"].isin(train_engines)].copy()
    val_split = train_df[train_df["unit_number"].isin(val_engines)].copy()

    logger.info(
        f"Engine split: {len(train_engines)} train engines ({len(train_split)} rows), "
        f"{len(val_engines)} validation engines ({len(val_split)} rows)."
    )
    return train_split, val_split, train_engines, val_engines


class CMapssPreprocessor:
    """Production preprocessor for C-MAPSS sensor data."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.scaler: Optional[MinMaxScaler] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "CMapssPreprocessor":
        """Fits feature selection and scaler strictly on the training partition.

        Args:
            train_df: Training partition dataframe.

        Returns:
            self
        """
        all_candidate = []
        if self.config.include_operational_settings:
            all_candidate.extend(SETTING_COLUMNS)
        all_candidate.extend(SENSOR_COLUMNS)

        # Detect constant sensors strictly on train split
        self.dropped_features = identify_constant_features(
            train_df,
            feature_cols=all_candidate,
            variance_threshold=self.config.drop_constant_threshold,
        )
        self.selected_features = [c for c in all_candidate if c not in self.dropped_features]
        logger.info(f"Retained {len(self.selected_features)} dynamic features: {self.selected_features}")

        if self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=self.config.feature_range)

        self.scaler.fit(train_df[self.selected_features].to_numpy())
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms features using the fitted scaler.

        Args:
            df: Input dataframe.

        Returns:
            pd.DataFrame: Scaled dataframe with index and target columns preserved.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        df_out = df.copy()
        scaled_values = self.scaler.transform(df[self.selected_features].to_numpy())

        # Overwrite selected features with scaled versions
        for idx, col in enumerate(self.selected_features):
            df_out[col] = scaled_values[:, idx]

        return df_out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits preprocessor on train_df and transforms it."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Path) -> None:
        """Saves fitted preprocessor and scaler to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved fitted preprocessor to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "CMapssPreprocessor":
        """Loads fitted preprocessor from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        loaded = joblib.load(filepath)
        logger.info(f"Loaded preprocessor from: {filepath}")
        return loaded


def create_sliding_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_length: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates temporal sliding windows for deep learning models.

    For each engine, windows of shape (window_length, num_features) are extracted.
    The target is the RUL value at the final step of each window.

    Args:
        df: Scaled DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name (RUL or RUL_capped).
        window_length: Number of time cycles per sequence window.

    Returns:
        Tuple of:
            - X: 3D array of shape (num_windows, window_length, num_features)
            - y: 1D array of shape (num_windows,)
            - engine_ids: 1D array of engine IDs for each window
    """
    x_list = []
    y_list = []
    engine_list = []

    for unit_id, group in df.groupby("unit_number"):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        n_samples = len(group)

        if n_samples < window_length:
            # Pad beginning if an engine has fewer cycles than window length
            pad_length = window_length - n_samples
            pad_features = np.repeat(features[:1], pad_length, axis=0)
            features = np.vstack([pad_features, features])
            targets = np.pad(targets, (pad_length, 0), mode="edge")
            n_samples = window_length

        num_windows = n_samples - window_length + 1
        for i in range(num_windows):
            x_window = features[i : i + window_length]
            y_val = targets[i + window_length - 1]
            x_list.append(x_window)
            y_list.append(y_val)
            engine_list.append(unit_id)

    X = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    engine_ids = np.array(engine_list, dtype=np.int32)

    logger.debug(f"Generated sliding sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y, engine_ids


def create_test_evaluation_sequences(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_length: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the final temporal sequence of each test engine for benchmark evaluation.

    The official NASA C-MAPSS competition evaluates RUL prediction on the
    LAST available cycle of each test engine.

    Args:
        test_df: Scaled test DataFrame.
        feature_cols: Feature columns.
        target_col: Target column (RUL or RUL_capped).
        window_length: Temporal sequence window length.

    Returns:
        Tuple of:
            - X_test_last: shape (num_test_engines, window_length, num_features)
            - y_test_last: shape (num_test_engines,) true final RULs
            - engine_ids: test engine IDs
    """
    x_list = []
    y_list = []
    engine_list = []

    for unit_id, group in test_df.groupby("unit_number"):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        n_samples = len(group)

        if n_samples < window_length:
            pad_length = window_length - n_samples
            pad_features = np.repeat(features[:1], pad_length, axis=0)
            features = np.vstack([pad_features, features])
            targets = np.pad(targets, (pad_length, 0), mode="edge")

        # Take strictly the LAST window_length cycles
        x_last = features[-window_length:]
        y_last = targets[-1]

        x_list.append(x_last)
        y_list.append(y_last)
        engine_list.append(unit_id)

    X_test_last = np.array(x_list, dtype=np.float32)
    y_test_last = np.array(y_list, dtype=np.float32)
    engine_ids = np.array(engine_list, dtype=np.int32)

    logger.info(
        f"Extracted final test evaluation windows for {len(engine_ids)} test engines: "
        f"X shape {X_test_last.shape}, y shape {y_test_last.shape}"
    )
    return X_test_last, y_test_last, engine_ids

