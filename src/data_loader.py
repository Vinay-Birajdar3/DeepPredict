"""NASA C-MAPSS dataset loader, validator, and RUL label generator.

Supports FD001-FD004 subsets with full data integrity checks:
- Proper space-separated column assignment
- Missing/duplicate value detection
- Engine cycle continuity and monotonicity validation
- Piecewise linear (capped) and uncapped Remaining Useful Life (RUL) engineering
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import (
    ALL_RAW_COLUMNS,
    INDEX_COLUMNS,
    SETTING_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
    TARGET_CAPPED_COLUMN,
    DatasetConfig,
)
from src.logging_config import setup_logger

logger = setup_logger("DataLoader")


def load_raw_dataset_file(file_path: Path) -> pd.DataFrame:
    """Reads a raw C-MAPSS whitespace-delimited file into a DataFrame.

    Handles variable whitespace delimiters and trailing whitespace columns.

    Args:
        file_path: Path to the raw .txt file.

    Returns:
        pd.DataFrame: Loaded dataframe with standard column names.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file has fewer columns than the expected 26.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {file_path}")

    logger.info(f"Loading raw file: {file_path.name}")
    # sep=r"\s+" handles arbitrary spaces; header=None ensures raw indices
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    # Some C-MAPSS files have trailing empty spaces resulting in extra NaN columns
    if df.shape[1] > len(ALL_RAW_COLUMNS):
        extra_cols = df.columns[len(ALL_RAW_COLUMNS):]
        # Verify that extra columns are solely NaNs before dropping
        if df[extra_cols].isna().all().all():
            logger.debug(f"Dropping {len(extra_cols)} empty trailing columns created by delimiters.")
            df = df.iloc[:, :len(ALL_RAW_COLUMNS)]
        else:
            raise ValueError(
                f"File {file_path.name} contains {df.shape[1]} non-empty columns, "
                f"expected {len(ALL_RAW_COLUMNS)}"
            )
    elif df.shape[1] < len(ALL_RAW_COLUMNS):
        raise ValueError(
            f"File {file_path.name} has only {df.shape[1]} columns; "
            f"expected at least {len(ALL_RAW_COLUMNS)}"
        )

    df.columns = ALL_RAW_COLUMNS

    # Cast index columns to integer
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)

    return df


def validate_dataset_integrity(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, any]:
    """Performs rigorous data quality and integrity validations.

    Checks:
    - Missing or null values
    - Duplicate engine cycles
    - Monotonically increasing cycles per engine
    - Engine ID continuity

    Args:
        df: Loaded DataFrame.
        dataset_name: Identifier for logging.

    Returns:
        Dict: Validation report dictionary.

    Raises:
        ValueError: If any integrity constraint is violated.
    """
    report = {
        "dataset_name": dataset_name,
        "total_rows": int(len(df)),
        "num_engines": int(df["unit_number"].nunique()),
        "null_counts": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated(subset=INDEX_COLUMNS).sum()),
    }

    # 1. Null checks
    total_nulls = sum(report["null_counts"].values())
    if total_nulls > 0:
        raise ValueError(f"Integrity check failed: {dataset_name} contains {total_nulls} null values.")

    # 2. Duplicate index checks
    if report["duplicate_rows"] > 0:
        raise ValueError(
            f"Integrity check failed: {dataset_name} contains {report['duplicate_rows']} "
            f"duplicate (unit_number, time_in_cycles) entries."
        )

    # 3. Monotonicity & continuous cycle verification per engine
    for unit_id, group in df.groupby("unit_number"):
        cycles = group["time_in_cycles"].to_numpy()
        if cycles[0] != 1:
            raise ValueError(f"Engine {unit_id} starts at cycle {cycles[0]} instead of cycle 1.")
        diffs = np.diff(cycles)
        if not np.all(diffs == 1):
            raise ValueError(f"Engine {unit_id} has non-continuous cycles: {cycles}")

    logger.info(
        f"Validation PASSED for {dataset_name}: {report['total_rows']} rows, "
        f"{report['num_engines']} engines, 0 nulls, 0 duplicates."
    )
    return report


def compute_training_rul(
    df: pd.DataFrame,
    rul_cap: Optional[int] = 125,
) -> pd.DataFrame:
    """Calculates Remaining Useful Life (RUL) for run-to-failure training data.

    RUL = max_cycle_for_engine - current_cycle
    Optionally computes piecewise linear (capped) RUL:
    RUL_capped = min(RUL, rul_cap)

    Args:
        df: Training DataFrame containing unit_number and time_in_cycles.
        rul_cap: Upper bound for RUL clipping (None for uncapped only).

    Returns:
        pd.DataFrame: Training DataFrame enriched with TARGET_COLUMN and TARGET_CAPPED_COLUMN.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df[TARGET_COLUMN] = max_cycles - df["time_in_cycles"]

    if rul_cap is not None and rul_cap > 0:
        df[TARGET_CAPPED_COLUMN] = df[TARGET_COLUMN].clip(upper=float(rul_cap))
    else:
        df[TARGET_CAPPED_COLUMN] = df[TARGET_COLUMN]

    return df


def load_test_ground_truth_rul(file_path: Path) -> np.ndarray:
    """Loads the test set ground truth RUL vector from RUL_FD00x.txt.

    Args:
        file_path: Path to RUL_FD00x.txt.

    Returns:
        np.ndarray: Array of true RUL values for each test engine at its last cycle.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"RUL ground truth file not found: {file_path}")

    rul_series = pd.read_csv(file_path, sep=r"\s+", header=None)[0]
    return rul_series.to_numpy(dtype=float)


def compute_test_rul(
    test_df: pd.DataFrame,
    rul_truth: np.ndarray,
    rul_cap: Optional[int] = 125,
) -> pd.DataFrame:
    """Aligns test ground truth RUL across all observed test cycles.

    For test engine i with final observed cycle T_i and true RUL R_i:
    At cycle t <= T_i:
        RUL(t) = R_i + (T_i - t)

    Args:
        test_df: Test DataFrame containing unit_number and time_in_cycles.
        rul_truth: 1D array of ground truth RUL at the last observed cycle.
        rul_cap: Upper bound for RUL clipping.

    Returns:
        pd.DataFrame: Test DataFrame enriched with TARGET_COLUMN and TARGET_CAPPED_COLUMN.
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    
    # Map engine unit_number (1-indexed) to rul_truth (0-indexed)
    unit_ids = test_df["unit_number"].to_numpy()
    if len(rul_truth) < test_df["unit_number"].nunique():
        raise ValueError(
            f"Ground truth RUL has {len(rul_truth)} entries, but test set has "
            f"{test_df['unit_number'].nunique()} engines."
        )

    # Truth RUL at end of test trajectory for each row's engine
    final_ruls = rul_truth[unit_ids - 1]
    
    # RUL at current cycle: final_rul + (max_cycle - current_cycle)
    test_df[TARGET_COLUMN] = final_ruls + (max_cycles - test_df["time_in_cycles"])

    if rul_cap is not None and rul_cap > 0:
        test_df[TARGET_CAPPED_COLUMN] = test_df[TARGET_COLUMN].clip(upper=float(rul_cap))
    else:
        test_df[TARGET_CAPPED_COLUMN] = test_df[TARGET_COLUMN]

    return test_df


def load_and_prepare_cmapss(
    config: Optional[DatasetConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, any]]:
    """Loads, validates, and prepares both training and test datasets.

    Args:
        config: Optional DatasetConfig. Defaults to FD001.

    Returns:
        Tuple of (train_df, test_df, test_final_rul_truth, metadata_dict)
    """
    if config is None:
        config = DatasetConfig()

    train_path = config.raw_dir / config.train_file
    test_path = config.raw_dir / config.test_file
    rul_path = config.raw_dir / config.rul_file

    raw_train = load_raw_dataset_file(train_path)
    train_report = validate_dataset_integrity(raw_train, f"{config.subset} Train")
    train_df = compute_training_rul(raw_train, rul_cap=config.rul_cap)

    raw_test = load_raw_dataset_file(test_path)
    test_report = validate_dataset_integrity(raw_test, f"{config.subset} Test")
    test_rul_truth = load_test_ground_truth_rul(rul_path)
    test_df = compute_test_rul(raw_test, test_rul_truth, rul_cap=config.rul_cap)

    metadata = {
        "subset": config.subset,
        "rul_cap": config.rul_cap,
        "train_engines": int(train_df["unit_number"].nunique()),
        "test_engines": int(test_df["unit_number"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_max_cycles": train_df.groupby("unit_number")["time_in_cycles"].max().describe().to_dict(),
        "test_max_cycles": test_df.groupby("unit_number")["time_in_cycles"].max().describe().to_dict(),
    }

    logger.info(
        f"C-MAPSS {config.subset} fully loaded. "
        f"Train: {metadata['train_engines']} engines ({metadata['train_rows']} rows). "
        f"Test: {metadata['test_engines']} engines ({metadata['test_rows']} rows)."
    )
    return train_df, test_df, test_rul_truth, metadata

