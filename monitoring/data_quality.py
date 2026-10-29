"""Data quality monitoring and schema validation module for DeepPredict."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.config import SENSOR_COLUMNS, INDEX_COLUMNS
from src.logging_config import setup_logger

logger = setup_logger("DataQuality")


@dataclass
class QualityCheckResult:
    """Stores the outcome of data quality inspections."""
    is_valid: bool
    total_records: int
    null_count: int
    inf_count: int
    duplicate_count: int
    missing_columns: List[str]
    out_of_range_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataQualityMonitor:
    """Monitors raw and incoming sensor streams against baseline distributions."""

    def __init__(self, reference_df: Optional[pd.DataFrame] = None, sigma_threshold: float = 4.0):
        self.sigma_threshold = sigma_threshold
        self.sensor_bounds: Dict[str, Dict[str, float]] = {}

        if reference_df is not None:
            self.fit_reference_bounds(reference_df)

    def fit_reference_bounds(self, df: pd.DataFrame) -> None:
        """Computes statistical sensor boundaries [mean - k*std, mean + k*std] from reference data."""
        numeric_cols = [c for c in df.columns if c in SENSOR_COLUMNS]
        for col in numeric_cols:
            mean = float(df[col].mean())
            std = float(df[col].std())
            # For zero-variance sensors, std can be 0
            margin = self.sigma_threshold * std if std > 0 else 1e-3
            self.sensor_bounds[col] = {
                "lower": mean - margin,
                "upper": mean + margin,
                "mean": mean,
                "std": std,
            }
        logger.info(f"Fitted sensor bounds for {len(self.sensor_bounds)} sensors.")

    def inspect_dataframe(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
    ) -> QualityCheckResult:
        """Performs a comprehensive quality audit on an input DataFrame.

        Args:
            df: DataFrame to validate.
            expected_columns: Expected column names.

        Returns:
            QualityCheckResult: Detailed quality audit report.
        """
        if expected_columns is None:
            expected_columns = SENSOR_COLUMNS

        errors = []
        warnings = []

        # 1. Schema check
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # 2. Nulls & Infs
        null_count = int(df.isna().sum().sum())
        if null_count > 0:
            errors.append(f"Found {null_count} null/NaN values across records.")

        numeric_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(numeric_df.to_numpy()).sum())
        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values across numeric features.")

        # 3. Duplicate checks on index columns if present
        duplicate_count = 0
        if set(INDEX_COLUMNS).issubset(df.columns):
            duplicate_count = int(df.duplicated(subset=INDEX_COLUMNS).sum())
            if duplicate_count > 0:
                warnings.append(f"Found {duplicate_count} duplicate engine cycle timestamps.")

        # 4. Out-of-bounds anomaly check
        out_of_range = {}
        for col, bounds in self.sensor_bounds.items():
            if col in df.columns:
                series = df[col]
                below = int((series < bounds["lower"]).sum())
                above = int((series > bounds["upper"]).sum())
                if below > 0 or above > 0:
                    out_of_range[col] = {
                        "below_lower": below,
                        "above_upper": above,
                        "lower_limit": bounds["lower"],
                        "upper_limit": bounds["upper"],
                    }
                    warnings.append(
                        f"Sensor {col} has {below + above} out-of-range observations "
                        f"outside [{bounds['lower']:.2f}, {bounds['upper']:.2f}]"
                    )

        is_valid = len(errors) == 0

        return QualityCheckResult(
            is_valid=is_valid,
            total_records=len(df),
            null_count=null_count,
            inf_count=inf_count,
            duplicate_count=duplicate_count,
            missing_columns=missing_cols,
            out_of_range_features=out_of_range,
            warnings=warnings,
            errors=errors,
        )

