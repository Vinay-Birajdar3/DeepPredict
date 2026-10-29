"""Data and sensor drift detection engine for DeepPredict.

Implements rigorous statistical drift detection using:
- Two-sample Kolmogorov-Smirnov (KS) test (distribution shape change)
- Population Stability Index (PSI) (binned entropy shift)
- Wasserstein Distance (Earth Mover's Distance)
- Overall dataset drift determination
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.logging_config import setup_logger

logger = setup_logger("DriftDetection")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculates the Population Stability Index (PSI) between two continuous distributions.

    PSI Categories:
    - PSI < 0.10: No significant distribution change.
    - 0.10 <= PSI < 0.20: Moderate distribution drift detected.
    - PSI >= 0.20: Significant distribution shift; model recalibration recommended.

    Args:
        expected: Reference sample values (e.g., training data).
        actual: Production/inference sample values.
        num_bins: Number of quantiles to segment the baseline.
        eps: Small epsilon to prevent division by zero or log(0).

    Returns:
        float: Calculated PSI value.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Handle constant feature case
    if np.all(expected == expected[0]) and np.all(actual == actual[0]):
        return 0.0 if expected[0] == actual[0] else 1.0

    # Determine quantile bins based on reference data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges in case of low variance

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon smoothing
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


@dataclass
class FeatureDriftResult:
    """Statistical drift test results for an individual sensor feature."""
    feature: str
    has_drifted: bool
    ks_statistic: float
    p_value: float
    psi: float
    wasserstein_distance: float
    drift_reason: str


@dataclass
class DatasetDriftReport:
    """Aggregated drift summary across all analyzed sensors."""
    dataset_has_drifted: bool
    drifted_feature_count: int
    total_feature_count: int
    drifted_features: List[str]
    feature_metrics: Dict[str, FeatureDriftResult]
    p_value_threshold: float
    psi_threshold: float
    share_of_drifted_features: float


class DriftDetector:
    """Detects sensor distribution shifts between training and inference data."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_cols: List[str],
        p_val_threshold: float = 0.05,
        psi_threshold: float = 0.20,
        dataset_drift_ratio: float = 0.30,
    ):
        self.feature_cols = [c for c in feature_cols if c in reference_df.columns]
        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        self.dataset_drift_ratio = dataset_drift_ratio
        self.reference_data: Dict[str, np.ndarray] = {
            col: reference_df[col].dropna().to_numpy() for col in self.feature_cols
        }

    def detect_drift(self, current_df: pd.DataFrame) -> DatasetDriftReport:
        """Runs KS test, PSI, and Wasserstein tests on current data against reference.

        Args:
            current_df: Recent operational sensor data.

        Returns:
            DatasetDriftReport: Full drift evaluation.
        """
        results = {}
        drifted_cols = []

        for col in self.feature_cols:
            if col not in current_df.columns:
                continue

            ref_arr = self.reference_data[col]
            curr_arr = current_df[col].dropna().to_numpy()

            if len(curr_arr) == 0:
                continue

            # 1. Kolmogorov-Smirnov Test
            ks_res = stats.ks_2samp(ref_arr, curr_arr)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            # 2. Population Stability Index (PSI)
            psi = calculate_psi(ref_arr, curr_arr)

            # 3. Wasserstein Distance
            w_dist = float(stats.wasserstein_distance(ref_arr, curr_arr))

            # Flag as drifted if either KS test rejects null hypothesis OR PSI is high
            has_drifted = (p_val < self.p_val_threshold) or (psi >= self.psi_threshold)

            reason_parts = []
            if p_val < self.p_val_threshold:
                reason_parts.append(f"KS p-value {p_val:.4e} < {self.p_val_threshold}")
            if psi >= self.psi_threshold:
                reason_parts.append(f"PSI {psi:.3f} >= {self.psi_threshold}")

            drift_reason = ", ".join(reason_parts) if reason_parts else "Distribution stable"

            if has_drifted:
                drifted_cols.append(col)

            results[col] = FeatureDriftResult(
                feature=col,
                has_drifted=has_drifted,
                ks_statistic=ks_stat,
                p_value=p_val,
                psi=psi,
                wasserstein_distance=w_dist,
                drift_reason=drift_reason,
            )

        drift_ratio = len(drifted_cols) / len(self.feature_cols) if self.feature_cols else 0.0
        dataset_drifted = drift_ratio >= self.dataset_drift_ratio

        logger.info(
            f"Drift Audit: {len(drifted_cols)}/{len(self.feature_cols)} features drifted "
            f"({drift_ratio*100:.1f}%). Dataset Drift Status: {dataset_drifted}"
        )

        return DatasetDriftReport(
            dataset_has_drifted=dataset_drifted,
            drifted_feature_count=len(drifted_cols),
            total_feature_count=len(self.feature_cols),
            drifted_features=drifted_cols,
            feature_metrics=results,
            p_value_threshold=self.p_val_threshold,
            psi_threshold=self.psi_threshold,
            share_of_drifted_features=drift_ratio,
        )

