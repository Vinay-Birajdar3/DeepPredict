"""Evaluation metrics for Remaining Useful Life (RUL) estimation.

Implements standard regression metrics (MAE, RMSE, R^2) and the NASA PHM 2008
data challenge asymmetric prognostics scoring function.
"""

from typing import Dict, Union
import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Coefficient of Determination (R^2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the NASA / PHM 2008 Prognostics Scoring function.

    Error Convention:
        d = y_pred - y_true

    Interpretation:
        - d < 0 (Early prediction): The model predicts failure sooner than it
          actually happens. Safe, but causes early maintenance.
          Penalty: s_i = exp(-d / 13) - 1
        - d >= 0 (Late prediction): The model predicts failure later than it
          actually happens. Catastrophic risk of in-flight failure!
          Penalty: s_i = exp(d / 10) - 1

    Total Score: S = sum(s_i)
    Lower score indicates superior prognostics performance.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        float: Asymmetric PHM score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true

    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_all_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Computes a comprehensive dictionary of regression and prognostics metrics.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dict[str, float]: Dictionary with MAE, RMSE, R2, PHM, Mean Bias, Max Error.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    errors = y_p - y_t

    return {
        "mae": mean_absolute_error(y_t, y_p),
        "rmse": root_mean_squared_error(y_t, y_p),
        "r2": r2_score(y_t, y_p),
        "phm_score": phm_score(y_t, y_p),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(np.abs(errors))),
        "sample_count": int(len(y_t)),
    }

