"""Unit tests for prognostics and regression evaluation metrics."""

import math
import numpy as np
import pytest

from src.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    phm_score,
    compute_all_metrics,
)


def test_mae_and_rmse_known_values():
    y_true = np.array([100.0, 50.0, 20.0])
    y_pred = np.array([90.0, 55.0, 20.0])

    # Errors: [-10, +5, 0]
    # Absolute errors: [10, 5, 0] -> Mean = 5.0
    # Squared errors: [100, 25, 0] -> Mean = 125/3 = 41.666... -> Sqrt = 6.45497...
    assert math.isclose(mean_absolute_error(y_true, y_pred), 5.0, rel_tol=1e-5)
    assert math.isclose(root_mean_squared_error(y_true, y_pred), math.sqrt(125.0 / 3.0), rel_tol=1e-5)


def test_phm_score_asymmetry():
    """Verifies that late prediction is penalized more heavily than early prediction of same magnitude."""
    y_true = np.array([50.0])

    # Late prediction: d = 60 - 50 = +10 -> exp(10/10) - 1 = exp(1) - 1 ≈ 1.718
    y_late = np.array([60.0])
    score_late = phm_score(y_true, y_late)

    # Early prediction: d = 40 - 50 = -10 -> exp(10/13) - 1 ≈ exp(0.7692) - 1 ≈ 1.158
    y_early = np.array([40.0])
    score_early = phm_score(y_true, y_early)

    assert score_late > score_early, f"Late score {score_late} must exceed early score {score_early}"
    assert math.isclose(score_late, math.exp(1.0) - 1.0, rel_tol=1e-4)
    assert math.isclose(score_early, math.exp(10.0 / 13.0) - 1.0, rel_tol=1e-4)


def test_perfect_predictions():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])

    metrics = compute_all_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["phm_score"] == 0.0
    assert metrics["r2"] == 1.0

