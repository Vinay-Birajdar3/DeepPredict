"""Evaluation and diagnostics module for DeepPredict.

Computes metrics (MAE, RMSE, R2, PHM Score), generates comparison tables,
and renders publication-grade evaluation figures (actual vs predicted, residuals, error distributions).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EVALUATION_DIR, FIGURES_DIR
from src.metrics import compute_all_metrics, phm_score
from src.logging_config import setup_logger

logger = setup_logger("Evaluation")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Plots and saves Actual vs. Predicted RUL scatter with ideal 45-degree reference line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", s=45, label="Predictions")
    min_val = min(y_true.min(), y_pred.min(), 0)
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")

    plt.title(f"{model_name}: Actual vs Predicted RUL", fontsize=14, fontweight="bold")
    plt.xlabel("Ground Truth RUL (Cycles)", fontsize=12)
    plt.ylabel("Predicted RUL (Cycles)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_actual_vs_pred.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Plots and saves Residuals (Error = Predicted - Actual) vs Ground Truth RUL."""
    residuals = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuals, alpha=0.7, edgecolors="k", s=45, color="teal")
    plt.axhline(0, color="crimson", linestyle="--", lw=2)

    # Shaded risk zones: positive residual = late prediction (dangerous)
    plt.fill_between(
        [y_true.min(), y_true.max()], 0, max(residuals.max() + 5, 20),
        color="red", alpha=0.08, label="Late Prediction (High Risk Zone)"
    )
    plt.fill_between(
        [y_true.min(), y_true.max()], min(residuals.min() - 5, -20), 0,
        color="green", alpha=0.08, label="Early Prediction (Safe Zone)"
    )

    plt.title(f"{model_name}: Prognostic Residuals Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Ground Truth RUL (Cycles)", fontsize=12)
    plt.ylabel("Residual (Predicted - Actual)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_residuals.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Plots histogram and KDE of prediction errors."""
    residuals = y_pred - y_true
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=25, color="royalblue", edgecolor="black")
    plt.axvline(0, color="crimson", linestyle="--", lw=2, label="Zero Error")
    plt.axvline(float(np.mean(residuals)), color="orange", linestyle="-.", lw=2,
                label=f"Mean Bias ({np.mean(residuals):.2f})")

    plt.title(f"{model_name}: Prediction Error Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction Error (Predicted - Actual Cycles)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_error_dist.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    engine_ids: Optional[np.ndarray] = None,
    config_dict: Optional[Dict[str, any]] = None,
) -> Dict[str, any]:
    """Generates full evaluation metrics, plots, and persists summary artifacts.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.
        model_name: Name of the model.
        engine_ids: Optional array of test engine identifiers.
        config_dict: Optional hyperparameters / configuration dict.

    Returns:
        Dict: Complete evaluation report.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    metrics = compute_all_metrics(y_t, y_p)

    # Generate and save diagnostic plots
    fig_actual = plot_actual_vs_predicted(y_t, y_p, model_name)
    fig_resid = plot_residuals(y_t, y_p, model_name)
    fig_dist = plot_error_distribution(y_t, y_p, model_name)

    # Engine-level error breakdown
    engine_results = []
    if engine_ids is not None:
        for u_id, yt, yp in zip(engine_ids, y_t, y_p):
            err = yp - yt
            engine_results.append({
                "unit_number": int(u_id),
                "true_rul": float(yt),
                "pred_rul": float(yp),
                "error": float(err),
                "abs_error": float(abs(err)),
                "phm_penalty": float(phm_score(np.array([yt]), np.array([yp]))),
            })
        # Sort by absolute error descending to highlight highest error engines
        engine_results.sort(key=lambda x: x["abs_error"], reverse=True)

    report = {
        "model_name": model_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "figures": {
            "actual_vs_predicted": str(fig_actual),
            "residuals": str(fig_resid),
            "error_distribution": str(fig_dist),
        },
        "hyperparameters": config_dict or {},
        "top_10_highest_error_engines": engine_results[:10] if engine_results else [],
    }

    # Persist JSON report
    report_file = EVALUATION_DIR / f"{model_name.lower().replace(' ', '_')}_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Evaluation for {model_name} completed. "
        f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, "
        f"PHM Score: {metrics['phm_score']:.2f}, R2: {metrics['r2']:.4f}"
    )
    return report

