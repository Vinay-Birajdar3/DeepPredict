"""Model explainability and error diagnostics for DeepPredict.

Provides:
- Sensor correlation analysis with degradation targets.
- Tree-based feature importance.
- SHAP values for model-agnostic feature contributions.
- Residual and high-error root-cause diagnostics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR
from src.logging_config import setup_logger

logger = setup_logger("Explainability")


def analyze_sensor_correlations(
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str = "RUL",
    save_path: Optional[Path] = None,
) -> pd.Series:
    """Computes and visualizes Pearson correlation between sensors and degradation (RUL).

    Args:
        train_df: Cleaned training DataFrame.
        sensor_cols: List of sensor columns.
        target_col: Degradation target column.
        save_path: Output image path.

    Returns:
        pd.Series: Sorted correlation values.
    """
    corr_series = train_df[sensor_cols].apply(lambda col: col.corr(train_df[target_col]))
    corr_sorted = corr_series.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ["crimson" if c < 0 else "navy" for c in corr_sorted.values]
    bars = plt.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor="black")
    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title(f"Sensor Correlation with {target_col}", fontsize=14, fontweight="bold")
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Sensor", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "sensor_correlations.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation analysis figure to: {save_path}")
    return corr_sorted


def plot_tree_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "RandomForest",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Plots top Gini / split feature importances for tree-based baselines."""
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    top_df = df_imp.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="teal", edgecolor="black")
    plt.title(f"Top {top_n} Feature Importances ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"{model_name.lower()}_feature_importance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance figure to: {save_path}")
    return df_imp


def compute_shap_explanations(
    tree_model,
    X_sample: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Computes SHAP values for tree-based baseline model if shap library is installed."""
    try:
        import shap
    except ImportError:
        logger.warning("shap package not installed; skipping SHAP computation.")
        return None

    logger.info("Computing SHAP values for tree baseline...")
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    if save_path is None:
        save_path = FIGURES_DIR / "shap_summary_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {save_path}")
    return shap_values


def analyze_high_error_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
    threshold_percentile: float = 90.0,
) -> pd.DataFrame:
    """Isolates and characterizes top error engines for targeted maintenance review.

    Distinguishes:
    - Dangerous Late Predictions (RUL overestimated, engine at risk of unexpected failure)
    - Conservative Early Predictions (RUL underestimated, unnecessary early maintenance)
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    df_err = pd.DataFrame({
        "unit_number": engine_ids,
        "true_rul": y_true,
        "pred_rul": y_pred,
        "error": errors,
        "abs_error": abs_errors,
        "risk_category": np.where(errors > 0, "Late (Dangerous)", "Early (Conservative)"),
    })

    cutoff = np.percentile(abs_errors, threshold_percentile)
    high_errors = df_err[df_err["abs_error"] >= cutoff].sort_values(by="abs_error", ascending=False)
    logger.info(f"Identified {len(high_errors)} high-error engines (>= {threshold_percentile}th percentile error).")
    return high_errors

