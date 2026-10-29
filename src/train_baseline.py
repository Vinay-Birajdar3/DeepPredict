"""Baseline Machine Learning model training script (Random Forest) for DeepPredict."""

from pathlib import Path
import json
import numpy as np

from src.config import DatasetConfig, PreprocessingConfig, TARGET_CAPPED_COLUMN, MODELS_DIR
from src.data_loader import load_and_prepare_cmapss
from src.preprocessing import (
    CMapssPreprocessor,
    split_train_val_engines,
    create_sliding_sequences,
    create_test_evaluation_sequences,
)
from src.train import train_baseline_rf
from src.explainability import plot_tree_feature_importance, compute_shap_explanations
from src.logging_config import setup_logger

logger = setup_logger("BaselineTrainer")


def run_baseline():
    logger.info("Starting Baseline Machine Learning pipeline...")
    # Load dataset
    ds_config = DatasetConfig(rul_cap=125, random_seed=42)
    train_df, test_df, test_rul_truth, meta = load_and_prepare_cmapss(ds_config)

    # Engine split
    train_split, val_split, _, _ = split_train_val_engines(train_df, val_ratio=0.20, random_seed=42)

    # Preprocessing
    prep_config = PreprocessingConfig(window_length=30)
    preprocessor = CMapssPreprocessor(prep_config)
    preprocessor.fit(train_split)

    train_scaled = preprocessor.transform(train_split)
    val_scaled = preprocessor.transform(val_split)
    test_scaled = preprocessor.transform(test_df)

    selected_features = preprocessor.selected_features
    target_col = TARGET_CAPPED_COLUMN

    X_train, y_train, _ = create_sliding_sequences(train_scaled, selected_features, target_col, 30)
    X_val, y_val, _ = create_sliding_sequences(val_scaled, selected_features, target_col, 30)
    X_test_last, y_test_last, test_ids = create_test_evaluation_sequences(test_scaled, selected_features, target_col, 30)

    # Train Random Forest baseline
    rf_results = train_baseline_rf(
        X_train_seq=X_train,
        y_train=y_train,
        X_val_seq=X_val,
        y_val=y_val,
        X_test_seq=X_test_last,
        y_test=y_test_last,
        feature_names=selected_features,
        n_estimators=100,
        random_seed=42,
    )

    # Feature importances
    plot_tree_feature_importance(
        rf_results["feature_names"],
        rf_results["importances"],
        model_name="RandomForest",
    )

    logger.info("Random Forest baseline training and evaluation complete.")
    return rf_results


if __name__ == "__main__":
    run_baseline()

