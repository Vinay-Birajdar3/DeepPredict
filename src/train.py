"""Model training pipeline for DeepPredict.

Supports:
- Baseline Machine Learning (Random Forest)
- Deep Learning (LSTM, 1D CNN, Transformer)
- Zero data leakage validation splits
- Early stopping, learning rate scheduling, checkpointing
- MLflow tracking and artifact logging
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    DatasetConfig,
    ModelConfig,
    PreprocessingConfig,
    MODELS_DIR,
    FIGURES_DIR,
    MLRUNS_DIR,
    TARGET_COLUMN,
    TARGET_CAPPED_COLUMN,
)
from src.data_loader import load_and_prepare_cmapss
from src.dataset import get_dataloaders
from src.evaluate import evaluate_predictions
from src.feature_engineering import extract_tabular_window_features
from src.logging_config import setup_logger
from src.metrics import compute_all_metrics
from src.models import build_model
from src.preprocessing import (
    CMapssPreprocessor,
    split_train_val_engines,
    create_sliding_sequences,
    create_test_evaluation_sequences,
)

logger = setup_logger("Trainer")


def set_seed(seed: int = 42) -> None:
    """Sets reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_baseline_rf(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    X_test_seq: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    random_seed: int = 42,
) -> Dict[str, any]:
    """Trains a Random Forest baseline on engineered sequence features."""
    logger.info("Extracting tabular window features for Random Forest...")
    X_tr_tab = extract_tabular_window_features(X_train_seq, feature_names)
    X_val_tab = extract_tabular_window_features(X_val_seq, feature_names)
    X_te_tab = extract_tabular_window_features(X_test_seq, feature_names)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        random_state=random_seed,
        n_jobs=-1,
    )
    logger.info("Training Random Forest baseline...")
    start_time = time.time()
    rf.fit(X_tr_tab, y_train)
    duration = time.time() - start_time

    val_preds = rf.predict(X_val_tab)
    test_preds = rf.predict(X_te_tab)

    val_metrics = compute_all_metrics(y_val, val_preds)
    test_report = evaluate_predictions(y_test, test_preds, "Baseline_RandomForest")

    logger.info(
        f"Random Forest finished in {duration:.1f}s. "
        f"Val MAE: {val_metrics['mae']:.2f}, Test MAE: {test_report['metrics']['mae']:.2f}, "
        f"Test RMSE: {test_report['metrics']['rmse']:.2f}, PHM: {test_report['metrics']['phm_score']:.2f}"
    )

    return {
        "model": rf,
        "feature_names": X_tr_tab.columns.tolist(),
        "importances": rf.feature_importances_,
        "val_metrics": val_metrics,
        "test_metrics": test_report["metrics"],
    }


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Plots and persists epoch-by-epoch training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss (MSE)")
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss (MSE)")
    plt.title(f"{model_name}: Learning Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"{model_name.lower()}_learning_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def train_deep_learning_model(
    model_type: str = "lstm",
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
    window_length: int = 30,
    rul_cap: Optional[int] = 125,
    seed: int = 42,
    device_str: str = "cpu",
    use_mlflow: bool = True,
) -> Dict[str, any]:
    """Complete deep learning training lifecycle for LSTM, 1D CNN, or Transformer.

    Returns:
        Dict: Training results, model metadata, and test evaluation metrics.
    """
    set_seed(seed)
    device = torch.device(device_str)
    logger.info(f"Initiating training for {model_type.upper()} on device: {device}")

    # 1. Load C-MAPSS dataset
    ds_config = DatasetConfig(rul_cap=rul_cap, random_seed=seed)
    train_df, test_df, test_rul_truth, ds_meta = load_and_prepare_cmapss(ds_config)
    target_col = TARGET_CAPPED_COLUMN if rul_cap else TARGET_COLUMN

    # 2. Leakage-free train/val engine splitting
    train_split, val_split, tr_ids, val_ids = split_train_val_engines(
        train_df, val_ratio=0.20, random_seed=seed
    )

    # 3. Fit preprocessor strictly on train engines
    prep_config = PreprocessingConfig(window_length=window_length)
    preprocessor = CMapssPreprocessor(prep_config)
    preprocessor.fit(train_split)
    preprocessor_path = MODELS_DIR / "scaler.joblib"
    preprocessor.save(preprocessor_path)

    # Transform all splits using training scaler
    train_scaled = preprocessor.transform(train_split)
    val_scaled = preprocessor.transform(val_split)
    test_scaled = preprocessor.transform(test_df)

    # 4. Generate temporal sequence windows
    selected_features = preprocessor.selected_features
    X_train, y_train, _ = create_sliding_sequences(train_scaled, selected_features, target_col, window_length)
    X_val, y_val, _ = create_sliding_sequences(val_scaled, selected_features, target_col, window_length)
    X_test_last, y_test_last, test_engine_ids = create_test_evaluation_sequences(
        test_scaled, selected_features, target_col, window_length
    )

    input_dim = len(selected_features)
    logger.info(
        f"Input shape: {X_train.shape[1:]} ({input_dim} sensor channels, window {window_length}). "
        f"Train windows: {len(X_train)}, Val windows: {len(X_val)}, Test engines: {len(X_test_last)}."
    )

    # 5. Build Model
    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        window_length=window_length,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # Training loop with Early Stopping
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    model_save_path = MODELS_DIR / f"best_{model_type.lower()}_model.pt"

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * len(y_b)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_true_list = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                running_val_loss += loss.item() * len(y_b)
                val_preds_list.extend(preds.cpu().numpy().tolist())
                val_true_list.extend(y_b.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            val_mae = float(np.mean(np.abs(np.array(val_preds_list) - np.array(val_true_list))))
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} - Train MSE: {epoch_train_loss:.2f} | "
                f"Val MSE: {epoch_val_loss:.2f} | Val MAE: {val_mae:.2f}"
            )

        # Checkpointing on validation performance
        if epoch_val_loss < best_val_loss - 0.05:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": model_type,
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "window_length": window_length,
                    "input_dim": input_dim,
                },
                model_save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    training_duration = time.time() - start_train_time
    logger.info(f"Training completed in {training_duration:.1f}s. Best Val Loss: {best_val_loss:.2f} at epoch {best_epoch}.")

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Learning curve plot
    fig_loss = plot_loss_curves(train_losses, val_losses, f"{model_type.upper()}")

    # 6. Benchmark Evaluation on Official Test Engines (last cycle)
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_last, dtype=torch.float32).to(device)
        raw_test_preds = model(test_tensor).cpu().numpy()
        # Physical constraint: RUL >= 0
        final_test_preds = np.clip(raw_test_preds, a_min=0.0, a_max=None)

    eval_report = evaluate_predictions(
        y_true=y_test_last,
        y_pred=final_test_preds,
        model_name=f"{model_type.upper()}_FD001",
        engine_ids=test_engine_ids,
        config_dict={
            "model_type": model_type,
            "window_length": window_length,
            "epochs_run": len(train_losses),
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "rul_cap": rul_cap,
            "training_duration_sec": training_duration,
        },
    )

    # Save model metadata JSON for API & Dashboard
    metadata = {
        "model_name": f"{model_type.upper()}_FD001",
        "model_type": model_type,
        "window_length": window_length,
        "input_features": selected_features,
        "input_dimension": input_dim,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_metrics": eval_report["metrics"],
        "training_duration": float(training_duration),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "1.0",
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # MLflow Tracking (optional/fallback safe)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(str(MLRUNS_DIR))
            mlflow.set_experiment("DeepPredict_Prognostics")
            with mlflow.start_run(run_name=f"{model_type.upper()}_FD001"):
                mlflow.log_params({
                    "model_type": model_type,
                    "window_length": window_length,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "rul_cap": rul_cap if rul_cap else "uncapped",
                    "epochs": epochs,
                })
                mlflow.log_metrics({
                    "val_loss": best_val_loss,
                    "test_mae": eval_report["metrics"]["mae"],
                    "test_rmse": eval_report["metrics"]["rmse"],
                    "test_phm_score": eval_report["metrics"]["phm_score"],
                    "test_r2": eval_report["metrics"]["r2"],
                    "training_duration": training_duration,
                })
                mlflow.log_artifact(str(model_save_path))
                mlflow.log_artifact(str(fig_loss))
                logger.info("Successfully logged run and artifacts to local MLflow.")
        except Exception as e:
            logger.warning(f"MLflow tracking failed (non-blocking): {e}")

    return {
        "metadata": metadata,
        "eval_report": eval_report,
        "model_path": str(model_save_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPredict Model Training")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "cnn_1d", "transformer", "rf"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_length", type=int, default=30)
    parser.add_argument("--rul_cap", type=int, default=125)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_deep_learning_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_length=args.window_length,
        rul_cap=args.rul_cap,
        learning_rate=args.lr,
    )
