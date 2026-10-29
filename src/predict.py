"""Inference pipeline for Remaining Useful Life (RUL) estimation."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from src.config import ServiceConfig
from src.preprocessing import CMapssPreprocessor
from src.models import build_model
from src.logging_config import setup_logger
from monitoring.prediction_monitor import global_prediction_monitor

logger = setup_logger("Predictor")


def determine_health_status(rul: float) -> str:
    """Classifies operational health risk based on predicted RUL cycles."""
    if rul > 80.0:
        return "HEALTHY"
    elif rul >= 30.0:
        return "WARNING"
    else:
        return "CRITICAL"


class RULPredictor:
    """Production predictor for online and batch RUL inference."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        service_config = ServiceConfig()
        self.model_path = model_path or service_config.model_path
        self.preprocessor_path = preprocessor_path or service_config.scaler_path
        self.metadata_path = metadata_path or service_config.metadata_path
        self.is_production = service_config.is_production

        self.model = None
        self.preprocessor: Optional[CMapssPreprocessor] = None
        self.metadata: Dict[str, Any] = {}
        self.is_ready: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model weights, scaler, and configuration."""
        if not self.model_path.exists():
            if self.is_production:
                raise FileNotFoundError(f"Model weights not found at {self.model_path}")
            logger.warning(
                f"Model weights not found at {self.model_path}. "
                f"Model must be trained before inference."
            )
            return

        if not self.preprocessor_path.exists():
            if self.is_production:
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
            logger.warning(f"Preprocessor not found at {self.preprocessor_path}.")
            return

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            if self.is_production:
                raise FileNotFoundError(f"Model metadata not found at {self.metadata_path}")
            self.metadata = {
                "model_type": "lstm",
                "window_length": 30,
                "input_dim": 14,
                "version": "1.0",
            }

        # Load Preprocessor
        self.preprocessor = CMapssPreprocessor.load(self.preprocessor_path)
        if not self.preprocessor.is_fitted or self.preprocessor.scaler is None:
            raise ValueError("Saved preprocessor must be fitted and contain a scaler.")

        # Build and load PyTorch model
        input_dim = len(self.preprocessor.selected_features)
        model_type = self.metadata.get("model_type", "lstm")
        window_length = self.metadata.get("window_length", 30)
        declared_features = self.metadata.get("input_features")
        if declared_features is not None and declared_features != self.preprocessor.selected_features:
            raise ValueError("Model metadata features do not match the saved preprocessor.")
        declared_dim = self.metadata.get("input_dimension", self.metadata.get("input_dim"))
        if declared_dim is not None and declared_dim != input_dim:
            raise ValueError("Model metadata input dimension does not match the saved preprocessor.")
        if not isinstance(window_length, int) or window_length < 1:
            raise ValueError("Model metadata window_length must be a positive integer.")

        self.model = build_model(
            model_type=model_type,
            input_dim=input_dim,
            window_length=window_length,
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.is_ready = True
        logger.info(f"RULPredictor successfully initialized with {model_type} model.")

    def predict_sequence(
        self,
        sequence: Union[np.ndarray, List[List[float]]],
        engine_id: Optional[int] = None,
        ground_truth_rul: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Performs single sequence RUL inference.

        Args:
            sequence: 2D array of shape (window_length, num_features).
            engine_id: Optional engine identifier.
            ground_truth_rul: Optional actual RUL for live metric monitoring.

        Returns:
            Dict containing predicted_rul, raw_rul, health_status, latency_ms.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Model is not ready or trained model files are missing. "
                "Please run training first."
            )

        start_time = time.perf_counter()
        seq_arr = np.asarray(sequence, dtype=np.float32)

        expected_len = self.metadata.get("window_length", 30)
        expected_features = len(self.preprocessor.selected_features)

        if seq_arr.ndim != 2:
            raise ValueError(f"Expected 2D sequence (window_length, features), got ndim={seq_arr.ndim}")

        if seq_arr.shape[0] != expected_len:
            raise ValueError(
                f"Expected sequence length {expected_len}, got {seq_arr.shape[0]}"
            )
        if seq_arr.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {seq_arr.shape[1]}"
            )
        if not np.isfinite(seq_arr).all():
            raise ValueError("Sequence values must all be finite numbers.")
        if ground_truth_rul is not None and (
            not np.isfinite(ground_truth_rul) or ground_truth_rul < 0
        ):
            raise ValueError("ground_truth_rul must be a finite non-negative number.")

        # Scale sequence features
        scaled_seq = self.preprocessor.scaler.transform(seq_arr)
        tensor_in = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_output = self.model(tensor_in).item()
        if not np.isfinite(raw_output):
            raise RuntimeError("Model produced a non-finite RUL prediction.")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Monitored prediction with negative clipping
        constrained_rul = global_prediction_monitor.record_prediction(
            raw_rul=raw_output,
            latency_ms=latency_ms,
            model_version=self.metadata.get("version", "v1.0"),
            ground_truth_rul=ground_truth_rul,
        )

        health = determine_health_status(constrained_rul)

        return {
            "engine_id": engine_id,
            "predicted_rul": round(constrained_rul, 2),
            "raw_rul": round(raw_output, 2),
            "health_status": health,
            "model_version": self.metadata.get("version", "v1.0"),
            "latency_ms": round(latency_ms, 2),
            "validation_status": "VALID",
        }
