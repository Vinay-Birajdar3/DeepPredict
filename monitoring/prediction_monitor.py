"""Inference observability and prediction health monitoring for DeepPredict."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.logging_config import setup_logger

logger = setup_logger("PredictionMonitor")


@dataclass
class PredictionLogEntry:
    """Record of a single model inference call."""
    timestamp: str
    predicted_rul: float
    raw_predicted_rul: float
    clipped_negative: bool
    latency_ms: float
    model_version: str
    status: str
    ground_truth_rul: Optional[float] = None


class PredictionMonitor:
    """Monitors live inference distribution, latency, errors, and degradation."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: List[PredictionLogEntry] = []
        self.error_count: int = 0
        self.total_requests: int = 0

    def record_prediction(
        self,
        raw_rul: float,
        latency_ms: float,
        model_version: str = "lstm_v1",
        ground_truth_rul: Optional[float] = None,
    ) -> float:
        """Processes and logs a model prediction, safely handling non-physical negative RULs.

        In turbofan prognostics, Remaining Useful Life cannot physically be negative.
        If a model outputs a negative prediction (e.g. -2.4 cycles), we clip it to 0.0
        and log an anomaly flag for prediction drift observability.

        Args:
            raw_rul: Unconstrained model output.
            latency_ms: Prediction latency in milliseconds.
            model_version: Model identifier.
            ground_truth_rul: Optional actual RUL if known.

        Returns:
            float: Constrained RUL (clipped to >= 0.0).
        """
        self.total_requests += 1
        clipped_negative = raw_rul < 0.0
        constrained_rul = float(max(0.0, raw_rul))

        entry = PredictionLogEntry(
            timestamp=datetime.now().isoformat(),
            predicted_rul=constrained_rul,
            raw_predicted_rul=float(raw_rul),
            clipped_negative=clipped_negative,
            latency_ms=float(latency_ms),
            model_version=model_version,
            status="SUCCESS",
            ground_truth_rul=ground_truth_rul,
        )

        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if clipped_negative:
            logger.warning(
                f"Negative RUL predicted ({raw_rul:.2f} cycles). "
                f"Clipped to 0.0 cycles for operational safety."
            )

        return constrained_rul

    def record_error(self) -> None:
        """Increments failed request / exception count."""
        self.total_requests += 1
        self.error_count += 1

    def get_summary_statistics(self) -> Dict[str, any]:
        """Calculates live inference health and distribution metrics."""
        if not self.history:
            return {
                "total_predictions": 0,
                "total_errors": self.error_count,
                "error_rate": 0.0,
                "negative_predictions_clipped": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "prediction_mean": 0.0,
                "prediction_std": 0.0,
                "rolling_mae": None,
                "rolling_rmse": None,
            }

        predictions = np.array([e.predicted_rul for e in self.history])
        latencies = np.array([e.latency_ms for e in self.history])
        neg_count = sum(1 for e in self.history if e.clipped_negative)

        # Ground truth evaluation when labels are supplied
        paired_errors = [
            e.predicted_rul - e.ground_truth_rul
            for e in self.history
            if e.ground_truth_rul is not None
        ]

        rolling_mae = float(np.mean(np.abs(paired_errors))) if paired_errors else None
        rolling_rmse = float(np.sqrt(np.mean(np.array(paired_errors) ** 2))) if paired_errors else None

        return {
            "total_predictions": len(self.history),
            "total_errors": self.error_count,
            "error_rate": float(self.error_count / max(1, self.total_requests)),
            "negative_predictions_clipped": neg_count,
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "rolling_mae": rolling_mae,
            "rolling_rmse": rolling_rmse,
        }


# Global singleton instance for API and Dashboard observability
global_prediction_monitor = PredictionMonitor()

