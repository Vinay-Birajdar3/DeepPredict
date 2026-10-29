"""Configuration module for DeepPredict predictive maintenance pipeline.

Centralizes paths, column definitions, hyperparameters, and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = REPORTS_DIR / "evaluation"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# Dataset column definitions
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
ALL_RAW_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Target columns
TARGET_COLUMN = "RUL"
TARGET_CAPPED_COLUMN = "RUL_capped"

# Descriptive names for C-MAPSS sensors
SENSOR_DESCRIPTIONS = {
    "s_1": "Fan inlet temperature (°R)",
    "s_2": "LPC outlet temperature (°R)",
    "s_3": "HPC outlet temperature (°R)",
    "s_4": "LPT outlet temperature (°R)",
    "s_5": "Fan inlet pressure (psia)",
    "s_6": "Bypass-duct pressure (psia)",
    "s_7": "HPC outlet pressure (psia)",
    "s_8": "Physical fan speed (rpm)",
    "s_9": "Physical core speed (rpm)",
    "s_10": "Engine pressure ratio (P50/P2)",
    "s_11": "HPC outlet static pressure (psia)",
    "s_12": "Ratio of fuel flow to Ps30 (pps/psia)",
    "s_13": "Corrected fan speed (rpm)",
    "s_14": "Corrected core speed (rpm)",
    "s_15": "Bypass ratio",
    "s_16": "Burner fuel-air ratio",
    "s_17": "Bleed enthalpy",
    "s_18": "Demanded fan speed (rpm)",
    "s_19": "Demanded corrected fan speed (rpm)",
    "s_20": "HPT coolant bleed (lbm/s)",
    "s_21": "LPT coolant bleed (lbm/s)",
}


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    subset: str = "FD001"
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    train_file: str = "train_FD001.txt"
    test_file: str = "test_FD001.txt"
    rul_file: str = "RUL_FD001.txt"
    rul_cap: Optional[int] = 125
    validation_engine_ratio: float = 0.20
    random_seed: int = 42


@dataclass
class PreprocessingConfig:
    """Data preprocessing and feature engineering configuration."""
    window_length: int = 30
    drop_constant_threshold: float = 1e-4
    scaler_type: str = "minmax"  # 'minmax' or 'standard'
    feature_range: tuple = (0.0, 1.0)
    # Selected dynamic sensors for FD001 (sensors with degradation signal)
    # Will also be dynamically validated by drop_constant_sensors
    candidate_sensors: List[str] = field(
        default_factory=lambda: [
            "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
            "s_11", "s_12", "s_13", "s_14", "s_15", "s_17",
            "s_20", "s_21",
        ]
    )
    include_operational_settings: bool = False  # FD001 has single condition


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_type: str = "lstm"  # 'baseline_rf', 'lstm', 'cnn_1d'
    window_length: int = 30
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 0.1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42
    device: str = "cpu"  # default to cpu for rock-solid portability


@dataclass
class ServiceConfig:
    """FastAPI service configuration."""
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    app_env: str = os.getenv("APP_ENV", "development").lower()
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(MODELS_DIR)))
    model_path: Path = model_dir / "best_lstm_model.pt"
    scaler_path: Path = model_dir / "scaler.joblib"
    metadata_path: Path = model_dir / "model_metadata.json"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
