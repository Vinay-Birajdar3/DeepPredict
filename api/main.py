"""FastAPI REST service for real-time Remaining Useful Life (RUL) inference."""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = PROJECT_ROOT / "static"

from api.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse
from src.config import ServiceConfig
from src.logging_config import setup_logger
from src.predict import RULPredictor, determine_health_status
from monitoring.prediction_monitor import global_prediction_monitor
from monitoring.data_quality import DataQualityMonitor

logger = setup_logger("API")
service_config = ServiceConfig()

# Global predictor & dataset cache
predictor: Optional[RULPredictor] = None
data_quality_monitor: Optional[DataQualityMonitor] = None
_train_df_cache: Optional[pd.DataFrame] = None
_test_df_cache: Optional[pd.DataFrame] = None
_test_truth_cache: Optional[np.ndarray] = None
_test_engines_cache: Dict[int, dict] = {}


def _get_test_dataset():
    """Lazily loads and caches the test dataset for the fleet explorer."""
    global _train_df_cache, _test_df_cache, _test_truth_cache, _test_engines_cache
    if _test_df_cache is None:
        from src.data_loader import load_and_prepare_cmapss
        from src.config import DatasetConfig
        train_df, test_df, test_rul_truth, _ = load_and_prepare_cmapss(DatasetConfig())
        _train_df_cache = train_df
        _test_df_cache = test_df
        _test_truth_cache = test_rul_truth
        
        for unit_id in sorted(test_df["unit_number"].unique()):
            unit_df = test_df[test_df["unit_number"] == unit_id]
            truth = float(test_rul_truth[unit_id - 1])
            _test_engines_cache[int(unit_id)] = {
                "unit_id": int(unit_id),
                "total_cycles": int(len(unit_df)),
                "ground_truth_rul": truth,
                "status": determine_health_status(truth),
            }
    return _test_df_cache, _test_truth_cache, _test_engines_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches model artifacts on service startup."""
    global predictor, data_quality_monitor
    logger.info("DeepPredict API service starting up...")
    try:
        predictor = RULPredictor()
        data_quality_monitor = DataQualityMonitor()
        if predictor.is_ready:
            logger.info("Predictor successfully loaded model weights.")
        else:
            logger.warning("Predictor initialized without trained weights. Run training to enable inference.")
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        if service_config.is_production:
            raise
        predictor = None
    yield
    logger.info("DeepPredict API service shutting down.")


app = FastAPI(
    title="DeepPredict: Turbofan Prognostics API",
    description=(
        "Production REST API for Remaining Useful Life (RUL) estimation of "
        "aircraft turbofan engines using Deep Learning (LSTM) and NASA C-MAPSS dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(response: Response):
    """Returns service operational health and model readiness status."""
    is_ready = predictor is not None and predictor.is_ready
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=is_ready,
        model_name=predictor.metadata.get("model_name") if is_ready else None,
        model_version=predictor.metadata.get("version") if is_ready else None,
    )


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Reports that the API process is responsive, independent of model readiness."""
    return {"status": "alive"}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Returns loaded model architecture, selected features, and test performance metrics."""
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model has not been trained yet. Please execute "
                "'py -m src.train --model_type lstm' to train and save the model."
            ),
        )

    return ModelInfoResponse(
        model_name=predictor.metadata.get("model_name", "LSTM_FD001"),
        model_type=predictor.metadata.get("model_type", "lstm"),
        window_length=predictor.metadata.get("window_length", 30),
        input_features=predictor.preprocessor.selected_features,
        input_dimension=len(predictor.preprocessor.selected_features),
        training_metrics={"best_val_loss": predictor.metadata.get("best_val_loss", 0.0)},
        test_metrics=predictor.metadata.get("test_metrics"),
        created_at=predictor.metadata.get("created_at", datetime.now().isoformat()),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(request: PredictionRequest):
    """Estimates the Remaining Useful Life (RUL) for a turbofan sensor sequence window."""
    if predictor is None or not predictor.is_ready:
        global_prediction_monitor.record_error()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is untrained or unavailable. Please run training first "
                "using 'py -m src.train --model_type lstm'."
            ),
        )
    if service_config.is_production and request.ground_truth_rul is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ground_truth_rul is only accepted in non-production environments.",
        )

    # 1. Parse input format
    selected_features = predictor.preprocessor.selected_features
    window_length = predictor.metadata.get("window_length", 30)

    try:
        if request.sequence is not None:
            seq_arr = np.array(request.sequence, dtype=np.float32)
            if seq_arr.ndim != 2:
                raise ValueError(
                    "Provided sequence must be a 2D array with shape "
                    "(window_length, features)"
                )
            if seq_arr.shape[1] != len(selected_features):
                raise ValueError(
                    f"Provided sequence has {seq_arr.shape[1]} features, but model expects "
                    f"{len(selected_features)} features: {selected_features}"
                )
        elif request.readings is not None:
            if len(request.readings) != window_length:
                raise ValueError(
                    f"Expected exactly {window_length} cycle readings, received {len(request.readings)}"
                )
            matrix = []
            for cycle_dict in request.readings:
                row = []
                for feat in selected_features:
                    if feat not in cycle_dict:
                        raise ValueError(f"Missing required sensor reading: '{feat}'")
                    row.append(float(cycle_dict[feat]))
                matrix.append(row)
            seq_arr = np.array(matrix, dtype=np.float32)
        else:
            raise ValueError("Either 'sequence' or 'readings' must be provided in request body.")

        # 2. Run inference
        result = predictor.predict_sequence(
            sequence=seq_arr,
            engine_id=request.engine_id,
            ground_truth_rul=request.ground_truth_rul,
        )

        return PredictionResponse(**result)

    except ValueError as val_err:
        global_prediction_monitor.record_error()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as e:
        global_prediction_monitor.record_error()
        logger.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed.",
        )


@app.get("/monitoring/status", tags=["Observability"])
async def get_monitoring_status():
    """Returns live prediction statistics, latency percentiles, and error rate."""
    return global_prediction_monitor.get_summary_statistics()


@app.get("/api/monitoring/drift", tags=["Observability"])
async def get_drift_report():
    """Calculates and returns KS-test and PSI sensor drift metrics."""
    _require_demo_mode()
    global _train_df_cache
    test_df, _, _ = _get_test_dataset()
    if _train_df_cache is None:
        from src.data_loader import load_and_prepare_cmapss
        from src.config import DatasetConfig
        train_df, _, _, _ = load_and_prepare_cmapss(DatasetConfig())
        _train_df_cache = train_df

    features = predictor.preprocessor.selected_features if (predictor and predictor.is_ready) else [
        "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
    ]
    
    from monitoring.drift_detection import DriftDetector
    detector = DriftDetector(_train_df_cache, features)
    report = detector.detect_drift(test_df)
    
    feature_list = []
    for f in features:
        if f in report.feature_metrics:
            m = report.feature_metrics[f]
            feature_list.append({
                "sensor": f,
                "has_drifted": m.has_drifted,
                "ks_statistic": round(m.ks_statistic, 4),
                "p_value": round(m.p_value, 6),
                "psi": round(m.psi, 4),
                "wasserstein_distance": round(m.wasserstein_distance, 4),
                "drift_reason": m.drift_reason,
            })
            
    return {
        "dataset_has_drifted": report.dataset_has_drifted,
        "drifted_feature_count": report.drifted_feature_count,
        "total_feature_count": report.total_feature_count,
        "features": feature_list,
    }


@app.get("/api/engines", tags=["Fleet"])
async def list_engines():
    """Lists all 100 test engines with cycle counts and ground truth status."""
    _require_demo_mode()
    _, _, engines_cache = _get_test_dataset()
    return list(engines_cache.values())


@app.get("/api/engines/{unit_id}", tags=["Fleet"])
async def get_engine_detail(unit_id: int):
    """Returns degradation sensor trajectory and inference window for a specific engine."""
    _require_demo_mode()
    test_df, test_truth, engines_cache = _get_test_dataset()
    if unit_id not in engines_cache:
        raise HTTPException(status_code=404, detail=f"Engine {unit_id} not found (must be 1-100)")

    unit_df = test_df[test_df["unit_number"] == unit_id]
    features = predictor.preprocessor.selected_features if (predictor and predictor.is_ready) else [
        "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
    ]

    records = []
    for _, row in unit_df.iterrows():
        rec = {"cycle": int(row["time_in_cycles"])}
        for f in features:
            rec[f] = round(float(row[f]), 3)
        records.append(rec)

    window_length = predictor.metadata.get("window_length", 30) if predictor else 30
    last_window_df = unit_df[features].tail(window_length)

    if len(last_window_df) < window_length:
        pad_count = window_length - len(last_window_df)
        top_row = last_window_df.iloc[[0] * pad_count]
        last_window_df = pd.concat([top_row, last_window_df], axis=0)

    last_window = last_window_df.to_numpy(dtype=float).tolist()

    return {
        "unit_id": unit_id,
        "total_cycles": engines_cache[unit_id]["total_cycles"],
        "ground_truth_rul": engines_cache[unit_id]["ground_truth_rul"],
        "status": engines_cache[unit_id]["status"],
        "features": features,
        "trajectory": records,
        "last_window": last_window,
    }


@app.post("/api/engines/{unit_id}/predict", tags=["Inference"])
async def predict_engine(unit_id: int):
    """Executes immediate RUL prediction on the test engine's operational data."""
    _require_demo_mode()
    if predictor is None or not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Predictor is not ready")

    test_df, test_truth, engines_cache = _get_test_dataset()
    if unit_id not in engines_cache:
        raise HTTPException(status_code=404, detail=f"Engine {unit_id} not found")

    unit_df = test_df[test_df["unit_number"] == unit_id]
    features = predictor.preprocessor.selected_features
    window_length = predictor.metadata.get("window_length", 30)

    last_window_df = unit_df[features].tail(window_length)
    if len(last_window_df) < window_length:
        pad_count = window_length - len(last_window_df)
        top_row = last_window_df.iloc[[0] * pad_count]
        last_window_df = pd.concat([top_row, last_window_df], axis=0)

    seq_matrix = last_window_df.to_numpy(dtype=np.float32)
    true_rul = engines_cache[unit_id]["ground_truth_rul"]

    pred_res = predictor.predict_sequence(
        sequence=seq_matrix,
        engine_id=unit_id,
        ground_truth_rul=true_rul,
    )

    pred_rul = float(pred_res["predicted_rul"])
    error = round(pred_rul - true_rul, 2)

    return {
        "engine_id": unit_id,
        "predicted_rul": pred_rul,
        "ground_truth_rul": true_rul,
        "error": error,
        "health_status": pred_res["health_status"],
        "latency_ms": pred_res["latency_ms"],
        "model_version": pred_res["model_version"],
    }


# Static files mount and root route
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui() -> Response:
    """Serves the production inference dashboard or development demo dashboard."""
    if service_config.is_production:
        return FileResponse(STATIC_DIR / "production.html")

    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "DeepPredict API active. Static UI not built yet."}


def _require_demo_mode() -> None:
    if service_config.is_production:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This demo endpoint is disabled in production.",
        )


if __name__ == "__main__":
    import uvicorn
    cfg = ServiceConfig()
    uvicorn.run("api.main:app", host=cfg.api_host, port=cfg.api_port, reload=False)
