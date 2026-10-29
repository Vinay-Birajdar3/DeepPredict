# Changelog

This file records notable project capabilities and changes. No earlier release history was available when this changelog was created.

## 2026-10-28

### Documented baseline

- Added project architecture documentation covering the React UI, FastAPI service, model pipeline, monitoring components, and local startup.
- Recorded the existing application capabilities:
  - React UI served by FastAPI from `static/index.html`.
  - FastAPI endpoints for health, model information, RUL inference, fleet data, and monitoring.
  - C-MAPSS data loading, preprocessing, model training, and evaluation workflows.
  - PyTorch LSTM inference with saved model, scaler, and metadata artifacts.
  - Prediction, data-quality, and sensor-drift monitoring.

### UI simplification

- Removed the Streamlit dashboard and its dedicated service configuration.
- Kept the React UI served by FastAPI at `http://localhost:8000/`.
- Removed Streamlit from the Python dependencies and container image setup.
