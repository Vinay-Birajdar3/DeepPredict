"""Integration tests for FastAPI endpoints."""

import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient
from api.main import app
from api import main as api_main


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "model_loaded" in data


def test_health_endpoint_reports_unready_model(client, monkeypatch):
    monkeypatch.setattr(api_main, "predictor", None)

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    assert client.get("/live").status_code == 200


def test_predict_endpoint_missing_body(client):
    response = client.post("/predict", json={})
    # Should fail validation (either 422 or 503 if model unloaded)
    assert response.status_code in (422, 503)


def test_predict_endpoint_rejects_empty_sequence(client):
    response = client.post("/predict", json={"sequence": []})
    assert response.status_code == 422


def test_predict_endpoint_rejects_wrong_feature_count(client):
    response = client.post(
        "/predict",
        json={"sequence": [[0.0] * 13 for _ in range(30)]},
    )
    assert response.status_code == 422


def test_demo_routes_are_disabled_in_production(client, monkeypatch):
    monkeypatch.setattr(api_main.service_config, "app_env", "production")

    assert client.get("/api/engines").status_code == 404
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_production_prediction_rejects_client_supplied_truth(client, monkeypatch):
    monkeypatch.setattr(api_main.service_config, "app_env", "production")
    monkeypatch.setattr(api_main, "predictor", SimpleNamespace(is_ready=True))
    response = client.post(
        "/predict",
        json={
            "sequence": [[0.0] * 14 for _ in range(30)],
            "ground_truth_rul": 10,
        },
    )

    assert response.status_code == 422
    assert "only accepted in non-production" in response.json()["detail"]


def test_monitoring_status_endpoint(client):
    response = client.get("/monitoring/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "error_rate" in data
