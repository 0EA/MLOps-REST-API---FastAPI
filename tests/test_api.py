"""Tests for the Iris API.

All tests train their own model into a pytest tmp_path directory (via
MODEL_ARTIFACTS_DIR) so they never touch -- and can never overwrite -- the
demo artifacts in <repo_root>/artifacts.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import FEATURE_ORDER, MODEL_FILENAME
from scripts.train import train as train_model

VALID_PAYLOAD = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}


@pytest.fixture
def trained_artifacts_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "artifacts"
    train_model(output_dir)
    return output_dir


@pytest.fixture
def client(trained_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(trained_artifacts_dir))
    with TestClient(app) as test_client:
        yield test_client


def test_health_after_startup(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_after_startup(client: TestClient):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "iris-logreg-1.0.0"
    assert body["feature_order"] == FEATURE_ORDER
    assert set(body["target_classes"]) == {"setosa", "versicolor", "virginica"}
    assert 0.0 <= body["held_out_accuracy"] <= 1.0
    assert body["random_seed"] == 42


def test_valid_prediction_shape(client: TestClient):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert body["predicted_species"] in {"setosa", "versicolor", "virginica"}
    assert body["model_version"] == "iris-logreg-1.0.0"

    probs = body["probabilities"]
    assert set(probs.keys()) == {"setosa", "versicolor", "virginica"}
    for value in probs.values():
        assert 0.0 <= value <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_prediction_matches_direct_pipeline_inference(client: TestClient, trained_artifacts_dir: Path):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    api_body = resp.json()

    pipeline = joblib.load(trained_artifacts_dir / MODEL_FILENAME)
    feature_vector = np.array([[VALID_PAYLOAD[name] for name in FEATURE_ORDER]], dtype=float)

    direct_label = pipeline.predict(feature_vector)[0]
    direct_proba = pipeline.predict_proba(feature_vector)[0]
    classes_ = pipeline.named_steps["classifier"].classes_

    import json as _json

    metadata = _json.loads((trained_artifacts_dir / "metadata.json").read_text())
    target_classes = metadata["target_classes"]
    class_index_to_name = {cls: target_classes[i] for i, cls in enumerate(classes_)}

    expected_species = class_index_to_name[direct_label]
    expected_probs = {
        class_index_to_name[cls]: pytest.approx(float(p), abs=1e-9)
        for cls, p in zip(classes_, direct_proba)
    }

    assert api_body["predicted_species"] == expected_species
    for species, expected_prob in expected_probs.items():
        assert api_body["probabilities"][species] == expected_prob


def test_missing_field_returns_422(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    del payload["petal_width_cm"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_nonpositive_measurement_returns_422(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    payload["sepal_length_cm"] = 0.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422

    payload["sepal_length_cm"] = -2.5
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_nonnumeric_value_returns_422(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    payload["sepal_length_cm"] = "not-a-number"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_non_finite_value_returns_422(client: TestClient):
    # JSON has no literal for Infinity/NaN, but Python's json module will
    # happily emit the nonstandard tokens `Infinity`/`NaN` that httpx's
    # content= bypasses json=; send raw bytes to exercise this edge case.
    payload = dict(VALID_PAYLOAD)
    payload["sepal_length_cm"] = float("inf")
    resp = client.post(
        "/predict",
        content=__import__("json").dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_extra_field_returns_422(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    payload["unexpected_field"] = "nope"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_startup_fails_with_actionable_message_when_artifacts_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(empty_dir))

    with pytest.raises(RuntimeError, match="train.py"):
        with TestClient(app):
            pass
