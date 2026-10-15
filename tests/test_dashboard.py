"""Integration coverage for the new browser dashboard route and static assets.

Only functional checks (status codes, content-type, key landmarks) -- no
snapshot testing of cosmetic HTML/CSS details.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.train import train as train_model


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "artifacts"
    train_model(output_dir)
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(output_dir))
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_root_serves_html(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "FastAPI" in resp.text
    assert "ML Inference Lab" in resp.text


def test_dashboard_root_excluded_from_openapi_schema(client: TestClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/" not in schema["paths"]
    assert "/predict" in schema["paths"]


def test_static_stylesheet_served(client: TestClient):
    resp = client.get("/static/styles.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]


def test_static_script_served(client: TestClient):
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    body = resp.text
    assert "fetch(" in body
    # The dashboard must never compute predictions client-side.
    assert "/predict" in body


def test_docs_and_openapi_still_reachable(client: TestClient):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
