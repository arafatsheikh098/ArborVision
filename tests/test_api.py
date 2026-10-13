"""Tests for the FastAPI surface (no torch required for the lightweight ones).

These tests use FastAPI's ``TestClient`` against the live app instance,
exercising the route registration and the JSON contract. The ``/predict``
endpoint is exercised using a small in-memory JPEG so we don't need torch.
"""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest

# Skip if FastAPI/httpx aren't available
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
_testclient = pytest.importorskip("fastapi.testclient")

from arborvision.api.server import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    return _testclient.TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "arborvision_version" in data
    assert "device" in data
    assert "n_models" in data


def test_list_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    names = {m["name"] for m in models}
    assert "resnet50" in names
    assert "maskrcnn" in names
    for m in models:
        assert m["task"] in ("classification", "instance_segmentation", "object_detection")
        assert m["device"]  # non-empty


def test_benchmark_unknown_model_rejected(client):
    r = client.post(
        "/benchmark",
        json={
            "model_names": ["not-a-model"],
            "image_paths": ["x.png"],
        },
    )
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


def test_benchmark_missing_image_paths_rejected(client):
    r = client.post(
        "/benchmark",
        json={"model_names": ["resnet50"], "image_paths": []},
    )
    assert r.status_code == 400


def test_predict_rejects_unknown_model(client):
    # Build a tiny JPEG in memory
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    files = {"files": ("test.jpg", BytesIO(buf.tobytes()), "image/jpeg")}
    r = client.post(
        "/predict",
        files=files,
        data={"model_names": "not-a-model"},
    )
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


def test_predict_rejects_no_files(client):
    r = client.post(
        "/predict",
        data={"model_names": "resnet50"},
    )
    assert r.status_code in (400, 422)