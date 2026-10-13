"""Smoke tests for the model registry and Prediction serialization.

These tests do NOT require torch / torchvision to be installed — they
exercise the registry, the dataclass serialization round-trip, and the
base API contract using the ``DummyAdapter`` defined below.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from arborvision.models import (
    ModelAdapter,
    ModelMetadata,
    Prediction,
    get_model,
    list_models,
    register,
    unregister,
)
from arborvision.models.base import TaskType


class DummyAdapter(ModelAdapter):
    """A toy adapter used to exercise the contract without real models."""

    name = "dummy"
    task: TaskType = "classification"

    def __init__(self) -> None:
        self.warmed = False
        self.calls = 0

    def predict(self, image: np.ndarray) -> Prediction:
        self.calls += 1
        return Prediction(
            labels=["dummy_label"],
            scores=[0.99],
            boxes=[],
            masks=[],
            topk_labels=["dummy_label", "other"],
            topk_scores=[0.99, 0.01],
        )

    def warmup(self) -> None:
        self.warmed = True

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            task=self.task,
            description="A dummy adapter for tests.",
            device="cpu",
            weights="none",
        )


def test_registry_contains_builtin_models():
    names = list_models()
    # Built-ins should at least include the two shipped adapters
    assert "resnet50" in names
    assert "maskrcnn" in names


def test_register_and_unregister_dummy():
    register("dummy", DummyAdapter)
    try:
        assert "dummy" in list_models()
        m = get_model("dummy")
        assert m.warmed is False
        m.warmup()
        assert m.warmed is True
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        out = m.timed_predict(img)
        assert out.latency_ms >= 0
        assert out.labels == ["dummy_label"]
    finally:
        unregister("dummy")


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("definitely-not-a-model")


def test_register_duplicate_raises():
    register("dup", DummyAdapter)
    try:
        with pytest.raises(ValueError):
            register("dup", DummyAdapter)
    finally:
        unregister("dup")


def test_prediction_serialization_roundtrip():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    pred = Prediction(
        labels=["tree", "rock"],
        scores=[0.9, 0.5],
        boxes=[(0, 0, 2, 2), (2, 2, 4, 4)],
        masks=[mask, mask],
        latency_ms=12.5,
        topk_labels=["tree", "rock", "shrub"],
        topk_scores=[0.9, 0.5, 0.1],
    )
    d = pred.to_dict()
    assert d["labels"] == ["tree", "rock"]
    assert d["boxes"] == [[0, 0, 2, 2], [2, 2, 4, 4]]
    assert len(d["masks_b64"]) == 2
    # Round-trip back
    restored = Prediction.from_dict(d)
    assert restored.labels == pred.labels
    assert restored.scores == pred.scores
    assert restored.boxes == pred.boxes
    assert restored.latency_ms == pytest.approx(pred.latency_ms)
    assert len(restored.masks) == len(pred.masks)
    assert np.array_equal(restored.masks[0], pred.masks[0])