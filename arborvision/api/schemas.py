"""Pydantic request/response models for the FastAPI surface."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata for a registered model."""

    name: str
    task: str
    description: str
    device: str
    weights: str
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class PredictionResponse(BaseModel):
    model: str
    latency_ms: float
    labels: list[str]
    scores: list[float]
    boxes: list[list[int]] = Field(default_factory=list)
    n_masks: int = 0
    topk_labels: list[str] = Field(default_factory=list)
    topk_scores: list[float] = Field(default_factory=list)


class MultiPredictResponse(BaseModel):
    """Result of running several models over a batch of images."""

    predictions: dict[str, list[PredictionResponse]]  # model_name -> per-image list
    total_latency_ms: float


class BenchmarkRequest(BaseModel):
    model_names: list[str]
    image_paths: list[str] | None = None
    ground_truth: list[str] | None = None
    score_threshold: float = 0.5
    n_warmup: int = 2
    n_runs: int = 10
    output_dir: Optional[str] = None
    save_overlays: bool = True


class ExperimentSummary(BaseModel):
    experiment_id: str
    created_at: str
    output_dir: str
    models: list[str]
    n_images: int


class ExperimentsListResponse(BaseModel):
    experiments: list[ExperimentSummary]


class HealthResponse(BaseModel):
    status: str
    arborvision_version: str
    device: str
    n_models: int