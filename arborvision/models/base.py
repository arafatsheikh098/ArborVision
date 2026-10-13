"""ModelAdapter ABC and shared dataclasses.

Every model in ArborVision conforms to the :class:`ModelAdapter` contract.
The runner, API, and Streamlit UI all interact with models *only* through
this interface — there is no need to special-case Mask R-CNN vs. ResNet etc.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


TaskType = Literal["classification", "instance_segmentation", "object_detection"]


@dataclass
class Prediction:
    """A single model's output on a single image.

    For pure classification, leave ``boxes`` and ``masks`` empty and put a
    single label + score in ``labels`` / ``scores``.

    For instance segmentation, populate ``boxes`` and ``masks`` in lockstep:
    ``masks[i]`` is the binary mask for the instance whose box is
    ``boxes[i]`` and label is ``labels[i]``.
    """

    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    latency_ms: float = 0.0
    # Optional extras: top-K class probabilities for classification
    topk_labels: list[str] = field(default_factory=list)
    topk_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (masks are base64'd)."""
        import base64

        masks_b64: list[str] = []
        for m in self.masks:
            if m is None:
                masks_b64.append("")
                continue
            raw = np.asarray(m, dtype=np.uint8).tobytes()
            masks_b64.append(base64.b64encode(raw).decode("ascii"))
        mask_shapes: list[list[int]] = [list(np.asarray(m).shape) for m in self.masks if m is not None]
        return {
            "labels": list(self.labels),
            "scores": [float(s) for s in self.scores],
            "boxes": [list(b) for b in self.boxes],
            "masks_b64": masks_b64,
            "mask_shapes": mask_shapes,
            "latency_ms": float(self.latency_ms),
            "topk_labels": list(self.topk_labels),
            "topk_scores": [float(s) for s in self.topk_scores],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        import base64

        masks: list[np.ndarray] = []
        b64_list = d.get("masks_b64", []) or []
        shapes = d.get("mask_shapes", []) or []
        for b64, shape in zip(b64_list, shapes):
            if not b64:
                masks.append(None)
                continue
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            masks.append(arr)
        return cls(
            labels=list(d.get("labels", [])),
            scores=[float(s) for s in d.get("scores", [])],
            boxes=[tuple(b) for b in d.get("boxes", [])],
            masks=masks,
            latency_ms=float(d.get("latency_ms", 0.0)),
            topk_labels=list(d.get("topk_labels", [])),
            topk_scores=[float(s) for s in d.get("topk_scores", [])],
        )


@dataclass
class ModelMetadata:
    """Human-readable info about a model, surfaced in the UI / API."""

    name: str
    task: TaskType
    description: str
    device: str
    weights: str  # e.g. "COCO v1.0", "ImageNet"
    extra: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """Abstract base class that every model adapter inherits from."""

    name: str
    task: TaskType

    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference on a single RGB uint8 image (H, W, 3)."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass to remove cold-start cost.

        Subclasses should pick a representative input size; this is called
        once before timing benchmarks begin.
        """

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return descriptive metadata for this adapter."""

    def timed_predict(self, image: np.ndarray) -> Prediction:
        """Wrap :meth:`predict` with wall-clock timing.

        The returned Prediction's ``latency_ms`` is set to the measured
        runtime. We use ``time.perf_counter`` for monotonicity.
        """
        t0 = time.perf_counter()
        out = self.predict(image)
        out.latency_ms = (time.perf_counter() - t0) * 1000.0
        return out