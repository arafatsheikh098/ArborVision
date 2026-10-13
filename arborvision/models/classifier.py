"""ResNet-50 ImageNet classifier adapter.

Used as the classification baseline for ArborVision. Produces top-K class
probabilities per image and surfaces the top-1 as the predicted label.

For tree-species classification out of the box, the ImageNet labels are
generic. To benchmark a *fine-tuned* tree model, set
``state_dict_path`` to a checkpoint produced by your own training script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

from ..config import DEFAULT_TOP_K, get_device
from .base import ModelAdapter, ModelMetadata, Prediction


class ResNet50Adapter(ModelAdapter):
    """ResNet-50 classification adapter (ImageNet weights by default)."""

    name = "resnet50"
    task = "classification"

    def __init__(
        self,
        device: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        state_dict_path: str | Path | None = None,
    ) -> None:
        self.device = device or get_device()
        self.top_k = top_k
        self._weights_meta = ResNet50_Weights.DEFAULT

        self.model = resnet50(weights=self._weights_meta)
        if state_dict_path is not None:
            sd = torch.load(str(state_dict_path), map_location=self.device)
            self.model.load_state_dict(sd)
        self.model.eval()
        self.model.to(self.device)

        # ImageNet preprocessing (resize, center crop, normalize)
        self._preprocess = self._weights_meta.transforms()
        self._categories: list[str] = list(self._weights_meta.meta.get("categories", []))

    def warmup(self) -> None:
        """Run a dummy forward pass at 224x224 to remove cold-start."""
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        with torch.no_grad():
            _ = self._forward(dummy)

    def predict(self, image: np.ndarray) -> Prediction:
        """Return top-K ImageNet labels + scores for the input image.

        The returned Prediction's top-1 label/score is also written into
        ``labels`` / ``scores`` so the unified comparison UI works
        consistently.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image.shape}")

        with torch.no_grad():
            logits = self._forward(image)  # (1, num_classes)

        probs = torch.softmax(logits, dim=1)[0]
        k = min(self.top_k, probs.shape[0])
        top_probs, top_idx = probs.topk(k)
        top_labels = [
            self._categories[i] if i < len(self._categories) else f"class_{i}"
            for i in top_idx.tolist()
        ]
        top_scores = [float(p) for p in top_probs.tolist()]

        return Prediction(
            labels=[top_labels[0]] if top_labels else [],
            scores=[top_scores[0]] if top_scores else [],
            boxes=[],
            masks=[],
            topk_labels=top_labels,
            topk_scores=top_scores,
        )

    def _forward(self, image: np.ndarray) -> torch.Tensor:
        # Apply ImageNet preprocessing (resize/center-crop/normalize) to a PIL
        # image derived from the numpy array. The preprocess call returns a
        # (C, H, W) float tensor ready to be batched.
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        return self.model(tensor)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            task=self.task,
            description="ResNet-50 ImageNet classifier (torchvision pretrained)",
            device=self.device,
            weights=str(self._weights_meta),
            extra={
                "top_k": self.top_k,
                "num_classes": len(self._categories),
            },
        )