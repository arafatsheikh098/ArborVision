"""torchvision Mask R-CNN adapter (instance segmentation).

Wraps the pretrained COCO Mask R-CNN ResNet-50 FPN model behind ArborVision's
ModelAdapter interface. Outputs boxes, labels, scores, and per-instance
binary masks (HxW uint8) for all detections above ``score_threshold``.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

from ..config import get_device
from .base import ModelAdapter, ModelMetadata, Prediction


# COCO instance category names (excluding the background at index 0).
# Source: torchvision's bundled metadata.
_COCO_INSTANCE_CATEGORY_NAMES: tuple[str, ...] = (
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class MaskRCNNAdapter(ModelAdapter):
    """Adapter for torchvision's pretrained Mask R-CNN ResNet-50 FPN."""

    name = "maskrcnn"
    task = "instance_segmentation"

    def __init__(
        self,
        device: str | None = None,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> None:
        self.device = device or get_device()
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._weights_meta = weights
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        # Read category names from the weights metadata if available
        cats = weights.meta.get("categories", None)
        if cats:
            # torchvision's category list already includes __background__ at 0
            self._category_names: list[str] = list(cats)
        else:
            self._category_names = list(_COCO_INSTANCE_CATEGORY_NAMES)

    def warmup(self) -> None:
        """Run a dummy forward on a 256x256 RGB image to remove cold-start."""
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        with torch.no_grad():
            _ = self._forward(dummy)

    def predict(self, image: np.ndarray) -> Prediction:
        """Run Mask R-CNN on a single RGB uint8 image.

        Returns a Prediction with boxes, labels, scores, and binary masks
        (thresholded at ``self.mask_threshold``) for all detections whose
        score is >= ``self.score_threshold``.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image.shape}")

        with torch.no_grad():
            output = self._forward(image)

        labels: list[str] = []
        scores: list[float] = []
        boxes: list[tuple[int, int, int, int]] = []
        masks: list[np.ndarray] = []

        for i in range(len(output["labels"])):
            score = float(output["scores"][i].item())
            if score < self.score_threshold:
                continue
            label_idx = int(output["labels"][i].item())
            label = self._category_names[label_idx] if label_idx < len(self._category_names) else str(label_idx)
            box = tuple(int(v) for v in output["boxes"][i].tolist())
            mask = (output["masks"][i, 0] >= self.mask_threshold).cpu().numpy().astype(np.uint8)
            labels.append(label)
            scores.append(score)
            boxes.append(box)  # (x1, y1, x2, y2)
            masks.append(mask)

        return Prediction(labels=labels, scores=scores, boxes=boxes, masks=masks)

    def _forward(self, image: np.ndarray) -> dict:
        """Convert numpy -> torch and run the torchvision model.

        Mask R-CNN expects a list of (C, H, W) float tensors normalized to
        [0, 1]; torchvision handles the rest internally.
        """
        # H, W, C uint8 RGB  ->  C, H, W float32 [0, 1]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        return self.model([tensor])[0]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            task=self.task,
            description="torchvision Mask R-CNN ResNet-50 FPN (COCO pretrained)",
            device=self.device,
            weights=str(self._weights_meta),
            extra={
                "score_threshold": self.score_threshold,
                "mask_threshold": self.mask_threshold,
                "categories": self._category_names,
            },
        )