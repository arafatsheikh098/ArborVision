"""Drawing bounding boxes + segmentation masks onto images.

All drawing routines return RGB uint8 ndarrays (the same channel order as
`loader.load_image`), so they can be passed straight to matplotlib /
Streamlit display functions.
"""

from __future__ import annotations

import colorsys
import random
from typing import Iterable, Sequence

import cv2
import numpy as np


# A small fixed palette so the same class name gets the same color across
# runs (helps eyeballing the confusion matrix).
_FIXED_PALETTE: list[tuple[int, int, int]] = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 128, 255),
    (255, 196, 64),
    (255, 64, 196),
    (64, 255, 255),
    (196, 64, 255),
    (128, 255, 128),
    (255, 180, 128),
    (128, 180, 255),
    (255, 128, 220),
    (180, 255, 180),
    (220, 220, 100),
    (100, 220, 220),
    (220, 100, 220),
]


def _color_for(label: str) -> tuple[int, int, int]:
    """Deterministic color for a label string."""
    if not label:
        return _FIXED_PALETTE[0]
    h = hash(label) % len(_FIXED_PALETTE)
    return _FIXED_PALETTE[h]


def _rng_for(label: str) -> random.Random:
    """Independent RNG seeded by label, for stable randomization per class."""
    return random.Random(label)


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of `image`.

    Parameters
    ----------
    image : (H, W, 3) uint8 RGB image
    boxes : sequence of (x1, y1, x2, y2) in pixel coords
    labels : one label per box
    scores : optional confidence per box (will be drawn in label text)
    """
    out = image.copy()
    if not boxes:
        return out
    boxes = np.asarray(boxes, dtype=int)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = max(int(x2), 0), max(int(y2), 0)
        color = _color_for(labels[i] if i < len(labels) else "")
        # OpenCV expects BGR for cv2.rectangle when image is BGR, but here we
        # work in RGB throughout — so we hand the RGB color straight in.
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        text = labels[i] if i < len(labels) else ""
        if scores is not None and i < len(scores):
            text = f"{text} {scores[i]:.4f}" if text else f"{scores[i]:.4f}"
        if not text:
            continue
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty1 = max(y1 - th - baseline - 2, 0)
        cv2.rectangle(out, (x1, ty1), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 1, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_masks(
    image: np.ndarray,
    masks: Iterable[np.ndarray],
    labels: Sequence[str] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay segmentation masks on a copy of `image`.

    Each mask is HxW bool/uint8; non-zero pixels are tinted with the label's
    color at the given alpha.
    """
    out = image.astype(np.float32)
    masks = list(masks)
    if not masks:
        return out.astype(np.uint8)

    H, W = out.shape[:2]
    for i, m in enumerate(masks):
        if m is None:
            continue
        mask = np.asarray(m)
        if mask.shape != (H, W):
            # Resize mask to image shape with nearest neighbor
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0
        if not mask_bool.any():
            continue
        label = labels[i] if labels is not None and i < len(labels) else ""
        color = np.array(_color_for(label), dtype=np.float32)
        # Blend: where mask is True, mix in the color at alpha
        for c in range(3):
            out[..., c] = np.where(
                mask_bool,
                (1 - alpha) * out[..., c] + alpha * color[c],
                out[..., c],
            )
    return out.astype(np.uint8)


def draw_prediction(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    masks: Sequence[np.ndarray] | None = None,
    score_threshold: float = 0.0,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """One-shot helper: draw masks (under) then boxes (over) on `image`."""
    if masks is not None and len(masks) > 0:
        keep = [i for i, s in enumerate(scores or [1.0] * len(masks)) if s >= score_threshold]
        if keep:
            image = draw_masks(
                image,
                [masks[i] for i in keep],
                labels=[labels[i] for i in keep],
                alpha=mask_alpha,
            )
    if boxes is not None and len(boxes) > 0:
        keep = [i for i, s in enumerate(scores or [1.0] * len(boxes)) if s >= score_threshold]
        if keep:
            image = draw_boxes(
                image,
                [boxes[i] for i in keep],
                labels=[labels[i] for i in keep],
                scores=[scores[i] for i in keep] if scores else None,
            )
    return image


def random_color_palette(n: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Generate `n` visually distinct RGB colors using HSV spacing."""
    rng = random.Random(seed)
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        h = (i + rng.random() * 0.3) / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.95)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette