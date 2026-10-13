"""Image loading utilities.

Images are decoded with OpenCV (BGR) and converted to RGB float32 uint8 arrays
for downstream model consumption. EXIF orientation is honored so portrait
phone photos aren't rotated wrong.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def is_image_path(path: Path) -> bool:
    """Return True if `path` looks like a supported image file."""
    return path.suffix.lower() in SUPPORTED_EXTS


def iter_image_paths(root: Path) -> list[Path]:
    """Recursively collect image paths under `root`, skipping hidden files."""
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and is_image_path(p):
            out.append(p)
    return out


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array.

    Handles EXIF orientation by loading twice (Pillow for orientation, then
    OpenCV for the actual decode). Falls back to OpenCV-only decode if Pillow
    fails.

    Parameters
    ----------
    path : str | Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image with shape (H, W, 3), dtype uint8, RGB channel order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Step 1: apply EXIF orientation via Pillow (no actual decode work here)
    try:
        with Image.open(path) as im_pil:
            im_pil = ImageOps.exif_transpose(im_pil)
            im_pil = im_pil.convert("RGB")
            arr = np.asarray(im_pil)
            return arr  # RGB uint8
    except Exception:
        # Fall back to OpenCV decode
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb


def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode an in-memory image upload (e.g. from Streamlit or FastAPI).

    Returns an RGB uint8 ndarray. Raises ValueError on decode failure.
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded bytes as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)