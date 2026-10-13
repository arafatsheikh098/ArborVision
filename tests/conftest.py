"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rgb_image_2x2() -> np.ndarray:
    """A 2x2 RGB image (used for metric smoke tests)."""
    return np.zeros((2, 2, 3), dtype=np.uint8)


@pytest.fixture
def rgb_image_32x32() -> np.ndarray:
    """A 32x32 RGB image with a known pattern (white square in the middle)."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24, :] = 255
    return img


@pytest.fixture
def binary_mask_square() -> np.ndarray:
    """A 32x32 mask with a 16x16 square in the middle."""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 1
    return m


@pytest.fixture
def binary_mask_circle() -> np.ndarray:
    """A 32x32 mask with a different shape (smaller, top-left)."""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[4:18, 4:18] = 1
    return m