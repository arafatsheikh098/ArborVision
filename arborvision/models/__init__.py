"""Model adapter interface and registry.

To plug in a new model:

1. Subclass :class:`ModelAdapter` in a new module under this package.
2. Register it in :mod:`arborvision.models.registry` (or use
   :func:`registry.register` at import time).
3. Done — runner, API, and UI auto-discover it.
"""

from __future__ import annotations

from .base import ModelAdapter, ModelMetadata, Prediction
from .registry import get_model, list_models, register, unregister

__all__ = [
    "ModelAdapter",
    "ModelMetadata",
    "Prediction",
    "get_model",
    "list_models",
    "register",
    "unregister",
]