"""Central configuration for ArborVision.

Defines filesystem locations, device selection, and global defaults.
Edit the constants here to retarget storage or change the default device.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root is the directory that contains this file's parent's parent.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Data directories (all relative to project root)
DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
RESULTS_DIR: Path = DATA_DIR / "results"
TREE_SPECIES_DIR: Path = DATA_DIR / "tree_species"

# Inference defaults
DEFAULT_SCORE_THRESHOLD: float = 0.5
DEFAULT_MASK_THRESHOLD: float = 0.5
DEFAULT_TOP_K: int = 5

# Benchmark defaults
DEFAULT_N_WARMUP: int = 2
DEFAULT_N_RUNS: int = 10


def get_device() -> str:
    """Return the best available torch device.

    Order of preference: cuda (NVIDIA) > mps (Apple Silicon) > cpu.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def ensure_data_dirs() -> None:
    """Create the standard data directories if they don't exist."""
    for d in (UPLOADS_DIR, RESULTS_DIR, TREE_SPECIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str) -> str:
    """Read an env var with a default."""
    return os.environ.get(name, default)