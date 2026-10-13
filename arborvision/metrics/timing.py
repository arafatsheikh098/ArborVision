"""Wall-clock inference timing with warmup.

We report mean / median / p95 latency over ``n_runs`` timed calls, after
``n_warmup`` untimed calls that prime caches / lazy compilation. All times
are in milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_N_RUNS, DEFAULT_N_WARMUP
from ..models.base import ModelAdapter


@dataclass
class TimingStats:
    """Latency summary for a single (model, image) pair."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    n_runs: int
    n_warmup: int

    def to_dict(self) -> dict:
        return {
            "mean_ms": float(self.mean_ms),
            "median_ms": float(self.median_ms),
            "p95_ms": float(self.p95_ms),
            "min_ms": float(self.min_ms),
            "max_ms": float(self.max_ms),
            "n_runs": int(self.n_runs),
            "n_warmup": int(self.n_warmup),
        }


def benchmark(
    model: ModelAdapter,
    image: np.ndarray,
    n_runs: int = DEFAULT_N_RUNS,
    n_warmup: int = DEFAULT_N_WARMUP,
) -> TimingStats:
    """Measure inference latency for `model` on `image`.

    `n_warmup` untimed calls run first (plus an explicit ``model.warmup()``
    if available) to remove cold-start bias.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if n_warmup < 0:
        raise ValueError("n_warmup must be >= 0")

    # Explicit warmup (full pass) + N untimed calls for caches.
    model.warmup()
    for _ in range(n_warmup):
        _ = model.predict(image)

    samples_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(image)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(samples_ms)
    return TimingStats(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )