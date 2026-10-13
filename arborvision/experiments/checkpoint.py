"""Experiment reproducibility config.

We capture enough state that a future re-run of the same experiment produces
the same metrics (modulo model stochasticity, which is mostly absent in
inference-only benchmarks):

- Model name, weights identifier
- Device used
- Score / mask thresholds
- Image transforms (referenced by name, e.g. "torchvision ResNet50 weights")
- Package versions (torch, torchvision, arborvision)
- Git SHA of the repo (if available)
- Wall-clock timestamp

Serialized as JSON next to the results.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """The full configuration of a benchmark run."""

    experiment_id: str
    model_names: list[str]
    device: str
    score_threshold: float
    mask_threshold: float
    n_warmup: int
    n_runs: int
    image_paths: list[str]
    ground_truth_labels: list[str] | None
    created_at: str
    package_versions: dict[str, str] = field(default_factory=dict)
    git_sha: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        # Tolerate missing fields with sensible defaults.
        return cls(
            experiment_id=d.get("experiment_id", "unknown"),
            model_names=list(d.get("model_names", [])),
            device=d.get("device", "cpu"),
            score_threshold=float(d.get("score_threshold", 0.5)),
            mask_threshold=float(d.get("mask_threshold", 0.5)),
            n_warmup=int(d.get("n_warmup", 2)),
            n_runs=int(d.get("n_runs", 10)),
            image_paths=list(d.get("image_paths", [])),
            ground_truth_labels=d.get("ground_truth_labels"),
            created_at=d.get("created_at", ""),
            package_versions=dict(d.get("package_versions", {})),
            git_sha=d.get("git_sha"),
            extra=dict(d.get("extra", {})),
        )


def _git_sha() -> str | None:
    """Best-effort current HEAD SHA, or None if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions() -> dict[str, str]:
    """Collect versions of relevant packages; missing packages are skipped."""
    versions: dict[str, str] = {}
    for pkg in ("torch", "torchvision", "numpy", "opencv-python", "scikit-learn", "streamlit"):
        try:
            mod = __import__(pkg.replace("-", "_").split("[")[0])
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    try:
        from .. import __version__ as arborvision_version

        versions["arborvision"] = arborvision_version
    except Exception:
        pass
    versions["python"] = platform.python_version()
    return versions


def make_config(
    model_names: list[str],
    device: str,
    image_paths: list[str],
    ground_truth_labels: list[str] | None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    n_warmup: int = 2,
    n_runs: int = 10,
    experiment_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig populated with current environment info."""
    if experiment_id is None:
        experiment_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ExperimentConfig(
        experiment_id=experiment_id,
        model_names=list(model_names),
        device=device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        n_warmup=n_warmup,
        n_runs=n_runs,
        image_paths=[str(p) for p in image_paths],
        ground_truth_labels=list(ground_truth_labels) if ground_truth_labels is not None else None,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        package_versions=_package_versions(),
        git_sha=_git_sha(),
        extra=extra or {},
    )