"""Export experiment artifacts to JSON / CSV / ZIP."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


def export_csv(result: dict, path: Path) -> Path:
    """Write per-image predictions to a flat CSV for spreadsheet analysis."""
    rows: list[dict] = []
    for m in result.get("models", []):
        for pr in m.get("per_image", []):
            labels = pr.get("prediction", {}).get("labels", []) or [""]
            scores = pr.get("prediction", {}).get("scores", []) or [0.0]
            rows.append(
                {
                    "model": m.get("name"),
                    "image_path": pr.get("image_path"),
                    "image_label": pr.get("image_label") or "",
                    "top1_label": labels[0],
                    "top1_score": float(scores[0]) if scores else 0.0,
                    "n_detections": len(pr.get("prediction", {}).get("labels", [])),
                    "latency_ms": float(pr.get("latency_ms", 0.0)),
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_zip(experiment_dir: Path, zip_path: Path | None = None) -> Path:
    """Bundle an entire experiment directory into a single ZIP.

    Includes the JSON config, JSON results, CSV metrics, and all overlay
    PNGs. Other files are added if present.
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {experiment_dir}")

    if zip_path is None:
        zip_path = experiment_dir / "experiment.zip"
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in experiment_dir.rglob("*"):
            if p.is_file() and p != zip_path and not p.name.endswith(".zip"):
                arcname = p.relative_to(experiment_dir)
                zf.write(p, arcname=str(arcname))
    return zip_path


def write_results_json(result: dict, path: Path) -> Path:
    """Write a structured results dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return path