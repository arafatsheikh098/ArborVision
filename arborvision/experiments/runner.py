"""End-to-end experiment orchestration.

Given a list of images, a list of model names, and optional ground truth,
runs every model on every image, computes per-model and aggregate metrics,
saves overlay PNGs, and writes a structured result tree on disk.

The runner can also be called from the CLI: see ``__main__`` at the bottom.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import __version__ as ARBORVISION_VERSION
from ..config import (
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_N_RUNS,
    DEFAULT_N_WARMUP,
    DEFAULT_SCORE_THRESHOLD,
    ensure_data_dirs,
    get_device,
    RESULTS_DIR,
)
from ..data.loader import iter_image_paths, load_image
from ..data.overlays import draw_prediction
from ..data.tree_dataset import TreeDataset, load_tree_dataset
from ..metrics.classification import compute_classification_metrics
from ..metrics.segmentation import compute_segmentation_metrics
from ..metrics.timing import benchmark
from ..models import get_model, list_models
from ..models.base import Prediction
from .checkpoint import ExperimentConfig, make_config
from .export import export_zip


@dataclass
class PerImageResult:
    """Predictions + per-image timing for one model on one image."""

    image_path: str
    image_label: str | None
    prediction: dict  # JSON-friendly
    latency_ms: float


@dataclass
class ModelResult:
    """All results for a single model in an experiment."""

    name: str
    metadata: dict
    per_image: list[PerImageResult]
    classification_metrics: dict | None
    segmentation_metrics: dict | None
    timing: dict  # per-image latency stats

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "per_image": [
                {
                    "image_path": r.image_path,
                    "image_label": r.image_label,
                    "prediction": r.prediction,
                    "latency_ms": r.latency_ms,
                }
                for r in self.per_image
            ],
            "classification_metrics": self.classification_metrics,
            "segmentation_metrics": self.segmentation_metrics,
            "timing": self.timing,
        }


@dataclass
class ExperimentResult:
    """Top-level container for a complete experiment run."""

    config: ExperimentConfig
    models: list[ModelResult]
    output_dir: Path

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "arborvision_version": ARBORVISION_VERSION,
            "models": [m.to_dict() for m in self.models],
        }


def _build_overlay(
    image: np.ndarray,
    pred: Prediction,
    score_threshold: float,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Draw boxes + masks on a fresh copy of `image`."""
    return draw_prediction(
        image,
        boxes=pred.boxes,
        labels=pred.labels,
        scores=pred.scores,
        masks=pred.masks,
        score_threshold=score_threshold,
        mask_alpha=mask_alpha,
    )


def _save_overlay(image: np.ndarray, path: Path) -> None:
    """Save an RGB image to disk as a PNG (parent dirs are created)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _per_image_timing(model, image: np.ndarray, n_warmup: int, n_runs: int) -> dict:
    """Run a small timing benchmark on a single image, return stats dict."""
    stats = benchmark(model, image, n_runs=n_runs, n_warmup=n_warmup)
    return stats.to_dict()


def run_experiment(
    image_paths: Iterable[Path],
    model_names: list[str],
    ground_truth: list[str] | None = None,
    output_dir: Path | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_runs: int = DEFAULT_N_RUNS,
    save_overlays: bool = True,
) -> ExperimentResult:
    """Run a multi-model benchmark over a set of images.

    Parameters
    ----------
    image_paths
        Paths of images to process.
    model_names
        Names of registered models (see :mod:`arborvision.models.registry`).
    ground_truth
        Optional list of labels (one per image) for classification metrics.
        Length must match ``image_paths``.
    output_dir
        Directory to write results to. If None, a timestamped subdirectory
        is created under ``arborvision.config.RESULTS_DIR``.
    score_threshold
        Detection score threshold applied to instance segmentation.
    n_warmup, n_runs
        Passed to the timing benchmark.

    Returns
    -------
    ExperimentResult
        The full structured result.
    """
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("No images supplied to run_experiment().")
    if ground_truth is not None and len(ground_truth) != len(image_paths):
        raise ValueError(
            f"ground_truth length {len(ground_truth)} != image_paths length {len(image_paths)}"
        )

    invalid = [m for m in model_names if m not in list_models()]
    if invalid:
        raise ValueError(f"Unknown model(s): {invalid}. Available: {list_models()}")

    ensure_data_dirs()
    if output_dir is None:
        from datetime import datetime, timezone

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = RESULTS_DIR / ts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)

    config = make_config(
        model_names=model_names,
        device=get_device(),
        image_paths=image_paths,
        ground_truth_labels=ground_truth,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )
    config.to_json(output_dir / "config.json")

    model_results: list[ModelResult] = []

    for mname in tqdm(model_names, desc="Models"):
        model = get_model(mname)
        model.warmup()
        meta = model.metadata().to_dict() if hasattr(model.metadata(), "to_dict") else {
            "name": model.name,
            "task": model.task,
            "device": getattr(model, "device", "unknown"),
        }

        per_image: list[PerImageResult] = []
        all_pred_top1: list[str] = []  # for classification metrics
        all_gt: list[str] = []  # aligns with all_pred_top1
        seg_preds: list[tuple[list[np.ndarray], list[str]]] = []
        seg_gts: list[tuple[list[np.ndarray], list[str]]] = []
        timing_per_image: list[dict] = []

        for idx, img_path in enumerate(
            tqdm(list(image_paths), desc=f"  {mname}", leave=False)
        ):
            try:
                image = load_image(img_path)
            except Exception as e:
                print(f"  [warn] failed to load {img_path}: {e}", file=sys.stderr)
                continue

            pred = model.timed_predict(image)

            # Per-image timing (a small benchmark over multiple runs)
            stats = _per_image_timing(model, image, n_warmup=n_warmup, n_runs=n_runs)
            timing_per_image.append(
                {
                    "image_path": str(img_path),
                    **{k: v for k, v in stats.items() if k != "n_runs" and k != "n_warmup"},
                }
            )

            label = ground_truth[idx] if ground_truth is not None else None

            per_image.append(
                PerImageResult(
                    image_path=str(img_path),
                    image_label=label,
                    prediction=pred.to_dict(),
                    latency_ms=pred.latency_ms,
                )
            )

            # Aggregate for classification metrics
            if model.task == "classification" and pred.topk_labels:
                # Map "gt -> pred" by top-1
                all_pred_top1.append(pred.topk_labels[0])
                all_gt.append(label or "")
            elif model.task == "classification" and pred.labels:
                all_pred_top1.append(pred.labels[0])
                all_gt.append(label or "")

            # Aggregate for segmentation metrics (use predicted masks + labels)
            if model.task == "instance_segmentation":
                seg_preds.append((list(pred.masks), list(pred.labels)))
                # We don't have GT masks from the loader — leave seg_gts empty
                # unless caller populated it.
                seg_gts.append(([], []))

            # Save overlay image
            if save_overlays:
                overlay = _build_overlay(
                    image,
                    pred,
                    score_threshold=score_threshold,
                )
                # Build a unique filename by combining the parent class dir
                # name with the image stem. This avoids collisions when
                # multiple classes contain images with the same filename
                # (e.g. img_1.jpg under both oak/ and maple/).
                parts = [p for p in img_path.parts[-2:] if p]
                stem = "__".join(parts) if len(parts) > 1 else Path(img_path).stem
                _save_overlay(overlay, output_dir / "overlays" / mname / f"{stem}.png")

        # Aggregate metrics
        cls_metrics: dict | None = None
        if model.task == "classification" and all_gt and any(all_gt):
            # Drop rows where ground truth is empty
            pairs = [(g, p) for g, p in zip(all_gt, all_pred_top1) if g]
            if pairs:
                labels_sorted = sorted({g for g, _ in pairs} | {p for _, p in pairs})
                rep = compute_classification_metrics(
                    [g for g, _ in pairs],
                    [p for _, p in pairs],
                    labels=labels_sorted,
                )
                cls_metrics = rep.to_dict()

        seg_metrics: dict | None = None
        if model.task == "instance_segmentation" and seg_preds:
            rep = compute_segmentation_metrics(seg_preds, seg_gts)
            seg_metrics = rep.to_dict()

        timing_agg = {
            "per_image": timing_per_image,
            "mean_ms": float(np.mean([t["mean_ms"] for t in timing_per_image]))
            if timing_per_image
            else 0.0,
            "p95_ms": float(np.mean([t["p95_ms"] for t in timing_per_image]))
            if timing_per_image
            else 0.0,
        }

        model_results.append(
            ModelResult(
                name=mname,
                metadata=meta,
                per_image=per_image,
                classification_metrics=cls_metrics,
                segmentation_metrics=seg_metrics,
                timing=timing_agg,
            )
        )

    result = ExperimentResult(config=config, models=model_results, output_dir=output_dir)

    # Persist full result
    (output_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))

    # Per-image CSV
    rows: list[dict] = []
    for mr in model_results:
        for pr in mr.per_image:
            top1 = pr.prediction.get("labels", [""])[0] if pr.prediction.get("labels") else ""
            top1_score = pr.prediction.get("scores", [0.0])[0] if pr.prediction.get("scores") else 0.0
            rows.append(
                {
                    "model": mr.name,
                    "image_path": pr.image_path,
                    "image_label": pr.image_label or "",
                    "top1_label": top1,
                    "top1_score": top1_score,
                    "n_detections": len(pr.prediction.get("labels", [])),
                    "latency_ms": pr.latency_ms,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)

    # Auto-export a zip of everything
    try:
        export_zip(output_dir)
    except Exception as e:
        print(f"  [warn] failed to create experiment zip: {e}", file=sys.stderr)

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m arborvision.experiments.runner ...``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="arborvision-runner",
        description="Run a multi-model computer vision benchmark.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory of images (recursive). Subdirs become class labels if --labeled is set.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to benchmark. See arborvision.models.registry.list_models().",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/results/<UTC timestamp>).",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="Treat subdirectories of --images as class labels.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Detection score threshold (default 0.5).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help="Untimed warmup calls per image (default 2).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Timed runs per image for latency stats (default 10).",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip writing overlay PNGs.",
    )

    args = parser.parse_args(argv)
    images_root = Path(args.images)
    output_dir = Path(args.output) if args.output else None

    if args.labeled:
        ds: TreeDataset = load_tree_dataset(images_root)
        image_paths = [s.path for s in ds.samples]
        ground_truth = [s.label for s in ds.samples]
    else:
        image_paths = iter_image_paths(images_root)
        ground_truth = None

    if not image_paths:
        print(f"No images found under {images_root}", file=sys.stderr)
        return 1

    result = run_experiment(
        image_paths=image_paths,
        model_names=args.models,
        ground_truth=ground_truth,
        output_dir=output_dir,
        score_threshold=args.score_threshold,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        save_overlays=not args.no_overlays,
    )
    print(f"Experiment {result.config.experiment_id} complete.")
    print(f"Results: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())