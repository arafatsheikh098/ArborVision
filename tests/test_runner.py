"""Tests for the experiment runner, config, and export.

These tests use the ``DummyAdapter`` from ``test_models`` and write
artifacts into a temporary directory that is cleaned up automatically.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from arborvision.experiments.checkpoint import ExperimentConfig, make_config
from arborvision.experiments.export import export_csv, export_zip
from arborvision.experiments.runner import run_experiment
from arborvision.models import register, unregister
from tests.test_models import DummyAdapter


@pytest.fixture
def tree_image_dir(tmp_path: Path) -> Path:
    """Create a small labeled tree-image dataset on disk."""
    root = tmp_path / "trees"
    (root / "oak").mkdir(parents=True)
    (root / "maple").mkdir(parents=True)
    for cls, idxs in [("oak", [1, 2]), ("maple", [1])]:
        for i in idxs:
            img = np.full((48, 48, 3), 255 if cls == "oak" else 100, dtype=np.uint8)
            cv2.imwrite(str(root / cls / f"img_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return root


@pytest.fixture
def register_dummy():
    register("dummy", DummyAdapter)
    try:
        yield
    finally:
        unregister("dummy")


def test_make_config_populates_environment():
    cfg = make_config(
        model_names=["dummy"],
        device="cpu",
        image_paths=["a.jpg", "b.jpg"],
        ground_truth_labels=["oak", "maple"],
    )
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model_names == ["dummy"]
    assert cfg.image_paths == ["a.jpg", "b.jpg"]
    assert cfg.ground_truth_labels == ["oak", "maple"]
    assert cfg.package_versions  # at least contains "python"
    assert cfg.created_at  # ISO timestamp


def test_config_roundtrip_json(tmp_path: Path):
    cfg = make_config(
        model_names=["a", "b"],
        device="cpu",
        image_paths=["x.png"],
        ground_truth_labels=None,
    )
    p = tmp_path / "cfg.json"
    cfg.to_json(p)
    loaded = ExperimentConfig.from_dict(json.loads(p.read_text()))
    assert loaded.model_names == ["a", "b"]
    assert loaded.image_paths == ["x.png"]
    assert loaded.ground_truth_labels is None


def test_run_experiment_writes_artifacts(tree_image_dir, register_dummy, tmp_path):
    image_paths = sorted(tree_image_dir.rglob("*.jpg"))
    out_dir = tmp_path / "exp"
    result = run_experiment(
        image_paths=image_paths,
        model_names=["dummy"],
        ground_truth=["oak", "oak", "maple"],
        output_dir=out_dir,
        n_warmup=1,
        n_runs=2,
        save_overlays=True,
    )
    assert (out_dir / "config.json").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "metrics.csv").exists()
    # Overlays should be under overlays/dummy/*.png
    overlay_files = list((out_dir / "overlays" / "dummy").glob("*.png"))
    assert len(overlay_files) == len(image_paths)
    # Bundle zip was auto-created
    assert (out_dir / "experiment.zip").exists()

    # The CSV has the right number of rows
    import pandas as pd

    df = pd.read_csv(out_dir / "metrics.csv")
    assert len(df) == len(image_paths)
    assert set(df["model"]) == {"dummy"}

    # Classification metrics should be populated because we passed labels.
    # The dummy adapter always predicts "dummy_label", so accuracy vs the
    # real ground truth ("oak", "maple") is 0 — but the labels and confusion
    # matrix should still be populated.
    m = result.models[0]
    assert m.classification_metrics is not None
    assert m.classification_metrics["accuracy"] == 0.0
    assert "oak" in m.classification_metrics["labels"]
    assert "maple" in m.classification_metrics["labels"]
    assert "dummy_label" in m.classification_metrics["labels"]


def test_run_experiment_rejects_unknown_model(tree_image_dir, tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        run_experiment(
            image_paths=[next(tree_image_dir.rglob("*.jpg"))],
            model_names=["not-a-real-model"],
        )


def test_run_experiment_rejects_empty_images(tmp_path):
    with pytest.raises(ValueError, match="No images"):
        run_experiment(image_paths=[], model_names=["dummy"])


def test_export_zip_includes_all_files(tree_image_dir, register_dummy, tmp_path):
    out_dir = tmp_path / "exp_zip"
    run_experiment(
        image_paths=[next(tree_image_dir.rglob("*.jpg"))],
        model_names=["dummy"],
        output_dir=out_dir,
        n_warmup=0,
        n_runs=1,
        save_overlays=True,
    )
    zip_path = out_dir / "experiment.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any("config.json" in n for n in names)
    assert any("results.json" in n for n in names)
    assert any("metrics.csv" in n for n in names)
    assert any("overlays/" in n for n in names)


def test_export_csv_writes_rows(register_dummy, tmp_path):
    result_dict = {
        "models": [
            {
                "name": "dummy",
                "per_image": [
                    {
                        "image_path": "a.jpg",
                        "image_label": "oak",
                        "latency_ms": 12.0,
                        "prediction": {
                            "labels": ["tree"],
                            "scores": [0.9],
                            "boxes": [],
                        },
                    }
                ],
            }
        ]
    }
    p = tmp_path / "out.csv"
    export_csv(result_dict, p)
    text = p.read_text()
    assert "dummy,a.jpg,oak" in text
    assert "tree" in text