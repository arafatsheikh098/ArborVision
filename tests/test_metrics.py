"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from arborvision.metrics.classification import (
    compute_classification_metrics,
    topk_accuracy,
)
from arborvision.metrics.segmentation import (
    SegmentationReport,
    _mask_iou,
    compute_segmentation_metrics,
)


# ---- classification ---------------------------------------------------------


def test_compute_classification_metrics_perfect():
    rep = compute_classification_metrics(
        y_true=["a", "a", "b", "b"],
        y_pred=["a", "a", "b", "b"],
        labels=["a", "b"],
    )
    assert rep.accuracy == 1.0
    assert rep.precision == [1.0, 1.0]
    assert rep.recall == [1.0, 1.0]
    assert rep.f1 == [1.0, 1.0]
    assert rep.macro_f1 == 1.0
    assert rep.support == [2, 2]
    assert rep.confusion.tolist() == [[2, 0], [0, 2]]


def test_compute_classification_metrics_imperfect():
    rep = compute_classification_metrics(
        y_true=["a", "a", "b", "b"],
        y_pred=["a", "b", "b", "a"],
        labels=["a", "b"],
    )
    assert rep.accuracy == 0.5
    assert rep.precision == [0.5, 0.5]
    assert rep.recall == [0.5, 0.5]
    assert rep.macro_f1 == 0.5


def test_compute_classification_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_classification_metrics(["a", "b"], ["a"])


def test_compute_classification_metrics_empty():
    rep = compute_classification_metrics([], [], labels=["a", "b"])
    assert rep.accuracy == 0.0
    assert rep.macro_f1 == 0.0
    assert rep.weighted_f1 == 0.0


def test_topk_accuracy():
    preds = [["a", "b", "c"], ["b", "a", "c"], ["z", "y", "x"]]
    gts = ["a", "c", "z"]
    assert topk_accuracy(preds, gts, k=1) == pytest.approx(2 / 3)
    assert topk_accuracy(preds, gts, k=3) == 1.0


# ---- segmentation -----------------------------------------------------------


def test_mask_iou_identical():
    m = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    assert _mask_iou(m, m) == 1.0


def test_mask_iou_disjoint():
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert _mask_iou(a, b) == 0.0


def test_mask_iou_partial():
    a = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    # intersection=2, union=4 -> 0.5
    assert _mask_iou(a, b) == 0.5


def test_mask_iou_shape_mismatch_returns_zero():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((2, 2), dtype=np.uint8)
    assert _mask_iou(a, b) == 0.0


def test_segmentation_metrics_perfect_match(binary_mask_square):
    rep = compute_segmentation_metrics(
        images_preds=[([binary_mask_square], ["tree"])],
        images_gts=[([binary_mask_square.copy()], ["tree"])],
    )
    assert rep.precision == [1.0]
    assert rep.recall == [1.0]
    assert rep.f1 == [1.0]
    assert rep.map50 == 1.0
    assert rep.support == [1]


def test_segmentation_metrics_no_gt_returns_coverage_only(binary_mask_square):
    rep = compute_segmentation_metrics(
        images_preds=[([binary_mask_square], ["tree"])],
        images_gts=None,
    )
    assert rep.precision == []
    assert rep.recall == []
    assert rep.map50 == 0.0
    assert rep.mean_mask_coverage > 0.0


def test_segmentation_metrics_class_mismatch_no_match(binary_mask_square):
    rep = compute_segmentation_metrics(
        images_preds=[([binary_mask_square], ["oak"])],
        images_gts=[([binary_mask_square.copy()], ["maple"])],
    )
    # Different labels -> no matching; both labels are tracked, precision
    # and recall are 0 for each.
    assert sorted(rep.labels) == ["maple", "oak"]
    for p, r, f in zip(rep.precision, rep.recall, rep.f1):
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0


def test_segmentation_metrics_partial_overlap(binary_mask_square, binary_mask_circle):
    # Two GT instances, one prediction matches first GT, second prediction
    # is a different class (no match).
    rep = compute_segmentation_metrics(
        images_preds=[
            (
                [binary_mask_square, binary_mask_circle],
                ["tree", "rock"],
            )
        ],
        images_gts=[
            (
                [binary_mask_square.copy(), binary_mask_circle.copy()],
                ["tree", "tree"],
            )
        ],
    )
    # Only "tree" class has matching pairs
    assert "tree" in rep.labels
    # For "tree": 1 TP (square match), 0 FP, 1 FN (circle not matched) -> p=1, r=0.5
    tree_idx = rep.labels.index("tree")
    assert rep.precision[tree_idx] == 1.0
    assert rep.recall[tree_idx] == 0.5
    assert rep.f1[tree_idx] == pytest.approx(2 * 1 * 0.5 / (1 + 0.5))


def test_segmentation_metrics_return_type():
    rep = compute_segmentation_metrics([], None)
    assert isinstance(rep, SegmentationReport)
    d = rep.to_dict()
    assert "map50" in d
    assert "mean_mask_coverage" in d