"""Instance segmentation metrics.

We compute:

- Per-class precision, recall, F1 (matching a prediction to the GT instance
  with the highest IoU above ``iou_thr``).
- mAP@0.5 across all classes.
- Mean mask coverage (fraction of image pixels covered by predicted masks).
- Average best-IoU per image (a quality-of-fit proxy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationReport:
    """Per-class and aggregate segmentation metrics."""

    labels: list[str]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]  # number of GT instances per class
    map50: float
    mean_best_iou: float
    mean_mask_coverage: float

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "precision": [float(x) for x in self.precision],
            "recall": [float(x) for x in self.recall],
            "f1": [float(x) for x in self.f1],
            "support": [int(x) for x in self.support],
            "map50": float(self.map50),
            "mean_best_iou": float(self.mean_best_iou),
            "mean_mask_coverage": float(self.mean_mask_coverage),
        }


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two binary masks of identical shape. Returns 0 on mismatch."""
    if a.shape != b.shape:
        return 0.0
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _greedy_match(
    pred_masks: list[np.ndarray],
    pred_labels: list[str],
    gt_masks: list[np.ndarray],
    gt_labels: list[str],
    iou_thr: float,
) -> tuple[set[int], set[int]]:
    """Greedy IoU-based matching.

    Returns matched prediction and GT index sets. The match is restricted to
    same-class pairs. For each GT, the highest-IoU same-class prediction
    above the threshold is matched first; matched predictions cannot be
    reused.
    """
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    # Build a list of (iou, pred_idx, gt_idx) sorted by IoU descending
    candidates: list[tuple[float, int, int]] = []
    for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
        for pi, (pm, pl) in enumerate(zip(pred_masks, pred_labels)):
            if pl != gl:
                continue
            iou = _mask_iou(pm, gm)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    candidates.sort(key=lambda t: -t[0])

    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    return matched_pred, matched_gt


def compute_segmentation_metrics(
    images_preds: list[tuple[list[np.ndarray], list[str]]],
    images_gts: list[tuple[list[np.ndarray], list[str]]] | None = None,
    iou_thr: float = 0.5,
) -> SegmentationReport:
    """Aggregate instance segmentation metrics.

    Parameters
    ----------
    images_preds
        Per-image list of (predicted_masks, predicted_labels).
    images_gts
        Per-image list of (gt_masks, gt_labels). If None, only mask coverage
        and best-IoU are reported (no precision/recall).
    iou_thr
        IoU threshold for a prediction to count as a true positive.

    Returns
    -------
    SegmentationReport
    """
    has_gt = images_gts is not None
    if has_gt and len(images_preds) != len(images_gts):
        raise ValueError("images_preds and images_gts must align")

    all_labels: set[str] = set()
    coverage_values: list[float] = []
    best_iou_values: list[float] = []

    per_class_tp: dict[str, int] = {}
    per_class_fp: dict[str, int] = {}
    per_class_fn: dict[str, int] = {}

    for idx, (pred_masks, pred_labels) in enumerate(images_preds):
        if pred_labels and has_gt:
            # Only track labels for per-class metrics when GT is available.
            all_labels.update(pred_labels)

        # Mask coverage: fraction of non-zero pixels across all pred masks.
        if pred_masks:
            stacked = np.zeros_like(pred_masks[0], dtype=np.uint8) if pred_masks else None
            for m in pred_masks:
                if stacked is None:
                    stacked = (m > 0).astype(np.uint8)
                else:
                    stacked = np.logical_or(stacked, m > 0).astype(np.uint8)
            if stacked is not None and stacked.size:
                coverage_values.append(float(stacked.mean()))

        if not has_gt:
            continue

        gt_masks, gt_labels = images_gts[idx]
        if gt_labels:
            all_labels.update(gt_labels)

        # Per-class matching
        if not gt_masks and not pred_masks:
            continue
        matched_pred, matched_gt = _greedy_match(
            pred_masks, pred_labels, gt_masks, gt_labels, iou_thr
        )

        # Track TP / FP / FN per class label
        for pi, pl in enumerate(pred_labels):
            if pi in matched_pred:
                per_class_tp[pl] = per_class_tp.get(pl, 0) + 1
            else:
                per_class_fp[pl] = per_class_fp.get(pl, 0) + 1
        for gi, gl in enumerate(gt_labels):
            if gi not in matched_gt:
                per_class_fn[gl] = per_class_fn.get(gl, 0) + 1

        # Best-IoU per image (averaged across GT instances)
        if gt_masks and pred_masks:
            ious: list[float] = []
            for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
                best = 0.0
                for pi, (pm, pl) in enumerate(zip(pred_masks, pred_labels)):
                    if pl != gl:
                        continue
                    best = max(best, _mask_iou(pm, gm))
                ious.append(best)
            if ious:
                best_iou_values.append(float(np.mean(ious)))

    # Build per-class report
    labels_sorted = sorted(all_labels)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    support: list[int] = []
    ap_values: list[float] = []
    for lbl in labels_sorted:
        tp = per_class_tp.get(lbl, 0)
        fp = per_class_fp.get(lbl, 0)
        fn = per_class_fn.get(lbl, 0)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(tp + fn)
        ap_values.append(p)  # AP@0.5 = precision at IoU 0.5 (single-threshold proxy)

    map50 = float(np.mean(ap_values)) if ap_values else 0.0
    mean_best_iou = float(np.mean(best_iou_values)) if best_iou_values else 0.0
    mean_mask_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0

    return SegmentationReport(
        labels=labels_sorted,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        map50=map50,
        mean_best_iou=mean_best_iou,
        mean_mask_coverage=mean_mask_coverage,
    )