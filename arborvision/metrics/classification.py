"""Classification metrics: confusion matrix, precision, recall, F1.

Built on top of scikit-learn's battle-tested implementations so we don't
reinvent the wheel (and so behavior matches common ML tooling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationReport:
    """Aggregated classification metrics across a run."""

    labels: list[str]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    accuracy: float
    confusion: np.ndarray  # shape (n_labels, n_labels), rows=true, cols=pred
    macro_f1: float
    weighted_f1: float

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "precision": [float(x) for x in self.precision],
            "recall": [float(x) for x in self.recall],
            "f1": [float(x) for x in self.f1],
            "support": [int(x) for x in self.support],
            "accuracy": float(self.accuracy),
            "confusion": self.confusion.tolist(),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
        }


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> ClassificationReport:
    """Compute the standard classification metric bundle.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth class labels.
    y_pred : list[str]
        Predicted class labels.
    labels : list[str] | None
        Optional explicit ordering of class labels. If omitted, the union of
        true and predicted labels (sorted) is used.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length (got {len(y_true)} vs {len(y_pred)})"
        )
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # sklearn's confusion_matrix / precision_recall_fscore_support refuse
    # empty input arrays. Handle that edge case explicitly so callers can
    # call us with no samples (e.g. an experiment with zero predictions).
    if len(y_true) == 0:
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        return ClassificationReport(
            labels=list(labels),
            precision=[0.0 for _ in labels],
            recall=[0.0 for _ in labels],
            f1=[0.0 for _ in labels],
            support=[0 for _ in labels],
            accuracy=0.0,
            confusion=cm,
            macro_f1=0.0,
            weighted_f1=0.0,
        )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    cm = _sk_confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    accuracy = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )
    weighted_f1 = float(weighted[2])

    return ClassificationReport(
        labels=list(labels),
        precision=[float(x) for x in precision],
        recall=[float(x) for x in recall],
        f1=[float(x) for x in f1],
        support=[int(x) for x in support],
        accuracy=accuracy,
        confusion=cm,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
    )


def topk_accuracy(
    topk_labels: list[list[str]],
    y_true: list[str],
    k: int = 1,
) -> float:
    """Fraction of samples whose true label appears in their top-K predictions."""
    if len(topk_labels) != len(y_true):
        raise ValueError("topk_labels and y_true must align")
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for preds, gt in zip(topk_labels, y_true) if gt in preds[:k])
    return hits / max(len(y_true), 1)