"""Dependency-light classification metrics for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    sample_count: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def classification_metrics(
    truth: list[int],
    predictions: list[int],
    labels: list[str],
) -> ClassificationMetrics:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have equal length")
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for expected, predicted in zip(truth, predictions, strict=True):
        if 0 <= expected < size and 0 <= predicted < size:
            matrix[expected][predicted] += 1
    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    correct = sum(matrix[index][index] for index in range(size))
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(size) if row != index)
        fn = sum(matrix[index][column] for column in range(size) if column != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": float(support),
        }
    count = len(truth)
    macro_precision = sum(precisions) / max(1, size)
    macro_recall = sum(recalls) / max(1, size)
    macro_f1 = sum(f1_values) / max(1, size)
    return ClassificationMetrics(
        sample_count=count,
        accuracy=round(correct / count if count else 0.0, 6),
        balanced_accuracy=round(macro_recall, 6),
        macro_precision=round(macro_precision, 6),
        macro_recall=round(macro_recall, 6),
        macro_f1=round(macro_f1, 6),
        per_class=per_class,
        confusion_matrix=matrix,
    )
