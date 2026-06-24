"""
metrics.py — Metric computation based on TP/FP/FN.
"""

from dataclasses import dataclass


@dataclass
class CompletenessMetrics:
    """Completeness (Stage 1) metrics"""
    TP: int = 0
    FP: int = 0
    FN: int = 0

    @property
    def precision(self) -> float:
        if self.TP + self.FP == 0:
            return 0.0
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        if self.TP + self.FN == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        return {
            "TP": self.TP, "FP": self.FP, "FN": self.FN,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class AccuracyMetrics:
    """Parameter Accuracy (Stage 2) metrics"""
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0  # No matched blocks -> 0% (prevents score inflation)
        return self.correct / self.total

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class OrderMetrics:
    """Execution Order (Stage 3) metrics"""
    correct_order: int = 0
    total_blocks: int = 0
    constraint_violations: int = 0

    @property
    def order_accuracy(self) -> float:
        if self.total_blocks == 0:
            return 0.0  # No matched blocks -> 0% (prevents score inflation)
        return self.correct_order / self.total_blocks

    def to_dict(self) -> dict:
        return {
            "correct_order": self.correct_order,
            "total_blocks": self.total_blocks,
            "order_accuracy": round(self.order_accuracy, 4),
            "constraint_violations": self.constraint_violations,
        }


def aggregate_completeness(file_metrics: list) -> dict:
    """Sum a list of per-file CompletenessMetrics"""
    total = CompletenessMetrics()
    for m in file_metrics:
        total.TP += m.TP
        total.FP += m.FP
        total.FN += m.FN
    return total.to_dict()


def aggregate_accuracy(file_metrics: list) -> dict:
    """Sum a list of per-file AccuracyMetrics"""
    total = AccuracyMetrics()
    for m in file_metrics:
        total.correct += m.correct
        total.total += m.total
    return total.to_dict()


def aggregate_order(file_metrics: list) -> dict:
    """Sum a list of per-file OrderMetrics"""
    total = OrderMetrics()
    for m in file_metrics:
        total.correct_order += m.correct_order
        total.total_blocks += m.total_blocks
        total.constraint_violations += m.constraint_violations
    return total.to_dict()
