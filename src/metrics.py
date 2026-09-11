"""
metrics.py
==========
Metriques d'evaluation (Equation 11.1 du memoire) :
    Precision = TP / (TP+FP)
    Rappel    = TP / (TP+FN)
    F1        = 2 * Precision * Rappel / (Precision + Rappel)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "loss": round(self.loss, 4),
        }

    def __str__(self) -> str:
        return (f"acc={self.accuracy:.3f} prec={self.precision:.3f} "
                f"recall={self.recall:.3f} F1={self.f1:.3f} loss={self.loss:.4f}")


def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_hat = (p_pred >= threshold).astype(np.float64)

    tp = float(np.sum((y_hat == 1) & (y_true == 1)))
    tn = float(np.sum((y_hat == 0) & (y_true == 0)))
    fp = float(np.sum((y_hat == 1) & (y_true == 0)))
    fn = float(np.sum((y_hat == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1e-9) if (tp + fp) > 0 else 0.0
    recall = tp / max(tp + fn, 1e-9) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-9) if (precision + recall) > 0 else 0.0

    eps = 1e-9
    p_clip = np.clip(p_pred, eps, 1 - eps)
    loss = float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))

    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, loss=loss)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=200) > 0.8).astype(np.float64)
    p_pred = np.clip(y_true * 0.8 + rng.normal(0, 0.2, size=200), 0, 1)
    m = compute_metrics(y_true, p_pred)
    print("[metrics.py]", m)
    assert 0 <= m.accuracy <= 1 and 0 <= m.f1 <= 1
    print("[metrics.py] auto-test OK.")
