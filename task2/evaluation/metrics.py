"""Metrics used for source validation and final target evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute the assignment's accuracy and macro F1 metrics."""
    return {
        "accuracy": float((labels == predictions).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
