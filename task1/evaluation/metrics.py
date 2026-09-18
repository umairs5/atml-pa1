"""Task 1 classification metrics."""

from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    return {
        "accuracy": (predictions.cpu() == labels.cpu()).float().mean().item(),
        "macro_f1": f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro"),
        "mean_maximum_confidence": probabilities.max(dim=1).values.mean().item(),
    }
