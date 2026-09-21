"""Reuse the fixed ResNet-18 definition shared with Task 2."""

from task2.models.backbone import PACSResNet18, freeze_batch_norm_statistics

__all__ = ["PACSResNet18", "freeze_batch_norm_statistics"]
