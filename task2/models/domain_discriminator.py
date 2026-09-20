"""Domain discriminator and gradient reversal used by DANN and CDAN."""

from __future__ import annotations

import torch
import torch.nn as nn
from math import exp


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return -ctx.strength * gradients, None


def gradient_reverse(inputs: torch.Tensor, strength: float) -> torch.Tensor:
    """Leave activations unchanged and reverse their backward gradient."""
    return _GradientReverse.apply(inputs, strength)


def grl_strength(progress: float, maximum: float = 1.0) -> float:
    """Return the assignment's gradual gradient-reversal schedule."""
    progress = min(max(progress, 0.0), 1.0)
    return maximum * (2.0 / (1.0 + exp(-10.0 * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    """Two-layer binary domain classifier used after gradient reversal."""

    def __init__(self, input_dimension: int, hidden_dimension: int = 256, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
