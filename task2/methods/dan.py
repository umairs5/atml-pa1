"""Deep Adaptation Network objective using multi-kernel MMD."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _squared_distances(features: torch.Tensor) -> torch.Tensor:
    squared_norms = (features * features).sum(dim=1, keepdim=True)
    return (squared_norms + squared_norms.T - 2.0 * features @ features.T).clamp_min(0.0)


def multi_kernel_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    width_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Compute squared MMD using RBF widths tied to the combined-batch median."""
    combined = torch.cat([source_features, target_features], dim=0)
    distances = _squared_distances(combined)
    nonzero_distances = distances[distances > 0]
    median_distance = nonzero_distances.median().clamp_min(torch.finfo(distances.dtype).eps)
    kernel = sum(
        torch.exp(-distances / (multiplier * median_distance))
        for multiplier in width_multipliers
    )

    source_count = source_features.shape[0]
    source_kernel = kernel[:source_count, :source_count]
    target_kernel = kernel[source_count:, source_count:]
    cross_kernel = kernel[:source_count, source_count:]
    return source_kernel.mean() + target_kernel.mean() - 2.0 * cross_kernel.mean()


def dan_update(model, optimizer, source_images, source_labels, target_images, progress, mmd_lambda: float):
    """Take one classification-plus-feature-alignment DAN update."""
    del progress
    optimizer.zero_grad()
    source_features, source_logits = model(source_images)
    target_features, _ = model(target_images)
    classification = functional.cross_entropy(source_logits, source_labels)
    alignment = multi_kernel_mmd(source_features, target_features)
    total = classification + mmd_lambda * alignment
    total.backward()
    optimizer.step()
    return {
        "classification_loss": classification.detach().item(),
        "alignment_loss": alignment.detach().item(),
        "total_loss": total.detach().item(),
    }
