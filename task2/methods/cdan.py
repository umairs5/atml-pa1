"""Conditional Domain-Adversarial Network objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from task2.models.domain_discriminator import gradient_reverse, grl_strength


def conditional_representation(features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return vec(feature outer predicted-class probabilities) for each image."""
    probabilities = torch.softmax(logits, dim=1)
    return torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)


def cdan_update(
    model,
    optimizer,
    source_images,
    source_labels,
    target_images,
    progress,
    discriminator,
    maximum_grl_strength: float,
):
    """Take one CDAN update without detaching features or predicted probabilities."""
    optimizer.zero_grad()
    source_features, source_logits = model(source_images)
    target_features, target_logits = model(target_images)
    classification = functional.cross_entropy(source_logits, source_labels)

    conditional_features = torch.cat(
        [
            conditional_representation(source_features, source_logits),
            conditional_representation(target_features, target_logits),
        ]
    )
    domain_labels = torch.cat(
        [
            torch.zeros(source_features.shape[0], dtype=torch.long, device=conditional_features.device),
            torch.ones(target_features.shape[0], dtype=torch.long, device=conditional_features.device),
        ]
    )
    strength = grl_strength(progress, maximum_grl_strength)
    domain_logits = discriminator(gradient_reverse(conditional_features, strength))
    domain = functional.cross_entropy(domain_logits, domain_labels)
    total = classification + domain
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(discriminator.parameters()), max_norm=1.0
    )
    optimizer.step()
    return {
        "classification_loss": classification.detach().item(),
        "domain_loss": domain.detach().item(),
        "grl_strength": strength,
        "total_loss": total.detach().item(),
    }
