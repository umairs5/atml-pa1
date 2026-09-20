"""Domain-Adversarial Neural Network objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from task2.models.domain_discriminator import gradient_reverse, grl_strength


def dann_update(
    model,
    optimizer,
    source_images,
    source_labels,
    target_images,
    progress,
    discriminator,
    maximum_grl_strength: float,
    gradient_clip_norm: float,
):
    """Take one DANN update with a scheduled gradient reversal layer."""
    optimizer.zero_grad()
    source_features, source_logits = model(source_images)
    target_features, _ = model(target_images)
    classification = functional.cross_entropy(source_logits, source_labels)

    features = functional.normalize(torch.cat([source_features, target_features], dim=0), dim=1)
    domain_labels = torch.cat(
        [
            torch.zeros(source_features.shape[0], dtype=torch.long, device=features.device),
            torch.ones(target_features.shape[0], dtype=torch.long, device=features.device),
        ]
    )
    strength = grl_strength(progress, maximum_grl_strength)
    domain_logits = discriminator(gradient_reverse(features, strength))
    domain = functional.cross_entropy(domain_logits, domain_labels)
    total = classification + domain
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(discriminator.parameters()), max_norm=gradient_clip_norm
    )
    optimizer.step()
    return {
        "classification_loss": classification.detach().item(),
        "domain_loss": domain.detach().item(),
        "grl_strength": strength,
        "total_loss": total.detach().item(),
    }
