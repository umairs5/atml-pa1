"""Source-pairwise MMD alignment for domain generalization."""

from itertools import combinations

import torch
import torch.nn.functional as functional

from task2.methods.dan import multi_kernel_mmd


def dan_dg_update(model, optimizer, source_batches: dict, device: torch.device, mmd_lambda: float):
    """Align each pair of observed source domains while training on all labels."""
    optimizer.zero_grad()
    features_by_domain = {}
    logits, labels = [], []
    for domain, (images, batch_labels) in source_batches.items():
        features, domain_logits = model(images.to(device, non_blocking=True))
        features_by_domain[domain] = features
        logits.append(domain_logits)
        labels.append(batch_labels.to(device, non_blocking=True))

    classification = functional.cross_entropy(torch.cat(logits), torch.cat(labels))
    pair_losses = [
        multi_kernel_mmd(features_by_domain[first], features_by_domain[second])
        for first, second in combinations(features_by_domain, 2)
    ]
    alignment = torch.stack(pair_losses).mean()
    total = classification + mmd_lambda * alignment
    total.backward()
    optimizer.step()
    return {
        "classification_loss": classification.detach().item(),
        "alignment_loss": alignment.detach().item(),
        "total_loss": total.detach().item(),
    }
