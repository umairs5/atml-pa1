"""Non-adaptive Sharpness-Aware Minimization updates."""

import torch
import torch.nn.functional as functional


def _source_loss(model, source_batches: dict, device: torch.device) -> torch.Tensor:
    logits, labels = [], []
    for images, batch_labels in source_batches.values():
        _, domain_logits = model(images.to(device, non_blocking=True))
        logits.append(domain_logits)
        labels.append(batch_labels.to(device, non_blocking=True))
    return functional.cross_entropy(torch.cat(logits), torch.cat(labels))


def sam_update(model, optimizer, source_batches: dict, device: torch.device, rho: float):
    """Use the prescribed two-pass SAM update over a balanced source batch."""
    optimizer.zero_grad()
    first_loss = _source_loss(model, source_batches, device)
    first_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = torch.norm(torch.stack([gradient.norm(p=2) for gradient in gradients]), p=2)
    scale = rho / (gradient_norm + torch.finfo(gradient_norm.dtype).eps)
    perturbations = []
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                perturbation = parameter.grad * scale
                parameter.add_(perturbation)
                perturbations.append((parameter, perturbation))

    optimizer.zero_grad()
    perturbed_loss = _source_loss(model, source_batches, device)
    perturbed_loss.backward()
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.sub_(perturbation)
    optimizer.step()
    return {
        "classification_loss": first_loss.detach().item(),
        "perturbed_classification_loss": perturbed_loss.detach().item(),
    }
