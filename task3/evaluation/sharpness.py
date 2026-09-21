"""Common local sharpness diagnostic for fixed source-validation batches."""

import numpy as np
import torch
import torch.nn.functional as functional


def _fixed_validation_batch(loaders: dict, device: torch.device, seed: int = 6304):
    """Draw 32 deterministic validation examples from each observed source domain."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for loader in loaders.values():
        selected = rng.choice(len(loader.dataset), size=32, replace=False)
        examples = [loader.dataset[int(index)] for index in selected]
        images.append(torch.stack([example[0] for example in examples]))
        labels.append(torch.tensor([example[1] for example in examples]))
    return torch.cat(images).to(device), torch.cat(labels).to(device)


def sharpness_proxy(model, loaders: dict, device: torch.device, rho: float = 0.05) -> float:
    """Measure the specified loss increase after one normalized ascent perturbation."""
    images, labels = _fixed_validation_batch(loaders, device)
    model.eval()
    model.zero_grad(set_to_none=True)
    _, logits = model(images)
    base_loss = functional.cross_entropy(logits, labels)
    base_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    norm = torch.norm(torch.stack([gradient.norm(p=2) for gradient in gradients]), p=2)
    scale = rho / (norm + torch.finfo(norm.dtype).eps)
    perturbations = []
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                perturbation = parameter.grad * scale
                parameter.add_(perturbation)
                perturbations.append((parameter, perturbation))
    with torch.no_grad():
        _, perturbed_logits = model(images)
        perturbed_loss = functional.cross_entropy(perturbed_logits, labels)
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.sub_(perturbation)
    model.zero_grad(set_to_none=True)
    return float((perturbed_loss - base_loss).item())
