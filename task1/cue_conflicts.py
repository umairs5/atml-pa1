"""Utilities for creating reproducible AdaIN cue-conflict candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


def load_adain(adain_root: str | Path, device: torch.device):
    """Load the encoder and decoder supplied by pytorch-AdaIN."""
    adain_root = Path(adain_root).resolve()
    if not adain_root.exists():
        raise FileNotFoundError(f"AdaIN source directory not found: {adain_root}")
    sys.path.insert(0, str(adain_root))
    import net as adain_net
    from function import adaptive_instance_normalization

    model_dir = adain_root / "models"
    encoder = adain_net.vgg
    decoder = adain_net.decoder
    encoder.load_state_dict(torch.load(model_dir / "vgg_normalised.pth", map_location=device, weights_only=False))
    decoder.load_state_dict(torch.load(model_dir / "decoder.pth", map_location=device, weights_only=False))
    encoder = nn.Sequential(*list(encoder.children())[:31]).to(device).eval()
    decoder = decoder.to(device).eval()
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return encoder, decoder, adaptive_instance_normalization


def stylize(
    content: torch.Tensor,
    style: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    adaptive_instance_normalization,
    alpha: float,
) -> torch.Tensor:
    """Transfer style statistics while retaining the content activations."""
    with torch.no_grad():
        content_features = encoder(content)
        style_features = encoder(style)
        target = adaptive_instance_normalization(content_features, style_features)
        target = alpha * target + (1.0 - alpha) * content_features
        return decoder(target)


def adain_transform(size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
