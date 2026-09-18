"""Deterministic common-image interventions for Task 1."""

from __future__ import annotations

from PIL import ImageOps
from torchvision.transforms import functional as transform_functional


def grayscale(image):
    """Remove chromatic information while retaining three RGB channels."""
    return ImageOps.grayscale(image).convert("RGB")


def fixed_hue_rotation(image, hue_factor: float = 0.25):
    """Rotate every image hue by a fixed fraction of the color wheel."""
    return transform_functional.adjust_hue(image, hue_factor)
