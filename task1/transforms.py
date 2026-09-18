"""Deterministic common-image interventions for Task 1."""

from __future__ import annotations

import numpy as np
from PIL import ImageOps
from torchvision.transforms import functional as transform_functional


def grayscale(image):
    """Remove chromatic information while retaining three RGB channels."""
    return ImageOps.grayscale(image).convert("RGB")


def fixed_hue_rotation(image, hue_factor: float = 0.25):
    """Rotate every image hue by a fixed fraction of the color wheel."""
    return transform_functional.adjust_hue(image, hue_factor)


def translate_reflect(image, dx: int, dy: int):
    """Shift a common-size image using reflection padding and a shifted crop."""
    if dx == 0 and dy == 0:
        return image.copy()
    pixels = np.asarray(image)
    padding = max(abs(dx), abs(dy))
    padded = np.pad(
        pixels,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="reflect",
    )
    top = padding - dy
    left = padding - dx
    height, width = pixels.shape[:2]
    return transform_functional.to_pil_image(padded[top : top + height, left : left + width])


def patch_permutation(image_id: int, seed: int, grid_size: int = 4) -> np.ndarray:
    """Create one deterministic non-identity patch permutation for an image."""
    permutation = np.random.default_rng(seed + int(image_id)).permutation(grid_size * grid_size)
    if np.array_equal(permutation, np.arange(grid_size * grid_size)):
        permutation = np.roll(permutation, 1)
    return permutation


def patch_shuffle(image, image_id: int, seed: int, grid_size: int = 4):
    """Shuffle a square common-size image on a pixel-space grid."""
    pixels = np.asarray(image)
    height, width, channels = pixels.shape
    if height % grid_size or width % grid_size:
        raise ValueError("Image dimensions must divide evenly into the patch grid.")
    patch_height, patch_width = height // grid_size, width // grid_size
    patches = pixels.reshape(grid_size, patch_height, grid_size, patch_width, channels)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(grid_size * grid_size, patch_height, patch_width, channels)
    permutation = patch_permutation(image_id, seed, grid_size)
    shuffled = patches[permutation].reshape(grid_size, grid_size, patch_height, patch_width, channels)
    shuffled = shuffled.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)
    return transform_functional.to_pil_image(shuffled)
