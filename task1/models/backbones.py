"""Frozen pretrained backbones used in Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)


@dataclass
class FrozenBackbone:
    name: str
    model: nn.Module
    feature_dimension: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    is_clip: bool = False

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_clip:
            return F.normalize(self.model.encode_image(images), dim=-1)
        return self.model(images)

    def image_transform(
        self,
        size: int,
        intervention: Callable | None = None,
    ) -> transforms.Compose:
        steps: list[Callable] = [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
        if intervention is not None:
            steps.append(intervention)
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.normalization_mean,
                    std=self.normalization_std,
                ),
            ]
        )
        return transforms.Compose(steps)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_backbone(name: str, device: torch.device) -> FrozenBackbone:
    """Load a required pretrained backbone with its final representation exposed."""
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        return FrozenBackbone(
            name=name,
            model=_freeze(model).to(device),
            feature_dimension=2048,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
        )

    if name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        return FrozenBackbone(
            name=name,
            model=_freeze(model).to(device),
            feature_dimension=768,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
        )

    if name == "clip_vit_b_32":
        import open_clip

        # OpenAI's ViT-B/32 checkpoint was trained with QuickGELU.
        model = open_clip.create_model("ViT-B-32-quickgelu", pretrained="openai")
        return FrozenBackbone(
            name=name,
            model=_freeze(model).to(device),
            feature_dimension=model.visual.output_dim,
            normalization_mean=tuple(open_clip.OPENAI_DATASET_MEAN),
            normalization_std=tuple(open_clip.OPENAI_DATASET_STD),
            is_clip=True,
        )

    raise ValueError(f"Unknown backbone: {name}")


def clip_zero_shot_logits(
    backbone: FrozenBackbone,
    image_features: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Evaluate CLIP using the assignment's fixed prompt template."""
    if not backbone.is_clip:
        raise ValueError("Zero-shot evaluation is only defined for the CLIP backbone.")

    import open_clip

    prompts = [f"a photo of a {class_name}." for class_name in class_names]
    tokens = open_clip.get_tokenizer("ViT-B-32")(prompts).to(device)
    with torch.no_grad():
        text_features = F.normalize(backbone.model.encode_text(tokens), dim=-1)
        return backbone.model.logit_scale.exp() * image_features.to(device) @ text_features.T
