"""Fine-tunable ImageNet ResNet-18 backbone for PACS."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PACSResNet18(nn.Module):
    """Expose 512-D pre-classifier features and seven-class logits."""

    feature_dimension = 512

    def __init__(self, num_classes: int = 7):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(network.children())[:-1])
        self.classifier = nn.Linear(self.feature_dimension, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images).flatten(1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        return features, self.classifier(features)


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    """Keep pretrained BatchNorm running statistics fixed while training affine terms."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
