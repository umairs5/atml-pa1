"""Feature extraction and linear-probe training for frozen Task 1 backbones."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.models.backbones import FrozenBackbone


def extract_features(
    backbone: FrozenBackbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, indices = [], [], []
    backbone.model.eval()
    with torch.no_grad():
        for images, batch_labels, batch_indices in loader:
            features.append(backbone.extract(images.to(device)).cpu())
            labels.append(batch_labels.cpu())
            indices.append(batch_indices.cpu())
    return torch.cat(features), torch.cat(labels), torch.cat(indices)


def train_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    feature_dimension: int,
    class_count: int,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, list[dict]]:
    torch.manual_seed(seed)
    head = nn.Linear(feature_dimension, class_count).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config["head_batch_size"],
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=config["head_batch_size"],
        shuffle=False,
    )

    best_state, best_validation_accuracy = None, float("-inf")
    epochs_without_improvement, history = 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        train_loss_sum, train_correct, train_count = 0.0, 0, 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_labels)
            train_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            train_count += len(batch_labels)

        head.eval()
        validation_loss_sum, validation_correct, validation_count = 0.0, 0, 0
        with torch.no_grad():
            for batch_features, batch_labels in validation_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                logits = head(batch_features)
                loss = criterion(logits, batch_labels)
                validation_loss_sum += loss.item() * len(batch_labels)
                validation_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                validation_count += len(batch_labels)

        validation_accuracy = validation_correct / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "train_accuracy": train_correct / train_count,
                "validation_loss": validation_loss_sum / validation_count,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break

    head.load_state_dict(best_state)
    return head, history


def load_linear_head(checkpoint_path: str, device: torch.device) -> nn.Linear:
    """Restore a selected Task 1 linear head for intervention evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head = nn.Linear(checkpoint["feature_dimension"], len(checkpoint["class_names"])).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head
