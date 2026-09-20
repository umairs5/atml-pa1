"""Shared training loop for Task 2 adaptation methods."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from common.seed import seed_everything
from shared.pacs_protocol import load_pacs_protocol
from task2.data import build_training_loaders, build_validation_loaders, cycle_loaders
from task2.evaluation.metrics import classification_metrics
from task2.models.backbone import freeze_batch_norm_statistics


@torch.no_grad()
def validate(model, loaders: dict, device: torch.device) -> dict[str, float]:
    """Validate separately on each source domain and average macro F1."""
    model.eval()
    scores = {}
    for domain, loader in loaders.items():
        labels, predictions = [], []
        for images, batch_labels in loader:
            _, logits = model(images.to(device))
            labels.extend(batch_labels.numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
        metrics = classification_metrics(np.asarray(labels), np.asarray(predictions))
        scores[f"{domain}_accuracy"] = metrics["accuracy"]
        scores[f"{domain}_macro_f1"] = metrics["macro_f1"]
    scores["mean_source_validation_macro_f1"] = float(
        np.mean([scores[f"{domain}_macro_f1"] for domain in loaders])
    )
    return scores


def concatenate_source_batches(source_batches: dict, device: torch.device):
    """Combine equal-size source-domain batches into a balanced classification batch."""
    images = torch.cat([batch[0] for batch in source_batches.values()]).to(device)
    labels = torch.cat([batch[1] for batch in source_batches.values()]).to(device)
    return images, labels


def run_training(config: dict, model, update_step, device: torch.device, extra_modules=()) -> Path:
    """Train one fixed-method experiment and select by source validation macro F1."""
    seed_everything(config["seed"])
    protocol = load_pacs_protocol(config["data"]["protocol"])
    source_loaders, target_loader = build_training_loaders(config, protocol)
    validation_loaders = build_validation_loaders(config, protocol)
    output_dir = Path(config["output"]["root"]) / config["method"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trainable_parameters = list(model.parameters())
    for module in extra_modules:
        trainable_parameters.extend(module.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    history = []
    best_score = float("-inf")
    epochs_without_improvement = 0
    max_epochs = config["training"]["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        model.train()
        for module in extra_modules:
            module.train()
        freeze_batch_norm_statistics(model)
        epoch_losses: dict[str, list[float]] = {}
        for step, (source_batches, target_images) in enumerate(cycle_loaders(source_loaders, target_loader)):
            source_images, source_labels = concatenate_source_batches(source_batches, device)
            target_images = target_images.to(device)
            steps_per_epoch = max(len(loader) for loader in source_loaders.values())
            progress = ((epoch - 1) * steps_per_epoch + step) / (max_epochs * steps_per_epoch)
            losses = update_step(model, optimizer, source_images, source_labels, target_images, progress)
            for name, value in losses.items():
                epoch_losses.setdefault(name, []).append(float(value))

        scores = validate(model, validation_loaders, device)
        row = {"epoch": epoch, **{name: float(np.mean(values)) for name, values in epoch_losses.items()}, **scores}
        history.append(row)
        score = scores["mean_source_validation_macro_f1"]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": config}, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
                break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    return output_dir / "best.pt"
