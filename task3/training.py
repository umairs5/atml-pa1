"""Common source-only training and source-validation selection for Task 3."""

import csv
from pathlib import Path

import numpy as np
import torch

from common.seed import seed_everything
from shared.pacs_protocol import load_pacs_protocol
from task2.evaluation.metrics import classification_metrics
from task3.data import build_training_loaders, build_validation_loaders, cycle_source_loaders
from task3.models.backbone import freeze_batch_norm_statistics


@torch.no_grad()
def validate(model, loaders: dict, device: torch.device) -> dict[str, float]:
    """Measure every observed source domain before any Sketch evaluation."""
    model.eval()
    scores = {}
    for domain, loader in loaders.items():
        labels, predictions = [], []
        for images, batch_labels in loader:
            _, logits = model(images.to(device, non_blocking=True))
            labels.extend(batch_labels.numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
        metrics = classification_metrics(np.asarray(labels), np.asarray(predictions))
        scores[f"{domain}_accuracy"] = metrics["accuracy"]
        scores[f"{domain}_macro_f1"] = metrics["macro_f1"]
    source_f1 = [scores[f"{domain}_macro_f1"] for domain in loaders]
    scores["mean_source_validation_macro_f1"] = float(np.mean(source_f1))
    scores["worst_source_validation_macro_f1"] = float(np.min(source_f1))
    return scores


def run_training(config: dict, model, update_step, device: torch.device) -> Path:
    """Train a source-only method and select its checkpoint by mean source macro-F1."""
    seed_everything(config["seed"])
    protocol = load_pacs_protocol(config["data"]["protocol"])
    source_loaders = build_training_loaders(config, protocol)
    validation_loaders = build_validation_loaders(config, protocol)
    output_dir = Path(config["output"]["root"]) / config["method"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    history, best_score, stale_epochs = [], float("-inf"), 0

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        freeze_batch_norm_statistics(model)
        losses: dict[str, list[float]] = {}
        for source_batches in cycle_source_loaders(source_loaders):
            values = update_step(model, optimizer, source_batches, device)
            for name, value in values.items():
                losses.setdefault(name, []).append(float(value))
        scores = validate(model, validation_loaders, device)
        row = {"epoch": epoch, **{name: float(np.mean(values)) for name, values in losses.items()}, **scores}
        history.append(row)
        score = scores["mean_source_validation_macro_f1"]
        print(f"Epoch {epoch:02d}: source validation macro-F1={score:.4f}")
        if score > best_score:
            best_score, stale_epochs = score, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": config}, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config["training"]["early_stopping_patience"]:
                break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    return output_dir / "best.pt"
