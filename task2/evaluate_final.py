"""Evaluate fixed Task 2 checkpoints without leaking target labels into training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset, PACSUnlabeledDataset, final_target_samples
from shared.pacs_protocol import load_pacs_protocol
from task2.config import load_config
from task2.data import build_validation_loaders
from task2.evaluation.domain_separability import domain_separability_accuracy
from task2.evaluation.metrics import classification_metrics
from task2.models.backbone import PACSResNet18
from task2.transforms import evaluation_transform


@torch.no_grad()
def collect_outputs(model, loader, device: torch.device, labeled: bool):
    """Collect logits, features, and labels when the supplied loader has labels."""
    model.eval()
    features, predictions, labels = [], [], []
    for batch in loader:
        if labeled:
            images, batch_labels = batch
            labels.extend(batch_labels.numpy())
        else:
            images = batch
        batch_features, logits = model(images.to(device))
        features.append(batch_features.cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(features), np.asarray(predictions), np.asarray(labels)


def load_model(checkpoint_path: str | Path, num_classes: int, device: torch.device):
    """Load the classification network from a checkpoint selected on source validation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PACSResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    """Parse repeated method_name=checkpoint_path command-line values."""
    checkpoints = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("Each --checkpoint must use method_name=path/to/best.pt")
        checkpoints[name] = Path(path)
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task2/configs/source_only.yaml")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--target-labels-released", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.target_labels_released:
        raise SystemExit(
            "Target metrics are disabled. Run this only after all method choices are fixed, "
            "then pass --target-labels-released."
        )

    config = load_config(args.config)
    protocol = load_pacs_protocol(config["data"]["protocol"])
    device = torch.device(args.device)
    validation_loaders = build_validation_loaders(config, protocol)
    transform = evaluation_transform(config["data"]["resize_size"], config["data"]["crop_size"])
    target_loader = DataLoader(
        PACSLabeledDataset(config["data"]["root"], final_target_samples(config["data"]["root"]), transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    unlabeled_target_loader = DataLoader(
        PACSUnlabeledDataset(config["data"]["root"], protocol["target"]["paths"], transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    output_dir = Path(config["output"]["root"]) / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, class_rows = [], []
    checkpoints = parse_checkpoints(args.checkpoint)
    for method_name, checkpoint_path in checkpoints.items():
        model = load_model(checkpoint_path, len(config["data"]["classes"]), device)
        source_features = []
        source_scores = []
        for loader in validation_loaders.values():
            features, predictions, labels = collect_outputs(model, loader, device, labeled=True)
            source_features.append(features)
            source_scores.append(classification_metrics(labels, predictions)["macro_f1"])
        target_features, target_predictions, target_labels = collect_outputs(model, target_loader, device, labeled=True)
        unlabeled_features, _, _ = collect_outputs(model, unlabeled_target_loader, device, labeled=False)
        target_metrics = classification_metrics(target_labels, target_predictions)
        metric_rows.append(
            {
                "method": method_name,
                "mean_source_validation_macro_f1": float(np.mean(source_scores)),
                "target_accuracy": target_metrics["accuracy"],
                "target_macro_f1": target_metrics["macro_f1"],
                "domain_separability_accuracy": domain_separability_accuracy(
                    np.concatenate(source_features), unlabeled_features
                ),
            }
        )
        confusion = confusion_matrix(target_labels, target_predictions, labels=range(len(config["data"]["classes"])))
        np.savetxt(output_dir / f"{method_name}_target_confusion.csv", confusion, fmt="%d", delimiter=",")
        for label, class_name in enumerate(config["data"]["classes"]):
            mask = target_labels == label
            class_rows.append(
                {
                    "method": method_name,
                    "class_name": class_name,
                    "target_accuracy": float((target_predictions[mask] == label).mean()),
                    "support": int(mask.sum()),
                }
            )

    for path, rows in [(output_dir / "metrics.csv", metric_rows), (output_dir / "per_class_accuracy.csv", class_rows)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
