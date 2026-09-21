"""Evaluate frozen Task 3 checkpoints on Sketch only after selection is complete."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset, final_target_samples
from shared.pacs_protocol import load_pacs_protocol
from task2.evaluation.metrics import classification_metrics
from task2.transforms import evaluation_transform
from task3.config import load_config
from task3.data import build_validation_loaders
from task3.evaluation.sharpness import sharpness_proxy
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.models.backbone import PACSResNet18


@torch.no_grad()
def collect_outputs(model, loader, device: torch.device):
    """Collect frozen-model features, predictions, and labels from one labeled loader."""
    model.eval()
    features, predictions, labels = [], [], []
    for images, batch_labels in loader:
        batch_features, logits = model(images.to(device, non_blocking=True))
        features.append(batch_features.cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy())
    return np.concatenate(features), np.asarray(predictions), np.asarray(labels)


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    """Parse repeated method=checkpoint command-line options."""
    checkpoints = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("Each --checkpoint must be method=path/to/best.pt")
        checkpoints[name] = Path(path)
    return checkpoints


def load_model(path: Path, classes: int, device: torch.device):
    """Load a model checkpoint selected exclusively on source validation."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PACSResNet18(num_classes=classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task3/configs/erm.yaml")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--sketch-labels-released", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.sketch_labels_released:
        raise SystemExit("Sketch metrics are disabled until all Task 3 settings and checkpoints are fixed.")

    config = load_config(args.config)
    device = torch.device(args.device)
    protocol = load_pacs_protocol(config["data"]["protocol"])
    source_loaders = build_validation_loaders(config, protocol)
    transform = evaluation_transform(config["data"]["resize_size"], config["data"]["crop_size"])
    sketch_loader = DataLoader(
        PACSLabeledDataset(config["data"]["root"], final_target_samples(config["data"]["root"]), transform),
        batch_size=64, shuffle=False, num_workers=0
    )
    output_dir = Path(config["output"]["root"]) / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, class_rows = [], []
    for method, checkpoint_path in parse_checkpoints(args.checkpoint).items():
        model = load_model(checkpoint_path, len(config["data"]["classes"]), device)
        source_features, source_scores = {}, []
        row = {"method": method}
        for domain, loader in source_loaders.items():
            features, predictions, labels = collect_outputs(model, loader, device)
            source_features[domain] = features
            scores = classification_metrics(labels, predictions)
            source_scores.append(scores)
            row[f"{domain}_accuracy"] = scores["accuracy"]
            row[f"{domain}_macro_f1"] = scores["macro_f1"]
        _, sketch_predictions, sketch_labels = collect_outputs(model, sketch_loader, device)
        sketch_scores = classification_metrics(sketch_labels, sketch_predictions)
        row.update({
            "mean_source_validation_accuracy": float(np.mean([score["accuracy"] for score in source_scores])),
            "mean_source_validation_macro_f1": float(np.mean([score["macro_f1"] for score in source_scores])),
            "worst_source_validation_accuracy": float(np.min([score["accuracy"] for score in source_scores])),
            "worst_source_validation_macro_f1": float(np.min([score["macro_f1"] for score in source_scores])),
            "sketch_accuracy": sketch_scores["accuracy"],
            "sketch_macro_f1": sketch_scores["macro_f1"],
            "source_domain_separability": source_domain_separability(source_features),
            "sharpness_proxy": sharpness_proxy(model, source_loaders, device),
        })
        metric_rows.append(row)
        matrix = confusion_matrix(sketch_labels, sketch_predictions, labels=range(len(config["data"]["classes"])))
        np.savetxt(output_dir / f"{method}_sketch_confusion.csv", matrix, fmt="%d", delimiter=",")
        for label, class_name in enumerate(config["data"]["classes"]):
            mask = sketch_labels == label
            class_rows.append({"method": method, "class_name": class_name, "sketch_accuracy": float((sketch_predictions[mask] == label).mean()), "support": int(mask.sum())})

    erm_accuracy = next(row["sketch_accuracy"] for row in metric_rows if row["method"] == "erm")
    for row in metric_rows:
        row["sketch_accuracy_change_from_erm"] = row["sketch_accuracy"] - erm_accuracy
    for path, rows in [(output_dir / "metrics.csv", metric_rows), (output_dir / "per_class_accuracy.csv", class_rows)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
