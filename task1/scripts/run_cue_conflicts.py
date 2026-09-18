"""Evaluate frozen Task 1 classifiers on the reviewed cue-conflict set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.data.stl10 import load_splits
from task1.models.backbones import clip_zero_shot_logits, load_backbone
from task1.training.linear_probe import load_linear_head


class CueConflictDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path, transform, class_to_id: dict[str, int]) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int):
        row = self.rows[position]
        image = Image.open(self.image_dir / f"{row['candidate_id']}.png").convert("RGB")
        return (
            self.transform(image),
            self.class_to_id[row["content_class"]],
            self.class_to_id[row["style_class"]],
        )


def _features(backbone, loader: DataLoader, device: torch.device):
    features, content_labels, style_labels = [], [], []
    with torch.no_grad():
        for images, content, style in loader:
            features.append(backbone.extract(images.to(device)).cpu())
            content_labels.append(content)
            style_labels.append(style)
    return torch.cat(features), torch.cat(content_labels), torch.cat(style_labels)


def _cue_metrics(logits: torch.Tensor, content_labels: torch.Tensor, style_labels: torch.Tensor) -> dict:
    predictions = logits.argmax(dim=1).cpu()
    shape = predictions == content_labels
    texture = predictions == style_labels
    shape_count = int(shape.sum().item())
    texture_count = int(texture.sum().item())
    coverage_count = shape_count + texture_count
    return {
        "shape_count": shape_count,
        "texture_count": texture_count,
        "other_count": int((~(shape | texture)).sum().item()),
        "shape_accuracy": shape.float().mean().item(),
        "texture_accuracy": texture.float().mean().item(),
        "other_prediction_rate": (~(shape | texture)).float().mean().item(),
        "shape_bias": shape_count / coverage_count if coverage_count else float("nan"),
        "coverage": coverage_count / len(predictions),
        "mean_maximum_confidence": torch.softmax(logits, dim=1).max(dim=1).values.mean().item(),
    }


def _prediction_rows(rows: list[dict], logits: torch.Tensor, classes: list[str], backbone: str, classifier: str) -> list[dict]:
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1).tolist()
    output = []
    for row, prediction, confidence in zip(rows, predictions, probabilities.max(dim=1).values.tolist()):
        predicted_class = classes[prediction]
        role = "shape" if predicted_class == row["content_class"] else "texture" if predicted_class == row["style_class"] else "other"
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "direction": row["direction"],
                "content_class": row["content_class"],
                "style_class": row["style_class"],
                "backbone": backbone,
                "classifier": classifier,
                "predicted_class": predicted_class,
                "prediction_role": role,
                "maximum_confidence": confidence,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output"]["cue_candidate_dir"])
    with (output_dir / "selection_metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["selection_status"] == "selected"]
    expected_count = len(config["cue_conflicts"]["pairs"]) * 2 * config["cue_conflicts"]["selected_per_direction"]
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} selected candidates, found {len(rows)}.")

    splits = load_splits(config["output"]["split_path"])
    class_to_id = {name: index for index, name in enumerate(splits["classes"])}
    results = {"seed": config["seed"], "candidate_count": len(rows), "models": {}}
    table_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for backbone_name in config["models"]:
        print(f"Running {backbone_name}")
        backbone = load_backbone(backbone_name, device)
        dataset = CueConflictDataset(rows, output_dir / "images", backbone.image_transform(config["image"]["size"]), class_to_id)
        loader = DataLoader(dataset, batch_size=config["training"]["feature_batch_size"], shuffle=False, num_workers=config["training"]["num_workers"], pin_memory=torch.cuda.is_available())
        features, content_labels, style_labels = _features(backbone, loader, device)
        checkpoint = Path(config["output"]["checkpoint_dir"]) / f"{backbone_name}_linear_head.pt"
        head = load_linear_head(str(checkpoint), device)
        classifiers = {"linear_head": head(features.to(device)).cpu()}
        if backbone.is_clip:
            classifiers["zero_shot"] = clip_zero_shot_logits(backbone, features, splits["classes"], device).cpu()
        results["models"][backbone_name] = {}
        for classifier, logits in classifiers.items():
            metrics = _cue_metrics(logits, content_labels, style_labels)
            results["models"][backbone_name][classifier] = metrics
            table_rows.append({"backbone": backbone_name, "classifier": classifier, **metrics})
            prediction_rows.extend(_prediction_rows(rows, logits, splits["classes"], backbone_name, classifier))
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = Path(config["output"]["cue_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with Path(config["output"]["cue_table_path"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    with Path(config["output"]["cue_predictions_path"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"Saved cue-conflict metrics to {metrics_path}")


if __name__ == "__main__":
    main()
