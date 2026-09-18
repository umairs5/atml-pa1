"""Evaluate grayscale and fixed hue rotation on the frozen Task 1 models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.data.stl10 import IndexedSTL10, load_splits, load_stl10
from task1.evaluation.metrics import classification_metrics
from task1.models.backbones import clip_zero_shot_logits, load_backbone
from task1.training.linear_probe import extract_features, load_linear_head
from task1.transforms import fixed_hue_rotation, grayscale


def _feature_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_logits(
    clean_logits: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    metrics = classification_metrics(logits, labels)
    clean_predictions = clean_logits.argmax(dim=1)
    predictions = logits.argmax(dim=1)
    metrics["accuracy_change_from_clean"] = metrics["accuracy"] - classification_metrics(clean_logits, labels)["accuracy"]
    metrics["prediction_consistency"] = (clean_predictions == predictions).float().mean().item()
    return metrics


def _to_rows(backbone: str, classifier: str, metrics_by_condition: dict) -> list[dict]:
    rows = []
    for condition, metrics in metrics_by_condition.items():
        rows.append({"backbone": backbone, "classifier": classifier, "condition": condition, **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_path = Path(config["output"]["split_path"])
    if not split_path.exists():
        raise FileNotFoundError(f"Missing saved split: {split_path}. Run run_clean_baseline.py first.")
    splits = load_splits(split_path)
    dataset_config = config["dataset"]
    test_dataset = load_stl10(dataset_config["root"], "test", download=False)
    image_size = config["image"]["size"]
    training_config = config["training"]
    hue_factor = config["color_interventions"]["hue_factor"]
    conditions = {
        "clean": None,
        "grayscale": grayscale,
        config["color_interventions"]["additional_color_name"]: lambda image: fixed_hue_rotation(image, hue_factor),
    }
    results = {
        "seed": config["seed"],
        "common_image_size": image_size,
        "interventions": {
            "grayscale": "Convert the common 224x224 RGB image to grayscale and replicate it over three channels.",
            config["color_interventions"]["additional_color_name"]: {
                "hue_factor": hue_factor,
                "description": "Rotate hue by 90 degrees after common resizing and before model normalization.",
            },
        },
        "models": {},
    }
    table_rows: list[dict] = []

    for backbone_name in config["models"]:
        print(f"\nRunning {backbone_name}")
        backbone = load_backbone(backbone_name, device)
        checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"{backbone_name}_linear_head.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing selected head: {checkpoint_path}. Run run_clean_baseline.py first.")
        linear_head = load_linear_head(str(checkpoint_path), device)
        condition_features: dict[str, torch.Tensor] = {}
        condition_labels: dict[str, torch.Tensor] = {}
        for condition_name, intervention in conditions.items():
            transform = backbone.image_transform(image_size, intervention=intervention)
            loader = _feature_loader(
                IndexedSTL10(test_dataset, splits["selected_test_indices"], transform),
                training_config["feature_batch_size"],
                training_config["num_workers"],
            )
            features, labels, _ = extract_features(backbone, loader, device)
            condition_features[condition_name] = features
            condition_labels[condition_name] = labels

        labels = condition_labels["clean"]
        with torch.no_grad():
            linear_logits = {
                condition: linear_head(features.to(device)).cpu()
                for condition, features in condition_features.items()
            }
        linear_metrics = {
            condition: _evaluate_logits(linear_logits["clean"], logits, labels)
            for condition, logits in linear_logits.items()
        }
        model_result = {"linear_head": linear_metrics}
        table_rows.extend(_to_rows(backbone_name, "linear_head", linear_metrics))

        if backbone.is_clip:
            zero_shot_logits = {
                condition: clip_zero_shot_logits(backbone, features, splits["classes"], device).cpu()
                for condition, features in condition_features.items()
            }
            zero_shot_metrics = {
                condition: _evaluate_logits(zero_shot_logits["clean"], logits, labels)
                for condition, logits in zero_shot_logits.items()
            }
            model_result["zero_shot"] = {
                "prompt_template": "a photo of a {class}.",
                "metrics": zero_shot_metrics,
            }
            table_rows.extend(_to_rows(backbone_name, "zero_shot", zero_shot_metrics))

        results["models"][backbone_name] = model_result
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = Path(config["output"]["color_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table_path = Path(config["output"]["color_table_path"])
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved color metrics to {metrics_path} and {table_path}")


if __name__ == "__main__":
    main()
