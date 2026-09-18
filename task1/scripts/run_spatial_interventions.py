"""Evaluate translation and patch-structure interventions for Task 1."""

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
from task1.transforms import patch_shuffle, translate_reflect


def _loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(clean_logits: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> dict:
    metrics = classification_metrics(logits, labels)
    metrics["accuracy_change_from_clean"] = metrics["accuracy"] - classification_metrics(clean_logits, labels)["accuracy"]
    metrics["prediction_consistency"] = (
        clean_logits.argmax(dim=1) == logits.argmax(dim=1)
    ).float().mean().item()
    return metrics


def _mean_metrics(metrics: list[dict]) -> dict:
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def _translation_intervention(dx: int, dy: int):
    return lambda image: translate_reflect(image, dx=dx, dy=dy)


def _evaluate_classifier(
    backbone,
    linear_head,
    conditions: dict,
    test_dataset,
    test_indices: list[int],
    config: dict,
    device: torch.device,
    classifier: str,
    class_names: list[str],
) -> tuple[dict, torch.Tensor]:
    features_by_condition, labels = {}, None
    for name, transform in conditions.items():
        dataset = IndexedSTL10(test_dataset, test_indices, transform)
        features, condition_labels, _ = extract_features(
            backbone,
            _loader(dataset, config["feature_batch_size"], config["num_workers"]),
            device,
        )
        features_by_condition[name] = features
        labels = condition_labels
    if classifier == "linear_head":
        with torch.no_grad():
            logits_by_condition = {
                name: linear_head(features.to(device)).cpu()
                for name, features in features_by_condition.items()
            }
    else:
        logits_by_condition = {
            name: clip_zero_shot_logits(backbone, features, class_names, device).cpu()
            for name, features in features_by_condition.items()
        }
    metrics = {
        name: _evaluate(logits_by_condition["clean"], logits, labels)
        for name, logits in logits_by_condition.items()
    }
    return metrics, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = load_splits(config["output"]["split_path"])
    test_dataset = load_stl10(config["dataset"]["root"], "test", download=False)
    training_config = config["training"]
    image_size = config["image"]["size"]
    spatial_config = config["spatial_interventions"]
    directions = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
    results = {"seed": config["seed"], "translation": {}, "patch_shuffle": {}}
    table_rows: list[dict] = []

    for backbone_name in config["models"]:
        print(f"\nRunning {backbone_name}")
        backbone = load_backbone(backbone_name, device)
        clean_transform = backbone.image_transform(image_size)
        checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"{backbone_name}_linear_head.pt"
        linear_head = load_linear_head(str(checkpoint_path), device)
        classifiers = ["linear_head"] + (["zero_shot"] if backbone.is_clip else [])
        results["translation"][backbone_name] = {}
        results["patch_shuffle"][backbone_name] = {}

        for classifier in classifiers:
            translation_result = {}
            for displacement in spatial_config["translation_displacements"]:
                if displacement == 0:
                    conditions = {"clean": clean_transform}
                else:
                    conditions = {"clean": clean_transform}
                    for direction, (x_sign, y_sign) in directions.items():
                        conditions[direction] = backbone.image_transform(
                            image_size,
                            intervention=_translation_intervention(displacement * x_sign, displacement * y_sign),
                        )
                metrics, _ = _evaluate_classifier(
                    backbone,
                    linear_head,
                    conditions,
                    test_dataset,
                    splits["selected_test_indices"],
                    training_config,
                    device,
                    classifier,
                    splits["classes"],
                )
                if displacement == 0:
                    averaged = metrics["clean"]
                    directional = {"clean": averaged}
                else:
                    directional = {direction: metrics[direction] for direction in directions}
                    averaged = _mean_metrics(list(directional.values()))
                translation_result[str(displacement)] = {"directional": directional, "average": averaged}
                table_rows.append(
                    {
                        "backbone": backbone_name,
                        "classifier": classifier,
                        "intervention": "translation",
                        "displacement_pixels": displacement,
                        "condition": "average",
                        **averaged,
                    }
                )

            resize = backbone.common_resize(image_size)
            normalize = backbone.tensor_normalization()

            def patch_transform(image, image_id):
                return normalize(
                    patch_shuffle(
                        resize(image),
                        image_id=image_id,
                        seed=config["seed"],
                        grid_size=spatial_config["patch_grid_size"],
                    )
                )

            patch_dataset = IndexedSTL10(
                test_dataset,
                splits["selected_test_indices"],
                patch_transform,
                index_aware_transform=True,
            )
            clean_dataset = IndexedSTL10(test_dataset, splits["selected_test_indices"], clean_transform)
            patch_features, patch_labels, _ = extract_features(
                backbone,
                _loader(patch_dataset, training_config["feature_batch_size"], training_config["num_workers"]),
                device,
            )
            clean_features, _, _ = extract_features(
                backbone,
                _loader(clean_dataset, training_config["feature_batch_size"], training_config["num_workers"]),
                device,
            )
            if classifier == "linear_head":
                with torch.no_grad():
                    clean_logits = linear_head(clean_features.to(device)).cpu()
                    patch_logits = linear_head(patch_features.to(device)).cpu()
            else:
                clean_logits = clip_zero_shot_logits(backbone, clean_features, splits["classes"], device).cpu()
                patch_logits = clip_zero_shot_logits(backbone, patch_features, splits["classes"], device).cpu()
            patch_metrics = _evaluate(clean_logits, patch_logits, patch_labels)
            results["translation"][backbone_name][classifier] = translation_result
            results["patch_shuffle"][backbone_name][classifier] = patch_metrics
            table_rows.append(
                {
                    "backbone": backbone_name,
                    "classifier": classifier,
                    "intervention": "patch_shuffle_4x4",
                    "displacement_pixels": "",
                    "condition": "patch_shuffle",
                    **patch_metrics,
                }
            )

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = Path(config["output"]["spatial_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table_path = Path(config["output"]["spatial_table_path"])
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved spatial metrics to {metrics_path} and {table_path}")


if __name__ == "__main__":
    main()
