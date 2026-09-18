"""Train the required Task 1 linear heads and evaluate clean STL-10 images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.data.stl10 import IndexedSTL10, load_splits, load_stl10, make_splits, save_splits
from task1.evaluation.metrics import classification_metrics
from task1.models.backbones import clip_zero_shot_logits, load_backbone
from task1.training.linear_probe import extract_features, train_linear_head


def _feature_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _load_or_create_splits(config: dict, train_dataset, test_dataset) -> dict:
    split_path = Path(config["output"]["split_path"])
    if split_path.exists():
        return load_splits(split_path)
    dataset_config = config["dataset"]
    splits = make_splits(
        train_dataset,
        test_dataset,
        config["seed"],
        dataset_config["validation_fraction"],
        dataset_config["test_images_per_class"],
    )
    save_splits(splits, split_path)
    return splits


def _save_head(head: torch.nn.Linear, backbone_name: str, config: dict, class_names: list[str]) -> None:
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"{backbone_name}_linear_head.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone_name,
            "class_names": class_names,
            "feature_dimension": head.in_features,
            "state_dict": head.state_dict(),
        },
        checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_config = config["dataset"]
    train_dataset = load_stl10(dataset_config["root"], "train", dataset_config["download"])
    test_dataset = load_stl10(dataset_config["root"], "test", dataset_config["download"])
    splits = _load_or_create_splits(config, train_dataset, test_dataset)
    training_config = config["training"]
    image_size = config["image"]["size"]
    class_names = splits["classes"]
    results = {
        "seed": config["seed"],
        "dataset": "STL10",
        "selected_test_count": len(splits["selected_test_indices"]),
        "common_image_size": image_size,
        "models": {},
    }

    for backbone_name in config["models"]:
        print(f"\nRunning {backbone_name}")
        backbone = load_backbone(backbone_name, device)
        transform = backbone.image_transform(image_size)
        train_loader = _feature_loader(
            IndexedSTL10(train_dataset, splits["train_indices"], transform),
            training_config["feature_batch_size"],
            training_config["num_workers"],
        )
        validation_loader = _feature_loader(
            IndexedSTL10(train_dataset, splits["validation_indices"], transform),
            training_config["feature_batch_size"],
            training_config["num_workers"],
        )
        test_loader = _feature_loader(
            IndexedSTL10(test_dataset, splits["selected_test_indices"], transform),
            training_config["feature_batch_size"],
            training_config["num_workers"],
        )
        train_features, train_labels, _ = extract_features(backbone, train_loader, device)
        validation_features, validation_labels, _ = extract_features(backbone, validation_loader, device)
        test_features, test_labels, test_indices = extract_features(backbone, test_loader, device)
        head, history = train_linear_head(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            backbone.feature_dimension,
            len(class_names),
            training_config,
            config["seed"],
            device,
        )
        with torch.no_grad():
            logits = head(test_features.to(device)).cpu()
        model_result = {
            "linear_head": classification_metrics(logits, test_labels),
            "training_history": history,
            "test_indices": test_indices.tolist(),
        }
        _save_head(head, backbone_name, config, class_names)

        if backbone.is_clip:
            zero_shot_logits = clip_zero_shot_logits(backbone, test_features, class_names, device).cpu()
            model_result["zero_shot"] = {
                "prompt_template": "a photo of a {class}.",
                **classification_metrics(zero_shot_logits, test_labels),
            }
        results["models"][backbone_name] = model_result
        print(model_result["linear_head"])
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = Path(config["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
