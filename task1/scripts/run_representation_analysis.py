"""Measure representation stability and plot clean-versus-intervention embeddings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from matplotlib.lines import Line2D
from PIL import Image
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.data.stl10 import IndexedSTL10, load_splits, load_stl10
from task1.models.backbones import load_backbone
from task1.training.linear_probe import extract_features
from task1.transforms import fixed_hue_rotation, grayscale, patch_shuffle, translate_reflect


class CueAndContentDataset(Dataset):
    def __init__(self, rows, image_dir: Path, test_dataset, transform, use_content: bool) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.test_dataset = test_dataset
        self.transform = transform
        self.use_content = use_content

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int):
        row = self.rows[position]
        if self.use_content:
            image, label = self.test_dataset[int(row["content_index"])]
        else:
            image = Image.open(self.image_dir / f"{row['candidate_id']}.png").convert("RGB")
            label = int(row["content_label"])
        return self.transform(image), int(label), position


def _loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _cosine(clean: torch.Tensor, changed: torch.Tensor) -> tuple[float, float]:
    values = functional.cosine_similarity(clean, changed, dim=1)
    return values.mean().item(), values.std(unbiased=False).item()


def _plot_pair(axis, clean, changed, labels, title: str, seed: int) -> None:
    values = torch.cat([clean, changed]).numpy()
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(values)
    labels = labels.numpy()
    colours = plt.get_cmap("tab10")(labels)
    count = len(labels)
    axis.scatter(embedding[:count, 0], embedding[:count, 1], c=colours, s=8, alpha=0.42, marker="o", label="clean")
    axis.scatter(embedding[count:, 0], embedding[count:, 1], c=colours, s=8, alpha=0.42, marker="x", label="intervened")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_figure(
    backbone_name: str,
    features: dict[str, torch.Tensor],
    labels: torch.Tensor,
    cue_clean,
    cue_features,
    cue_labels,
    class_names: list[str],
    figure_dir: Path,
    seed: int,
) -> None:
    pairs = [
        ("grayscale", features["grayscale"], labels),
        ("hue rotation (90°)", features["hue_rotation_90_degrees"], labels),
        ("translation (32 px right)", features["translation_right_32"], labels),
        ("4×4 patch shuffle", features["patch_shuffle_4x4"], labels),
        ("cue conflict", cue_features, cue_labels),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.8))
    for axis, (title, changed, pair_labels) in zip(axes.flat, pairs):
        reference = cue_clean if title == "cue conflict" else features["clean"]
        _plot_pair(axis, reference, changed, pair_labels, title, seed)
    legend_axis = axes.flat[5]
    legend_axis.axis("off")
    colour_map = plt.get_cmap("tab10")
    class_handles = [
        Line2D([], [], marker="o", linestyle="None", color=colour_map(index), markersize=7, label=class_name)
        for index, class_name in enumerate(class_names)
    ]
    legend_axis.legend(
        handles=class_handles,
        title="Ground-truth class",
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    condition_handles = [
        Line2D([], [], marker="o", linestyle="None", color="0.25", markersize=7, label="clean"),
        Line2D([], [], marker="x", linestyle="None", color="0.25", markersize=8, markeredgewidth=1.4, label="intervened"),
    ]
    figure.legend(
        handles=condition_handles,
        title="Feature condition",
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        title_fontsize=11,
        markerscale=1.2,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.suptitle(f"{backbone_name}: UMAP representations", y=0.98)
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    figure.savefig(figure_dir / f"{backbone_name}_representation_umap.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(config["output"]["split_path"])
    test_dataset = load_stl10(config["dataset"]["root"], "test", download=False)
    selected_indices = splits["selected_test_indices"]
    class_to_id = {name: index for index, name in enumerate(splits["classes"])}
    cue_dir = Path(config["output"]["cue_candidate_dir"])
    with (cue_dir / "selection_metadata.csv").open(newline="", encoding="utf-8") as handle:
        cue_rows = [row for row in csv.DictReader(handle) if row["selection_status"] == "selected"]
    for row in cue_rows:
        row["content_label"] = class_to_id[row["content_class"]]

    rows: list[dict] = []
    figure_dir = Path(config["output"]["representation_figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_size = config["image"]["size"]
    training_config = config["training"]
    directions = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}

    for backbone_name in config["models"]:
        print(f"Running {backbone_name}")
        backbone = load_backbone(backbone_name, device)
        clean_transform = backbone.image_transform(image_size)
        resize = backbone.common_resize(image_size)
        normalize = backbone.tensor_normalization()
        conditions = {
            "clean": clean_transform,
            "grayscale": backbone.image_transform(image_size, intervention=grayscale),
            "hue_rotation_90_degrees": backbone.image_transform(
                image_size,
                intervention=lambda image: fixed_hue_rotation(image, config["color_interventions"]["hue_factor"]),
            ),
            "translation_right_32": backbone.image_transform(
                image_size, intervention=lambda image: translate_reflect(image, dx=32, dy=0)
            ),
        }
        for direction, (x_sign, y_sign) in directions.items():
            conditions[f"translation_{direction}_32"] = backbone.image_transform(
                image_size, intervention=lambda image, x=x_sign, y=y_sign: translate_reflect(image, dx=32 * x, dy=32 * y)
            )

        def patch_transform(image, image_id):
            return normalize(patch_shuffle(resize(image), image_id=image_id, seed=config["seed"], grid_size=4))

        features = {}
        labels = None
        for condition, transform in conditions.items():
            output, condition_labels, _ = extract_features(
                backbone,
                _loader(IndexedSTL10(test_dataset, selected_indices, transform), training_config["feature_batch_size"], training_config["num_workers"]),
                device,
            )
            features[condition] = output
            labels = condition_labels
        patch_features, _, _ = extract_features(
            backbone,
            _loader(IndexedSTL10(test_dataset, selected_indices, patch_transform, index_aware_transform=True), training_config["feature_batch_size"], training_config["num_workers"]),
            device,
        )
        features["patch_shuffle_4x4"] = patch_features

        cue_clean, cue_labels, _ = extract_features(
            backbone,
            _loader(CueAndContentDataset(cue_rows, cue_dir / "images", test_dataset, clean_transform, use_content=True), training_config["feature_batch_size"], training_config["num_workers"]),
            device,
        )
        cue_features, _, _ = extract_features(
            backbone,
            _loader(CueAndContentDataset(cue_rows, cue_dir / "images", test_dataset, clean_transform, use_content=False), training_config["feature_batch_size"], training_config["num_workers"]),
            device,
        )

        for condition in ("grayscale", "hue_rotation_90_degrees", "patch_shuffle_4x4"):
            mean, standard_deviation = _cosine(features["clean"], features[condition])
            rows.append({"backbone": backbone_name, "intervention": condition, "sample_count": len(labels), "mean_cosine_similarity": mean, "standard_deviation": standard_deviation})
        translation_scores = torch.stack([
            functional.cosine_similarity(features["clean"], features[f"translation_{direction}_32"], dim=1)
            for direction in directions
        ])
        rows.append({"backbone": backbone_name, "intervention": "translation_32px_average", "sample_count": len(labels), "mean_cosine_similarity": translation_scores.mean().item(), "standard_deviation": translation_scores.std(unbiased=False).item()})
        mean, standard_deviation = _cosine(cue_clean, cue_features)
        rows.append({"backbone": backbone_name, "intervention": "cue_conflicts", "sample_count": len(cue_labels), "mean_cosine_similarity": mean, "standard_deviation": standard_deviation})
        _save_figure(
            backbone_name,
            features,
            labels,
            cue_clean,
            cue_features,
            cue_labels,
            splits["classes"],
            figure_dir,
            config["seed"],
        )
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table_path = Path(config["output"]["representation_table_path"])
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved representation stability to {table_path}")


if __name__ == "__main__":
    main()
