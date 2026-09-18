"""Create the compact Task 1 result figures from saved CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DISPLAY_NAMES = {
    ("resnet50", "linear_head"): "ResNet-50",
    ("vit_b_16", "linear_head"): "ViT-B/16",
    ("clip_vit_b_32", "linear_head"): "CLIP linear",
    ("clip_vit_b_32", "zero_shot"): "CLIP zero-shot",
}


def _label(row) -> str:
    return DISPLAY_NAMES[(row.backbone, row.classifier)]


def _grouped_bars(axis, frame, categories, value_column, title, y_label) -> None:
    methods = list(DISPLAY_NAMES.values())
    x = np.arange(len(categories))
    width = 0.19
    for index, method in enumerate(methods):
        values = []
        for category in categories:
            selected = frame[(frame["method"] == method) & (frame["category"] == category)]
            values.append(selected[value_column].iloc[0] if len(selected) else np.nan)
        axis.bar(x + (index - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, categories, rotation=10)
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    metrics_dir = Path(config["output"]["color_table_path"]).parent
    figure_dir = Path(config["output"]["representation_figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    color = pd.read_csv(metrics_dir / "color_interventions.csv")
    spatial = pd.read_csv(metrics_dir / "spatial_interventions.csv")
    cue = pd.read_csv(metrics_dir / "cue_conflicts.csv")

    color = color.assign(method=color.apply(_label, axis=1), category=color["condition"])
    patch = spatial[spatial["intervention"] == "patch_shuffle_4x4"].copy()
    patch = patch.assign(method=patch.apply(_label, axis=1), category="patch shuffle")
    compact = pd.concat([color[["method", "category", "accuracy"]], patch[["method", "category", "accuracy"]]])
    figure, axis = plt.subplots(figsize=(10, 5))
    _grouped_bars(axis, compact, ["clean", "grayscale", "hue_rotation_90_degrees", "patch shuffle"], "accuracy", "Color and patch-structure accuracy", "Top-1 accuracy")
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "task1_color_patch_accuracy.png", dpi=180)
    plt.close(figure)

    translation = spatial[spatial["intervention"] == "translation"].copy()
    translation["method"] = translation.apply(_label, axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for method, subset in translation.groupby("method"):
        subset = subset.sort_values("displacement_pixels")
        axes[0].plot(subset["displacement_pixels"], subset["accuracy"], marker="o", label=method)
        axes[1].plot(subset["displacement_pixels"], subset["prediction_consistency"], marker="o", label=method)
    axes[0].set_ylabel("Top-1 accuracy")
    axes[1].set_ylabel("Prediction consistency")
    for axis, title in zip(axes, ["Translation accuracy", "Translation consistency"]):
        axis.set_xlabel("Displacement (pixels)")
        axis.set_xticks([0, 8, 16, 32])
        axis.set_ylim(0.94, 1.005)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "task1_translation_curves.png", dpi=180)
    plt.close(figure)

    cue["method"] = cue.apply(_label, axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(cue))
    axes[0].bar(x, cue["shape_accuracy"], label="shape", color="#4c78a8")
    axes[0].bar(x, cue["texture_accuracy"], bottom=cue["shape_accuracy"], label="texture", color="#f58518")
    axes[0].bar(x, cue["other_prediction_rate"], bottom=cue["shape_accuracy"] + cue["texture_accuracy"], label="other", color="#bab0ac")
    axes[0].set_xticks(x, cue["method"], rotation=18, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of 200 conflicts")
    axes[0].set_title("Cue-conflict decisions")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, cue["shape_bias"], 0.36, label="shape bias", color="#54a24b")
    axes[1].bar(x + 0.18, cue["coverage"], 0.36, label="coverage", color="#eeca3b")
    axes[1].set_xticks(x, cue["method"], rotation=18, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Shape bias among intended decisions")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "task1_cue_conflicts.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
