"""Generate and document AdaIN cue-conflict candidates before model evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.cue_conflicts import adain_transform, load_adain, stylize
from task1.data.stl10 import load_splits, load_stl10


def _directions(pairs: list[list[str]]) -> list[tuple[str, str]]:
    return [(content, style) for first, second in pairs for content, style in ((first, second), (second, first))]


def _candidate_plan(train_dataset, test_dataset, splits: dict, config: dict) -> list[dict]:
    rng = np.random.default_rng(config["seed"])
    cue_config = config["cue_conflicts"]
    class_to_id = {name: index for index, name in enumerate(splits["classes"])}
    test_labels = np.asarray(test_dataset.labels)
    train_labels = np.asarray(train_dataset.labels)
    selected_test_indices = np.asarray(splits["selected_test_indices"])
    rows: list[dict] = []
    for content_class, style_class in _directions(cue_config["pairs"]):
        content_id, style_id = class_to_id[content_class], class_to_id[style_class]
        content_indices = selected_test_indices[test_labels[selected_test_indices] == content_id]
        if len(content_indices) < cue_config["candidates_per_direction"]:
            raise ValueError(f"Not enough selected test images for {content_class}.")
        chosen_content = rng.choice(content_indices, size=cue_config["candidates_per_direction"], replace=False)
        style_indices = np.flatnonzero(train_labels == style_id)
        chosen_style = rng.choice(style_indices, size=cue_config["candidates_per_direction"], replace=True)
        direction = f"{content_class}_to_{style_class}"
        for slot, (content_index, style_index) in enumerate(zip(chosen_content, chosen_style)):
            rows.append(
                {
                    "candidate_id": f"{direction}_{slot:02d}",
                    "direction": direction,
                    "content_class": content_class,
                    "style_class": style_class,
                    "content_index": int(content_index),
                    "style_index": int(style_index),
                    "slot": slot,
                    "review_status": "pending",
                }
            )
    return rows


def _save_contact_sheet(rows: list[dict], test_dataset, image_dir: Path, path: Path) -> None:
    columns = 5
    figure, axes = plt.subplots(math.ceil(len(rows) / columns), columns, figsize=(15, 3 * math.ceil(len(rows) / columns)))
    for axis, row in zip(np.asarray(axes).reshape(-1), rows):
        content, _ = test_dataset[row["content_index"]]
        stylized = Image.open(image_dir / f"{row['candidate_id']}.png")
        combined = np.concatenate([np.asarray(content.resize((128, 128))), np.asarray(stylized.resize((128, 128)))], axis=1)
        axis.imshow(combined)
        axis.set_title(f"{row['slot']:02d}: {row['content_index']} / {row['style_index']}", fontsize=8)
        axis.axis("off")
    for axis in np.asarray(axes).reshape(-1)[len(rows):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    parser.add_argument("--adain-root", default="/content/pytorch-AdaIN")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output"]["cue_candidate_dir"])
    image_dir = output_dir / "images"
    sheet_dir = output_dir / "contact_sheets"
    image_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_stl10(config["dataset"]["root"], "train", download=False)
    test_dataset = load_stl10(config["dataset"]["root"], "test", download=False)
    splits = load_splits(config["output"]["split_path"])
    rows = _candidate_plan(train_dataset, test_dataset, splits, config)
    encoder, decoder, adaptive_instance_normalization = load_adain(arguments.adain_root, device)
    transform = adain_transform(config["cue_conflicts"]["adain_input_size"])

    for number, row in enumerate(rows, start=1):
        content_image, _ = test_dataset[row["content_index"]]
        style_image, _ = train_dataset[row["style_index"]]
        content = transform(content_image).unsqueeze(0).to(device)
        style = transform(style_image).unsqueeze(0).to(device)
        output = stylize(content, style, encoder, decoder, adaptive_instance_normalization, config["cue_conflicts"]["alpha"])
        pixels = (output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(image_dir / f"{row['candidate_id']}.png")
        row["pixel_standard_deviation"] = float(pixels.std() / 255)
        print(f"Generated {number}/{len(rows)}: {row['candidate_id']}")

    metadata_path = output_dir / "candidate_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for direction in sorted({row["direction"] for row in rows}):
        _save_contact_sheet([row for row in rows if row["direction"] == direction], test_dataset, image_dir, sheet_dir / f"{direction}.png")
    protocol = {
        "accept": "content object remains recognisable and the transferred texture or colour is visibly present",
        "reject": "content object is unrecognisable, output is severely distorted, or style transfer is not visible",
        "candidates_per_direction": config["cue_conflicts"]["candidates_per_direction"],
        "selected_per_direction": config["cue_conflicts"]["selected_per_direction"],
        "status": "pending_visual_review",
    }
    (output_dir / "screening_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} candidates to {output_dir}")


if __name__ == "__main__":
    main()
