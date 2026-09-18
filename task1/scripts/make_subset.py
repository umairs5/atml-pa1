"""Create and save the Task 1 STL-10 split and common test subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.seed import seed_everything
from task1.data.stl10 import load_stl10, make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    seed_everything(config["seed"])

    dataset_config = config["dataset"]
    train_dataset = load_stl10(dataset_config["root"], "train", dataset_config["download"])
    test_dataset = load_stl10(dataset_config["root"], "test", dataset_config["download"])
    splits = make_splits(
        train_dataset,
        test_dataset,
        config["seed"],
        dataset_config["validation_fraction"],
        dataset_config["test_images_per_class"],
    )
    save_splits(splits, config["output"]["split_path"])
    print(f"Saved {len(splits['selected_test_indices'])} test identifiers to {config['output']['split_path']}")
    print(splits["selected_test_class_counts"])


if __name__ == "__main__":
    main()
