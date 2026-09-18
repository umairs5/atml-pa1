"""STL-10 split creation and model-agnostic image access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import STL10


def load_stl10(root: str | Path, split: str, download: bool) -> STL10:
    return STL10(root=str(root), split=split, download=download, transform=None)


def make_splits(
    train_dataset: STL10,
    test_dataset: STL10,
    seed: int,
    validation_fraction: float,
    test_images_per_class: int,
) -> dict:
    """Create the required stratified training split and balanced test subset."""
    train_labels = np.asarray(train_dataset.labels)
    test_labels = np.asarray(test_dataset.labels)
    all_train_indices = np.arange(len(train_dataset))
    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_labels,
    )

    generator = np.random.default_rng(seed)
    selected_test_indices: list[int] = []
    class_counts: dict[str, int] = {}
    for class_id, class_name in enumerate(test_dataset.classes):
        indices = np.flatnonzero(test_labels == class_id)
        selected_count = min(test_images_per_class, len(indices))
        selected = generator.choice(indices, size=selected_count, replace=False)
        selected_test_indices.extend(int(index) for index in selected)
        class_counts[class_name] = selected_count

    return {
        "dataset": "STL10",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_images_per_class_requested": test_images_per_class,
        "classes": list(train_dataset.classes),
        "train_indices": sorted(int(index) for index in train_indices),
        "validation_indices": sorted(int(index) for index in validation_indices),
        "selected_test_indices": sorted(selected_test_indices),
        "selected_test_class_counts": class_counts,
    }


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class IndexedSTL10(Dataset):
    """Apply a common image transform to an explicit list of STL-10 examples."""

    def __init__(
        self,
        dataset: STL10,
        indices: list[int],
        transform: Callable,
        index_aware_transform: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.index_aware_transform = index_aware_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        original_index = self.indices[position]
        image, label = self.dataset[original_index]
        if self.index_aware_transform:
            image = self.transform(image, original_index)
        else:
            image = self.transform(image)
        return image, int(label), int(original_index)
