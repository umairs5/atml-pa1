"""Dataset discovery and loaders for the PACS benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


PACS_CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
PACS_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def require_pacs_root(root: str | Path) -> Path:
    """Return a validated PACS root with one directory per domain."""
    root = Path(root)
    missing = [domain for domain in PACS_DOMAINS if not (root / domain).is_dir()]
    if missing:
        expected = root / "photo"
        raise FileNotFoundError(
            "PACS was not found. Put the extracted PACS_Original folder at "
            f"{root}. Expected, for example: {expected}"
        )
    return root


def discover_domain_samples(root: str | Path, domain: str) -> list[dict[str, object]]:
    """Discover labeled images for one PACS domain in a stable order."""
    root = require_pacs_root(root)
    if domain not in PACS_DOMAINS:
        raise ValueError(f"Unknown PACS domain: {domain}")

    samples: list[dict[str, object]] = []
    for label, class_name in enumerate(PACS_CLASSES):
        class_dir = root / domain / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected PACS class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "label": label,
                    }
                )
    if not samples:
        raise RuntimeError(f"No images found under {root / domain}")
    return samples


def discover_domain_paths(root: str | Path, domain: str) -> list[str]:
    """Return image paths without exposing their class labels."""
    return [str(sample["path"]) for sample in discover_domain_samples(root, domain)]


class PACSLabeledDataset(Dataset):
    """PACS dataset for source training or source validation."""

    def __init__(self, root: str | Path, samples: Iterable[dict[str, object]], transform=None):
        self.root = require_pacs_root(root)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self.root / str(sample["path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])


class PACSUnlabeledDataset(Dataset):
    """Sketch images used during transductive adaptation without labels."""

    def __init__(self, root: str | Path, paths: Iterable[str], transform=None):
        self.root = require_pacs_root(root)
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.root / self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
