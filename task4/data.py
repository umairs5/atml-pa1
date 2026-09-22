"""Fixed, evaluation-safe CIFAR-10/CIFAR-100 protocol construction."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

NEAR = ("bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel")
FAR = ("bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe")
NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

def transforms_for(method: str, train: bool):
    steps = []
    if train:
        steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if method == "gcsc": steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    return transforms.Compose([*steps, transforms.ToTensor(), NORMALIZE])

def make_split(root: str, split_path: str, seed: int, fraction: float) -> dict:
    path = Path(split_path)
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    base = datasets.CIFAR10(root, train=True, download=True)
    indices = np.arange(len(base.targets))
    train, valid = train_test_split(indices, test_size=fraction, random_state=seed, stratify=base.targets)
    payload = {"seed": seed, "train_indices": sorted(map(int, train)), "validation_indices": sorted(map(int, valid))}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

def unknown_indices(root: str):
    data = datasets.CIFAR100(root, train=False, download=True)
    names = data.classes
    groups = {"near": NEAR, "far": FAR}
    return data, {key: [i for i, label in enumerate(data.targets) if names[label] in group] for key, group in groups.items()}
