"""Source-only PACS data loaders for domain generalization."""

from itertools import cycle

import torch
from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset
from task2.transforms import evaluation_transform, training_transform


def _loader_options(config: dict) -> dict:
    workers = config["data"]["num_workers"]
    return {"num_workers": workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": workers > 0}


def build_training_loaders(config: dict, protocol: dict) -> dict[str, DataLoader]:
    """Create balanced labeled loaders without ever loading the Sketch domain."""
    transform = training_transform(config["data"]["resize_size"], config["data"]["crop_size"])
    return {
        domain: DataLoader(
            PACSLabeledDataset(config["data"]["root"], protocol["source"][domain]["train"], transform),
            batch_size=config["training"]["source_per_domain"], shuffle=True, drop_last=True, **_loader_options(config)
        )
        for domain in config["data"]["source_domains"]
    }


def build_validation_loaders(config: dict, protocol: dict) -> dict[str, DataLoader]:
    """Create deterministic validation loaders for the observed source domains."""
    transform = evaluation_transform(config["data"]["resize_size"], config["data"]["crop_size"])
    return {
        domain: DataLoader(
            PACSLabeledDataset(config["data"]["root"], protocol["source"][domain]["validation"], transform),
            batch_size=64, shuffle=False, **_loader_options(config)
        )
        for domain in config["data"]["source_domains"]
    }


def cycle_source_loaders(loaders: dict[str, DataLoader]):
    """Yield one eight-image batch from every source domain per optimization step."""
    iterators = {domain: cycle(loader) for domain, loader in loaders.items()}
    for _ in range(max(len(loader) for loader in loaders.values())):
        yield {domain: next(iterator) for domain, iterator in iterators.items()}
