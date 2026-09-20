"""Data loader construction for domain-balanced PACS experiments."""

from __future__ import annotations

from itertools import cycle

from torch.utils.data import DataLoader

from shared.pacs import PACSLabeledDataset, PACSUnlabeledDataset
from task2.transforms import evaluation_transform, training_transform


def build_training_loaders(config: dict, protocol: dict) -> tuple[dict[str, DataLoader], DataLoader]:
    """Build one shuffled source loader per domain and one target loader."""
    data_config = config["data"]
    training_config = config["training"]
    train_transform = training_transform(data_config["resize_size"], data_config["crop_size"])
    source_loaders = {}
    for domain in data_config["source_domains"]:
        dataset = PACSLabeledDataset(
            data_config["root"],
            protocol["source"][domain]["train"],
            transform=train_transform,
        )
        source_loaders[domain] = DataLoader(
            dataset,
            batch_size=training_config["source_per_domain"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
    target_dataset = PACSUnlabeledDataset(
        data_config["root"], protocol["target"]["paths"], transform=train_transform
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=training_config["target_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    return source_loaders, target_loader


def cycle_loaders(loaders: dict[str, DataLoader], target_loader: DataLoader):
    """Yield 8 examples from every source and 24 target examples on each update."""
    source_iterators = {domain: cycle(loader) for domain, loader in loaders.items()}
    target_iterator = cycle(target_loader)
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    for _ in range(steps_per_epoch):
        source_batches = {domain: next(iterator) for domain, iterator in source_iterators.items()}
        yield source_batches, next(target_iterator)


def build_validation_loaders(config: dict, protocol: dict) -> dict[str, DataLoader]:
    """Build deterministic source validation loaders, one for each source domain."""
    data_config = config["data"]
    transform = evaluation_transform(data_config["resize_size"], data_config["crop_size"])
    return {
        domain: DataLoader(
            PACSLabeledDataset(
                data_config["root"],
                protocol["source"][domain]["validation"],
                transform=transform,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )
        for domain in data_config["source_domains"]
    }
