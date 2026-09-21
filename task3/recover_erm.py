"""Recover the missing Task 2 ERM checkpoint without loading Sketch.

This is only for a lost checkpoint: it preserves the Task 2 Source-only
configuration, initialization, augmentation, optimizer, source batches, and
source-validation selection rule while deliberately using Task 3's loaders.
Unlike the Task 2 shared adaptation loop, those loaders never construct a
Sketch dataset or batch.
"""

import argparse

import torch
import torch.nn.functional as functional

from task2.config import load_config
from task3.models.backbone import PACSResNet18
from task3.training import run_training


def source_only_update(model, optimizer, source_batches: dict, device: torch.device) -> dict[str, float]:
    """Apply the Task 2 ERM cross-entropy update to one balanced source batch."""
    optimizer.zero_grad()
    logits, labels = [], []
    for images, batch_labels in source_batches.values():
        _, batch_logits = model(images.to(device, non_blocking=True))
        logits.append(batch_logits)
        labels.append(batch_labels.to(device, non_blocking=True))
    classification = functional.cross_entropy(torch.cat(logits), torch.cat(labels))
    classification.backward()
    optimizer.step()
    return {"classification_loss": classification.detach().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task2/configs/source_only.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["method"]["name"] != "source_only":
        raise ValueError("ERM recovery requires task2/configs/source_only.yaml.")
    model = PACSResNet18(num_classes=len(config["data"]["classes"])).to(args.device)
    checkpoint = run_training(config, model, source_only_update, torch.device(args.device))
    print(f"Recovered ERM checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
