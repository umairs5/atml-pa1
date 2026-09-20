"""Train one Task 2 method under the fixed PACS protocol."""

from __future__ import annotations

import argparse

import torch

from task2.config import load_config
from task2.methods.source_only import source_only_loss
from task2.models.backbone import PACSResNet18
from task2.training import run_training


def source_only_update(model, optimizer, source_images, source_labels, target_images, progress):
    """Perform one source-only update while retaining the common batch schedule."""
    del target_images, progress
    optimizer.zero_grad()
    _, source_logits = model(source_images)
    classification = source_only_loss(source_logits, source_labels)
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
        raise NotImplementedError("This entry point currently supports source_only only.")
    model = PACSResNet18(num_classes=len(config["data"]["classes"])).to(args.device)
    checkpoint = run_training(config, model, source_only_update, torch.device(args.device))
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
