"""Train one Task 2 method under the fixed PACS protocol."""

from __future__ import annotations

import argparse
from functools import partial

import torch

from task2.config import load_config
from task2.methods.cdan import cdan_update
from task2.methods.dan import dan_update
from task2.methods.dann import dann_update
from task2.methods.source_only import source_only_loss
from task2.models.backbone import PACSResNet18
from task2.models.domain_discriminator import DomainDiscriminator
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
    model = PACSResNet18(num_classes=len(config["data"]["classes"])).to(args.device)
    method = config["method"]
    method_name = method.get("base_method", method["name"])
    extra_modules = ()

    if method_name == "source_only":
        update_step = source_only_update
    elif method_name == "dan":
        update_step = partial(dan_update, mmd_lambda=method["mmd_lambda"])
    elif method_name == "dann":
        discriminator = DomainDiscriminator(
            model.feature_dimension,
            method["discriminator_hidden_dim"],
            method["discriminator_dropout"],
        ).to(args.device)
        update_step = partial(
            dann_update,
            discriminator=discriminator,
            maximum_grl_strength=method["max_grl_strength"],
        )
        extra_modules = (discriminator,)
    elif method_name == "cdan":
        discriminator = DomainDiscriminator(
            model.feature_dimension * len(config["data"]["classes"]),
            method["discriminator_hidden_dim"],
            method["discriminator_dropout"],
        ).to(args.device)
        update_step = partial(
            cdan_update,
            discriminator=discriminator,
            maximum_grl_strength=method["max_grl_strength"],
        )
        extra_modules = (discriminator,)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    checkpoint = run_training(
        config,
        model,
        update_step,
        torch.device(args.device),
        extra_modules=extra_modules,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
