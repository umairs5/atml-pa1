"""Train a Task 3 source-only domain-generalization method."""

import argparse
from functools import partial

import torch

from task3.config import load_config
from task3.methods.dan_dg import dan_dg_update
from task3.methods.sam import sam_update
from task3.models.backbone import PACSResNet18
from task3.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    method = config["method"]
    method_name = method.get("base_method", method["name"])
    if method_name not in {"dan_dg", "sam"}:
        raise ValueError("ERM is the fixed Task 2 checkpoint and must not be retrained.")
    model = PACSResNet18(num_classes=len(config["data"]["classes"])).to(args.device)
    if method_name == "dan_dg":
        update_step = partial(dan_dg_update, mmd_lambda=method["mmd_lambda"])
    else:
        update_step = partial(sam_update, rho=method["rho"])
    checkpoint = run_training(config, model, update_step, torch.device(args.device))
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
