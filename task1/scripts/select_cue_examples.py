"""Choose diverse cue-conflict examples for qualitative inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


METHODS = [
    ("resnet50", "linear_head", "resnet50"),
    ("vit_b_16", "linear_head", "vit_b_16"),
    ("clip_vit_b_32", "linear_head", "clip_linear"),
    ("clip_vit_b_32", "zero_shot", "clip_zero_shot"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    predictions = pd.read_csv(config["output"]["cue_predictions_path"])
    wide = None
    for backbone, classifier, name in METHODS:
        subset = predictions[(predictions.backbone == backbone) & (predictions.classifier == classifier)]
        subset = subset[["candidate_id", "direction", "content_class", "style_class", "predicted_class", "prediction_role", "maximum_confidence"]].rename(
            columns={
                "predicted_class": f"{name}_prediction",
                "prediction_role": f"{name}_role",
                "maximum_confidence": f"{name}_confidence",
            }
        )
        wide = subset if wide is None else wide.merge(subset, on=["candidate_id", "direction", "content_class", "style_class"])
    wide["role_disagreement"] = wide[[f"{name}_role" for _, _, name in METHODS]].nunique(axis=1)
    choices = []
    for direction, group in wide.groupby("direction", sort=True):
        choices.append(group.sort_values(["role_disagreement", "resnet50_confidence"], ascending=[False, False]).iloc[0])
    selected = pd.DataFrame(choices).sort_values("direction")
    selected.to_csv(config["output"]["cue_example_manifest_path"], index=False)
    print(f"Saved {len(selected)} cue-conflict examples to {config['output']['cue_example_manifest_path']}")


if __name__ == "__main__":
    main()
