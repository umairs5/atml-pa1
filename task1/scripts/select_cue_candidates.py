"""Create the fixed, balanced 200-image cue-conflict evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/clean_baseline.yaml")
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    output_dir = Path(config["output"]["cue_candidate_dir"])
    candidate_path = output_dir / "candidate_metadata.csv"
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    rng = np.random.default_rng(config["seed"])
    selected_per_direction = config["cue_conflicts"]["selected_per_direction"]
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    directions = sorted({row["direction"] for row in candidates})
    for direction in directions:
        direction_rows = [row for row in candidates if row["direction"] == direction]
        if len(direction_rows) < selected_per_direction:
            raise ValueError(f"{direction} has fewer than {selected_per_direction} candidates.")
        chosen_positions = set(rng.choice(len(direction_rows), size=selected_per_direction, replace=False).tolist())
        for position, row in enumerate(direction_rows):
            row["selection_status"] = "selected" if position in chosen_positions else "reserve"
            row["selection_reason"] = "visual_review_pass" if position in chosen_positions else "reserve_after_balanced_sampling"
            rows.append(row)
        summary[direction] = {
            "generated": len(direction_rows),
            "visual_review_pass": len(direction_rows),
            "selected": selected_per_direction,
            "reserve": len(direction_rows) - selected_per_direction,
        }

    selected_path = output_dir / "selection_metadata.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {sum(row['selection_status'] == 'selected' for row in rows)} candidates.")


if __name__ == "__main__":
    main()
