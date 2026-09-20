"""Summarize per-class target changes relative to Source-only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task2/results/final_evaluation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    with (results_dir / "per_class_accuracy.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = {
        row["class_name"]: float(row["target_accuracy"])
        for row in rows
        if row["method"] == "source_only"
    }
    if not baseline:
        raise ValueError("per_class_accuracy.csv does not contain a source_only row.")

    summary = []
    for row in rows:
        accuracy = float(row["target_accuracy"])
        summary.append(
            {
                "method": row["method"],
                "class_name": row["class_name"],
                "target_accuracy": accuracy,
                "change_from_source_only": accuracy - baseline[row["class_name"]],
                "support": row["support"],
            }
        )
    with (results_dir / "per_class_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
