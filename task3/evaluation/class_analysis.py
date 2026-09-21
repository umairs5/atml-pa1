"""Summarize per-class Sketch changes relative to the shared ERM baseline."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task3/results/final_evaluation")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    with (results_dir / "per_class_accuracy.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = {row["class_name"]: float(row["sketch_accuracy"]) for row in rows if row["method"] == "erm"}
    if not baseline:
        raise ValueError("per_class_accuracy.csv does not contain ERM results.")
    summary = [
        {**row, "change_from_erm": float(row["sketch_accuracy"]) - baseline[row["class_name"]]}
        for row in rows
    ]
    with (results_dir / "per_class_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
