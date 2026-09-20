"""Find the most common incorrect prediction for each target class."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task2/results/final_evaluation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for path in sorted(results_dir.glob("*_target_confusion.csv")):
        method = path.name.removesuffix("_target_confusion.csv")
        matrix = np.loadtxt(path, delimiter=",", dtype=int)
        for label, class_name in enumerate(DEFAULT_CLASSES):
            support = int(matrix[label].sum())
            incorrect = matrix[label].copy()
            incorrect[label] = 0
            predicted_label = int(incorrect.argmax())
            count = int(incorrect[predicted_label])
            rows.append(
                {
                    "method": method,
                    "true_class": class_name,
                    "most_common_wrong_prediction": DEFAULT_CLASSES[predicted_label],
                    "confusion_count": count,
                    "true_class_support": support,
                    "confusion_rate": count / support if support else 0.0,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No target confusion matrices found in {results_dir}")
    with (results_dir / "dominant_target_confusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
