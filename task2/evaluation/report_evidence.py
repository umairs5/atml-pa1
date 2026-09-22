"""Create Task 2 training curves and deterministic final-analysis case summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MAIN_METHODS = ("source_only", "dan", "dann", "cdan")


def write_selected_cases(results_dir: Path) -> None:
    deltas = pd.read_csv(results_dir / "per_class_deltas.csv")
    confusions = pd.read_csv(results_dir / "dominant_target_confusions.csv")
    selected = []
    for method in MAIN_METHODS[1:]:
        method_rows = deltas[deltas["method"] == method]
        for rank, (_, row) in enumerate(method_rows.nsmallest(2, "change_from_source_only").iterrows(), 1):
            selected.append({"method": method, "selection": "largest_degradation", "rank": rank, **row.to_dict()})
        for rank, (_, row) in enumerate(method_rows.nlargest(1, "change_from_source_only").iterrows(), 1):
            selected.append({"method": method, "selection": "largest_improvement", "rank": rank, **row.to_dict()})
    output = pd.DataFrame(selected).merge(
        confusions,
        left_on=["method", "class_name"],
        right_on=["method", "true_class"],
        how="left",
        validate="one_to_one",
    )
    output.to_csv(results_dir / "selected_failure_cases.csv", index=False)


def plot_histories(results_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, method in zip(axes.flat, MAIN_METHODS):
        history = pd.read_csv(results_dir.parent / method / "history.csv")
        epoch = history["epoch"]
        axis.plot(epoch, history["classification_loss"], label="Classification loss", color="#4c78a8")
        axis.set_title(method.replace("_", " ").upper())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Classification loss")
        axis.grid(alpha=0.25)
        diagnostic = "alignment_loss" if "alignment_loss" in history else "domain_loss" if "domain_loss" in history else None
        if diagnostic:
            secondary = axis.twinx()
            secondary.plot(epoch, history[diagnostic], label=diagnostic.replace("_", " ").title(), color="#f58518")
            secondary.set_ylabel(diagnostic.replace("_", " ").title())
            lines, labels = axis.get_legend_handles_labels()
            extra_lines, extra_labels = secondary.get_legend_handles_labels()
            axis.legend(lines + extra_lines, labels + extra_labels, loc="best")
        else:
            axis.legend(loc="best")
    figure.suptitle("Task 2 training curves", fontsize=14)
    figure.savefig(results_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task2/results/final_evaluation")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    plot_histories(results_dir)
    write_selected_cases(results_dir)


if __name__ == "__main__":
    main()
