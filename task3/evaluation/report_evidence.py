"""Create Task 3 curves, selected failure cases, and a cross-task integrity record."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")


def dominant_confusions(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_sketch_confusion.csv")):
        method = path.name.removesuffix("_sketch_confusion.csv")
        matrix = np.loadtxt(path, delimiter=",", dtype=int)
        for label, class_name in enumerate(CLASSES):
            support = int(matrix[label].sum())
            wrong = matrix[label].copy()
            wrong[label] = 0
            prediction = int(wrong.argmax())
            rows.append(
                {
                    "method": method,
                    "true_class": class_name,
                    "most_common_wrong_prediction": CLASSES[prediction],
                    "confusion_count": int(wrong[prediction]),
                    "true_class_support": support,
                    "confusion_rate": int(wrong[prediction]) / support if support else 0.0,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(results_dir / "dominant_sketch_confusions.csv", index=False)
    return output


def selected_cases(results_dir: Path, confusions: pd.DataFrame) -> None:
    deltas = pd.read_csv(results_dir / "per_class_deltas.csv")
    selected = []
    for method in ("dan_dg", "sam"):
        method_rows = deltas[deltas["method"] == method]
        for rank, (_, row) in enumerate(method_rows.nsmallest(2, "change_from_erm").iterrows(), 1):
            selected.append({"method": method, "selection": "largest_degradation", "rank": rank, **row.to_dict()})
        for rank, (_, row) in enumerate(method_rows.nlargest(2, "change_from_erm").iterrows(), 1):
            selected.append({"method": method, "selection": "largest_improvement", "rank": rank, **row.to_dict()})
    output = pd.DataFrame(selected).merge(
        confusions,
        left_on=["method", "class_name"],
        right_on=["method", "true_class"],
        how="left",
        validate="one_to_one",
    )
    output.to_csv(results_dir / "selected_failure_cases.csv", index=False)


def plot_histories(results_dir: Path, erm_history: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    specifications = (
        ("ERM (reused Task 2 checkpoint)", pd.read_csv(erm_history), None),
        ("DAN-DG", pd.read_csv(results_dir.parent / "dan_dg" / "history.csv"), "alignment_loss"),
        ("SAM", pd.read_csv(results_dir.parent / "sam" / "history.csv"), "perturbed_classification_loss"),
    )
    for axis, (title, history, diagnostic) in zip(axes, specifications):
        epoch = history["epoch"]
        axis.plot(epoch, history["classification_loss"], label="Classification loss", color="#4c78a8")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Classification loss")
        axis.grid(alpha=0.25)
        if diagnostic:
            secondary = axis.twinx()
            label = "MMD penalty" if diagnostic == "alignment_loss" else "Perturbed classification loss"
            secondary.plot(epoch, history[diagnostic], label=label, color="#f58518")
            secondary.set_ylabel(label)
            lines, labels = axis.get_legend_handles_labels()
            extra_lines, extra_labels = secondary.get_legend_handles_labels()
            axis.legend(lines + extra_lines, labels + extra_labels, loc="best")
        else:
            axis.legend(loc="best")
    figure.suptitle("Task 3 training curves", fontsize=14)
    figure.savefig(results_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def comparison_integrity_record(results_dir: Path, task2_metrics: Path, task2_erm_sha256: str, task3_erm_sha256: str) -> None:
    task2 = pd.read_csv(task2_metrics).set_index("method").loc["dan"]
    task3 = pd.read_csv(results_dir / "metrics.csv").set_index("method").loc["dan_dg"]
    shared = task2_erm_sha256.lower() == task3_erm_sha256.lower()
    rows = []
    for metric, task2_column, task3_column in (
        ("mean_source_validation_macro_f1", "mean_source_validation_macro_f1", "mean_source_validation_macro_f1"),
        ("Sketch accuracy", "target_accuracy", "sketch_accuracy"),
        ("Sketch macro-F1", "target_macro_f1", "sketch_macro_f1"),
    ):
        rows.append(
            {
                "metric": metric,
                "task2_target_aware_dan": task2[task2_column],
                "task3_target_free_dan_dg": task3[task3_column],
                "task3_minus_task2": task3[task3_column] - task2[task2_column],
                "task2_erm_checkpoint_sha256": task2_erm_sha256,
                "task3_erm_checkpoint_sha256": task3_erm_sha256,
                "shared_erm_checkpoint": shared,
                "comparison_status": "ready" if shared else "blocked_checkpoint_mismatch",
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "task2_dan_vs_task3_dan_dg.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task3/results/final_evaluation")
    parser.add_argument("--erm-history", default="task2/results/source_only/history.csv")
    parser.add_argument("--task2-metrics", default="task2/results/final_evaluation/metrics.csv")
    parser.add_argument("--task2-erm-sha256", required=True)
    parser.add_argument("--task3-erm-sha256", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    plot_histories(results_dir, Path(args.erm_history))
    confusions = dominant_confusions(results_dir)
    selected_cases(results_dir, confusions)
    comparison_integrity_record(results_dir, Path(args.task2_metrics), args.task2_erm_sha256, args.task3_erm_sha256)


if __name__ == "__main__":
    main()
