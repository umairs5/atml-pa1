"""Create Task 3 comparison figures from final evaluation CSV files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task3/results/final_evaluation")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    metrics = pd.read_csv(results_dir / "metrics.csv").set_index("method")

    methods = ["erm", "dan_dg", "sam"]
    main = metrics.loc[methods]
    labels = ["ERM", "DAN-DG", "SAM"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, column, title, ylabel in zip(
        axes,
        ["mean_source_validation_macro_f1", "worst_source_validation_macro_f1", "sketch_accuracy"],
        ["Mean source macro-F1", "Worst-source macro-F1", "Sketch accuracy"],
        ["Macro-F1", "Macro-F1", "Accuracy"],
    ):
        bars = axis.bar(labels, main[column], color="#4c78a8")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        for bar, value in zip(bars, main[column]):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    figure.savefig(results_dir / "main_method_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    study = metrics.loc[["dan_dg_lambda_0_1", "dan_dg", "dan_dg_lambda_10"]]
    lambdas = [0.1, 1.0, 10.0]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    axes[0].plot(lambdas, study["mean_source_validation_macro_f1"], marker="o", label="Mean source macro-F1")
    axes[0].plot(lambdas, study["sketch_accuracy"], marker="o", label="Sketch accuracy")
    axes[1].plot(lambdas, study["source_domain_separability"], marker="o", label="Source-domain separability")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(lambdas, ["0.1", "1", "10"])
        axis.set_xlabel("DAN-DG strength λ")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Probe accuracy")
    axes[0].set_title("DAN-DG strength study")
    axes[1].set_title("Observed-source alignment")
    figure.savefig(results_dir / "dan_dg_lambda_study.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
