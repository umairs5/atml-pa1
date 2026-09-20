"""Create the Task 2 comparison figures from final evaluation CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def labeled_bars(axis, labels, values, title, ylabel):
    bars = axis.bar(labels, values, color="#4c78a8")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.05)
    axis.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task2/results/final_evaluation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = pd.read_csv(results_dir / "metrics.csv")

    main_methods = ["source_only", "dan", "dann", "cdan"]
    main_results = metrics.set_index("method").loc[main_methods].reset_index()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    labels = main_results["method"].str.upper().str.replace("_", " ")
    labeled_bars(axes[0], labels, main_results["mean_source_validation_macro_f1"], "Source validation macro-F1", "Macro-F1")
    labeled_bars(axes[1], labels, main_results["target_accuracy"], "Sketch target accuracy", "Accuracy")
    labeled_bars(axes[2], labels, main_results["domain_separability_accuracy"], "Domain separability", "Probe accuracy")
    figure.savefig(results_dir / "main_method_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    study_methods = ["dan_lambda_0_1", "dan", "dan_lambda_10"]
    study = metrics.set_index("method").loc[study_methods].reset_index()
    lambdas = [0.1, 1.0, 10.0]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    axes[0].plot(lambdas, study["mean_source_validation_macro_f1"], marker="o", label="Source validation macro-F1")
    axes[0].plot(lambdas, study["target_accuracy"], marker="o", label="Sketch target accuracy")
    axes[1].plot(lambdas, study["domain_separability_accuracy"], marker="o", color="#f58518", label="Domain separability")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(lambdas, ["0.1", "1", "10"])
        axis.set_xlabel("MMD strength λ")
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Probe accuracy")
    axes[0].set_title("Performance across DAN strengths")
    axes[1].set_title("Alignment diagnostic across DAN strengths")
    figure.savefig(results_dir / "dan_lambda_study.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
