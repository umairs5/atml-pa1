"""Create Task 4 figures and failure tables from frozen evaluation caches."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from torchvision import datasets

from task4.scores import energy, mahalanobis, mls, msp


CIFAR10_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
PLAUSIBLE_NEAR = {
    "bus": {"truck"}, "pickup_truck": {"truck"},
    "motorcycle": {"automobile"}, "tractor": {"truck"},
    "wolf": {"dog"}, "fox": {"dog"}, "leopard": {"cat"},
    "camel": {"horse"},
}


def vanilla_statistics(cache: Path):
    known = np.load(cache / "vanilla_known_outputs.npz")
    train_features, train_labels = known["train_features"], known["train_labels"]
    means = np.stack([train_features[train_labels == label].mean(0) for label in range(10)])
    residuals = train_features - means[train_labels]
    variance = residuals.var(0) + 1e-6
    return known, means, variance


def scores(logits, features, means, variance):
    import torch

    logits_t, features_t = torch.tensor(logits), torch.tensor(features)
    return {
        "MSP": msp(logits_t).numpy(),
        "MLS": mls(logits_t).numpy(),
        "Mahalanobis": mahalanobis(features_t, torch.tensor(means), torch.tensor(variance)).numpy(),
        "Energy": energy(logits_t).numpy(),
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def roc_figure(results: Path, cache: Path):
    known, means, variance = vanilla_statistics(cache)
    near, far = np.load(cache / "vanilla_near_outputs.npz"), np.load(cache / "vanilla_far_outputs.npz")
    known_scores = scores(known["test_logits"][:, :10], known["test_features"], means, variance)
    unknown_scores = scores(
        np.r_[near["logits"][:, :10], far["logits"][:, :10]],
        np.r_[near["features"], far["features"]], means, variance,
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    labels = np.r_[np.zeros(len(known_scores["MSP"])), np.ones(len(unknown_scores["MSP"]))]
    for axis, name in zip(axes, ("MSP", "MLS", "Mahalanobis")):
        fpr, tpr, _ = roc_curve(labels, np.r_[known_scores[name], unknown_scores[name]])
        axis.plot(fpr, tpr, lw=2, label=f"AUROC = {auc(fpr, tpr):.3f}")
        axis.plot([0, 1], [0, 1], "--", color="0.55", lw=1)
        axis.set(title=name, xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
        axis.legend(loc="lower right", frameon=False)
    figure.savefig(results / "vanilla_posthoc_roc.png", dpi=220)
    plt.close(figure)


def failure_cases(results: Path, cache: Path, data_root: str):
    known, means, variance = vanilla_statistics(cache)
    threshold = float(np.quantile(scores(known["validation_logits"][:, :10], known["validation_features"], means, variance)["MLS"], 0.95))
    cifar100_names = datasets.CIFAR100(data_root, train=False, download=False).classes
    rows = []
    for group in ("near", "far"):
        unknown = np.load(cache / f"vanilla_{group}_outputs.npz")
        unknownness = scores(unknown["logits"][:, :10], unknown["features"], means, variance)["MLS"]
        accepted = np.flatnonzero(unknownness <= threshold)
        # Lowest unknownness means the most confident incorrect acceptance.
        for index in accepted[np.argsort(unknownness[accepted])[:3]]:
            unknown_name = cifar100_names[int(unknown["labels"][index])]
            prediction = CIFAR10_NAMES[int(unknown["logits"][index, :10].argmax())]
            plausible = group == "near" and prediction in PLAUSIBLE_NEAR.get(unknown_name, set())
            rows.append({
                "group": group,
                "unknown_class": unknown_name,
                "predicted_cifar10_class": prediction,
                "mls_unknownness": float(unknownness[index]),
                "vanilla_mls_threshold": threshold,
                "semantic_relation": "plausible" if plausible else "surprising",
            })
    write_csv(results / "vanilla_mls_accepted_failures.csv", rows)


def comparison_tables(results: Path):
    with (results / "osr_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    posthoc = [row for row in metrics if row["method"] == "vanilla" and row["score"] in {"msp", "mls", "energy", "mahalanobis"}]
    trained = [row for row in metrics if row["score"] in {"mls", "proser_placeholder"}]
    write_csv(results / "vanilla_posthoc_score_comparison.csv", posthoc)
    write_csv(results / "trained_model_comparison.csv", trained)


def training_curves(results: Path, root: Path):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    for method in ("vanilla", "gcsc", "proser"):
        with (root / method / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in history]
        loss_key = "total_loss" if method == "proser" else "train_loss"
        axes[0].plot(epochs, [float(row[loss_key]) for row in history], label=method)
        axes[1].plot(epochs, [float(row["validation_accuracy"]) for row in history], label=method)
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Known validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend(frameon=False)
    figure.savefig(results / "training_curves.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="task4/results/final_evaluation")
    parser.add_argument("--cache-dir", default="task4/cache")
    parser.add_argument("--data-root", default="data/cifar")
    parser.add_argument("--training-results", default="task4/results")
    args = parser.parse_args()
    results, cache = Path(args.results_dir), Path(args.cache_dir)
    results.mkdir(parents=True, exist_ok=True)
    roc_figure(results, cache)
    failure_cases(results, cache, args.data_root)
    comparison_tables(results)
    training_curves(results, Path(args.training_results))


if __name__ == "__main__":
    main()
