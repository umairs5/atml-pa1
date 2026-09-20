"""Label-free diagnostic for source-versus-target feature separability."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def domain_separability_accuracy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int = 6304,
) -> float:
    """Fit the specified balanced linear domain probe on equal feature counts."""
    count = min(len(source_features), len(target_features))
    if count < 2:
        raise ValueError("At least two source and target features are required.")
    rng = np.random.default_rng(seed)
    source = source_features[rng.choice(len(source_features), size=count, replace=False)]
    target = target_features[rng.choice(len(target_features), size=count, replace=False)]
    features = np.concatenate([source, target])
    domains = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
    train_features, test_features, train_domains, test_domains = train_test_split(
        features,
        domains,
        test_size=0.30,
        random_state=seed,
        stratify=domains,
    )
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    classifier.fit(train_features, train_domains)
    return float(classifier.score(test_features, test_domains))
