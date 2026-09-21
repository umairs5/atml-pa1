"""Source-domain probe used to measure observed-domain invariance."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def source_domain_separability(features_by_domain: dict[str, np.ndarray], seed: int = 6304) -> float:
    """Fit the required balanced three-way linear source-domain probe."""
    count = min(len(features) for features in features_by_domain.values())
    if count < 2:
        raise ValueError("Each source domain needs at least two validation features.")
    rng = np.random.default_rng(seed)
    features, domains = [], []
    for label, domain in enumerate(features_by_domain):
        domain_features = features_by_domain[domain]
        selected = rng.choice(len(domain_features), size=count, replace=False)
        features.append(domain_features[selected])
        domains.append(np.full(count, label, dtype=int))
    train_features, test_features, train_domains, test_domains = train_test_split(
        np.concatenate(features), np.concatenate(domains), test_size=0.30,
        random_state=seed, stratify=np.concatenate(domains)
    )
    probe = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial", random_state=seed)
    probe.fit(train_features, train_domains)
    return float(probe.score(test_features, test_domains))
