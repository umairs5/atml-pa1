"""The domain-balanced source-only ERM objective."""

import torch.nn.functional as functional


def source_only_loss(source_logits, source_labels):
    """Compute classification loss from the concatenated balanced source batch."""
    return functional.cross_entropy(source_logits, source_labels)
