"""Post-hoc unknownness scores; larger values mean more novel."""
from __future__ import annotations
import torch
def msp(logits): return 1 - logits.softmax(1).amax(1)
def mls(logits): return -logits.amax(1)
def energy(logits): return -torch.logsumexp(logits, 1)
def mahalanobis(features, means, diagonal_variance):
    return ((features[:, None] - means[None]) ** 2 / diagonal_variance[None, None]).sum(2).amin(1)
