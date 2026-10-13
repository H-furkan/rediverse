"""Pairwise-cosine diversity on cached embeddings (numpy only)."""
from __future__ import annotations

import numpy as np


def l2n(x, axis=-1):
    """Rows scaled to unit L2 norm (norm floored at 1e-12)."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def pair_mean_cos(E):
    """Mean over i<j of cos(E_i, E_j) for row-normalised E [R, D]."""
    R = E.shape[0]; G = E @ E.T
    return float((G.sum() - np.trace(G)) / (R * (R - 1)))


def cos_diversity(E):
    """1 - mean pairwise cosine of the rows of E (L2-normalised here)."""
    return 1.0 - pair_mean_cos(l2n(E))
