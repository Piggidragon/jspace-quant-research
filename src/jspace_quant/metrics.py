"""Metrics and statistics (pure numpy, no model access — unit-tested in tests/)."""

from __future__ import annotations

import numpy as np


def kl_divergence(logp_ref: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """KL(P_ref || P) per row, in nats. Both arguments are log-probabilities [n, V]."""
    return (np.exp(logp_ref) * (logp_ref - logp)).sum(axis=-1)


def topk_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of common tokens of two top-k lists [n, k] (order ignored)."""
    k = a.shape[1]
    return np.array([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, b)])


def pass_at_k(min_ranks: list[np.ndarray], k: int) -> float:
    """The paper's metric (data/evaluations/README.md): mean over items of the share of
    intermediate concepts whose best rank across layers is ≤ k.

    min_ranks[i] holds, for each intermediate concept of item i, its rank minimized over layers.
    """
    return float(np.mean([np.mean(r <= k) for r in min_ranks]))


def bootstrap_ci(values: np.ndarray, *, n: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over the first axis."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, *, n: int, seed: int,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap over clusters (e.g. paragraphs), since positions of one paragraph are correlated."""
    values = np.asarray(values, dtype=np.float64)
    ids = np.unique(clusters)
    groups = [values[clusters == c] for c in ids]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def workspace_band(hit_rate_per_layer: np.ndarray, *, frac: float = 0.5, exclude_last: bool = True) -> tuple[int, int]:
    """Workspace band: contiguous layer range around the peak hit rate in which the rate is
    ≥ frac · peak. The last block (= output) is excluded.

    Returns the (first, last) layer, inclusive.
    """
    r = np.asarray(hit_rate_per_layer, dtype=np.float64)
    if exclude_last:
        r = r[:-1]
    peak = int(r.argmax())
    thr = frac * r[peak]
    lo = peak
    while lo > 0 and r[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < len(r) - 1 and r[hi + 1] >= thr:
        hi += 1
    return lo, hi


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) of two activation matrices [n, d] over the same n
    positions; 1 = same geometry up to rotation and isotropic scaling."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    xty = np.linalg.norm(x.T @ y) ** 2
    return float(xty / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))
