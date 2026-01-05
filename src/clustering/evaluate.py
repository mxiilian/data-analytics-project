from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class ClusterMetrics:
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    n_clusters: int
    min_cluster_size: int
    max_cluster_frac: float


def _safe_metric(fn, X: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(fn(X, labels))
    except Exception:
        return float("nan")


def compute_cluster_metrics(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    fit_predict_full: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> ClusterMetrics:
    """
    Computes quality metrics + a stability estimate.

    Stability definition:
    - Fit the model on the full X -> baseline labels
    - For each bootstrap resample: fit model on bootstrap sample, then predict labels for full X
    - Compute ARI between baseline labels and bootstrap model labels on full X
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n_clusters = int(len(uniq))

    # Cluster size stats
    counts = np.bincount(labels - labels.min()) if labels.size else np.array([], dtype=int)
    # Fallback if labels are not 0..k-1
    if counts.size == 0 or counts.sum() != labels.size:
        _, counts = np.unique(labels, return_counts=True)
    min_cluster_size = int(counts.min()) if counts.size else 0
    max_cluster_frac = float(counts.max() / counts.sum()) if counts.size else 0.0

    sil = float("nan")
    dbi = float("nan")
    ch = float("nan")
    if n_clusters >= 2 and X.shape[0] >= 3:
        sil = _safe_metric(silhouette_score, X, labels)
        dbi = _safe_metric(davies_bouldin_score, X, labels)
        ch = _safe_metric(calinski_harabasz_score, X, labels)

    # Stability
    rng = np.random.default_rng(seed)
    stability_scores: list[float] = []
    baseline = labels
    n = X.shape[0]
    if n >= 5 and n_clusters >= 2:
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n, endpoint=False)
            Xb = X[idx]
            try:
                boot_labels_full = fit_predict_full(Xb, X)
            except Exception:
                # Some models (e.g. GMM) can fail to fit on degenerate bootstraps.
                continue

            if boot_labels_full is None:
                continue
            boot_labels_full = np.asarray(boot_labels_full)
            if boot_labels_full.shape[0] != n:
                continue
            try:
                stability_scores.append(float(adjusted_rand_score(baseline, boot_labels_full)))
            except Exception:
                continue

    stability_ari = float(np.nanmean(stability_scores)) if stability_scores else float("nan")

    return ClusterMetrics(
        silhouette=sil,
        davies_bouldin=dbi,
        calinski_harabasz=ch,
        stability_ari=stability_ari,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        max_cluster_frac=max_cluster_frac,
    )


def objective_score(
    metrics: ClusterMetrics,
    *,
    min_cluster_size: int = 2,
    max_cluster_frac: float = 0.85,
) -> float:
    """
    Single scalar objective to maximize.
    """
    if metrics.n_clusters < 2:
        return -1e9
    if metrics.min_cluster_size < min_cluster_size:
        return -1e9
    if metrics.max_cluster_frac > max_cluster_frac:
        return -1e9

    sil = metrics.silhouette if np.isfinite(metrics.silhouette) else -1.0
    dbi = metrics.davies_bouldin if np.isfinite(metrics.davies_bouldin) else 10.0
    stab = metrics.stability_ari if np.isfinite(metrics.stability_ari) else 0.0
    # Keep it simple (from plan): maximize silhouette and stability, penalize DB index.
    return float(sil - 0.2 * dbi + 0.5 * stab)


