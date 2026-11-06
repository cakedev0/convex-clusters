"""Normalization and k-means, i.e. what happens before the part we want to POC."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# All are per-sensor and affine, hence preserve the shape of distributions and the
# convexity of sets. See report.md for their properties.
SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "quantile_range": lambda: RobustScaler(quantile_range=(0.1, 99.9)),
}


@dataclass
class ClusteringResult:
    scaler: object
    kmeans: KMeans
    X_scaled: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray  # (n_clusters,), number of training samples per cluster


def fit_clusters(X, n_clusters=100, scaling="minmax", seed=0):
    """Normalize `X` and cluster it. Outliers are included in both steps."""
    scaler = SCALERS[scaling]()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(X_scaled)
    labels = kmeans.labels_
    return ClusteringResult(
        scaler=scaler,
        kmeans=kmeans,
        X_scaled=X_scaled,
        labels=labels,
        sizes=np.bincount(labels, minlength=n_clusters),
    )
