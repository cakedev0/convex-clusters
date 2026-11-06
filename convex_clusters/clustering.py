"""Normalization and k-means, i.e. what happens before the part we want to POC."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    scaler: StandardScaler
    kmeans: KMeans
    X_scaled: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray  # (n_clusters,), number of training samples per cluster


def fit_clusters(X, n_clusters=100, seed=0):
    """Standardize `X` and cluster it. Outliers are included in both steps."""
    scaler = StandardScaler()
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
