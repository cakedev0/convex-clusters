"""Alternatives to the hulls, built on the k-means of a fitted detector.

Each function returns a scorer mapping raw readings to a distance: the larger, the
more abnormal.
"""

import numpy as np


def centroid_scorer(detector):
    """Distance to the nearest k-means center."""

    def score(X):
        scaled = detector.scaler_.transform(X)
        return detector.clusters_.kmeans.transform(scaled).min(axis=1)

    return score


def _box_scorer(detector, boxes):
    lower = np.array([box[0] for box in boxes])
    upper = np.array([box[1] for box in boxes])

    def score(X):
        Q = detector.scaler_.transform(X)
        distance = np.full(len(Q), np.inf)
        for low, high in zip(lower, upper):
            excess = Q - np.clip(Q, low, high)
            distance = np.minimum(distance, np.linalg.norm(excess, axis=1))
        return distance

    return score


def box_scorer(detector):
    """Distance to the nearest axis-aligned box of a trimmed cluster."""
    return _box_scorer(
        detector,
        [(p.min(axis=0), p.max(axis=0)) for _, p in detector.trimmed_clusters()],
    )


def raw_box_scorer(detector):
    """Distance to the nearest box of a cluster, without trimming or dropping any.

    Zero means inside a box, so this can be used as is, without a threshold.
    """
    clusters = detector.clusters_
    boxes = []
    for c in np.flatnonzero(clusters.sizes > 0):
        points = clusters.X_scaled[clusters.labels == c]
        boxes.append((points.min(axis=0), points.max(axis=0)))
    return _box_scorer(detector, boxes)


def mahalanobis_scorer(detector):
    """Mahalanobis distance to the nearest trimmed cluster."""
    means, precisions = [], []
    for _, points in detector.trimmed_clusters():
        means.append(points.mean(axis=0))
        precisions.append(np.linalg.inv(np.cov(points.T)))

    def score(X):
        Q = detector.scaler_.transform(X)
        squared = np.full(len(Q), np.inf)
        for mean, precision in zip(means, precisions):
            residual = Q - mean
            squared = np.minimum(squared, np.sum((residual @ precision) * residual, axis=1))
        return np.sqrt(squared)

    return score
