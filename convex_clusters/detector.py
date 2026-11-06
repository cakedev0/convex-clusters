"""Training and prediction pipeline: alert when a reading is far from every hull."""

import numpy as np
from scipy.spatial.distance import cdist

from convex_clusters.clustering import fit_clusters
from convex_clusters.hull import distance_to_hull, select_vertices


class ConvexClusterDetector:
    """Raise an alert for readings outside the hulls of the k-means clusters.

    Parameters
    ----------
    n_clusters : int
        Number of k-means clusters.
    n_directions : int
        Random directions per cluster; a hull has at most twice as many vertices.
    min_cluster_size : int
        Smaller clusters get no hull: they are made of measurement errors.
    trim_quantile : float
        Within a cluster, only this fraction of the points closest to its center
        can be hull vertices. Extreme points are precisely where the measurement
        errors that landed in a large cluster are, and a single one would stretch
        the hull towards it.
    target_alert_rate : float
        Fraction of the calibration data above the alert threshold. As that data
        is contaminated too, this must be above the rate of measurement errors.
    calibration_fraction : float
        Fraction of the training data held out to set the threshold. The hulls
        can't be used on the data they are built from: these points are inside by
        construction, whereas new points of a healthy plant may not be.
    random_state : int
        Seed of the split, of k-means and of the directions.
    """

    def __init__(
        self,
        n_clusters=100,
        n_directions=100,
        min_cluster_size=100,
        trim_quantile=0.99,
        target_alert_rate=0.01,
        calibration_fraction=0.2,
        random_state=0,
    ):
        self.n_clusters = n_clusters
        self.n_directions = n_directions
        self.min_cluster_size = min_cluster_size
        self.trim_quantile = trim_quantile
        self.target_alert_rate = target_alert_rate
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        permutation = rng.permutation(len(X))
        n_calibration = round(self.calibration_fraction * len(X))
        X_calibration = X[permutation[:n_calibration]]
        X_train = X[permutation[n_calibration:]]

        clusters = fit_clusters(X_train, self.n_clusters, seed=self.random_state)
        self.clusters_ = clusters
        self.scaler_ = clusters.scaler

        self.centers_, self.hulls_ = [], []
        for center, points in self.trimmed_clusters():
            self.centers_.append(center)
            self.hulls_.append(select_vertices(points, self.n_directions, rng))
        self.centers_ = np.array(self.centers_)
        # Each hull fits in a ball around its center, which allows skipping it.
        self.radii_ = np.array(
            [
                np.linalg.norm(V - center, axis=1).max()
                for V, center in zip(self.hulls_, self.centers_)
            ]
        )

        distances = self.hull_distance(X_calibration)
        self.threshold_ = np.quantile(distances, 1 - self.target_alert_rate)
        return self

    def trimmed_clusters(self):
        """Yield the center and the trimmed points of each cluster large enough."""
        clusters = self.clusters_
        for c in np.flatnonzero(clusters.sizes >= self.min_cluster_size):
            center = clusters.kmeans.cluster_centers_[c]
            points = clusters.X_scaled[clusters.labels == c]
            distances = np.linalg.norm(points - center, axis=1)
            yield center, points[distances <= np.quantile(distances, self.trim_quantile)]

    def hull_distance(self, X):
        """Distance of each reading to the closest hull, in normalized units."""
        Q = self.scaler_.transform(X)
        to_center = cdist(Q, self.centers_)
        nearest = to_center.argmin(axis=1)

        # The hull of the nearest center gives a first bound...
        distance = np.empty(len(Q))
        for c, V in enumerate(self.hulls_):
            queries = np.flatnonzero(nearest == c)
            if queries.size:
                distance[queries], _ = distance_to_hull(Q[queries], V)

        # ... so only few other hulls can still be closer.
        for c, V in enumerate(self.hulls_):
            queries = np.flatnonzero(
                (nearest != c) & (to_center[:, c] - self.radii_[c] < distance)
            )
            if queries.size:
                upper, _ = distance_to_hull(Q[queries], V, stop_above=distance[queries])
                distance[queries] = np.minimum(distance[queries], upper)
        return distance

    def predict(self, X):
        """Return True for the readings that should raise an alert."""
        return self._decide(X)[0]

    def _decide(self, X):
        """Compare the distance to the threshold, without computing the distance.

        Hulls are visited from the nearest center to the farthest, and a reading is
        settled as soon as one hull is within the threshold. Strictly-inside would
        not work as a stopping rule: in this dimension, new healthy readings are
        almost never inside a hull. Hulls whose ball is beyond the threshold are
        skipped, so a reading far from everything is cheap too.

        Up to the solver's `tol`, this agrees with `hull_distance(X) > threshold_`.

        Returns the alerts and the number of hulls solved for each reading.
        """
        Q = self.scaler_.transform(X)
        to_center = cdist(Q, self.centers_)
        order = to_center.argsort(axis=1)
        reachable = to_center - self.radii_ <= self.threshold_

        alert = np.ones(len(Q), dtype=bool)
        n_solved = np.zeros(len(Q), dtype=int)
        stop_above = np.full(len(Q), self.threshold_)
        pending = np.arange(len(Q))
        for rank in range(len(self.hulls_)):
            hull_of_pending = order[pending, rank]
            pending_reachable = reachable[pending, hull_of_pending]
            for c in np.unique(hull_of_pending[pending_reachable]):
                queries = pending[(hull_of_pending == c) & pending_reachable]
                upper, _ = distance_to_hull(
                    Q[queries],
                    self.hulls_[c],
                    stop_above=stop_above[queries],
                    stop_below=self.threshold_,
                )
                n_solved[queries] += 1
                alert[queries[upper <= self.threshold_]] = False
            pending = pending[alert[pending]]
            if pending.size == 0:
                break
        return alert, n_solved
