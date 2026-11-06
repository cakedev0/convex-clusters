"""Implicit approximate convex hulls in high dimension.

An exact hull has up to n^(d/2) facets, out of reach at d=20. A hull is therefore
only represented by a few vertices, and is queried by optimizing over the convex
combinations of those vertices, without ever computing a facet.
"""

import numpy as np
from scipy.spatial.distance import cdist


def select_vertices(points, n_directions, rng):
    """Keep the extreme points of `points` along random directions, in O(n d m).

    The hull of the result is an inner approximation of the hull of `points`, with
    at most `2 * n_directions` vertices.
    """
    directions = rng.normal(size=(n_directions, points.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projections = points @ directions.T
    return points[np.unique([projections.argmin(axis=0), projections.argmax(axis=0)])]


def distance_to_hull(Q, V, stop_above=None, stop_below=None, tol=1e-2, max_iter=2000):
    """Euclidean distance of each row of `Q` to the convex hull of the rows of `V`.

    Solves `min ||V^T w - q||^2, w >= 0, sum(w) = 1` with Frank-Wolfe, batched over
    queries. Each iteration provides both bounds on the distance: the current
    iterate gives the upper one, and the linear subproblem gives a separating
    hyperplane, hence the lower one.

    Parameters
    ----------
    Q : ndarray of shape (n_queries, n_features)
    V : ndarray of shape (n_vertices, n_features)
    stop_above : ndarray of shape (n_queries,), default=None
        Give up on a query once its distance is known to be above this value.
    stop_below : float, default=None
        Give up on a query once its distance is known to be below this value.
    tol : float
        Stop once the bounds are this close.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    upper, lower : ndarrays of shape (n_queries,)
        Bounds on the distance. Without `stop_above` and `stop_below`,
        `upper - lower <= tol` unless `max_iter` was reached.
    """
    upper = np.empty(len(Q))
    lower = np.empty(len(Q))
    x = V[cdist(Q, V, "sqeuclidean").argmin(axis=1)]
    active = np.arange(len(Q))
    for _ in range(max_iter):
        residual = Q[active] - x[active]
        norm = np.linalg.norm(residual, axis=1)
        scores = residual @ V.T
        best = scores.argmax(axis=1)
        gap = scores[np.arange(len(active)), best] - np.sum(residual * x[active], axis=1)

        safe_norm = np.maximum(norm, np.finfo(float).tiny)
        upper[active] = norm
        lower[active] = np.maximum(norm - gap / safe_norm, 0.0)

        keep = norm - lower[active] > tol
        if stop_above is not None:
            keep &= lower[active] < stop_above[active]
        if stop_below is not None:
            keep &= norm > stop_below
        active = active[keep]
        if active.size == 0:
            break

        residual, best = residual[keep], best[keep]
        direction = V[best] - x[active]
        step = np.sum(residual * direction, axis=1) / np.sum(direction**2, axis=1)
        x[active] += np.clip(step, 0.0, 1.0)[:, None] * direction
    return upper, lower
