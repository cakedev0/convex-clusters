"""Time training and prediction of every method on one thread.

Run it under `benchlock`. The need is 10k readings per hour. For the hulls, compares
three ways to reach the same decision:

- all hulls: solve every hull for every reading, the naive cost.
- distance: `hull_distance`, the exact distance, with pruning.
- predict: stop at the first hull within the threshold.
"""

import argparse
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from convex_clusters.baselines import (
    box_scorer,
    centroid_scorer,
    mahalanobis_scorer,
    raw_box_scorer,
)
from convex_clusters.data import make_plant
from convex_clusters.detector import ConvexClusterDetector
from convex_clusters.hull import distance_to_hull, select_vertices

BUDGET_MS_PER_READING = 3600e3 / 10_000


def seconds(function, *args):
    start = perf_counter()
    result = function(*args)
    return result, perf_counter() - start


def ms_per_reading(function, X):
    result, elapsed = seconds(function, X)
    return result, 1e3 * elapsed / len(X)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--n-directions", type=int, default=100)
    parser.add_argument("--n-batch", type=int, default=10_000)
    parser.add_argument("--n-single", type=int, default=300)
    parser.add_argument("--n-naive", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    plant = make_plant(seed=args.seed)
    train = plant.sample(args.n_samples, seed=args.seed + 1)
    detector = ConvexClusterDetector(n_directions=args.n_directions, random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    with threadpool_limits(1):
        _, fit_time = seconds(detector.fit, train.X)
        _, selection_time = seconds(
            lambda: [
                select_vertices(points, args.n_directions, rng)
                for _, points in detector.trimmed_clusters()
            ]
        )
        calibration = plant.sample(args.n_samples // 5, seed=args.seed + 2).X
        _, calibration_time = seconds(detector.hull_distance, calibration)

        print(f"budget: {BUDGET_MS_PER_READING:.0f} ms/reading")
        print(f"\nhull training, one thread: {fit_time:.1f} s, of which")
        print(f"  trimming + vertex selection: {selection_time:.1f} s")
        print(f"  threshold calibration: {calibration_time:.1f} s")
        print(f"  normalization + k-means: {fit_time - selection_time - calibration_time:.1f} s")

        scorers = {}
        print("\nbuilding from the k-means, one thread:")
        for name, make_scorer in {
            "centroid": centroid_scorer,
            "box raw": raw_box_scorer,
            "box": box_scorer,
            "mahalanobis": mahalanobis_scorer,
        }.items():
            scorers[name], elapsed = seconds(make_scorer, detector)
            print(f"  {name}: {elapsed:.3f} s")

        n_vertices = sum(len(V) for V in detector.hulls_)
        print(f"\n{len(detector.hulls_)} hulls, {n_vertices} vertices in total")

        def all_hulls(X):
            Q = detector.scaler_.transform(X)
            return np.min([distance_to_hull(Q, V)[0] for V in detector.hulls_], axis=0)

        n = args.n_batch
        test_sets = {
            "healthy": plant.sample(n, 0.0, seed=args.seed + 3).X,
            "faults, severity 5": plant.sample_faults(n, 5, seed=args.seed + 4).X,
            "faults, severity 10": plant.sample_faults(n, 10, seed=args.seed + 5).X,
            "faults, severity 30": plant.sample_faults(n, 30, seed=args.seed + 6).X,
            "measurement errors": plant.sample(n, 1.0, seed=args.seed + 7).X,
        }

        print(
            f"\n{'ms/reading, one thread':<22}{'solved':>8}{'first':>7}"
            f"{'predict':>9}{'distance':>10}{'all hulls':>11}"
            + "".join(f"{name:>13}" for name in scorers)
            + f"{'single p50':>12}{'p99':>8}{'max':>8}"
        )
        for name, X in test_sets.items():
            (_, n_solved), predict_ms = ms_per_reading(detector._decide, X)
            _, distance_ms = ms_per_reading(detector.hull_distance, X)
            _, naive_ms = ms_per_reading(all_hulls, X[: args.n_naive])
            scorer_ms = [ms_per_reading(scorer, X)[1] for scorer in scorers.values()]
            single_ms = [
                ms_per_reading(detector.predict, x[None, :])[1] for x in X[: args.n_single]
            ]
            print(
                f"{name:<22}{n_solved.mean():>8.2f}{np.mean(n_solved == 1):>7.2f}"
                f"{predict_ms:>9.3f}{distance_ms:>10.3f}{naive_ms:>11.2f}"
                + "".join(f"{ms:>13.3f}" for ms in scorer_ms)
                + f"{np.median(single_ms):>12.2f}{np.quantile(single_ms, 0.99):>8.2f}"
                f"{np.max(single_ms):>8.2f}"
            )
    print("\nsolved: mean number of hulls solved per reading")
    print("first: fraction of readings settled by solving a single hull")
    print("single: `predict` on one reading at a time")


if __name__ == "__main__":
    main()
