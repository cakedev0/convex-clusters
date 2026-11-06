"""Time prediction on one thread, against the need of 10k readings per hour.

Run it under `benchlock`. Compares three ways to reach the same decision:

- all hulls: solve every hull for every reading, the naive cost.
- full distance: `hull_distance`, the exact distance, with pruning.
- predict: stop at the first hull within the threshold.
"""

import argparse
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from convex_clusters.baselines import box_scorer
from convex_clusters.data import make_plant
from convex_clusters.detector import ConvexClusterDetector
from convex_clusters.hull import distance_to_hull

BUDGET_MS_PER_READING = 3600e3 / 10_000


def timed(function, *args):
    start = perf_counter()
    result = function(*args)
    return result, 1e3 * (perf_counter() - start) / len(args[0])


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
    detector = ConvexClusterDetector(
        n_directions=args.n_directions, random_state=args.seed
    ).fit(train.X)
    box = box_scorer(detector)

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

    print(f"budget: {BUDGET_MS_PER_READING:.0f} ms/reading; {len(detector.hulls_)} hulls")
    print(
        f"\n{'ms/reading, one thread':<22}{'alerts':>8}{'solved':>8}{'first':>7}"
        f"{'predict':>9}{'distance':>10}{'all hulls':>11}{'box':>8}"
        f"{'single p50':>12}{'p99':>8}{'max':>8}"
    )
    with threadpool_limits(1):
        for name, X in test_sets.items():
            (alert, n_solved), predict_ms = timed(detector._decide, X)
            _, distance_ms = timed(detector.hull_distance, X)
            _, naive_ms = timed(all_hulls, X[: args.n_naive])
            _, box_ms = timed(box, X)

            single_ms = []
            for x in X[: args.n_single]:
                _, ms = timed(detector.predict, x[None, :])
                single_ms.append(ms)

            print(
                f"{name:<22}{alert.mean():>8.3f}{n_solved.mean():>8.2f}"
                f"{np.mean(n_solved == 1):>7.2f}"
                f"{predict_ms:>9.3f}{distance_ms:>10.3f}{naive_ms:>11.2f}{box_ms:>8.3f}"
                f"{np.median(single_ms):>12.2f}{np.quantile(single_ms, 0.99):>8.2f}"
                f"{np.max(single_ms):>8.2f}"
            )
    print("\nsolved: mean number of hulls solved per reading")
    print("first: fraction of readings settled by solving a single hull")


if __name__ == "__main__":
    main()
