"""Train the detector on a synthetic plant, then evaluate its alerts on new data.

Every method uses the same k-means. All but `box raw` have their threshold set for
the same alert rate on the same kind of contaminated data. `box raw` is the box
used as is: an alert for anything outside all the boxes, which contain every
training point, measurement errors included.
"""

import argparse
from time import perf_counter

import numpy as np

from convex_clusters.baselines import (
    box_scorer,
    centroid_scorer,
    mahalanobis_scorer,
    raw_box_scorer,
)
from convex_clusters.data import MEASUREMENT_ERRORS, make_plant
from convex_clusters.detector import ConvexClusterDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--n-directions", type=int, default=100)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--trim-quantile", type=float, default=0.99)
    parser.add_argument("--target-alert-rate", type=float, default=0.01)
    parser.add_argument("--outlier-fraction", type=float, default=0.005)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    plant = make_plant(n_features=args.n_features, seed=args.seed)
    train = plant.sample(
        args.n_samples, outlier_fraction=args.outlier_fraction, seed=args.seed + 1
    )
    detector = ConvexClusterDetector(
        n_clusters=args.n_clusters,
        n_directions=args.n_directions,
        min_cluster_size=args.min_cluster_size,
        trim_quantile=args.trim_quantile,
        target_alert_rate=args.target_alert_rate,
        random_state=args.seed,
    )
    start = perf_counter()
    detector.fit(train.X)
    fit_time = perf_counter() - start

    sizes = detector.clusters_.sizes
    n_vertices = [len(V) for V in detector.hulls_]
    print(f"train: {train.X.shape}, measurement errors: {train.is_measurement_error.sum()}")
    print(
        f"clusters: {(sizes == 0).sum()} empty, {(sizes <= 10).sum()} with <=10 samples,"
        f" {(sizes < args.min_cluster_size).sum()} without hull,"
        f" median size {np.median(sizes):.0f}"
    )
    print(f"hulls: {len(detector.hulls_)}, median vertices {np.median(n_vertices):.0f}")
    print(f"threshold: {detector.threshold_:.3f} (normalized units)")
    print(f"fit time: {fit_time:.1f}s")

    # Fresh data, contaminated like the training set, as the detector's held-out set.
    calibration = plant.sample(
        args.n_samples // 5, outlier_fraction=args.outlier_fraction, seed=args.seed + 2
    ).X
    methods = {"hull": (detector.hull_distance, detector.threshold_)}
    for name, make_scorer in {
        "box": box_scorer,
        "centroid": centroid_scorer,
        "mahalanobis": mahalanobis_scorer,
    }.items():
        scorer = make_scorer(detector)
        threshold = np.quantile(scorer(calibration), 1 - args.target_alert_rate)
        methods[name] = (scorer, threshold)
    methods["box raw"] = (raw_box_scorer(detector), 0.0)

    test_sets = {"healthy": plant.sample(args.n_test, 0.0, seed=args.seed + 3).X}
    errors = plant.sample(args.n_test, 1.0, seed=args.seed + 4)
    for kind, name in MEASUREMENT_ERRORS.items():
        test_sets[f"error: {name}"] = errors.X[errors.kind == kind]
    for severity in (3, 5, 10, 20):
        faults = plant.sample_faults(args.n_test, severity, seed=args.seed + 5 + severity)
        test_sets[f"fault: all sensors {severity}"] = faults.X
    for severity in (5, 10, 20):
        faults = plant.sample_faults(
            args.n_test, severity, single_sensor=True, seed=args.seed + 50 + severity
        )
        test_sets[f"fault: one sensor {severity}"] = faults.X

    print(f"\n{'alert rate':<24}" + "".join(f"{name:>13}" for name in methods))
    times = dict.fromkeys(methods, 0.0)
    n_scored = sum(len(X) for X in test_sets.values())
    for set_name, X in test_sets.items():
        rates = []
        for name, (scorer, threshold) in methods.items():
            start = perf_counter()
            distance = scorer(X)
            times[name] += perf_counter() - start
            rates.append(np.mean(distance > threshold))
        print(f"{set_name:<24}" + "".join(f"{rate:>13.3f}" for rate in rates))
    print(f"{'threshold':<24}" + "".join(f"{t:>13.3f}" for _, t in methods.values()))
    print(f"{'ms/sample':<24}" + "".join(f"{1e3 * times[n] / n_scored:>13.3f}" for n in methods))


if __name__ == "__main__":
    main()
