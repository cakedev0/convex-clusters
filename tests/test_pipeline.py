import numpy as np
import pytest
from scipy.optimize import nnls

from convex_clusters.baselines import box_scorer, raw_box_scorer
from convex_clusters.clustering import fit_clusters
from convex_clusters.data import FAULT, HEALTHY, make_plant
from convex_clusters.detector import ConvexClusterDetector
from convex_clusters.hull import distance_to_hull, select_vertices


def exact_distance(q, V, weight=1e4):
    """Distance to a hull with NNLS, the sum-to-one constraint being a stiff row."""
    A = np.vstack([V.T, weight * np.ones(len(V))])
    b = np.append(q, weight)
    w, _ = nnls(A, b, maxiter=10_000)
    return np.linalg.norm(V.T @ w - q)


def test_sample_shape_and_outlier_fraction():
    data = make_plant(n_features=7).sample(2000, outlier_fraction=0.01)
    assert data.X.shape == (2000, 7)
    assert data.is_measurement_error.sum() == 20
    assert np.all(data.kind[data.is_measurement_error] != HEALTHY)


def test_sample_reproducible():
    plant = make_plant()
    np.testing.assert_array_equal(plant.sample(500, seed=3).X, plant.sample(500, seed=3).X)
    assert not np.array_equal(plant.sample(500, seed=3).X, plant.sample(500, seed=4).X)
    assert not np.array_equal(
        make_plant(seed=1).sample(500, seed=3).X, plant.sample(500, seed=3).X
    )


def test_fit_clusters_sizes():
    data = make_plant(n_features=5).sample(2000)
    result = fit_clusters(data.X, n_clusters=10)
    assert result.sizes.sum() == 2000
    assert result.sizes.shape == (10,)


def test_select_vertices_are_input_points():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 6))
    vertices = select_vertices(points, 10, rng)
    assert len(vertices) <= 20
    assert all(np.any(np.all(points == v, axis=1)) for v in vertices)


def test_distance_known_geometry():
    # Unit square: distances are known in closed form.
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Q = np.array([[0.5, 0.5], [2.0, 0.5], [2.0, 2.0], [1.0, 1.0]])
    upper, lower = distance_to_hull(Q, V, tol=1e-3)
    expected = [0.0, 1.0, np.sqrt(2.0), 0.0]
    np.testing.assert_allclose(upper, expected, atol=1e-3)
    assert np.all(lower <= np.array(expected) + 1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_distance_matches_exact_solver_in_high_dimension(seed):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(100, 20))
    Q = rng.normal(scale=1.5, size=(30, 20))
    upper, lower = distance_to_hull(Q, V, tol=1e-2)
    exact = np.array([exact_distance(q, V) for q in Q])
    assert np.all(upper - lower <= 1e-2)
    # lower <= exact <= upper, up to the accuracy of the reference solver
    assert np.all(lower <= exact + 1e-3)
    assert np.all(exact <= upper + 1e-3)


def test_stop_above_keeps_valid_lower_bound():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(50, 10))
    Q = rng.normal(scale=3.0, size=(20, 10))
    _, lower = distance_to_hull(Q, V, stop_above=np.zeros(20))
    exact = np.array([exact_distance(q, V) for q in Q])
    assert np.all(lower <= exact + 1e-3)


def test_hull_distance_pruning_matches_brute_force():
    plant = make_plant(n_features=8)
    detector = ConvexClusterDetector(n_clusters=10, n_directions=30, random_state=0)
    detector.fit(plant.sample(5000).X)
    X = plant.sample(100, seed=1).X
    Q = detector.scaler_.transform(X)
    brute = np.min([distance_to_hull(Q, V)[0] for V in detector.hulls_], axis=0)
    np.testing.assert_allclose(detector.hull_distance(X), brute, atol=2e-2)


def test_predict_agrees_with_distance():
    plant = make_plant(n_features=8)
    detector = ConvexClusterDetector(n_clusters=10, n_directions=30, random_state=0)
    detector.fit(plant.sample(5000).X)
    X = np.vstack(
        [
            plant.sample(300, 0.0, seed=1).X,
            plant.sample_faults(300, severity=5, seed=2).X,
            plant.sample(100, 1.0, seed=3).X,
        ]
    )
    alert = detector.predict(X)
    distance = detector.hull_distance(X)
    # Both alert states must occur for the comparison to mean anything.
    assert 0 < alert.sum() < len(X)
    clear_cut = np.abs(distance - detector.threshold_) > 2e-2
    np.testing.assert_array_equal(alert[clear_cut], distance[clear_cut] > detector.threshold_)


def test_box_contains_hull():
    # A box holds all its cluster's points, so it can't be further than the hull.
    plant = make_plant(n_features=8)
    detector = ConvexClusterDetector(n_clusters=10, n_directions=30, random_state=0)
    detector.fit(plant.sample(5000).X)
    X = plant.sample_faults(200, severity=10, seed=1).X
    assert np.all(box_scorer(detector)(X) <= detector.hull_distance(X) + 1e-9)


def test_raw_box_contains_all_training_points():
    plant = make_plant(n_features=8)
    detector = ConvexClusterDetector(n_clusters=10, n_directions=30, random_state=0)
    X = plant.sample(5000).X
    detector.fit(X)
    train = detector.scaler_.inverse_transform(detector.clusters_.X_scaled)
    assert np.all(raw_box_scorer(detector)(train) < 1e-9)


def test_single_sensor_fault_moves_one_sensor():
    plant = make_plant(n_features=8)
    faults = plant.sample_faults(50, severity=5, single_sensor=True, seed=1)
    healthy = plant.sample(50, 0.0, seed=1)
    assert faults.X.shape == healthy.X.shape
    assert np.all(faults.kind == FAULT)


def test_detector_alerts_on_severe_faults_only():
    plant = make_plant(n_features=8)
    detector = ConvexClusterDetector(n_clusters=10, n_directions=50, random_state=0)
    detector.fit(plant.sample(10_000).X)
    assert detector.threshold_ > 0
    assert detector.predict(plant.sample(1000, 0.0, seed=1).X).mean() < 0.05
    assert detector.predict(plant.sample_faults(1000, severity=40, seed=2).X).mean() > 0.9
