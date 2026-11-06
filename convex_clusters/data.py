"""Synthetic sensor data from a simulated power plant.

Healthy readings come from a mixture of multivariate Gaussians, one per operating
regime (power level, plant state, outside temperature, ...), with correlated sensors
and very unequal regime frequencies. Two kinds of abnormal samples can be added:

- measurement errors: the plant is fine but a few sensors misread. They are rare and
  large, and pollute the training data, which is otherwise healthy.
- faults: the plant itself left its normal operating range. They only exist at
  prediction time and are what the system should raise an alert for.

Features are returned in raw "physical" units with very different scales, so
normalizing them is required before clustering.
"""

from dataclasses import dataclass

import numpy as np

HEALTHY = 0
SPIKE = 1
DROPOUT = 2
GAIN = 3
FAULT = 4
MEASUREMENT_ERRORS = {SPIKE: "spike", DROPOUT: "dropout", GAIN: "gain"}


@dataclass
class SensorData:
    X: np.ndarray  # (n_samples, n_features), raw units
    regime: np.ndarray  # (n_samples,), operating regime the sample was drawn from
    kind: np.ndarray  # (n_samples,), HEALTHY or what was done to the sample

    @property
    def is_measurement_error(self):
        return np.isin(self.kind, list(MEASUREMENT_ERRORS))


@dataclass
class Plant:
    """Fixed parameters of a plant, to draw any number of samples from.

    Each regime has covariance `L L^T + diag(psi^2)`. Everything but `scale` and
    `offset` is in "whitened" units, close to the noise level of a sensor.
    """

    weights: np.ndarray  # (n_regimes,)
    means: np.ndarray  # (n_regimes, n_features)
    loadings: np.ndarray  # (n_regimes, n_factors, n_features), the L above
    noise_std: np.ndarray  # (n_regimes, n_features), the psi above
    scale: np.ndarray  # (n_features,), physical scale of each sensor
    offset: np.ndarray  # (n_features,), physical offset of each sensor

    def _sample_whitened(self, n_samples, rng):
        n_regimes, n_factors, n_features = self.loadings.shape
        regime = rng.choice(n_regimes, size=n_samples, p=self.weights)
        Z = np.empty((n_samples, n_features))
        for r in range(n_regimes):
            mask = regime == r
            n = mask.sum()
            Z[mask] = (
                self.means[r]
                + rng.normal(size=(n, n_factors)) @ self.loadings[r]
                + rng.normal(size=(n, n_features)) * self.noise_std[r]
            )
        return Z, regime

    def _to_physical(self, Z):
        return self.offset + self.scale * Z

    def sample(self, n_samples, outlier_fraction=1e-4, seed=0):
        """Sample readings of a healthy plant, with some measurement errors."""
        rng = np.random.default_rng(seed)
        Z, regime = self._sample_whitened(n_samples, rng)
        X = self._to_physical(Z)
        kind = self._add_measurement_errors(X, outlier_fraction, rng)
        return SensorData(X=X, regime=regime, kind=kind)

    def sample_faults(self, n_samples, severity, single_sensor=False, seed=0):
        """Sample readings of a plant with a fault.

        A healthy reading gets displaced by `severity` (in whitened units).

        By default, along a random direction, which moves all the sensors together.
        Spread over 20 sensors, `severity=10` is only about 2 units per sensor,
        roughly one standard deviation of a single sensor, so it can't be seen
        sensor by sensor, only by breaking the correlations between them.

        With `single_sensor=True`, a single random sensor moves instead, which is
        the easy case for per-sensor bounds.
        """
        rng = np.random.default_rng(seed)
        Z, regime = self._sample_whitened(n_samples, rng)
        if single_sensor:
            directions = np.zeros_like(Z)
            sensors = rng.integers(Z.shape[1], size=n_samples)
            signs = rng.choice([-1.0, 1.0], size=n_samples)
            directions[np.arange(n_samples), sensors] = signs
        else:
            directions = rng.normal(size=Z.shape)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        X = self._to_physical(Z + severity * directions)
        return SensorData(X=X, regime=regime, kind=np.full(n_samples, FAULT))

    def _add_measurement_errors(self, X, outlier_fraction, rng):
        """Misread a few sensors of a random subset of samples, in place."""
        n_samples, n_features = X.shape
        kind = np.full(n_samples, HEALTHY)
        n_outliers = round(outlier_fraction * n_samples)
        indices = rng.choice(n_samples, size=n_outliers, replace=False)
        kind[indices] = rng.choice(list(MEASUREMENT_ERRORS), size=n_outliers)

        for i in indices:
            n_sensors = rng.integers(1, 4)
            sensors = rng.choice(n_features, size=n_sensors, replace=False)
            if kind[i] == SPIKE:
                amplitude = np.exp(rng.uniform(np.log(100), np.log(3000), size=n_sensors))
                sign = rng.choice([-1, 1], size=n_sensors)
                X[i, sensors] += sign * amplitude * self.scale[sensors]
            elif kind[i] == DROPOUT:
                X[i, sensors] = 0.0
            else:
                # Typically a unit error.
                X[i, sensors] *= rng.uniform(10, 100, size=n_sensors)
        return kind


def make_plant(
    n_features=20, n_regimes=8, n_factors=4, separation=2.0, imbalance=2.0, seed=0
):
    """Draw the parameters of a plant.

    Parameters
    ----------
    n_features : int
        Number of sensors.
    n_regimes : int
        Number of Gaussian components (operating regimes).
    n_factors : int
        Rank of the part of the covariance shared between sensors.
    separation : float
        Standard deviation of the regime means, in whitened units.
    imbalance : float
        Regime frequencies are proportional to `rank ** -imbalance`: with the
        defaults, from 65% for the most frequent regime to 1% for the rarest. The
        more frequent a regime, the tighter it is, from 0.1 to 1 times the spread
        of the rarest one: a plant sits at steady full power most of the time.
        k-means allocates centers by inertia, so this is what makes cluster sizes
        imbalanced, whereas unequal frequencies alone would even out.
    seed : int
        Seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    # A dead sensor reads 0: the offset makes that far from its normal readings.
    scale = np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=n_features))
    weights = np.arange(1.0, n_regimes + 1) ** -imbalance
    spread = np.geomspace(0.1, 1.0, n_regimes)
    return Plant(
        weights=weights / weights.sum(),
        means=rng.normal(scale=separation, size=(n_regimes, n_features)),
        loadings=spread[:, None, None]
        * rng.normal(scale=0.6, size=(n_regimes, n_factors, n_features)),
        noise_std=spread[:, None] * rng.uniform(0.3, 1.0, size=(n_regimes, n_features)),
        scale=scale,
        offset=scale * rng.uniform(10, 40, size=n_features),
    )
