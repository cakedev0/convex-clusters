# convex-clusters

POC of outlier detection for power plant monitoring. Readings are min-max scaled and
clustered with k-means, and an alert is raised when a reading is farther than a
threshold from the convex hull of every cluster. Hulls are sampled: each one keeps
only the extreme points along random directions, so no facet is ever computed.

```python
from convex_clusters.detector import ConvexClusterDetector

detector = ConvexClusterDetector(n_clusters=100, n_directions=100).fit(X_train)
alerts = detector.predict(X_new)  # True = raise an alert
```

`target_alert_rate` must be above the rate of measurement errors in `X_train`.

Setup uses [pixi](https://pixi.sh): `pixi run test` runs the tests, `pixi run poc`
scores all methods on synthetic data, and `pixi run bench` times them.

See [report.md](report.md) for the methods, their mathematical properties and costs.
