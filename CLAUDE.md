# convex-clusters

POC of outlier detection for power plant monitoring. Readings are min-max scaled and
clustered with k-means, then an alert is raised when a reading is too far from the
convex hull of every cluster. The current system of the person monitoring the plant
uses boxes instead of hulls, the axis-aligned min/max of each cluster: `box raw` in
`run_poc.py` is that system, and the baseline to beat. `box` is the same with this
POC's trimming and threshold.

## Context

- About 200k samples, 20 features, 100 clusters, 10k readings to score per hour.
- Sensors have very different scales, and the real system min-max scales them.
- Training data comes from times the plant was healthy, but contains outliers due
  to measurement errors. They should not widen what counts as normal.
- The plant has several regimes (% power, plant state, outside temperature, ...).
  Clusters represent them and also isolate measurement errors, which are almost
  always alone in their cluster. Clusters are highly imbalanced.
- Use scikit-learn's `KMeans`. Its relocation of empty clusters means that clusters
  end up tiny rather than empty.

## Commands

```
pixi run test    # tests
pixi run poc     # alert rates of the hulls and baselines on a synthetic plant
pixi run bench   # single-thread prediction time
```

The last two are CPU-heavy: run them under `benchlock`.

## Layout

- `convex_clusters/data.py`: `make_plant` draws a plant, `Plant.sample` healthy
  readings with measurement errors, `Plant.sample_faults` readings of a faulty plant.
- `convex_clusters/clustering.py`: per-sensor scaling (min-max by default) and `KMeans`.
- `convex_clusters/hull.py`: vertex selection along random directions, and distance
  to a hull with batched Frank-Wolfe.
- `convex_clusters/detector.py`: `ConvexClusterDetector`, the train/predict pipeline.
- `convex_clusters/baselines.py`: box, centroid and Mahalanobis scorers.

## Design notes

`report.md` covers the mathematical properties and costs of each method. It leaves
out detection performance on purpose: synthetic data says little about a real plant.
Keep it that way.

- Hulls are implicit, only a set of vertices: facets are out of reach at d=20.
- New healthy readings are almost never strictly inside a hull at d=20, even a hull
  of all the points of a cluster. Decisions rely on the threshold on the distance,
  so `inside` is never used as a stopping rule; `distance <= threshold` is.
- Only large errors get isolated by k-means. Smaller ones stay in large clusters, at
  their extreme points, hence the most likely to be picked as vertices. This is why
  clusters are trimmed before selecting them, even though most errors look isolated.
- The threshold is calibrated on held-out data, as training points are in the hulls
  by construction. That data is contaminated, so `target_alert_rate` has to be above
  the rate of measurement errors.
- Keep preprocessing per-sensor and affine: it preserves convexity, which the hull
  approach relies on.
- Distances are in normalized units, so the scaling sets their scale: anything absolute
  must be relative to the data, like `relative_tol`.
- The synthetic plant is tuned so that its structure looks like the real one (errors
  alone in clusters, imbalanced sizes), through tight frequent regimes and large
  errors. Its fault models are placeholders for real faults.
