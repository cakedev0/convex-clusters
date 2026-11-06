# convex-clusters

POC of outlier detection for power plant monitoring. Readings are normalized and
clustered with k-means, then an alert is raised when a reading is too far from the
convex hull of every cluster. The current system of the person monitoring the plant
uses boxes instead of hulls: `box` and `box raw` are the baselines to beat.

## Context

- About 200k samples, 20 features, 100 clusters, 10k readings to score per hour.
- Training data comes from times the plant was healthy, but contains outliers due
  to measurement errors. They should not widen what counts as normal.
- Use scikit-learn's `KMeans`. Its relocation of empty clusters means that clusters
  end up tiny rather than empty, unlike what was reported on the real data.

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
- `convex_clusters/clustering.py`: `StandardScaler` and `KMeans`.
- `convex_clusters/hull.py`: vertex selection along random directions, and distance
  to a hull with batched Frank-Wolfe.
- `convex_clusters/detector.py`: `ConvexClusterDetector`, the train/predict pipeline.
- `convex_clusters/baselines.py`: box, centroid and Mahalanobis scorers.

## Design notes

- Hulls are implicit, only a set of vertices: facets are out of reach at d=20.
- New healthy readings are almost never strictly inside a hull at d=20, even a hull
  of all the points of a cluster. Decisions rely on the threshold on the distance,
  so `inside` is never used as a stopping rule; `distance <= threshold` is.
- Measurement errors are the extreme points of their cluster, hence the most likely
  to be picked as vertices. This is why clusters are trimmed before selecting them.
- The threshold is calibrated on held-out data, as training points are in the hulls
  by construction. That data is contaminated, so `target_alert_rate` has to be above
  the rate of measurement errors.
- Assumptions to check on real data: renormalization is `StandardScaler`, boxes are
  axis-aligned, and the synthetic fault models are placeholders for real faults.
