# Detecting abnormal readings around k-means clusters

This report covers the mathematical properties and the computational cost of each
method. It deliberately says nothing about detection performance: the only data
available so far is synthetic, so such numbers would say little about a real plant.

## Setting

Notation: `n` training points, `d` sensors, `k` clusters, cluster `c` with points
`C_c` of size `n_c`, center `c̄`. Here `n ≈ 200k`, `d = 20`, `k = 100`, and 10k
readings have to be scored per hour, i.e. 360 ms per reading.

All methods share the same first steps: normalization (`O(nd)`) and k-means
(`O(nkdI)` for `I` iterations). They then differ only in the set `A_c` they attach
to each cluster. A reading `x` is normal if it is in the union of the `A_c`, or within
a distance `ε` of it:

```
alert(x)  ⇔  min_c dist(x, A_c) > ε
```

Three properties matter throughout:

- **Nesting.** With `S ⊂ C_c`: `conv(S) ⊂ conv(C_c) ⊂ box(C_c)`. The hull is the
  smallest convex set containing the cluster, the box the smallest axis-aligned one.
- **Extreme points versus moments.** Boxes and hulls are built from extreme values,
  which keep growing with `n` on unbounded distributions (about `σ√(2 ln n)` for a
  Gaussian), and which a single bad point can move arbitrarily. Centers and
  covariances are averages: they converge, and an error of size `a` only moves a
  center by `a/n_c`.
- **Contamination.** Training data contains measurement errors, which are extreme by
  nature. Methods built from extreme values need trimming to be usable. Clusters
  smaller than `min_cluster_size` are also dropped, as they are mostly such errors.

## Method 1: box (current system)

`A_c = Π_j [min_{C_c} x_j, max_{C_c} x_j]`, the per-sensor min/max of each cluster.
Used with `ε = 0`, this is `box raw` in the code. The `box` variant uses the trimming
and the threshold of the hull method; the distance is in closed form,
`‖x − clip(x, low, high)‖`.

**Strengths**

- Closed-form test, no solver, no numerical issue.
- Directly interpretable: a violated bound names the sensor at fault.
- Exact containment of all the training points of the cluster when untrimmed, with no
  threshold to choose.

**Weaknesses**

- It only encodes marginals. Within a cluster, any combination of in-range sensor
  values is accepted, whatever the correlations. Only the union over many clusters
  captures correlations, piecewise.
- It is very loose in high dimension, as most of its volume lies in corners with no
  data: the ball inscribed in a cube fills `π^(d/2) / (Γ(d/2+1) 2^d)` of it, about
  `2.5·10⁻⁸` at `d = 20`.
- It depends on the coordinate system: rotating the sensors changes the set.
- The untrimmed box has a breakdown point of `1/n_c`: one measurement error moves a
  face arbitrarily far. A cluster made of errors also yields a box of "normal" states.

**Cost:** training `O(nd)`, prediction `O(kd)` per reading, memory `2kd` floats.

## Method 2: distance to the nearest center

`A_c = {c̄}`, so the normal region is a union of balls of radius `ε`. It is the
baseline that uses the k-means output and nothing else.

**Strengths**

- Cheapest possible, nothing beyond k-means to compute or store.
- Stable: a center is an average, so it doesn't depend on extreme values.

**Weaknesses**

- Isotropic: it ignores the shape of clusters, so a low-variance direction counts as
  much as a high-variance one.
- A single `ε` for all clusters, whatever their spread.
- Every center counts as normal in this implementation, including those k-means put
  on measurement errors. Dropping small clusters first would be a one-line fix.

**Cost:** training none, prediction `O(kd)` per reading, memory `kd` floats.

## Method 3: sampled convex hull

`A_c = conv(S_c)`, where `S_c` holds the min and max of `C_c` (after trimming) along
`m` random unit directions, so `|S_c| = v ≤ 2m`.

**Why this is the set of interest.** If the physics says that any convex combination
of healthy states is healthy, then `conv(C_c)` is exactly what the data certifies as
healthy, with no distributional assumption. The box adds states that nothing
certifies, and an ellipsoid both adds and removes some.

**Why it is implicit.** A hull of `v` vertices can have `Θ(v^⌊d/2⌋)` facets, i.e.
`v¹⁰` at `d = 20`, so facets are out of reach. But membership and distance are convex
programs over the weights of the vertices, whose size doesn't involve facets:

```
dist(x, conv(S))² = min ‖Vᵀw − x‖²   s.t.  w ≥ 0, Σw = 1
```

**Solver.** Frank-Wolfe, batched over readings. With iterate `y` and residual
`r = x − y`, one iteration costs `O(vd)`, and picks the vertex maximizing `v·r`. With
gap `g = max_v (v − y)·r`, any point `z` of the hull satisfies `r·z ≤ r·y + g`, so:

```
‖r‖ − g/‖r‖  ≤  dist(x, conv(S))  ≤  ‖r‖
```

Both bounds are valid at every iteration, the lower one being a separating hyperplane.
This allows stopping as soon as the decision is known, rather than solving to
accuracy. The objective converges in `O(D²/t)` for a hull of diameter `D`, so reaching
`‖r‖ ≤ ε` from a point inside takes `O(D²/ε²)` iterations, *whatever `d` and `v`*.
This dimension-free bound is why a non-zero `ε` makes the problem easy.

**Avoiding most hulls.** Two facts cut the number of hulls to solve:

- Each `conv(S_c)` lies in the Voronoi cell of `c̄`, since the cell is convex and
  contains `C_c`. Hence hulls are disjoint, and a point strictly inside a hull has that
  cluster as its nearest center: at `ε = 0`, only the nearest hull has to be solved.
  With `ε > 0` this is no longer strict, but it justifies visiting hulls by increasing
  distance to their center, and stopping at the first one within `ε`.
- With `R_c = max_{v∈S_c} ‖v − c̄‖`, `dist(x, conv(S_c)) ≥ ‖x − c̄‖ − R_c`, so a hull is
  skipped without any solving when this exceeds `ε`. This is what makes readings far
  from everything cheap.

**Strengths**

- It is the tightest convex description, and uses correlations without assuming a
  shape: it can follow any convex cluster, not only symmetric ones.
- Vertex selection is a matrix product, `O(n_c d m)` for a cluster. Prediction depends
  on `d` only linearly, through the cost of an iteration.
- The set is equivariant to affine changes of the sensors, unlike the box. (The
  directions and `ε` are not, see below.)

**Weaknesses**

- *Strict membership is nearly empty in high dimension.* The hull of `n_c` Gaussian
  points is roughly a ball of radius `√(2 ln(n_c/d))`, whereas a new point has a norm
  around `√d`. At `n_c = 1600`, `d = 20`, that's about `3.0` against `4.5`, and matching
  them takes `n_c` of order `d·e^(d/2)`, about `4·10⁵` per cluster. Even if k-means
  cells are not Gaussian, the order of magnitude rules out relying on strict membership.
  The actual normal set is the Minkowski sum `conv(S_c) ⊕ B(ε)`, so **`ε` carries much of
  the decision**, and the elegant "in the hull means fine" reading does not hold as is.
- *`ε` is isotropic and global.* It is one Euclidean radius, in normalized units, for
  all clusters and directions. This partly undoes the shape-awareness of the hull.
- *No guarantee on the approximation.* `S_c` matches the support function of `C_c` only
  on the sampled directions. A worst-case guarantee needs a net of the sphere, of size
  about `(1/δ)^(d−1)` for a resolution `δ`, so `m` gives no bound on the Hausdorff
  distance. ε-kernels or greedy selection could give guarantees, at a higher cost.
- *Directions are biased towards long axes.* On a Gaussian cluster, the extreme point
  along `u` is around `Σu / √(uᵀΣu)`, which is dominated by the high-variance axes, so
  low-variance directions, where off-pattern readings go, get few vertices. Whitening
  each cluster before sampling directions would fix this (not implemented).
- *Selection is the opposite of robust.* It keeps precisely the extreme points, which
  is where measurement errors are, and one is enough to stretch a hull. Hence the
  trimming, but it is by Euclidean distance to the center, so it removes normal points
  on long axes and keeps errors that are small but off-axis. Trimming by Mahalanobis
  distance would address this (not implemented).
- *Solver accuracy degrades near the decision boundary.* Frank-Wolfe is sublinear when
  the projection is on a face of the hull, and iterations are capped at 2000. Readings
  with a distance close to `ε` cost the most, and are decided up to `tol = 10⁻²`.
- The alert gives no direct explanation, unlike a box bound, though the residual
  `x − y` does show which sensors contribute to the distance.

**Cost:** selection `O(ndm)`, calibration one exact pass on the held-out set, memory
`kvd` floats. Prediction per reading is `O(kd + k log k)` to order the centers, plus
`O(T v d)` per hull solved, for `T` iterations. Absolute worst case: `k` hulls solved
at the iteration cap, which is about 1 s by estimation, never approached in practice.

## Method 4: Mahalanobis distance

`A_c = {μ_c}` with the distance `‖x − μ_c‖²_Σ = (x − μ_c)ᵀ Σ_c⁻¹ (x − μ_c)`, on trimmed
clusters. The normal region is a union of ellipsoids.

**Link with PCA.** With `(u_i, λ_i)` the eigenpairs of `Σ_c`:

```
d²(x) = Σ_i (u_i · (x − μ_c))² / λ_i
```

This is the Euclidean distance after a PCA whitening, with all components kept. The
classic monitoring statistics are its parts: Hotelling's `T²` is this sum on the top
components, and the reconstruction error (SPE, or Q statistic) is the sum on the other
components, without the division by `λ_i`.

**Strengths**

- It uses the whole second-order structure, and divides by `λ_i`: moves that break
  correlations go along small `λ_i`, and count the most.
- Affine invariant: reading a sensor in other units or linearly combining sensors
  changes nothing, so it doesn't depend on the normalization within a cluster.
- It has a statistical meaning. For a Gaussian cluster, level sets are those of the
  density and `d² ~ χ²_d`, which gives a principled threshold. (k-means truncates
  clusters, so this is only approximate here, and the threshold is calibrated anyway.)
- Parameters are moments: they converge with `n_c` and need no solver.

**Weaknesses**

- The shape is imposed: an ellipsoid, convex and symmetric. For an arbitrary convex
  set, even the best ellipsoid is only tight up to a factor `d`: `E/d ⊂ K ⊂ E` (John).
  A strongly non-elliptical cluster is thus misdescribed, where a hull would adapt.
- It relies on inverting `Σ_c`. It needs `n_c ≫ d`, and small `λ_i` amplify estimation
  errors. A sensor that is constant or redundant within a cluster makes `Σ_c` singular:
  the current code has no regularization, and needs a ridge term for real data.
- The covariance is not robust either: an error of size `a` contributes `a²/n_c`. A
  robust estimator such as MCD would handle this, at a higher training cost.
- `min_c` compares distances between clusters of different volumes without accounting
  for them. A Gaussian mixture log-likelihood would add `log det Σ_c` and the weights.
- No certificate tied to the physics: it can accept states out of the hull and refuse
  states in it, so it doesn't implement "convex combinations of healthy are healthy".

**Cost:** training `O(nd² + kd³)`, prediction `O(kd²)` per reading, memory `kd²` floats.

## Summary of costs

Complexities, for `I` k-means iterations (shared), `m` directions, `v ≤ 2m` vertices,
`h` hulls solved and `T` Frank-Wolfe iterations per hull:

| Method | Training, after k-means | Prediction per reading | Memory (floats) |
|---|---|---|---|
| Box | `O(nd)` | `O(kd)` | `2kd` |
| Center | none | `O(kd)` | `kd` |
| Sampled hull | `O(ndm)` + calibration | `O(kd + k log k + h T v d)` | `kvd` |
| Mahalanobis | `O(nd² + kd³)` | `O(kd²)` | `kd²` |

Measured with `pixi run bench`, on one thread of a laptop, for 200k × 20 and 100
clusters, of which 89 have a hull, with 12333 vertices in total. Two runs of the same
benchmark differed by up to 2×, so only orders of magnitude are meaningful.

| Method | Building, after k-means | ms/reading in batch | Model size |
|---|---|---|---|
| Center | 0 | 0.001 | 16 kB |
| Box, with or without trimming | 0.07 s | 0.005–0.011 | 30 kB |
| Mahalanobis | 0.08 s | 0.005–0.010 | 300 kB |
| Sampled hull | 0.1 s + 10.8 s calibration | 0.003–0.055 | 2 MB |

Normalization and k-means take 9.2 s for all methods. The other methods need a
threshold too (except the raw box), but that's a pass of at most 0.4 s for them.

Details for the hull, depending on the readings:

| Readings | Hulls solved | ms/reading, `predict` | exact distance | all hulls | one at a time, median / max |
|---|---|---|---|---|---|
| Healthy | 1.04 | 0.010 | 0.46 | 3.6 | 0.14 / 0.68 |
| Close to the threshold (faults, severity 5) | 11.8 | 0.055 | 0.18 | 1.8 | 0.79 / 4.1 |
| Far from everything (faults, severity 30) | 0.05 | 0.003 | 0.07 | 0.6 | 0.59 / 1.1 |

Timings of the baselines in one-at-a-time mode were not measured. These depend on the
data through `h` and `T`, so they would vary on real data, but with 360 ms available
per reading, computation is not a constraint for any of these methods:

- Even solving every hull for every reading fits, so early exit isn't needed to meet
  the budget, though it makes a batch of 10k healthy readings take 0.1 s instead of 36 s.
- Calibration is the most expensive step of the hull method, since it needs the actual
  distances rather than a decision. This only matters at training time.
- The remaining criteria to choose between them are mathematical: whether the convex
  assumption holds on the real plant, and whether a Gaussian assumption holds instead.
