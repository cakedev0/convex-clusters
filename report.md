# Detecting abnormal readings around k-means clusters

This report covers the mathematical properties and the computational cost of each
method. It deliberately says nothing about detection performance: the only data
available so far is synthetic, so such numbers would say little about a real plant.

## Setting

Notation: $n$ training points, $d$ sensors, $k$ clusters, cluster $c$ with points
$C_c$ of size $n_c$ and center $\bar c$. Here $n \approx 200\text{k}$, $d = 20$,
$k = 100$, and 10k readings have to be scored per hour, i.e. 360 ms per reading.

All methods share the same first steps: normalization, in $O(nd)$, and k-means, in
$O(nkdI)$ for $I$ iterations. They then differ only in the set $A_c$ they attach to
each cluster. A reading $x$ is normal if it is in the union of the $A_c$, or within a
distance $\varepsilon$ of it:

$$
\text{alert}(x) \iff \min_c \mathrm{dist}(x, A_c) > \varepsilon
$$

Three properties matter throughout:

- **Nesting.** With $S \subset C_c$, we have
  $\mathrm{conv}(S) \subset \mathrm{conv}(C_c) \subset \mathrm{box}(C_c)$.
  The hull is the smallest convex set containing the cluster, the box the smallest
  axis-aligned one.
- **Extreme points versus moments.** Boxes and hulls are built from extreme values,
  which keep growing with $n$ on unbounded distributions (about $\sigma\sqrt{2 \ln n}$
  for a Gaussian), and which a single bad point can move arbitrarily. Centers and
  covariances are averages: they converge, and an error of size $a$ only moves a
  center by $a/n_c$.
- **Contamination.** Training data contains measurement errors, which are extreme by
  nature. Methods built from extreme values need trimming to be usable. Clusters
  smaller than `min_cluster_size` are also dropped, as they hold isolated errors.

## What the clusters are for

The clusters have two jobs at once, which interact.

**Representing regimes.** The state of a plant depends on the power level, the state
of the plant, the outside temperature and so on, so healthy readings do not form one
blob. Clusters cut them into pieces. As most of these conditions are continuous, a
regime is not necessarily one cluster: a union of convex pieces can approximate a
normal set which is not convex. Per-cluster hulls thus only assume convexity within
each cluster, the weakest of the hypotheses in
[Where convexity holds](#where-convexity-holds).

**Isolating measurement errors.** k-means gives a far point its own center when this
saves more inertia than the best other use of that center. Giving a center to an error
at distance $a$ saves about $a^2$. Splitting a cluster of $n_c$ points along its top
axis, of variance $\lambda_1$, saves about $\frac{2}{\pi} n_c \lambda_1$ if it is
Gaussian. So an error is isolated roughly when:

$$
a^2 > \frac{2}{\pi} n_c \lambda_1
$$

Several consequences follow:

- Only large errors are isolated. With $n_c = 2000$, the limit is at
  $a \approx 36 \sqrt{\lambda_1}$, so an error of 30 standard deviations can stay in a
  large cluster, where it is an extreme point. Errors being "almost always alone" may
  partly be a selection effect, as those that are not alone go unnoticed. This is why
  trimming is still needed, even though large errors are isolated.
- Each isolated error uses a center: only $k - E$ remain for regimes, with $E$ errors.
  This also bounds the contamination: if errors are almost always alone, then $E < k$,
  so their rate is below $k/n = 5 \cdot 10^{-4}$ here. `target_alert_rate` has to stay
  above that rate, which gives a useful order of magnitude for it.
- A cluster reduced to an error must not count as normal, otherwise a ball of radius
  $\varepsilon$ around this bad reading is. Hence `min_cluster_size`. Size alone cannot
  separate an error from a genuine regime seen only once, so this threshold is a choice,
  to be kept low. It would be better to confirm those clusters by hand.
- scikit-learn relocates empty clusters during training, so its clusters end up tiny
  rather than empty. Implementations without relocation can keep actual empty clusters.

**Imbalance.** At high resolution, the density of k-means centers follows
$p^{d/(d+2)}$ for a density $p$. For a Gaussian regime of frequency $w$ and spread
$\sigma$, this gives a number of centers and a cluster size, up to constants, of:

$$
k_r \propto (w \sigma^2)^{d/(d+2)}, \qquad \frac{n_c}{n} \propto \frac{w}{k_r}
$$

At $d = 20$ the exponent is 0.91, so centers go almost in proportion to the inertia of
a regime, and $n_c$ scales about as $w^{0.09} \sigma^{-1.8}$. This regime is only
indicative with 100 centers in dimension 20, but it shows that cluster sizes mostly
reflect tightness, not frequency: large clusters are tight regimes, such as steady
operation, and rare but wide regimes get many small clusters. Consequences:

- A regime with too little inertia gets no center of its own, and shares a cluster with
  a neighbor. Hull and box then span the gap between both, and certify states in between
  that may never occur. As the regime variables are measured, clustering within known
  regimes would avoid this (not implemented).
- Vertices are capped at $2m$ whatever $n_c$, so the largest clusters are described the
  most coarsely. $m$ could grow with $n_c$ at a small cost (not implemented).
- Small clusters give small hulls: the extent of a hull grows as $\sqrt{2 \ln n_c}$, and
  with $n_c \le d$ a hull is flat, of dimension at most $n_c - 1$. Boxes keep a non-zero
  width unless a sensor is constant in the cluster, but they get narrow. Covariances
  are singular, so Mahalanobis needs to skip those clusters.
- The threshold is global, and its calibration is dominated by frequent regimes. The
  alert rate is thus controlled overall, not per regime, and can be much higher on rare
  regimes. A threshold per cluster is possible, but needs enough calibration points in
  each cluster, precisely what rare regimes lack.

## Where convexity holds

Hulls rely on convex combinations of healthy readings being healthy. Where this holds
is not known, and it can be framed as a question of scale. Take a sensor that responds
smoothly to an operating variable $p$, such as power: $y = f(p)$. The midpoint of two
healthy readings at $p_1$ and $p_2$ misses the healthy curve by about:

$$
\frac{|f''| (p_1 - p_2)^2}{8}
$$

Convexity is a good approximation as long as this gap stays below the noise: between
nearby regimes, or over any range where the response is close to linear. It fails
between distant regimes on a curved response. Categorical plant states are a different
case: halfway between "shutdown" and "full power" may not be healthy, even if every
sensor responds linearly.

This gives three hypotheses, from weakest to strongest:

| Hypothesis | Hull | New regimes accepted |
|---|---|---|
| Within a cluster | one per cluster, as in this POC | only near known ones |
| Within a plant state | one per state | mixes of regimes of the same state |
| Global | one for all healthy data | any mix of known regimes |

The weakest is almost free, since a union of small convex pieces fits any shape. For
the same reason, it cannot accept a new regime between two clusters: each hull lies in
the Voronoi cell of its cluster (see Method 3), so the union of the hulls is much
smaller than the hull of their union. Everything in this report implements the first
hypothesis. [probabilistic_take.md](probabilistic_take.md) sketches a model for the
third.

A family in between (not implemented): score each reading against the hull of its $m$
nearest clusters. $m = 1$ is the first hypothesis and $m = k$ the third, so $m$ could be
chosen from data rather than decided upfront.

Data can show where convexity fails, but cannot prove that an unseen mix is healthy:

- Within each plant state, regressing each sensor on the operating variables (power,
  outside temperature), linear against quadratic, estimates $f''$, hence the scale above.
- Holding out a cluster whose center is inside the hull of the others, and finding the
  smallest $m$ that accepts its readings, measures it directly.
- Training on the first months and scoring the later ones tests what new regimes look
  like in practice, which is the claim itself.

## Preprocessing

Sensors have very different units, so Euclidean distances are meaningless without
rescaling. Per-sensor affine rescaling, $x'_j = (x_j - \ell_j) / s_j$, is the same as
choosing a metric:

$$
\lVert x' - y' \rVert^2 = \sum_j \frac{(x_j - y_j)^2}{s_j^2}
$$

So the scale $s_j$ sets the weight of sensor $j$ in k-means and in $\varepsilon$.
The location $\ell_j$ plays no role in distances.

**Why stick to affine maps.** They preserve the shape of each distribution, as modes,
skewness and tails only change units. They also preserve convexity:
$\mathrm{conv}(T(S)) = T(\mathrm{conv}(S))$ for $T$ affine. This is essential here: if
convex combinations of healthy states are healthy in sensor units, this remains true
after scaling, and a nonlinear map (log, Box-Cox, quantile transform) would break it.
For a given partition, the raw box and strict hull membership do not depend on $s_j$
at all. What depends on it is the partition, $\varepsilon$, and the random directions.

### Min-max scaling (current)

$\ell_j = \min x_j$ and $s_j = \max x_j - \min x_j$, on the training data. The unit is
the full observed range.

- **The worst error sets the scale.** One error at distance $a$ on sensor $j$ gives
  $s_j \approx a$. With $R_j$ the range without it, the weight of this sensor in squared
  distances is divided by $(a/R_j)^2$. With $a/R_j = 100$, that sensor nearly disappears
  from k-means, from the hulls and from $\varepsilon$, for all readings. The breakdown
  point is $1/n$.
- **The metric is partly random.** Sensor weights depend on the errors that happened to
  occur, so two sensors with identical behavior can weigh very differently. They also
  change between training windows, and $\varepsilon$ has no stable meaning across them.
- **Errors are capped.** An error is at distance at most about 1 per affected sensor,
  whatever its size. This interacts with isolation: $a^2$ is about 1, so isolating needs
  $\frac{2}{\pi} n_c \lambda_1 < 1$, i.e. clusters tight compared to the ranges, which
  is more likely when errors have themselves inflated the ranges.
- **Not bounded on new data.** New readings can leave $[0, 1]$, and must not be clipped,
  as this would hide what should raise an alert.
- It is simple, $O(nd)$, and has a clear meaning, with no assumption on distributions.

### Alternatives

All are per-sensor and affine, so all preserve shapes and convexity. They mainly differ
in how one error of size $a$ affects the scale, with $\sigma$ the sensor's spread:

| Scale $s_j$ | Effect of one error | Breakdown point | Limit |
|---|---|---|---|
| Min-max | $s_j \approx a$ | $1/n$ | The worst error sets the unit |
| Standard deviation | $\sqrt{\sigma^2 + a^2/n}$ | $1/n$ | Unit mixes regimes and noise |
| Quantile range $q_{1-p} - q_p$ | None | $p$ | Choice of $p$ |
| IQR, MAD | None | 25%, 50% | Fails on multimodal sensors |
| Pooled within-cluster deviation | None if isolated | $1/n$ otherwise | Needs a first clustering |
| Nominal range | None | Not applicable | Needs the specifications |

- **Quantile range, the recommended first step.** It is min-max made robust, with the
  closest meaning: a range. At $p = 0.001$, up to 200 errors per side and per sensor have
  no effect at all for $n = 200\text{k}$, far above the contamination bound above, and
  regimes rarer than $p$ just land out of $[0, 1]$, where nothing is lost as nothing is
  clipped. It costs one sort per sensor. It is available as
  `scaling="quantile_range"`, using `RobustScaler(quantile_range=(0.1, 99.9))`.
- **Standard deviation.** Much less sensitive than min-max: an error of $1000\sigma$ for
  $n = 200\text{k}$ gives $s_j \approx 2.4\sigma$, against $1000\sigma$. But the influence
  is not bounded, and its unit depends on how far apart regimes are on that sensor. It is
  available as `scaling="standard"`.
- **IQR and MAD.** They have the best breakdown points, but are not suited here. If one
  regime covers more than 75% of the time, the IQR only reflects that regime, and other
  regimes look huge. It is even zero for a sensor constant most of the time.
- **Pooled within-cluster deviation.** With
  $s_j^2 = \frac{1}{n} \sum_{c : n_c > 1} \sum_{x \in C_c} (x_j - \bar c_j)^2$, the unit is
  the noise of the sensor within a regime, and isolated errors do not contribute. It
  equalizes noise, rather than range, so sensors that follow regimes weigh more. It
  requires a first clustering, then a second one with the new scale.
- **Nominal range.** With $s_j$ taken from specifications, such as sensor span or expected
  operating range, nothing is estimated: it is immune to errors and stable across training
  windows, so $\varepsilon$ keeps its meaning. It requires domain knowledge, which is
  available here, and a sensor using little of its span gets little weight.
- **Two-pass min-max.** Run min-max and k-means, remove the isolated points, recompute
  min-max, and cluster again. This keeps the current convention for twice the k-means
  cost. Each pass may reveal errors that were masked by larger ones, so several may be
  needed, and errors that are not isolated still set the range.

With any robust scale, errors are no longer capped, so $a^2$ can be large. This makes
their isolation more systematic, and it also attracts k-means++ seeding, which samples
points with a probability proportional to their squared distance.

## Method 1: box (current system)

$A_c = \prod_j [\min_{C_c} x_j, \max_{C_c} x_j]$, the per-sensor min/max of each
cluster. Used with $\varepsilon = 0$, this is `box raw` in the code. The `box` variant
uses the trimming and the threshold of the hull method; with $\ell$ and $u$ the bounds
of the box, the distance is in closed form, $\lVert x - \mathrm{clip}(x, \ell, u) \rVert$.

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
  data: the ball inscribed in a cube fills $\pi^{d/2} / (\Gamma(d/2+1) \cdot 2^d)$ of it,
  about $2.5 \cdot 10^{-8}$ at $d = 20$.
- It depends on the coordinate system: rotating the sensors changes the set.
- The untrimmed box has a breakdown point of $1/n_c$: one measurement error moves a
  face arbitrarily far. A cluster made of errors also yields a box of "normal" states.

**Cost:** training $O(nd)$, prediction $O(kd)$ per reading, memory $2kd$ floats.

## Method 2: distance to the nearest center

$A_c = \lbrace \bar c \rbrace$, so the normal region is a union of balls of radius $\varepsilon$.
It is the baseline that uses the k-means output and nothing else.

**Strengths**

- Cheapest possible, nothing beyond k-means to compute or store.
- Stable: a center is an average, so it doesn't depend on extreme values.

**Weaknesses**

- Isotropic: it ignores the shape of clusters, so a low-variance direction counts as
  much as a high-variance one.
- A single $\varepsilon$ for all clusters, whatever their spread.
- Every center counts as normal in this implementation, including those k-means put
  on measurement errors. Dropping small clusters first would be a one-line fix.

**Cost:** training none, prediction $O(kd)$ per reading, memory $kd$ floats.

## Method 3: sampled convex hull

$A_c = \mathrm{conv}(S_c)$, where $S_c$ holds the min and max of $C_c$ (after
trimming) along $m$ random unit directions, so it has $v \le 2m$ vertices.

**Why this is the set of interest.** If convex combinations of healthy states within a
cluster are healthy, then $\mathrm{conv}(C_c)$ is exactly what the data
certifies as healthy, with no distributional assumption. The box adds states that
nothing certifies, and an ellipsoid both adds and removes some.

**Why it is implicit.** A hull of $v$ vertices can have $\Theta(v^{\lfloor d/2 \rfloor})$
facets, i.e. $v^{10}$ at $d = 20$, so facets are out of reach. But membership and
distance are convex programs over the weights of the vertices, whose size doesn't
involve facets. With $V$ the matrix of the vertices:

$$
\mathrm{dist}(x, \mathrm{conv}(S))^2 = \min_{w} \lVert V^\top w - x \rVert^2
\quad \text{s.t.} \quad w \ge 0, \quad \sum_i w_i = 1
$$

**Solver.** Frank-Wolfe, batched over readings. With iterate $y$ and residual
$r = x - y$, one iteration costs $O(vd)$, and picks the vertex $s \in S$ maximizing
$s \cdot r$. With gap $g = \max_{s \in S} (s - y) \cdot r$, any point $z$ of the hull
satisfies $r \cdot z \le r \cdot y + g$, so:

$$
\lVert r \rVert - \frac{g}{\lVert r \rVert} \le \mathrm{dist}(x, \mathrm{conv}(S)) \le \lVert r \rVert
$$

Both bounds are valid at every iteration, the lower one being a separating hyperplane.
This allows stopping as soon as the decision is known, rather than solving to
accuracy. The objective converges in $O(D^2/t)$ for a hull of diameter $D$, so reaching
$\lVert r \rVert \le \varepsilon$ from a point inside takes $O(D^2/\varepsilon^2)$
iterations, whatever $d$ and $v$. This dimension-free bound is why a non-zero
$\varepsilon$ makes the problem easy.

**Avoiding most hulls.** Two facts cut the number of hulls to solve:

- Each $\mathrm{conv}(S_c)$ lies in the Voronoi cell of $\bar c$, since the cell
  is convex and contains $C_c$. Hence hulls are disjoint, and a point strictly inside a
  hull has that cluster as its nearest center: at $\varepsilon = 0$, only the nearest
  hull has to be solved. With $\varepsilon > 0$ this is no longer strict, but it
  justifies visiting hulls by increasing distance to their center, and stopping at the
  first one within $\varepsilon$.
- With $R_c = \max_{s \in S_c} \lVert s - \bar c \rVert$, we have
  $\mathrm{dist}(x, \mathrm{conv}(S_c)) \ge \lVert x - \bar c \rVert - R_c$,
  so a hull is skipped without any solving when this exceeds $\varepsilon$. This is what
  makes readings far from everything cheap.

**Strengths**

- It is the tightest convex description, and uses correlations without assuming a
  shape: it can follow any convex cluster, not only symmetric ones.
- Vertex selection is a matrix product, $O(n_c d m)$ for a cluster. Prediction depends
  on $d$ only linearly, through the cost of an iteration.
- The set is equivariant to affine changes of the sensors, unlike the box. (The
  directions and $\varepsilon$ are not, see below.)

**Weaknesses**

- *Strict membership is nearly empty in high dimension.* The hull of $n_c$ Gaussian
  points is roughly a ball of radius $\sqrt{2 \ln(n_c/d)}$, whereas a new point has a
  norm around $\sqrt{d}$. At $n_c = 1600$, $d = 20$, that's about 3.0 against 4.5, and
  matching them takes $n_c$ of order $d \cdot e^{d/2}$, about $4 \cdot 10^5$ per cluster.
  Even if k-means cells are not Gaussian, the order of magnitude rules out relying on
  strict membership. The actual normal set is the Minkowski sum
  $\mathrm{conv}(S_c) \oplus B(\varepsilon)$, so $\varepsilon$ carries much of the
  decision, and the elegant "in the hull means fine" reading does not hold as is.
- *The threshold is isotropic and global.* It is one Euclidean radius, in normalized
  units, for all clusters and directions. This partly undoes the shape-awareness of the
  hull.
- *No guarantee on the approximation.* $S_c$ matches the support function of $C_c$ only
  on the sampled directions. A worst-case guarantee needs a net of the sphere, of size
  about $(1/\delta)^{d-1}$ for a resolution $\delta$, so $m$ gives no bound on the
  Hausdorff distance. $\varepsilon$-kernels or greedy selection could give guarantees,
  at a higher cost.
- *Directions are biased towards long axes.* On a Gaussian cluster, the extreme point
  along $u$ is around $\Sigma u / \sqrt{u^\top \Sigma u}$, which is dominated by the
  high-variance axes, so low-variance directions, where off-pattern readings go, get few
  vertices. Whitening each cluster before sampling directions would fix this (not
  implemented).
- *Selection is the opposite of robust.* It keeps precisely the extreme points, which
  is where measurement errors are, and one is enough to stretch a hull. Hence the
  trimming, but it is by Euclidean distance to the center, so it removes normal points
  on long axes and keeps errors that are small but off-axis. Trimming by Mahalanobis
  distance would address this (not implemented).
- *Solver accuracy degrades near the decision boundary.* Frank-Wolfe is sublinear when
  the projection is on a face of the hull, and iterations are capped at 2000. Readings
  with a distance close to $\varepsilon$ cost the most, and are decided up to
  `relative_tol`, 1% of the median radius of the hulls.
- The alert gives no direct explanation, unlike a box bound, though the residual
  $x - y$ does show which sensors contribute to the distance.

**Cost:** selection $O(ndm)$, calibration one exact pass on the held-out set, memory
$kvd$ floats. Prediction per reading is $O(kd + k \log k)$ to order the centers, plus
$O(Tvd)$ per hull solved, for $T$ iterations. Absolute worst case: $k$ hulls solved at
the iteration cap, which is about 1 s by estimation, never approached in practice.

## Method 4: Mahalanobis distance

$A_c = \lbrace \mu_c \rbrace$ with the distance
$\lVert x - \mu_c \rVert_{\Sigma_c}^2 = (x - \mu_c)^\top \Sigma_c^{-1} (x - \mu_c)$, on
trimmed clusters. The normal region is a union of ellipsoids.

**Link with PCA.** With $(u_i, \lambda_i)$ the eigenpairs of $\Sigma_c$:

$$
d^2(x) = \sum_i \frac{\left(u_i \cdot (x - \mu_c)\right)^2}{\lambda_i}
$$

This is the Euclidean distance after a PCA whitening, with all components kept. The
classic monitoring statistics are its parts: Hotelling's $T^2$ is this sum on the top
components, and the reconstruction error (SPE, or Q statistic) is the sum on the other
components, without the division by $\lambda_i$.

**Strengths**

- It uses the whole second-order structure, and divides by $\lambda_i$: moves that break
  correlations go along small $\lambda_i$, and count the most.
- Affine invariant: reading a sensor in other units or linearly combining sensors
  changes nothing, so it doesn't depend on the normalization within a cluster.
- It has a statistical meaning. For a Gaussian cluster, level sets are those of the
  density and $d^2 \sim \chi^2_d$, which gives a principled threshold. (k-means truncates
  clusters, so this is only approximate here, and the threshold is calibrated anyway.)
- Parameters are moments: they converge with $n_c$ and need no solver.

**Weaknesses**

- The shape is imposed: an ellipsoid, convex and symmetric. For an arbitrary convex
  set $K$, even the best ellipsoid $E$ is only tight up to a factor $d$:
  $E/d \subset K \subset E$, with the scaling around the center of $E$ (John's theorem).
  A strongly non-elliptical cluster is thus misdescribed, where a hull would adapt.
- It relies on inverting $\Sigma_c$. It needs $n_c \gg d$, and small $\lambda_i$ amplify
  estimation errors. The code skips clusters with $n_c \le 2d$, which leaves out the
  smallest regimes, a real limit with imbalanced clusters. It adds a small ridge term for
  sensors that are constant or redundant within a cluster, which makes the distance
  sensitive to the choice of this term along those sensors.
- The covariance is not robust either: an error of size $a$ contributes $a^2/n_c$. A
  robust estimator such as MCD would handle this, at a higher training cost.
- $\min_c$ compares distances between clusters of different volumes without accounting
  for them. A Gaussian mixture log-likelihood would add $\log \det \Sigma_c$ and the
  weights.
- No certificate tied to the physics: it can accept states out of the hull and refuse
  states in it, so it doesn't implement "convex combinations of healthy are healthy".

**Cost:** training $O(nd^2 + kd^3)$, prediction $O(kd^2)$ per reading, memory $kd^2$
floats.

## Summary of costs

Complexities, for $I$ k-means iterations (shared), $m$ directions, $v \le 2m$ vertices,
$h$ hulls solved and $T$ Frank-Wolfe iterations per hull:

| Method | Training, after k-means | Prediction per reading | Memory (floats) |
|---|---|---|---|
| Box | $O(nd)$ | $O(kd)$ | $2kd$ |
| Center | none | $O(kd)$ | $kd$ |
| Sampled hull | $O(ndm)$ + calibration | $O(kd + k \log k + hTvd)$ | $kvd$ |
| Mahalanobis | $O(nd^2 + kd^3)$ | $O(kd^2)$ | $kd^2$ |

Measured with `pixi run bench`, on one thread of a laptop, for 200k × 20 and 100
clusters with min-max scaling, of which 92 have a hull, with 7348 vertices in total.
Repeated runs of this benchmark have differed by up to 2×, so only orders of magnitude
are meaningful.

| Method | Building, after k-means | ms/reading in batch | Model size |
|---|---|---|---|
| Center | 0 | 0.001 | 16 kB |
| Box, with or without trimming | 0.1 s | 0.005–0.011 | 30 kB |
| Mahalanobis | 0.1 s | 0.005–0.010 | 300 kB |
| Sampled hull | 0.1 s + 4.2 s calibration | 0.005–0.023 | 1.2 MB |

Normalization and k-means take about 1 s for all methods. This varies widely with the
data, through the number of iterations: the same step took 9 s on more balanced clusters.
The other methods need a threshold too (except the raw box), but that's a pass of at
most 0.4 s for them. Other per-sensor scalings have the same order of cost as min-max,
at most a sort per sensor, except those needing a second k-means.

Details for the hull, depending on the readings:

| Readings | Hulls solved | ms/reading, `predict` | exact distance | all hulls | one at a time, median / max |
|---|---|---|---|---|---|
| Healthy | 1.01 | 0.006 | 0.21 | 1.2 | 0.19 / 0.55 |
| Close to the threshold (faults, severity 5) | 5.5 | 0.023 | 0.04 | 0.2 | 0.59 / 1.7 |
| Far from everything (faults, severity 30) | 0.24 | 0.005 | 0.02 | 0.2 | 0.61 / 1.0 |

Timings of the baselines in one-at-a-time mode were not measured. These depend on the
data through $h$ and $T$, so they would vary on real data, but with 360 ms available
per reading, computation is not a constraint for any of these methods:

- Even solving every hull for every reading fits, so early exit isn't needed to meet
  the budget, though it makes a batch of 10k healthy readings take 0.06 s instead of 12 s.
- Calibration is the most expensive step of the hull method, since it needs the actual
  distances rather than a decision. This only matters at training time.
- The remaining criteria to choose between them are mathematical: whether the convex
  assumption holds on the real plant, and whether a Gaussian assumption holds instead.
