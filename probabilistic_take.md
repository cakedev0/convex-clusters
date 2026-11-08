# A probabilistic take

Ideas only: none of this is implemented or tested, and costs are estimates. The
question is what changes if the problem is modeled with likelihoods, given one
property: **the plant can move to new regimes that are convex combinations of known
healthy regimes.**

## The property rules out density models

A Gaussian mixture says that each reading comes from one regime. A new regime between
two known ones then gets a low likelihood, and raises an alert. The same holds for any
density estimate (KDE, normalizing flows), since a new regime has no training density.
This includes per-cluster Mahalanobis (Method 4 of [report.md](report.md)), which is a
mixture model in all but name.

The current per-cluster hulls don't have the property either. Each hull lies in its
k-means Voronoi cell, so the hulls are disjoint, and a point between two separated
regimes is in none of them. The property describes the hull of the union, not the union
of the hulls:

$$\mathrm{conv}\left(\bigcup_r S_r\right) \neq \bigcup_r \mathrm{conv}(S_r)$$

The per-cluster hulls implement the weakest of the three hypotheses in
[report.md](report.md#where-convexity-holds), and this property is the strongest.

## The model: mixing inside each reading

The property is an admixture, the structure behind topic models, spectral unmixing and
archetypal analysis. Weights mix the regimes inside each reading, instead of picking
one regime:

$$x = \sum_r \pi_r \theta_r + e, \qquad \pi \in \Delta, \qquad e \sim \mathcal{N}(0, \Psi)$$

where $\Delta$ is the probability simplex, $\theta_r$ the position of regime $r$ and
$\Psi$ the noise covariance. With a flat prior on $\pi$, since unknown regimes are
allowed, maximizing the likelihood over $\pi$ gives:

$$-2 \log p(x \mid \pi^\star) = \min_{\pi \in \Delta} \; (x - \Theta^\top \pi)^\top \Psi^{-1} (x - \Theta^\top \pi) + \mathrm{const}$$

This is the squared **distance to one global hull, measured in the noise metric**. The
hull intuition comes out of the likelihood, with a few differences from the POC:

- **The threshold becomes a noise level.** The signal lies in the polytope and noise
  pushes it out, which explains why healthy readings are almost never strictly inside a
  hull. If the polytope spans $q$ dimensions, the squared distance of a reading whose
  signal is well inside it follows $\chi^2_{d-q}$. For example, $d = 20$ and $q = 7$
  put the 99.9% quantile at 34.5, a distance of about 5.9.
- **That threshold can be falsified.** Compare it with the observed distances: a
  mismatch shows that a noise assumption is wrong, which a quantile calibrated on the
  data can't show.
- **Vertices are denoised.** They estimate regime positions rather than extreme
  readings, so they don't drift outward with $n$ and are harder for errors to pull.
- **Sensor scaling stops mattering for the score.** The distance is invariant under
  invertible affine maps of the sensors, if $\Psi$ and $\Theta$ are mapped too. The
  k-means still depends on the scaling. Whitening isn't per-sensor, but it is linear,
  so it preserves convexity.
- **It should be cheap.** The score is one Frank-Wolfe solve over at most 100 vertices,
  instead of up to about 90 hulls, plus one $O(d^2)$ whitening. Reusing
  `distance_to_hull` should keep it within budget, but this hasn't been measured.

The marginal likelihood, integrating over $\pi$ instead of maximizing, adds a volume
term that is nearly constant inside the polytope. It should barely change decisions, so
the maximum is the simpler choice.

## Promising ideas, in order

### 1. Whitened distance to the global hull of cluster centers

Drop tiny clusters, use the remaining centers as $\Theta$, estimate $\Psi$ from the
pooled within-cluster covariance, and score with `distance_to_hull` on whitened data.
About 20 lines.

Three things to watch, all from using k-means centers as regime positions:

- **Centers undercover.** They are averages, so their hull is an inner approximation.
- **k-means shrinks the noise estimate.** Cutting a regime into clusters truncates its spread, so
  the pooled covariance is too small. Expect the $\chi^2$ threshold to be off, and
  calibrate on held-out data as a first step.
- **100 centers can span all 20 dimensions.** Many clusters split a single regime
  along its noise directions. Then $q = d$, $\chi^2_{d-q}$ is meaningless, and little
  stays outside the hull. The spectrum of the centers shows this. If so, reduce first:
  keep the top $q$ directions, or merge centers into true regimes by fitting archetypes
  (Cutler and Breiman, 1994).

### 2. Separating bad sensors from faults

Add a per-sensor error to the model: $x_j$ is arbitrary with a small probability. The
generalized likelihood ratio for "sensor $j$ is wrong" compares the distance with and
without that sensor:

$$\Lambda_j = d^2(x) - d^2(x_{-j}), \qquad \Lambda_j \sim \chi^2_1 \text{ approximately, if healthy}$$

where $d^2(x_{-j})$ drops sensor $j$ from $x$ and $\Theta$, and row and column $j$ from
$\Psi$. Three outcomes on an alert:

- One large $\Lambda_j$ that brings $d^2$ back to normal points to a measurement error
  on sensor $j$.
- No single sensor explaining the alert points to a fault, which matches faults being
  sensors that drift together.
- Testing 20 sensors needs a multiplicity correction. As a reference, the 99.9%
  quantile of $\chi^2_1$ is 10.8.

It costs at most 20 extra solves and only runs on alerts. It also addresses a limit of
the current design: measurement errors raise alerts at prediction time, as faults do.

The same component helps at training. A contamination term, or Student-t noise,
downweights errors through reweighting, instead of the hard trimming and
`min_cluster_size`.

### 3. Using time, if readings are ordered

So far each reading is scored independently. Accumulating $d^2 - \mathbb{E}[d^2]$ with a
CUSUM test ignores a one-off spike, and catches a slow drift whose individual readings
all stay below the threshold. Healthy readings are probably autocorrelated, which
inflates the false alert rate of a test that assumes independence, so the CUSUM needs
to be calibrated on real sequences.

### 4. Reporting the regime mixture

$\pi^\star$ would give three outcomes: known regime, new regime (near the hull with low
mixture density), or abnormal. It could also say "60% A + 40% B", but only if the
vertices are affinely independent, at most $q + 1$ of them. With 100 centers in 20
dimensions $\pi^\star$ isn't unique, even though the distance is, so this needs the
reduction from idea 1.

## Risks

- **Noise differs by regime.** Clusters are imbalanced because regimes differ in
  tightness, and one pooled $\Psi$ fits that badly: too loose in tight regimes and too
  strict in wide ones. $\Psi(\pi) = \sum_r \pi_r \Psi_r$ is the natural fix, but the
  log-determinant term makes the problem non-convex, so it should come second.
- **The hull can be too permissive.** If regimes span all 20 dimensions, many faults
  land inside it. By assumption, a fault that looks like a convex combination of healthy
  regimes can't be detected at all.
- **Physics.** Sensors that respond nonlinearly to load break exact linear mixing, and
  only an affine rescaling preserves it. A sensor could need a transform to be linear
  in the underlying state, but that is a physical modeling choice, not a preprocessing
  convenience.

## The experiment to run first

**Leave-one-regime-out on real data:** remove a cluster, fit on the rest, and see
whether its readings are accepted.

- Hold out clusters whose center is inside the hull of the other centers, which
  `distance_to_hull` can check. Holding out an extreme regime says nothing, since the
  property doesn't claim it should be accepted.
- Compare per-cluster hulls and the global hull. The property predicts that the global
  hull accepts these readings and the per-cluster hulls reject many of them.
- To measure how permissive the global hull is, also score real faults, if any were
  recorded.

This tests the property itself rather than a model of it, and needs no synthetic data.

## An open question on the plant

Can **any** two regimes combine? Power and temperature probably can, but a categorical
plant state may not: being halfway between "shutdown" and "full power" may not be
healthy. If so, the right model is one global hull per plant state, and the current
per-cluster hulls are a rough approximation of that.
