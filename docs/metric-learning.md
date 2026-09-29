# Upstream `enn` Metric Learning Note

Source checked: 2026-10-01.

This note summarizes the metric-learning work in upstream
`yubo-research/enn` branch `dsweet/more` at commit
`506e98c506eeb849cffbf53d9ddf3a3a799c6830`. It is a reading note for ENNX, not
an upstream reading note followed by the concrete ENNX integration below.

## What Problem It Addresses

The branch changes how disk BPANN decides what "near" means.

Nearest-neighbor search needs a distance rule. With ordinary squared distance,
every input dimension counts equally:

```text
d(x, q) = (x0 - q0)^2 + (x1 - q1)^2 + ... + (xd - qd)^2
```

That can be a poor geometry for optimization. Some input dimensions explain the
output strongly. Some are weak. Some may be nuisance coordinates, categorical
bits, seeds, or scale artifacts. If the distance metric treats all dimensions as
equally meaningful, the neighbor index can retrieve rows that are close in the
wrong dimensions.

The upstream branch adds an `AUTO` diagonal metric mode. "Diagonal" means one
weight per input dimension. The distance becomes:

```text
d(x, q) = w0 * dx0^2 + w1 * dx1^2 + ... + wd * dxd^2
```

Large weights make a dimension matter more in neighbor ranking. Small weights
make it matter less.

## Where It Is Allowed

The upstream branch restricts AUTO metric learning to the disk BPANN layout:

```text
IndexDriver::BpAnnDisk
disk storage
scale_x = false
```

It rejects AUTO on flat and in-memory layouts. That matters because the feature
is not just a fitter; it also changes the metric used by the disk index and must
decide whether the index can be rescaled or needs rebuilding.

## Main Upstream Modules

| Module | Role |
| --- | --- |
| `metric_auto.rs` | AUTO policy, reservoir, refit cadence, metric drift, rescale versus rebuild |
| `metric_weights.rs` | Converts dependence scores into per-dimension weights and handles tied dimensions |
| `metric_sobol.rs` | Binned first-order dependence estimator |
| `metric_loo.rs` | Leave-one-out predictive log-likelihood check |
| `reservoir.rs` | Bounded Algorithm R sample of observed `(x, y)` rows |
| `model/metric.rs` | Applies learned metric scales to disk BPANN |
| `py_metric.rs` | Python bindings for snapshots, manual weights, and diagnostics |

## The AUTO Loop

The branch uses this loop:

1. Observe new `(x, y)` rows as data is added.
2. Store a bounded sample of rows in a reservoir.
3. Wait until at least `MIN_DEPENDENCE_ROWS` rows are available.
4. Estimate how strongly each input dimension explains output variation.
5. Convert those dependence scores into positive metric weights.
6. Compare learned weights against the all-ones metric with leave-one-out
   predictive log likelihood.
7. Use the learned metric only if the held-out gain is positive.
8. Apply the metric as per-dimension scaling.
9. Rescale the existing disk index for small metric changes.
10. Rebuild or repartition when metric drift is large.

Useful constants from the branch:

| Constant | Value | Meaning |
| --- | ---: | --- |
| `MIN_DEPENDENCE_ROWS` | 100 | Do not fit dependence before this many rows |
| `AUTO_RESERVOIR_CAPACITY` | 1000 | Maximum retained rows for metric fitting |
| `AUTO_K` | 10 | Neighbor count used in LOO scoring |
| `AUTO_REFIT_GROWTH` | 1.5 | Refit after geometric growth in seen rows |
| `AUTO_RESCALE_TOL` | 0.01 | Ignore very small metric changes |
| `DEFAULT_REBUILD_DRIFT` | `ln(2)` | Rebuild threshold in log-weight drift |
| `AUTO_MIN_HELDOUT_GAIN` | 0.0 | Learned metric must beat all-ones metric |

## How The Weights Are Estimated

The branch uses a binned first-order Sobol-style dependence estimator.

For each input dimension, it ranks rows by that dimension, splits them into
bins, and asks how much output variance is explained by the difference between
bins.

The intuition is:

```text
high score: changing this input dimension tends to move y
low score: this input dimension does not explain much y variation
```

For multi-output `y`, the scores are averaged across output columns. Weak
apparent dependence is suppressed with a null threshold. If no dimension passes
the threshold, the branch falls back to uniform weights.

The scores are also adjusted by input spread, so a dimension does not get a
large role merely because it has large raw units.

## Tied Dimensions

The branch supports `tied_dims`: disjoint groups of dimensions that should be
treated as one metric unit.

This is useful for one-hot or structured categorical inputs. One column of a
one-hot group may look weak on its own, while the group as a whole represents a
meaningful choice.

The branch validates that tied groups are:

```text
non-empty
in range
disjoint
```

For a tied group, the metric code computes a group dependence score and assigns
the same metric unit to dimensions in that group.

## Why The Held-Out Check Matters

Dependence scores can be misleading, especially early in an optimization run.
The branch therefore checks whether the learned metric improves local
prediction.

It compares:

```text
loo_loglik(learned weights)
loo_loglik(all-ones weights)
```

The learned metric is used only when:

```text
loo_loglik(learned) - loo_loglik(all_ones) > 0
```

This is conservative. AUTO is allowed to do nothing when the data does not
support a learned metric.

## Why The Reservoir Matters

The branch does not keep every row forever for metric fitting. It uses Algorithm
R reservoir sampling with fixed capacity.

The reservoir first fills normally. After it reaches capacity, each new row has
a fair chance to replace an existing row.

That keeps metric fitting bounded:

```text
memory: O(capacity * (num_dim + num_outputs))
refit:  bounded by reservoir capacity
```

The reservoir still tracks total rows seen, so the refit schedule can grow with
the stream even though the retained sample stays bounded.

## Applying The Metric

For learned weights `w`, the branch computes:

```text
x_scale[j] = 1 / sqrt(w[j])
```

Searching over scaled inputs is equivalent to using weighted squared distance
over raw inputs.

Metric changes create an index-maintenance problem:

```text
small change: rescale the existing index
large change: rebuild or repartition the index
```

This is the core systems point. Metric learning is not only about estimating
weights. It also has to keep the disk neighbor index consistent with the metric
used for search.

## Python Surface

The branch exposes diagnostic and control hooks through Python:

```text
metric_snapshot
metric_set_weights
metric_configure
metric_tied
dependence_weights
auto_weights
sobol_index
group_sobol_index
loo_loglik
validate_tied_dims
```

The snapshot reports counters such as rows seen, refits, rescales, rebuilds,
held-out gain, whether learned weights are active, current weights, and built
weights.

## Relationship To The Build Measurements

The same upstream branch also changes release profile settings from
`codegen-units = 1` to `codegen-units = 2` while keeping `lto = true`.

That build-profile change is separate from metric learning. If we benchmark the
branch, we should not attribute build-time differences to metric-learning code
without isolating the profile change.

Use two comparisons:

```text
isolated cgu=2 effect:
  same source commit, cgu=1 versus cgu=2

full branch effect:
  main branch versus dsweet/more
```

## What To Measure

If ENNX studies this upstream idea, the minimum evidence should include:

| Question | Measurement |
| --- | --- |
| Does AUTO improve prediction? | LOO log likelihood and held-out prediction error |
| Does AUTO improve optimization? | Paired BO runs with shared seeds and objectives |
| Does AUTO hurt latency? | Per-round wall time, refit time, rescale/rebuild count |
| Does AUTO stabilize? | Weight trajectories and held-out gain over rounds |
| Does AUTO overfit early data? | Early-round and delayed-enable ablations |
| Do tied dimensions help? | Grouped versus ungrouped categorical inputs |

Every report should include the metric weights, tied groups, number of refits,
number of rescales, number of rebuilds, held-out gain, and whether AUTO actually
used learned weights.

## Reading

The upstream branch is best understood as a focused disk-BPANN feature:

```text
learn a diagonal distance metric from observed x-to-y dependence,
validate it with leave-one-out local prediction,
then update the disk neighbor index by rescaling or rebuilding.
```

It is not a general dense metric learner, not a neural embedding model, and not
a replacement for the surrogate. It changes the geometry used by nearest-neighbor
search.

## ENNX Integration

The pinned implementation is now additive code in ENNX:

| Path | Operation |
| --- | --- |
| `rust/crates/ennx/src/metric_auto.rs` | AUTO policy, refit cadence, uniform-baseline LOO gate, drift decisions |
| `metric_weights.rs`, `metric_sobol.rs` | Screened binned dependence, spread normalization, categorical tied groups |
| `metric_rows.rs`, `metric_rng.rs`, `metric_seed.rs` | Bounded Algorithm R reservoir with upstream NumPy-compatible PCG64/SeedSequence |
| `metric_loo.rs` | Upstream scale/noise grid and LOO score; direct diagonal distances instead of an allocated Gram matrix |
| `model/metric.rs` | `ENN::learn_metric(tied, seed)`, `metric_config`, `metric_weights`, `metric_snapshot`; automatic observation on `add` |
| `rust/crates/bpann/src/backend/metric.rs`, `index/metric.rs` | Rescale forest centroids, vector leaves and pending centroids; invalidate flat cache; rebuild on large drift |
| `rust/crates/ennx/src/bf16_metric.rs`, `bf16_metric.metal` | Batched resident family-distance reweighting, local radii, deterministic neighbors and noisy LOO scoring on Metal |

Ordinary disk AUTO remains opt-in through the Rust model API. It does not add a
TOML key or change the current tune CLI. Its reservoir stores ordinary input
rows, not billion-weight models. No default seed is inserted: enabling AUTO
requires the caller's seed. NumPy-stream tests cover reservoir replacement and
both 32-bit and 64-bit bounded integer draws.

The ordinary LOO implementation retains the upstream objective and grid search,
but accumulates squared coordinate differences directly, sharing neighbor lists
across output columns. This removes Gram-matrix storage and cancellation in the
norm/dot-product identity. Arithmetic is not promised bitwise identical to the
upstream Gram computation.

Disk metric metadata identifies the geometry of persisted pages. A pending
marker or a mismatched metric prevents reuse of those pages after reopening;
raw observation files are retained and the index can be rebuilt. This is not a
checkpoint of the AUTO reservoir, RNG, or refit schedule. Reopening an ordinary
ENN without enabling AUTO uses unit geometry. Rescaling uses the existing FP32
index representation; it is not a promise of FP64 or bitwise rebuild parity.

The resident path remains the existing four-family conditional likelihood
search. It is **not** the upstream full-coordinate Sobol estimator. It batches
the same metric candidates, uses the same sampled row IDs and observation
variances, and retains the existing regularizer, acceptance margin and damped
updates. Metal computes in FP32 with fast math disabled. Tests compare likelihood
scores with the FP64 CPU calculation and compare selected metric weights; no
bitwise-FP64 claim is made. Equal distances use observation ID as the tie-break.

Warm paired median timings from the CLI-built test binary on this machine
(nine paired samples, nine metric candidates, eight held-out rows). The laptop
was on AC with Low Power Mode enabled; these are preliminary observations, not
qualified performance results under the benchmark policy:

| History | Distances | CPU ms | Metal ms |
| ---: | --- | ---: | ---: |
| 16 | global | 0.148 | 0.483 |
| 16 | self-tuning | 0.230 | 0.591 |
| 64 | global | 0.615 | 0.540 |
| 64 | self-tuning | 1.912 | 1.177 |
| 128 | global | 1.814 | 0.908 |
| 128 | self-tuning | 7.175 | 1.529 |

Dispatch retains CPU fitting below H=64, where measured GPU launch overhead
dominates, and uses Metal at H=64..128. The current cutoff is machine-derived,
not a claim about every Apple GPU or fitting budget. Scratch is bounded by
history and family count, not model dimension: 862,864 shared-buffer bytes.
These measurements cover the metric fitter, not the complete generated-token
BO round. They do not establish a subsecond end-to-end run.

The upstream gate is LOO conditional prediction under a metric estimated from
the same reservoir labels. It is not an independent holdout or nested validation
of the whole metric-learning procedure. Independent episodes remain necessary
for optimization-quality claims.

## Billion-coordinate latent history

The resident full-weight optimizer applies the same conservative principle to
its tensor-family metric. With `[enn] geometry = "latent"`, each Rademacher
proposal contributes an analytic squared edge norm per tensor family. Those
family components form the fitting matrix; the current family policy compares
penalized conditional LOO scores against the current shape, not specifically
the upstream all-ones baseline gate. This avoids replaying
historical billion-coordinate tensors while preserving metric/index consistency.

This changes the declared geometry rather than approximating realized FP16
distances. Full candidate weights are still materialized and evaluated. The
`realized` mode remains the matched oracle for measuring whether the latent
geometry changes acquisition decisions or held-out optimization quality.
