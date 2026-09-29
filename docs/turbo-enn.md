# Pretraining BO runbook

Source checked: 2026-09-29.

```sh
./ennx tune examples/tuning/code-pretrain.toml
```

The higher-budget learned-family experiment is:

```sh
./ennx tune examples/tuning/code-pretrain-family-learned.toml
```

It runs 256 rounds with 32 neighbors and 64-row/64-candidate ENN fitting. The
pretraining resolver derives its random streams, chooses its content-addressed
output directory, and prints the artifact path.

The command accepts one TOML file. The active preset is FBT/PISA/MoE,
not the legacy dense LocalV1 study. Apple silicon and sufficient unified memory
for the full model/controller/scorer are required.

## Configuration

Read [the example](../examples/tuning/code-pretrain.toml) for current values.
Use a completed run's resolved study.toml to reproduce its settings.

| Section | Role |
| --- | --- |
| pretrain | Model and corpus presets |
| rounds | Selected-candidate round count, repetitions, and target wall time |
| perturbation | Independent Gaussian or Rademacher coordinates |
| acquisition | Selection rule; UCB beta controls its uncertainty bonus |
| surrogate | Neighbor count, distance scaling, initial uncertainty/output scales, LOOCV fitting budget |
| trust-region | Length bounds and tensor-family scaling mode |

The neighbor count also sets the initialization observation count.
fit_candidates and fit_samples control surrogate fitting, not proposal count:
the pool remains four. Fitting estimates epistemic/aleatoric scales; output
scale comes from the fitter's outcome standard deviation. This is point fitting,
not integration over a hyperparameter posterior.

`reps > 1` runs complete pretraining studies sequentially. The model weights
and immutable corpus stay fixed while proposal and acquisition streams are
derived independently for each repetition. The derivation excludes treatment
settings and the requested round/repetition budgets, so matching ablation cells
use the same streams and extending a run preserves its existing prefix. Do not
put seeds in a repeated study: explicit legacy seeds are rejected when
`reps > 1`. Each repetition writes `rep-NNN/result.toml`, tensor updates, and
controller records; the top-level `result.toml` reports aggregate latency and
goal status.

`fit_neighbors = true` treats the configured neighbor count as both the
initialization count and the upper bound for the fitted ENN neighborhood. After
fitting the uncertainty scales, the same leave-one-out predictive likelihood
scores every feasible neighbor count. The winner is used by subsequent
acquisition calls; it is reported with the fitted scales rather than hidden in
the implementation.

`distance_scaling = "self_tuning"` makes the learned squared metric local before
neighbor ranking and inverse-variance weighting. `local_scale_neighbors = 8`
uses each observation's eighth-neighbor squared distance as its local radius;
the query radius is computed from the same candidate-to-history distances on
the GPU. The CPU leave-one-out fitter uses the identical normalization. The
extra resident state is 128 FP32 radii, not copies of model weights. Omit both
fields for the original global metric.

## Reliability-aware controller

`[trust-region] method = "reliability"` selects the constant-state ENN
controller. Its `[trust-region.reliability]` table is required; no experiment
Controller policy is supplied through TOML, and experiment seeds are derived
from the resolved run configuration. The controller combines signals already
available after a paired evaluation:

- noise-screened concordance between the candidate's predicted and realized
  rank among its local evaluated neighbors;
- a discounted Beta posterior over concordant versus discordant rounds;
- an exponentially weighted, uncertainty-screened progress signal;
- the selected perturbation's realized metric radius divided by the
  incumbent's raw K-th-neighbor radius.

Nominal length is corrected through a learned scalar conversion from nominal
to realized radius, so tensor-family scaling and BF16 rounding remain inside
the feedback loop. Reliable progress expands smoothly. Unreliable stagnation
contracts smoothly. Reliable stagnation starts a bounded maximum-radius burst
that restricts selection to fresh directions; an accepted escape proposal
therefore relocates the incumbent without storing another model-sized vector.

The controller retains only scalar state. Logical ENN history remains bounded
at 128 observations and physical model residency remains two rows. Checked-in
experiments omit seeds: `./ennx tune` derives domain-separated deterministic
streams from the resolved experiment, and matching ablation cells receive
matching streams automatically.

The complete production factorial is:

```sh
./ennx tune examples/tuning/code-pretrain-ablation-global-turbo.toml
./ennx tune examples/tuning/code-pretrain-ablation-local-turbo.toml
./ennx tune examples/tuning/code-pretrain-ablation-global-reliability.toml
./ennx tune examples/tuning/code-pretrain-ablation-local-reliability.toml
```

Every reliability round reports concordance coverage, posterior mean and lower
bound, evidence mass, progress, density scale, normalized realized step,
nominal-to-realized conversion, escape state, action, and next length.

The active `tensor_family_learned` mode starts from each tensor's floored
initial RMS and fits four shared metric sensitivities: experts,
projections/embeddings, routers/feedback, and normalization. Each fit performs
a bounded coordinate search against ENN leave-one-out likelihood. Proposal
scales are inverse square roots of those sensitivities and are normalized over
tensor blocks, so this changes the search shape without changing the global
trust-region energy. It stores four FP32 distance components per history pair,
not historical billion-weight vectors.

`tensor_family_static` remains the fixed baseline: its family multipliers are
1.0, 0.75, 0.5, and 0.25 in the order above. `scalar` applies no family
multiplier. These are four shared group parameters, not per-coordinate
lengthscales or a hyperparameter posterior.

Legacy flat inputs remain supported where the parser accepts them; use the
sectioned example for new studies. Unsupported controls should fail rather
than silently change the workload.

## Actual round

1. Score the initial model once and record its reward and observation variance.
2. Load the round's training minibatch: two 4,096-token examples.
3. Compact logical history if full. Before sufficient observations exist,
   choose a scheduled pool member; otherwise use the configured ENN acquisition.
4. Materialize one full FP16 candidate and score both feedback passes. The
   production pretraining path also scores the incumbent on the identical
   minibatch, providing a common-random-number comparison.
5. Record negative mean sequence NLL. Estimate the paired improvement variance
   from contiguous 128-token block differences, while retaining the exact
   sequence-weighted mean as the optimization target.
6. During initialization, accept strict observed improvement without trust
   adaptation. During guided search, compare observed candidate reward against
   the ENN-predicted incumbent mean, requiring more than two combined standard
   errors. A screened deterioration counts as failure; otherwise inconclusive.
7. Store the observation; when eligible, fit the four family sensitivities and
   then the ordinary ENN scales, update the incumbent/trust state, and record
   round and tensor statistics.

Rejected candidates still supply observations. Paired improvements accumulate
from an initial zero anchor, making observations across changing minibatches
comparable without treating raw batch NLLs as the same objective. Contiguous
block means improve variance resolution but are not independent-token claims;
model-agreement diagnostics do not by themselves calibrate the rule.

The four-failure/three-success trust policy is experimental. Logical-history
compaction is distinct from a trust-region restart.

## History and reproducibility

See [history.md](history.md). There are two exact resident history rows and
up to 128 logical observations, not 128 exact resident vectors. Historical
distance approximations also enter the fitting matrix.

Artifacts include resolved configuration, source/run metadata, result records
and per-tensor updates. These are not a complete durable optimizer-resume or
historical-weight replay facility. Do not infer reproducibility from a seed alone.

## Measurement

Report complete wall time, scorer GPU time, ask/tell time, objective count,
accepted steps, realized FP16 changes, and fixed validation outcomes separately.
Trace mode splits controller commands and perturbs timing; it is for attribution.
The target requires every measured round to meet the wall-time bound, not only
the median. No current subsecond or learning-quality claim is established.

For controlled selection diagnostics, see [bo-audit.md](bo-audit.md).
For legacy dense configuration and historical measurements, see the
[archived runbook](archive/turbo-enn.md); those results do not describe this preset.
