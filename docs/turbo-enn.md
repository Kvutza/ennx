# Pretraining BO runbook

Source checked: 2026-10-01.

```sh
./ennx tune examples/tuning/code-pretrain.toml
```

The higher-budget learned-family experiment is:

```sh
./ennx tune examples/tuning/code-pretrain-family-learned.toml
```

The free-running 4,096-token generated-pretraining experiment is:

```sh
./ennx tune examples/tuning/code-pretrain-generated.toml
```

Unlike `code-pretrain.toml`, this path does not teacher-force the corpus. It
generates a complete continuation from its own prefix and uses negative target
NLL on those generated contexts as the BO reward. The NLL reduction is fused
with proposal readout; it does not add another model forward pass. One corpus
continuation is fixed for the run so candidate and incumbent observations remain
comparable, and the accepted portable checkpoint is written into the artifact.
The CLI decodes and prints the initial completion and every proposed completion
immediately after its measured round, including rejected proposals. Exact text
is also stored as `initial/completion.txt` and
`round-NNNN/completion.txt`; the final incumbent is `completion.txt`. Decoding,
file output, and terminal output are deliberately outside `round_ms`.

The example runs 100 rounds. The pretraining resolver derives its random
streams, chooses its content-addressed output directory, and prints the artifact
path.

The command accepts one TOML file. The active preset is FBT/PISA/MoE,
not the legacy dense LocalV1 experiment. Apple silicon and sufficient unified memory
for the full model/controller/scorer are required.

Prepare and validate the content-addressed corpus without allocating the model
or starting BO:

```sh
./ennx tune examples/tuning/code-generation-enn.toml --prepare
```

## Configuration

Read [the example](../examples/tuning/code-pretrain.toml) for current values.
Authored and resolved experiments share the closed, Rust-typed version 2 schema.
Unknown keys, unknown selectors, wrong value types, and incompatible choices
fail before corpus preparation or GPU allocation. Version 1 remains readable
for historical runs; newly resolved files use version 2. No CUE or generated
schema files are involved. Use a completed run's resolved experiment.toml to
reproduce its settings.

Only overrides need to be written. For example:

```toml
version = 2
experiment = "pretrain"
model = "fbt-pisa1-legacy-v1"
corpus = "stack-v3-python-pilot-v1"

[run]
rounds = 512
reps = 3
target_ms = 200

[proposal]
distribution = "gaussian"

[enn]
scaling = "self-tuning"
local-neighbors = 8

[acquisition]
method = "ucb"
beta = 0.75

[trust-region]
method = "reliability"
shape = "tensor-family-learned"
initial = 0.01
min = 0.0001
max = 0.1
```

This example changes optimizer settings; it does not enable free-running text
generation. For the full generated-token workload, use the checked-in
[coding experiment](../examples/tuning/code-generation-enn.toml), including its
`[generation]` policy. `target_ms` is a measured goal, not a latency guarantee.

| Section | Role |
| --- | --- |
| Root | `experiment`, model/corpus presets, optional output path |
| run | Selected-candidate rounds, repetitions, selection, validation interval, target milliseconds |
| data | Optional `train` and `validation` paths for already prepared data |
| proposal | Independent Gaussian/Rademacher distribution, total candidates, arms |
| objective | Moving-incumbent pairing or an explicit frozen-initial control variate |
| acquisition | Selection rule; UCB beta controls its uncertainty bonus |
| enn | Neighbor count, history geometry, distance scaling, initial uncertainty/output scales |
| enn.fit | LOOCV fitting candidates/samples, adaptive neighbor fitting |
| trust-region | Tagged `turbo`, `morbo`, or `reliability` policy with length bounds, tensor-family shape, and reliability fields |
| generation | Free-running length, sampling policy, corpus prompt, and generated-context reward |
| seeds | Optional explicit single-run streams; normally omitted |
| diagnostics | Optional trace, stage samples, and kernel trial sources |

The neighbor count also sets the initialization observation count.
`[enn.fit] candidates` and `samples` control surrogate fitting, not proposal
count. `[proposal] candidates` defaults to four total proposals; the current
GPU pool requires four, split evenly across `arms` (default one). Fitting
estimates epistemic/aleatoric scales; output
scale comes from the fitter's outcome standard deviation. This is point fitting,
not integration over a hyperparameter posterior. The unchanged optimizer
defaults are 10 neighbors, 30 fitting candidates, 10 samples, global distance
scaling, realized history geometry, UCB beta 2, and TuRBO control. A missing
`[run]` means 3 rounds, 1 repetition, and a 1000 ms target. Reliability policy
defaults come from the existing controller type. Resolved files materialize
defaults rather than requiring verbose authored files.

Acquisition choices have distinct typed payloads: Thompson accepts no UCB beta;
Pareto requires objective scales; augmented Chebyshev requires its preferences,
alpha, and seed domain. The MORBO trust region owns its region count,
rescalarization cadence, and clipping policy.
Vector acquisition also requires a vector-producing reward. A selector cannot
silently retain fields belonging to another method. The corpus preparer
receives a Rust-validated concrete request; it does not interpret version 2
keys or maintain a second version 2 schema. Parsing runs before timing, not
inside the BO loop.

`[objective] reference = "frozen_initial"` scores the initial model once for
every immutable minibatch before timing. A timed round then evaluates only the
candidate and uses its exact block-wise difference from the same-batch frozen
anchor. This removes the incumbent forward and batch-level offsets, but changes
the optimization target from improvement over the moving incumbent to
improvement over the initial model. The default remains `moving_incumbent`.

`[run] reps > 1` runs complete pretraining studies sequentially. The model weights
and immutable corpus stay fixed while proposal and acquisition streams are
derived independently for each repetition. The derivation excludes treatment
settings and the requested round/repetition budgets, so matching ablation cells
use the same streams and extending a run preserves its existing prefix. Do not
put seeds in a repeated experiment: explicit legacy seeds are rejected when
`reps > 1`. Each repetition writes `rep-NNN/result.toml`, tensor updates, and
controller records; the top-level `result.toml` reports aggregate latency and
goal status.

`[enn.fit] neighbors = true` treats the configured neighbor count as both the
initialization count and the upper bound for the fitted ENN neighborhood. After
fitting the uncertainty scales, the same leave-one-out predictive likelihood
scores every feasible neighbor count. The winner is used by subsequent
acquisition calls; it is reported with the fitted scales rather than hidden in
the implementation.

`[enn] scaling = "self_tuning"` makes the learned squared metric local before
neighbor ranking and inverse-variance weighting. `local_neighbors = 8`
uses each observation's eighth-neighbor squared distance as its local radius;
the query radius is computed from the same candidate-to-history distances on
the GPU. The CPU leave-one-out fitter uses the identical normalization. The
extra resident state is 128 FP32 radii, not copies of model weights. Omit both
fields for the original global metric.

## Reliability-aware controller

`[trust-region] method = "reliability"` selects the constant-state ENN controller.
Omitted policy fields use its typed defaults; overrides such as
`evidence-decay` belong directly in `[trust-region]`. Experiment seeds are
derived from the resolved run configuration. The controller combines signals already
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
the median. A single free-running generated-pretraining round has measured
500.886 ms, but the 200 ms target, sustained subsecond throughput, and
learning-quality improvement are not established.

For controlled selection diagnostics, see [bo-audit.md](bo-audit.md).
Legacy dense measurements do not describe this preset.
