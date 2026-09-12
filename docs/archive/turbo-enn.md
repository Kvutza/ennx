> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# TuRBO-ENN Tuning

Updated: 2026-09-29.

## Pretraining configuration

Run `./ennx tune examples/tuning/code-pretrain.toml`. The
[configuration](../../examples/tuning/code-pretrain.toml) selects these components:

| Section | Meaning |
| --- | --- |
| `[pretrain]` | Model and corpus presets; cache and run paths are resolved automatically. |
| `[rounds]` | BO round count and complete-round wall-time target in milliseconds. |
| `[perturbation]` | `distribution` selects independent Gaussian perturbations over all model weights. |
| `[acquisition]` | `method` selects upper confidence bound acquisition; `beta` controls exploration. |
| `[surrogate]` | ENN method, neighbor count, uncertainty/outcome scales, and fitting budget. |
| `[trust-region]` | `method = "turbo"` selects TuRBO; the section contains base length, bounds, and tensor scaling. |

`neighbors` also sets the number of observations gathered before ENN-guided
selection starts. `fit_candidates` is the number of random ENN hyperparameter
candidates; the fitter also considers its current parameters. `fit_samples`
limits the observations sampled for leave-one-out likelihood. Neither field
sets the weight proposal pool, which currently has four candidates. Each BO
round evaluates one selected model on two 4096-token examples with two feedback
passes.

The ENN fitter updates `epistemic_scale`, `aleatoric_scale`, and `y_scale` during
the run. They specify initial model uncertainty, observation noise, and outcome
scaling. The trust-region lengths are dimensionless. With
`shape = "tensor_family_static"`, each tensor's initial RMS (floored at 1e-6)
is multiplied by its fixed family factor and the proposal radius. Family factors
are 1.0 for experts, 0.75 for projections and embeddings, 0.5 for routers and
feedback, and 0.25 for normalization tensors.

`reps = 1` is the current supported execution pattern: the worker runs once.
The field is parsed, but scheduling multiple repetitions is not implemented.
Random streams are derived internally when explicit legacy seeds are absent.
Section titles use hyphens; field names retain underscores. Earlier flat
configurations and nested primitive selector tables remain accepted as legacy
inputs, but new files should use selector fields.

The sections below describe the legacy dense LocalV1 workload and its
experiments; its dimensions and historical measurements do not describe the
pretraining preset above.

## Dense workload entry points

```sh
./ennx tune --help
./ennx tune examples/tuning/turbo-enn.toml
./ennx tune examples/tuning/turbo-enn-one-round.toml
./ennx tune examples/tuning/turbo-enn-trace.toml
```

`tune` accepts exactly one TOML path; there are no target subcommands. The
existing KNN and proposal documents retain their `[knn]` and `[proposal]`
tables. A document without either table is parsed as TuRBO-ENN: `version = 1`
and the existing flat `ConfigOverrides` fields live at the document root. The
flat `study = "end_to_end"` field selects this complete-round workload; it is
required so a generic optimizer override document is not silently interpreted
as a 1.065B-parameter benchmark.
Execution controls extend that same schema; there is no `[turbo_enn]` wrapper,
nested experiment table, or parallel run schema.

The configured operation is a complete TuRBO-ENN round study, not a selector
for individual kernel experiments. Every run uses the production scorer and
records end-to-end round latency, losses, decisions, trust-region radii and
allocated memory. Every measured round must meet `target_round_ms` for
`goal_met = true`; missing the target is a completed measurement rather than a
runner failure. Kernel feasibility work remains in the low-level diagnostic
adapter and does not define the public configuration vocabulary.

The pre-existing `work_dir` field retains its ENN-storage meaning and is
rejected by this fixed resident workload because it cannot honor that setting.

Execution requires Apple silicon macOS and
substantially more than 4 GiB: the recorded full run allocated 16.1474 GiB.
Rounds, seeds, latency target and supported optimizer overrides are supplied by
the TOML. This fixed workload supports UCB and Thompson acquisition, resident
ENN neighbor and variance scales, and TuRBO length overrides. Fields it cannot
honor, including Pareto and fitter, candidate-count or storage controls, are
rejected rather than silently ignored.

[The example](../../examples/tuning/turbo-enn.toml) and the direct TuRBO-ENN
run parser in [config.rs](../../rust/crates/ennx/src/config.rs) define version one.
Unknown fields and invalid radii fail before GPU allocation. Relative `output`
paths resolve against the configuration file's directory. Once a unique run
directory has been initialized, the invocation records `study.toml`,
`source.txt`, and `run.log` (including Buck2 build identity and model/device
details); completed workers also record `result.toml` and `exit.txt`. The result
contains aggregate fields plus one `[[rounds]]` table per measured round, with
latency, losses, decision, proposal radius, resulting trust-region length,
success/failure counters, restarts and allocation. `source.txt`
reports `unavailable` if JJ identity cannot be read. Worker failures after log
initialization receive a structured `result.toml`. Configuration, output-root,
or artifact-initialization failures are reported on stderr and may leave no run
directory or a partial one. Replaying `study.toml` creates another child under
the same root.
No full-memory budget estimate is exposed yet. Workload dimensions are fixed;
seeds and radius bounds are tunable. Acceptance runs are uninstrumented by
default. Setting `trace = true` selects the perturbative logical-operation trace
documented in [native-profiling.md](native-profiling.md); use it for attribution,
not production round timing.

Retained ignored tests cover scorer/controller parity, full FFN scoring,
readout views, attention boundaries and exact-shape layout probes.
[tools/fbt-bo](../../tools/fbt-bo) remains their low-level diagnostic adapter.
Configured rounds run through the `turbo-enn-worker` Buck2 executable.

## Completed gate/up and attention experiments

The 2026-09-24 exact-shape run preserved every FP16 output and both output-tail
canaries, but its earlier collector gave the endpoint variant twice as many raw
samples as the middle variant. The historical medians below explain why the
route was retired, but they are not evidence that the corrected balanced
acceptance protocol passed:

| Sandwich order | Production MPS + SwiGLU | Fused candidate |
|---|---:|---:|
| Production / candidate / production | 99.509 ms | 629.076 ms |
| Candidate / production / candidate | 99.763 ms | 639.204 ms |

The retained diagnostic uses balanced production/candidate/production/candidate
and candidate/production/candidate/production orders. Each order has equal
warmup dispatches and seven measured observations per variant; each observation
averages that repetition's two timings. The diagnostic requirement was a 10% win in both orders. The candidate was
instead approximately six times slower. This closes the fused gate/up route
under the current kernel; it is not a configured tuning mode.

The subsequent bounded repair to `fbt_prefill_flash_attention_half_96` removed
an out-of-bounds output-stage write and twelve final normalization matrix
multiplies per SIMD group. Its correctness fixtures passed. One three-round
before/after pair changed the sample mean from 67.657810 to 63.727272 seconds,
but this is not evidence of a causal or repeatable speedup: machine state was
uncontrolled and fixed-seed mean losses differed by up to 0.000908883.

The diagonal `simdgroup_multiply` accumulator rescale has already been
replaced in `fbt_prefill_flash_attention_half_96`. The remaining bounded
attention target is its scratch-staged scalar rescale, not a generic attention
rewrite. See the [implementation traffic audit](bo-complexity.md) for current
counts; the old 0.992943 trillion matrix-operation estimate is not current.
First require complete output comparison against the retained FP64 attention
oracle at lengths 1/31/32/33/65/4096, causal and 2048-local masks, grouped
heads, gates and output canaries. Then compare the exact production shape in
baseline/candidate/baseline and candidate/baseline/candidate orders, with equal
warmups and seven observations per variant. Do not integrate unless both orders
improve by at least 10%, followed by `tools/fbt-bo --check` and a three-round
baseline/candidate/baseline test with matching printed losses, decisions and
radii. Abort if the exact-shape comparison cannot beat the baseline. The 10%
threshold is an experimental decision rule, not a predicted speedup. No
seconds-per-round ceiling is claimed because the available isolated flash
replay failed profile consistency.

## Workload

The executable specification is `run_round_study` in
[fbt_round.rs](../../rust/crates/ennx/src/fbt_round.rs). Default settings are:

- Random initialization with seed 42; synthetic tokens, not a dataset.
- Width 1536, FFN width 6656, 24 layers, vocabulary 100352.
- Sixteen query heads, eight KV heads, head dimension 96.
- Local window 2048; every sixth layer has full causal attention.
- Two examples of 4096 tokens; two-pass `ScoreMode::Fused`.
- Incumbent and one selected candidate scored on the same examples per round.
- Two model-scoring calls, four sequence scores, eight sequence-level
  transformer passes per round.

This is teacher-forced scoring, not autoregressive token generation.

## Proposal and acquisition

`model_search` visits every independent parameter tensor, assigning:

```text
scale = max(RMS(initial_tensor), 1e-6)
distance_weight = 1 / (tensor_element_count * scale * scale)
```

The scales are fixed at initialization. Search retains two historical rows.
The initial trust-region length is 0.01, with limits 0.0001 and 0.1.
The FBT study explicitly contracts after four consecutive non-improvements and
expands after three consecutive improvements under the shared TuRBO reward
comparison. Four is an experimental failure budget, independent of parameter
count; the generic controller still derives its default from dimension.
Startup and restart seed acquisition history with the measured incumbent reward
and variance, and TuRBO with that reward. Proposal radius and trust-region length
are reported separately.

[bf16_search.metal](../../rust/crates/ennx/src/bf16_search.metal) generates:

```text
persistent_direction = 0.75 * RMS_normalized_reference + sqrt(0.4375) * noise
fresh_direction = independent_noise
small_radius = clamp(trust_length / 2, minimum, maximum)
large_radius = clamp(trust_length * 2, minimum, maximum)
candidate = BF16(incumbent + scale * radius * direction)
```

The pool combines two directions and two radii. The pretraining workload selects
`gaussian` by default or `rademacher` with the flat `perturbation` field. Seeded
hashing plus Box--Muller implements Gaussian noise; seeded hashing plus sign-bit
extraction implements Rademacher noise. See the
[perturbation lab](../perturbation-lab.md) for their distinct mathematical and
performance contracts. Candidate distances use realized FP16 weights, weighted by
`distance_weight`. ENN-style Thompson acquisition selects one candidate.
[bf16_metal.rs](../../rust/crates/ennx/src/bf16_metal.rs) requires one arm and four
pool candidates in this path.

The pretraining path resolves its controller request through the same `Ask`
structure consumed by CPU, Metal, OpenCL and CUDA. The configurable fields are
`acquisition`, `k_neighbors`, `epistemic_scale`, `aleatoric_scale`, `y_scale`
and `acquisition_seed`; UCB carries its configured `beta`, while Thompson does
not use `beta`. The current study keeps two resident history rows, so the
requested neighbor count is capped at the live history length. The model scorer
and proposal materialization remain Metal-specific on Apple silicon; sharing
the controller ABI does not make the FBT/PISA scorer executable through OpenCL
or CUDA.

## Round ordering and decision

1. Before the timed loop, score the incumbent once on synthetic minibatch zero
   and initialize acquisition history and TuRBO from its negative mean NLL and
   estimated variance. This cost is reported as `initial_objective_ms`.
2. Generate a fresh two-example minibatch for the round.
3. Call `begin_ask` and `finish_ask`, then bind the selected proposal buffers.
4. Score only the candidate: two sequence losses and two FBT passes per
   sequence-level batch, reported as four transformer passes.
5. Call `tell_noisy`; restore the original model buffers on rejection or bind
   the accepted controller base on acceptance.

```text
candidate_reward = -mean(candidate_loss)
candidate_variance = (candidate_loss[0] - candidate_loss[1])^2 / 4
improvement = candidate_reward - stored_incumbent_reward
combined_variance = candidate_variance + stored_incumbent_variance
accept = improvement > 2 * sqrt(combined_variance)
```

The candidate and stored incumbent generally come from different synthetic
minibatches. The threshold is therefore an explicit conservative policy, not a
calibrated posterior-confidence guarantee. Rejected observations still enter
the retained history and surrogate; only an accepted observation replaces the
base and incumbent. Trust-region success or failure follows that acceptance
decision rather than an unrelated absolute-reward comparison.

## Timing

`TURBO_ENN` reports setup, the separate initial objective, acquisition,
binding, candidate scoring, decision/restoration, complete-round and loop
times. `initial_objective_ms` is excluded from `loop_seconds` and round timing.
Each timed round reports `objective_calls=1`, `sequence_scores=2`, and
`transformer_passes=4`. `ask_seconds` covers the complete synchronous ask.

`FBT_PREFILL` reports scoring wall time, pass times, encode/submit and completion
waits. The current scorer submits one command buffer per pass; its GPU intervals
are pass intervals, not per-layer timings. Normal-path logging therefore reports
only those pass intervals. Use the separate diagnostic procedure in
[native-profiling.md](native-profiling.md) for perturbative per-operation
breakdowns. See the [audit](bo-complexity.md). Do not substitute device intervals
for round wall time.

The first full-size one-observation run is
`results/turbo-enn/run-1790613032204-30491-0`. The separate initialization
objective took 36.165565 seconds. The three timed rounds took 35.517759,
34.561049, and 33.844689 seconds, for a 34.641166-second mean. Every round
reported one objective call, two sequence scores, four transformer passes, and
16.0615 GiB allocated. Candidate scorer times were 31.557117, 31.987804, and
32.342156 seconds. All candidates were rejected; round three's mean NLL was
0.005065 lower than the stored incumbent, but its 0.005065 reward improvement
did not exceed the 0.022028 two-standard-error threshold. The run validates the
new execution and artifact contract; it does not meet the one-second target.

A prior uninstrumented default-MPS performance run passed three rounds in
92.147856 seconds, build ID `e137d413-30bf-4d89-b3e7-84f0f1749c59`.
The complete-round times are 29.419889, 33.133651, and 29.594257 seconds;
the loop mean is 30.715952 seconds per round. The source log is
`.cache/fbt-bo-normal-power.log`, which reports the same 4K, batch-two workload
and 16.1474 GiB allocated. This is one recorded run, not a distribution or an
attributed speedup. The older 275.324042-second run remains historical evidence.

After the attention-output repair, another uninstrumented three-round sample
measured 67.303598, 62.856918 and 61.021262 seconds, or 63.727272 seconds per
round. The single before/after pair does not establish causality or repeatability.

After the 96-wide local-attention key-block start repair, the one-round
uninstrumented config
[`turbo-enn-one-round.toml`](../../examples/tuning/turbo-enn-one-round.toml)
completed successfully at
`results/turbo-enn-one-round/run-1790447634868-11743-0`. It measured
61.182787 seconds, allocated 16.1474 GiB, and printed the same losses,
decision and radius as the adjacent trace runs. This is a smoke benchmark for
the production path, not a distribution.

The subsequent one-round production run
`results/turbo-enn-one-round/run-1790447807157-13452-0` also completed with the
same printed losses, decision and radius, but measured 77.249205 seconds. Its
phase line showed `tell_seconds = 1.431390041` and `restore_seconds =
0.002904875`; the larger regression came primarily from a slow incumbent scorer
sample at 42.797854 seconds. This reinforces the need for repeated controlled
runs before treating a one-round wall time as a speed claim.

After rejected tells stopped revalidating unchanged reference scales, the next
one-round production run
`results/turbo-enn-one-round/run-1790450248171-16385-0` completed with unchanged
printed losses, decision and radius. It measured 81.136647 seconds because both
scorer calls were slow, but the phase line reported `tell_seconds =
0.393944250` and `restore_seconds = 0.010624416`. This is consistent with
removing avoidable rejected-tell CPU validation, but still not a controlled
wall-time distribution.

After readout blocking, the one-round production run
`results/turbo-enn-one-round/run-1790455459958-22814-0` completed with the same
printed losses, decision and radius. It measured 28.365439 seconds, allocated
16.0615 GiB, and reported scorer calls of 14.044583 and 12.946897 seconds.
The paired trace run
`results/turbo-enn-trace/run-1790455593754-23210-0` measured 28.190632 seconds.
The 1000 ms target is still missed; the current measured bottleneck order is
gate/up GEMM, flash attention, down GEMM, QKVG GEMM, then blocked readout.

The matching three-round configured run
`results/turbo-enn/run-1790456762994-24664-0` measured 28.297495, 27.888071 and
27.057729 seconds, for a 27.747765-second mean. This satisfies the requested
post-change check above two rounds but still reports `goal_met = false`.

The historical minibatch-reuse ablation used `minibatch_refresh = 3`.
Its config and runner option have been removed. The active runner now changes
the synthetic examples and scores only the candidate every round. In
`results/turbo-enn-reuse-minibatch/run-1790457009127-25442-0`, round 1 still
performed two objective calls and took 28.716530 seconds. Rounds 2 and 3 reused
the incumbent score on the same two examples, performed one objective call, and
took 14.257172 and 13.864263 seconds. This proves incumbent rescoring accounts
for about one scorer call when the objective minibatch is held fixed; it does
not solve the remaining per-candidate scorer cost or meet the 1000 ms target.

The historical one-pass ablation added `score_mode = "standard"` to the same
fixed-minibatch setup. Its config and runner option have been removed. This changes
the objective from the default two-pass FBT scorer and is only a cost-isolation
experiment. `results/turbo-enn-standard-reuse/run-1790457331619-26875-0`
measured 14.228903 seconds in round 1, then 7.693430 and 7.398349 seconds with
incumbent reuse. The paired trace
`results/turbo-enn-standard-reuse-trace/run-1790457376068-27248-0` shows that a
single one-pass candidate scorer is still dominated by gate/up, flash
attention, down projection, readout and QKVG. The subsecond target therefore
requires a multi-x improvement even after removing incumbent rescoring and the
second feedback-conditioned pass.

The compute audit now has an explicit [subsecond capacity
ledger](bo-complexity.md#subsecond-capacity-ledger). Using the repo's counted
dense operations and current measurements, a one-pass fixed-minibatch candidate
would still need about 19.337 dense TFLOP/s to fit in one second, while one
fused candidate would need about 36.227 dense TFLOP/s. The current effective
dense rate is about 2.6--2.7 TFLOP/s on these runs. This does not mean every
optimization is exhausted; it means the remaining path to subsecond must reduce
or share the scorer arithmetic itself, not rely on small epilogue fusions of
the existing dense full-4K workload.

The historical full-size interface qualification at source revision
`e9dd35cac759575089c989ef02effea5c82113ee` ran the example TOML and legacy
`--rounds 3` adapter sequentially under a 20 GiB session limit. The resolved
artifacts matched except for their intentionally unique output paths. All three
old/new NLL pairs, rejection decisions and radius values matched to printed
precision. The configured loop took 192.799135 seconds and the legacy loop took
186.358168 seconds; both exited successfully and allocated 16.1474-16.1552 GiB.
This validates interface equivalence, not a speedup, bitwise tensor parity,
all-token parity or equivalence to a pre-cleanup executable. The subsecond goal
remains unmet.

## Checks and limitations

After the measured-startup and failure-budget repair, the ten-round run
`results/turbo-enn-controller-fix/run-1790461457701-48158-0` completed with
a 28.459187-second mean (27.276512--29.986257 seconds), stable 16.0615 GiB
allocation and no accepted proposals. The log contains exactly twenty
batch-two, 4096-token scorer calls. Trust length contracted from 0.01 to 0.005
after round eight; the one-second target remains unmet. This is a functional
controller check, not an attributed speedup or optimization-quality result.
`./ennx test` passed all 30 targets and `sh tools/fbt-bo --check` passed all
five scorer diagnostics. The existing MPS batch-view parity test also exposed
missing origins for nonzero offsets; that layout defect was repaired.

`gpu_scorer`, `gpu_scorer_optimized` and `gpu_scorer_readout_views` check small
fixtures, proposal/restoration behavior and MPS buffer views. The optimized
fixture uses width 1536 and 16Q/8KV heads, but only two layers, FFN width 256,
64 tokens and vocabulary 97. Existing tolerances are 0.02 maximum token-loss
error and 0.003 mean-loss error. These are not bitwise equivalence claims.

The initial search value is the separately measured negative mean NLL before
the timed loop. Examples change with the round index; stored outcomes are not
all from one fixed batch. Trust-region updates now use the incumbent selected
by the noisy acceptance policy. The four-failure policy makes contraction
reachable but does not establish that this controller is statistically
calibrated for changing minibatches.
A passing check does not establish optimization quality, full-size numerical
equivalence, or subsecond performance.
