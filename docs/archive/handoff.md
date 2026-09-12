> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# FBT BO Handoff

Updated: 2026-09-29. Active execution: GPU-only Metal/MPS.
Machine checked: MacBook Air, Apple M4, 10-core GPU, 24 GB unified memory.

## Shared TuRBO-ENN control: 2026-09-28

The active pretraining path no longer constructs a private, hard-coded ENN
request. `ConfigOverrides::resident_enn` now resolves the flat TOML fields into
the shared `Ask` ABI used by the CPU, Metal, OpenCL and CUDA controller paths:
acquisition, neighbor count, epistemic scale, aleatoric scale, output scale,
beta and acquisition seed. The configured TuRBO length bounds and proposal seed
also drive the full-weight pretraining search. The Metal selector uses up to 128
logical outcomes and distances while retaining two complete FP16 weight rows.
The two resident rows correct exact distances; they do not cap logical ENN
neighbors.

The checked-in pretraining study explicitly selects UCB acquisition with
`beta = 1.0`, `k_neighbors = 10`, `epistemic_scale = 0.7`, `aleatoric_scale = 0.05`,
`y_scale = 1.0`, trust lengths `0.01/0.0001/0.1`, and seeds `123/456`.
Unsupported fitter, candidate-count and storage controls fail validation rather
than being ignored. `result.toml` records the resolved values. The model scorer
and full-weight proposal kernel are Metal on this machine; OpenCL and CUDA share
the controller parameter contract but are not alternate backends for this
macOS pretraining scorer.

## Selectable full-weight perturbations: 2026-09-28

The pretraining BO path now supports `gaussian` and `rademacher` through the
flat TOML `perturbation` field. Gaussian remains the parser default for replay
compatibility. Both laws perturb all 1,038,508,544 FP16 coordinates, use the
same tensor RMS scaling and trust radii, compute exact realized-row distances,
and regenerate the selected row from the same seed. The Rademacher Metal path
replaces per-pair Box--Muller `log`, `sqrt`, and `sincos` with sign extraction;
it is a different search distribution and must be judged on timing and
optimization behavior separately.

The full test suite reports 534 passing `ennx-unit` tests, including CPU/Metal
Rademacher coordinate parity and pool-to-materialization identity. The only
remaining suite build failure is the pre-existing `region-arithmetic` linker
incompatibility with new macOS SDK `arm64e.x1` TAPI entries. Design provenance,
licenses, and the controlled comparison protocol are in
[perturbation-lab.md](../perturbation-lab.md).

## Active 100-round result: 2026-09-29

The active full-weight pretraining command completed 100 rounds:

```sh
./ennx tune examples/tuning/code-pretrain.toml
```

Artifact: `.cache/ennx/runs/pretrain/e9ebf5ccdffe38d3217f/run-1790654604391-26725-0`.
It contains 100 round records and 166,000 tensor records: one record for each
of 1,660 tensor blocks in every round. Each proposal spans all 1,038,508,544
FP16 weights. Four proposals were accepted.

| Measurement | Seconds |
| --- | ---: |
| Complete loop | 126.243592 |
| Round wall median | 1.266180 |
| Round wall minimum | 1.114282 |
| Round wall maximum | 1.413137 |
| Scorer GPU median | 1.069627 |
| Controller ask mean | 0.198488 |
| Controller tell mean | 0.002753 |

After nine initialization rounds, trust evidence comprised two successes, six
failures and 83 inconclusive outcomes. Inconclusive observations preserve the
active counter. Four failures accumulated after the last success and contracted
the trust radius from `0.01` to `0.005` at round 77. Subsequent proposal radii
were `0.0025` and `0.01`. This is the first verified long run in which the noisy
controller adapts instead of remaining frozen.

The previous UCB artifact is
`.cache/ennx/runs/pretrain/6417d699926683d47c63/run-1790653898562-23658-0`:
133.213548 s loop, 1.313729 s wall median and 1.111135 s scorer median. The
scorer now binds eleven offset spans directly into the controller's contiguous
proposal row instead of blitting 2.077 GB into separate tensor buffers. Across
the initial objective and all 100 rounds, reward, variance and accept/reject
decision match the previous artifact exactly. The removed copy reduced loop
time by 6.969956 s, round median by 47.549 ms and scorer median by 41.507 ms.

The directly comparable Thompson artifact is
`.cache/ennx/runs/pretrain/287ca5de00bee2653807/run-1790652894404-18283-0`:
121.869254 s loop, 1.200673 s wall median, 1.009509 s scorer median and four
accepted proposals. The UCB run was slower, but scorer time also rose by about
0.10 s, so sequential run order and thermal state prevent assigning the whole
difference to acquisition. UCB selection itself does not add model FLOPs.

The redundant proposal-to-tensor copy is gone. The remaining median gap is
266.180 ms. The scorer alone is 69.627 ms over the complete-round target, and
controller ask plus tell average 201.242 ms. The controller still scans every
realized FP16 coordinate for four proposals and materializes the selected
2.077 GB row. An implicit Rademacher distance path is not automatically exact:
FP16 rounding and accepted-weight history mean it must prove selector parity or
explicitly change the acquisition metric. Scalar acquisition tuning cannot
close this systems gap.

The pretraining path now honors `trace = true` by using the controller's native
split-command Metal timings. A ten-round 4K trace is at
`.cache/ennx/runs/pretrain/2b4002c410be9d20b1b4/run-1790655029642-27654-0`.
Excluding one invalid cross-command envelope outlier, phase medians were
101.029 ms for the four-candidate full-weight pool, 15.638 ms for selection,
50.840 ms for selected-row materialization and 166.736 ms for the split-command
GPU envelope. Splitting the normal single command buffer perturbs timing; in
particular, the one-thread selection arithmetic cannot be interpreted as 15.6
ms of useful computation. The pool and materialization measurements establish
that the two full-coordinate kernels themselves consume about 152 ms in this
trace.

## Terminal output: 2026-09-28

The tune frontend now uses `anstream` and `anstyle` for an aligned live table.
The renderer runs behind a bounded nonblocking channel; a slow or closed
terminal cannot stall GPU work or artifact creation. `run.log` remains complete
even when display records are dropped, and `tensor-updates.jsonl` retains the
full per-tensor data without formatting it inside the measured loop. The loop
now records `loop_seconds`, including round reporting, in `result.toml`.

## Direct pretraining startup: 2026-09-28

`./ennx tune examples/tuning/code-pretrain.toml` now loads the corpus, initializes
the model/controller, scores the initial model once, and enters BO directly.
It no longer runs component timings, repeated synthetic model benchmarks, MPS
reference comparisons or the extra unfused initial score. GPU completion,
finite sequence-score, dataset and controller checks remain in the BO path.
The existing `./ennx tune examples/tuning/moe-layer.toml` command retains the
diagnostic suite and its fused/unfused comparison. No new switches are required.

Pretraining `result.toml` retains actual BO measurements and identifies
`diagnostics = false`; absent diagnostic measurements are omitted, not zeroed.
The full-round goal still requires the maximum measured round to meet the target.
Tensor-update logging is unchanged. Shared buffer/pipeline allocation has not
been split in this change; only diagnostic execution is removed from pretraining.

Verification: ten 4K rounds completed in 17.6 s of runner elapsed time, versus
43.2 s in the preceding diagnostic-first run. Initial score, round rewards,
variances, proposals and decisions match; all 16,600 tensor records are
byte-identical. No diagnostic log markers appear. The round median is
1.131868 s (previously 1.132413 s); this is a startup reduction, not a BO-kernel
optimization. All ten proposals were rejected.
Artifact: `.cache/ennx/runs/pretrain/0470b43b9b448109bd1a/run-1790642258491-74906-0`.

## Per-tensor perturbation records: 2026-09-28

Pretraining runs now write `tensor-updates.jsonl` alongside `result.toml`.
Each selected proposal has one record for each of the 1,660 tensor blocks,
with tensor name, family, layer/expert indices, size, seed, radius and acceptance.
Changed counts compare FP16 representations against the current incumbent.
`squared_change` is the sum of squared realized weight differences;
`rms_change` divides that sum by the tensor size before taking its square root.
`relative_rms` divides by the fixed **initial** tensor RMS, not its current RMS;
it is null for zero initial RMS. `proposal_scale` includes the 1e-6 floor.
`realized_requested_ratio` divides realized RMS by proposal scale times radius.
That ratio reflects finite-sample Gaussian variation and FP16 rounding, not
rounding alone. Rejected records describe attempted changes, not applied updates.

These records reuse existing GPU reductions: no extra model-sized scan or
candidate evaluation. Raw block statistics are retained during the run and JSONL
is written on successful completion. The terminal shows one compact row per
round; detailed tensor data stays in the artifact instead of being formatted
inside the measured loop. Final JSON serialization remains outside the timer.

Verification artifact:
`.cache/ennx/runs/pretrain/0f6f9768849d2404e888/run-1790641648675-72810-0`.
Ten 4K rounds produced 16,600 records. Every round covers 1,038,508,544 weights;
counts, proposal radii, rewards, variances and decisions match the prior run.
All ten were rejected. Round wall median was 1.132413 s, range 1.109738--1.257869 s;
this is a verification run, not evidence of an optimization. The statistics unit
test and full-coordinate controller test pass; the known SDK linker failure
still prevents a completely green full suite.

## Full-weight restoration: 2026-09-28

The active pretraining BO path now searches the actual **1,038,508,544 FP16
weights** of the PISA/MoE model. The earlier **1,065,494,016** count belongs to
the historical dense FBT model. Do not use the old count for this architecture.

The 2,688,896-coordinate Kronecker generating descriptor has been removed from
the actual-round path. Search uses per-coordinate Gaussian noise keyed by
candidate stream, tensor ID and element index. Four proposals combine two
independent directions with two radii. There is no persistent correlated
direction, factorization, sketch or sampled-coordinate distance. The selected
complete weight row is copied byte-for-byte into the scorer's FP16 tensors.
The tied embedding/readout is one tensor. Both feedback matrices and all
normalization weights participate. FP16 rounding can leave individual weights
unchanged; each round reports the actual changed-weight count.

The existing dense Metal controller retains **two complete history rows** and
reduces block-weighted distances over every realized weight coordinate. Fixed
tensor scales are initialization RMS values, with a 1e-6 floor. Acquisition
and noisy acceptance settings remain unchanged in this restoration; they have
not been calibrated for pretraining. Different minibatches and the stored
incumbent observation still limit interpretation of acceptance as learning.

The earlier 878 ms and 978 ms figures below describe the factorized experiment.
They do **not** establish full-weight subsecond BO. The restored full-weight
path has now completed ten 4K rounds. Component probes still use their legacy
Kronecker fixtures and must not be reported as full-round proposal timings.

Artifact: `.cache/ennx/runs/pretrain/1cdaece82d9c55e9caf3/run-1790640681229-70063-0`.

| Full-weight measurement | Seconds |
| --- | ---: |
| Round wall median (upper middle) | 1.115298 |
| Round wall minimum | 1.112607 |
| Round wall maximum (first round) | 1.297977 |
| Scorer command GPU median, including full-row copies | 0.877019 |
| Controller ask + tell wall median | 0.238353 |

All ten proposals were rejected. No round met the one-second target. The initial
reward is -9.013980865, matching the previous factorized run's unperturbed model;
fused/unfused initial NLL difference is zero. Every proposal visits all
1,038,508,544 parameters in 1,660 tensor blocks. Between 828,882,617 and
1,025,267,206 weights changed after FP16 rounding at the selected radii. Each
round logs proposal radius separately from the post-update trust length,
changed weights, ask/tell wall times and scorer GPU time. Acceptance/rejection
state transitions pass the focused controller test; full-size accepted-round
latency is still unmeasured. The median is 115 ms above target; that is a
measured gap, not a forecast that a particular optimization will close it.

`./ennx test` passes the full-coordinate FP16 CPU/GPU reference test, including
tile boundaries, both Gaussian streams, exact history distances, history
eviction, and acceptance/rejection. The suite has 31 passing targets and the
existing C++ `region-arithmetic` linker/SDK build failure. The active 100-round
result is recorded above.

## Corpus preparation: 2026-09-28

`examples/tuning/code-pretrain.toml` now requests **100 rounds**. Its corpus is
ready at `.cache/ennx/corpora/d82c1394da5c7c4ec6de`. Run the same `./ennx tune`
command below; no directory arguments or environment variables are needed.

The collector uses Rust Arrow/Parquet through Apache OpenDAL and its existing
`parquet_opendal` adapter. It projects only required columns, filters forks and
completed splits, and filters file metadata before decoding content. Shard order,
repository splits, license/vendor exclusions, Unicode character limits, global
content deduplication, tokenizer, and packing semantics are unchanged. Python
`collect` remains the fixture reference, not the production reader.

Tokenizer and token pools persist independently of round-specific packed streams.
Their identity includes source revision, selection rules, character targets, seed,
and tokenizer recipe. Atomic directory publication and checksums guard cache reuse.
The current 6.7 MiB pool stage is
`.cache/ennx/token-pools/c6d35c22e11f4ffd2072`.

Measured on this machine: the initial native scan took 271.2 seconds, tokenizer
training 0.6 seconds, and tokenization 1.7 seconds. That scan preceded the final
file-metadata predicate; final cold-path speedup is **not measured**. The exact
100-round cache resolved in 0.05 seconds. Expanding to 150 rounds from the same
token pools, including checksum validation and fresh packing, took 0.08 seconds.
The example still requests 100, not 150. Its completed timing is recorded at the
top of this handoff.

The tokenizer JSON, validation/test streams, and first ten training batches match
the previous Python-prepared corpus byte-for-byte. Focused verification: 24 Python
tests and two native unit tests pass. `./ennx test`: 31 targets passed, zero test
failures, one pre-existing C++ `region-arithmetic` build failure because bundled
`ld64.lld` cannot parse the SDK's `arm64e.x1` architecture. No SDK files were edited.

## Historical factorized result: 2026-09-28 kernel sprint

These measurements predate the full-weight restoration above. The command
now runs the restored full-weight implementation:

```sh
./ennx tune examples/tuning/code-pretrain.toml
```

The measurements below ran ten one-observation noisy BO rounds on the cached Stack v3 Python
pilot corpus: context 4,096, batch two, 24 distinct layers, two feedback passes,
width 512, PISA Q8/KV1/D64/block64/top8, and 32 balanced hash top-1 experts of
width 864. Candidate search uses 2,688,896 BF16 descriptor coordinates and
Kronecker updates materialized into FP16 weights. This is not the historical
dense billion-coordinate paired-objective runner. Corpus identity is
`.cache/ennx/corpora/1282dec2b55702e33f58`; held-out splits are not BO input.

Retained changes:

- PISA Q4 selection uses SIMD reductions instead of a serial lane-zero scan.
  Smaller-node tie breaking and invalid-node handling are preserved.
- Metal TensorOps gate/up projections feed SwiGLU through cooperative tensors,
  avoiding the intermediate gate/up buffer write and read in the optimized
  scorer. FP16 rounding before the float activation is retained. Tiles use
  128 rows, 64 columns, with a 32-column tail for expert width 864.
- Parity now compares every selected block and every Q4 output against Q1,
  including an all-zero tie fixture, and rejects non-finite differences.
  The fused activation is checked against the unfused implementation.

Ten-round wall measurements, milliseconds (harness median is the upper middle
sample for an even sample count):

| Run | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Original baseline A | 963.300 | 957.365 | 966.963 |
| SIMD selector only | 886.683 | 883.268 | 890.794 |
| Original baseline B, restored for comparison | 973.918 | 968.748 | 981.604 |
| Final SIMD selector plus fused SwiGLU | 878.834 | 876.077 | 883.239 |

Final wall reduction is 8.8-9.8% against the two baselines, not a 10x result.
All 4,194,304 Q4 attention values and 7,077,888 fused activation values have
zero numerical difference against their GPU references in these fixtures.
The sampled CPU attention comparison has maximum absolute error 0.000004112.
Initial fused/unfused model NLL difference is zero. All ten logged rewards,
variances, sequence NLLs, decisions, and radii match both baselines at recorded
precision. All ten proposals were rejected: this is latency/parity evidence,
not a demonstration of successful pretraining or the actual-round accept path.
Query caching and 64-row FFN tiles did not show a reliable gain and were removed.

Artifacts under `.cache/ennx/runs/pretrain/`, each with `result.toml` and `run.log`:

- Baseline A: `e47f9b9ec7ccbcf3a147/run-1790634889854-40972-0`
- Selector only: `dfc82206d2df3e741d4b/run-1790634981057-41408-0`
- Baseline B: `b7fda85f6b27877ee831/run-1790635385861-43271-0`
- Final: `551f7bf223a5ad2edca5/run-1790635719134-44796-0`

Next kernel targets remain shared selected-block reuse and candidate
materialization. The latter must preserve `half(float(base) + delta)` before
GEMM; replacing it with separate base and delta products changes rounding.
Existing component timings are separate repeated-same-layer probes, not an
additive profile of the actual distinct-layer BO round; the FFN component probe
still uses the unfused reference. Do not use these as a post-fusion attribution.

## Historical notes

## Goal and evidence boundary

The target is a complete full-parameter BO round below one second, with
repeated-round measurements. Pretraining from random initialization and
post-training are research goals, not demonstrated outcomes of the current
synthetic benchmark. The implemented controller uses local perturbations;
full-parameter reach does not establish global optimization.

## 2026-09-28 grouped-MoE NAX result

The proposed sparse-FFN feasibility gate is now implemented as
`./ennx tune examples/tuning/moe-layer.toml`. It uses 8,192 rows, width 512,
32 balanced top-1 experts, expert width 864, and exact dense full-rank
Kronecker candidate updates. The candidate is fused into the B-tile load of
MLX v0.32.1's Metal 4 NAX cooperative-tensor gather MatMul at
`BM64 x BN128 x BK128`; no candidate expert matrix is materialized. The pinned
MLX source and MIT license are under `rust/crates/ennx/src/third_party/mlx/`.

A temporary independent Metal harness compared the fused gate/up output with
materialize-then-NAX GEMM for 884,736 FP16 outputs. All outputs were bitwise
identical. Its task-shaped isolated medians were 6.344 ms for the fused
gate/up projection and 3.434 ms for fused down.

The repository's nine-sample run is
`results/moe-layer/run-1790611182190-22670-0`:

- routing plus grouping: 0.363 ms GPU;
- fused gate/up: 6.196 ms GPU;
- SwiGLU: 0.420 ms GPU;
- fused down: 3.392 ms GPU;
- ungroup plus gated residual: 0.281 ms GPU;
- complete layer: 11.281 ms GPU and 11.426 ms wall;
- 24 layers times two FBT passes: 548.440 ms by direct layer extrapolation.

The 548.440 ms figure is one objective call. The active one-observation runner
also performs one objective call per timed BO round, or 48 layer executions.
The isolated MoE projection therefore leaves 451.560 ms for attention,
projections, readout, feedback, loss, controller work, and all integration
overhead. The older paired runner required 96 layer executions and projected
to 1,096.880 ms for the FFN alone; that figure is historical rather than the
active round shape. The 14 ms per-layer gate remains only a component gate.

The layer's 14 ms gate passed. This is not an end-to-end BO result. The
extrapolation excludes attention, non-expert projections, normalization,
feedback construction, readout, loss, and controller work. It also uses
balanced synthetic routing and fixed finite values rather than a trained
router distribution. `goal_met = true` in this artifact means only that the
complete expert layer met its configured gate. The complete subsecond research
target remains unmet until the remaining scorer is implemented and measured
together at context 4,096, batch two, and two feedback passes.

## 2026-09-28 PISA and FP8 feasibility gate

The proposed 4K architecture uses width 512, 8 query heads, 4 KV heads, head
dimension 64, and PISA's published pretraining settings `C=64`, `K=8`, `g=2`.
For a full-size query, hierarchical routing evaluates 64 intermediate child
summaries plus 1,024 raw keys at the leaf level. Selected attention then
recomputes QK and applies probabilities to V over 512 keys. Counting a
multiply-add as two operations gives this upper scheduled work per layer-pass:

- QKV and output projections: 12.885 billion operations;
- PISA routing: 9.127 billion operations;
- selected QK plus probability-times-V: 8.590 billion operations;
- non-FFN subtotal: 30.602 billion operations.

The PISA attention subtotal is 17.717 billion operations. Dense causal
attention at the same batch, context, heads, and head dimension is 17.184
billion useful operations. PISA therefore does not reduce the 4K arithmetic
count under these settings; its benefit is asymptotic selection at longer
contexts. This matches the paper's H100 selection benchmark: at 4K, PISA Q4
selection is 0.701181 ms versus 0.167007 ms for BSA, while PISA becomes the
reported winner only from 32K. Source:
`https://arxiv.org/pdf/2609.31093`, Table 4 and Appendix C.

Across the active round's 48 layer executions, the measured MoE arithmetic plus
the projection and PISA counts total about 2.530 trillion operations. An
8K-vocabulary readout and two width-512 feedback projections bring the complete
one-observation estimate to about 2.607 trillion useful operations before
normalization, activation, softmax/LSE, Top-K, pooling, routing movement, loss,
or controller work. The round therefore requires more than 2.607 useful TF/s
sustained end to end. The older paired-round count was about 5.213 trillion.

An exact-shape MLX 0.32.1 probe did not establish FP8 as the missing speedup.
Its Metal MXFP8 path decodes packed weights into NAX operands. Against FP16,
MXFP8 was effectively unchanged for grouped gate/up, about 16% faster for
grouped down, slower for QKV, and about 14% faster for the output projection in
that run. macOS 27 exposes FP8 tensor data types, but a direct MPSGraph
`Float8E4M3` matmul was rejected by the runtime as an unsupported matmul
operand. The active Xcode is 26.1.1; its SDK does not expose the new Swift FP8
case, while the separate command-line 27.0 SDK cannot be linked by that older
toolchain. Native FP8 throughput on this M4 remains unmeasured and must not be
assumed to be 2x FP16.

Decision: do not implement the published two-stage PISA schedule yet. At 4K it
adds routing work without creating the required round budget. The next
feasibility gate is either a usable native FP8 matrix path measured on the exact
MoE/projection shapes, or an algorithmic change that reduces the `C=64`, `K=8`
attention work while preserving the intended BO and model semantics.

There is also an implementation boundary that must not be blurred. The live
end-to-end BO controller in `bf16_search.metal` generates a dense correlated
Gaussian direction for every BF16 parameter coordinate and materializes one
1.065-billion-coordinate candidate row. It does not generate the Kronecker
factors used by the isolated MoE probe. The MoE probe's width-512, 32-expert
model is likewise not wired into the current dense LocalV1 scorer.

Kronecker structure does not by itself remove the paired incumbent forward. For
one grouped expert matrix, the probe uses
`DeltaW[(ko,ki),(no,ni)] = outer[ko,no] * inner[ki,ni]`. A separate exact
factorized correction can evaluate `X * DeltaW` as two contractions. Across the
probe's gate/up and down projections this is 1.216 billion operations per layer,
versus 21.743 billion for their dense products. The existing NAX kernel is
already more economical: it adds each factor product to a loaded weight tile
and performs the original dense MMA, charging about 0.340 billion additional
source operations. After the perturbed embedding, however, incumbent and
candidate activations differ. Every later projection therefore still requires
two distinct base-weight matrix products for exact paired scoring. A paired
kernel may reuse weight-tile reads, but it cannot remove the second product.
The measured expert GEMMs have roughly 145--156 operations per logical byte, so
weight-read sharing is not evidence for the required twofold wall-time gain.

The active runner now implements one-observation noisy BO. It scores the initial
incumbent once outside the timed loop, then appends one independently sampled
candidate observation per round. Candidate reward is negative mean NLL;
variance is estimated from the two example losses. A candidate is accepted only
when its reward improvement over the stored incumbent exceeds two times the
square root of their summed variances. Rejected candidates still enter history,
while trust-region success/failure follows the accepted incumbent. This removes
the common-random-numbers paired comparison and halves scorer work, but the
two-example variance estimate and threshold are a policy rather than a
calibrated noisy-GP guarantee.

For the proposed width-512 MoE/PISA model, one objective call is about 2.607
trillion counted useful operations and 48 layer executions. The measured MoE
alone projects to 548.440 ms. The exact-shape FP16 QKV and output medians project
to roughly another 466 ms over 48 layers, already exhausting one second before
PISA attention, feedback, readout, normalization, routing, loss, or controller
work. Thus one-call noisy BO is necessary for this design's budget but not
sufficient with the measured FP16 component implementations. It must be paired
with a demonstrated matrix-throughput increase or a smaller 4K attention
schedule. Primary references: Eriksson et al., "Scalable Global Optimization via
Local Bayesian Optimization," NeurIPS 2019, and BoTorch's noisy-model
documentation.

The old MPS/materialize probe was removed rather than retained as a second
path. A split, ordered command-buffer submission removed a measured scheduler
penalty without adding CPU waits. The earlier plan to implement a complete
PISA non-FFN layer is superseded by the feasibility result above. A composed
non-FFN implementation follows only after a native FP8 measurement or an
algorithmic reduction establishes a complete-round budget that can close.

## Configuration and cleanup update

The controller removes the artificial zero-reward observation. A separate
initial incumbent score initializes search history and TuRBO before the timed
loop. Every measured round then performs one candidate objective call. The FBT
study explicitly sets a four-failure contraction budget; three
successes expand the length under the existing absolute-reward comparison.
Restart seeds fresh controller history from the current measured incumbent.
Round artifacts now distinguish proposal radius from trust-region length and
include success/failure counters and restarts. Earlier measurements below used
the previous controller and are historical performance evidence.

Validation: `./ennx test` passed all 30 targets (build
`e0d62622-8d51-4924-842a-e26f35d500c6`); `sh tools/fbt-bo --check` passed
all five scorer diagnostics (build `c8274671-5b00-48d8-b1e4-859b37393e60`).
An existing MPS batch-view test exposed missing row/column origins for nonzero
offsets; `Matrix::layout` now derives both origins from that offset, and the
sentinel and numerical checks pass.

The full ten-round run at
`results/turbo-enn-controller-fix/run-1790461457701-48158-0` completed in
284.594384 seconds: 28.459187 seconds/round mean, 27.276512 minimum and
29.986257 maximum. All rounds retained context 4096, batch two, two objective
calls and eight sequence-level transformer passes. Allocation stayed at
16.0615 GiB; all ten proposals were rejected. Trust length contracted from
0.01 to 0.005 after round eight. This verifies reachable adaptation, not
optimization quality or a speedup; `goal_met = false`.
Round four counted as a controller success despite losing to its paired
incumbent: candidate NLL 12.000589912 versus incumbent 11.999115366. Absolute
rewards from changing minibatches still drive adaptation independently of the
paired acceptance decision. That statistical limitation remains unresolved.

Cleanup on 2026-09-26 removed the reduced-context and single-pass study options.
The current runner fixes context at 4,096 tokens and scores one candidate with
two FBT passes every round, after one separately reported initialization score.
Historical paired and reuse artifacts remain available, but their timings do
not describe the active observation policy. Use
`./ennx tune examples/tuning/turbo-enn.toml`.
The unused threadgroup-memory scratch crate, its duplicate standalone source,
and an editor backup were removed from the source tree.

The active BO runner now accepts versioned typed TOML; see the
[runbook](turbo-enn.md). The paired TensorOps FFN path and fusion-only tests
were removed after the regression documented below. The normal MPS path,
feedback scoring, reference parity and accept/reject restoration tests remain.
The removed implementation is recoverable from JJ revision
`f9cfd9cbdd6c6a278b35d75a7fb2890b8eb37f24`.

## 2026-09-28 one-observation full-size result

The active implementation passed all 30 `./ennx test` targets, including 528
core unit tests and a focused noisy-decision boundary test. The full 4K,
batch-two, three-round run is
`results/turbo-enn/run-1790613032204-30491-0`. It completed successfully with
`observation_policy = "one_observation_noisy_bo"`.

- separate initial objective: 36.165565 seconds;
- timed rounds: 35.517759, 34.561049, and 33.844689 seconds;
- timed-round mean: 34.641166 seconds;
- each round: one objective, two sequence scores, four transformer passes;
- candidate scorer mean: 31.962359 seconds;
- allocation: 16.0615 GiB;
- accepted candidates: zero of three;
- target result: `goal_met = false`.

Round three had candidate NLL 12.002395344 versus stored incumbent NLL
12.007460594. It was rejected because the 0.005065 reward improvement did not
exceed the 0.022028 threshold from the summed incumbent and candidate variance.
This is expected under the implemented conservative rule and demonstrates that
"lower observed NLL" is not synonymous with acceptance under noisy BO.

The active dense round still contains about 36.226607 trillion counted useful
operations and achieved about 1.046 effective counted TFLOP/s end to end in
this run. Reaching one second with the same arithmetic would require 36.227
counted TFLOP/s plus uncounted scalar and controller work. One-observation BO
removes one scorer call, but it does not by itself close the scorer gap.

The follow-up artifact review removed both redundant `bench_gemm_shapes` probes,
the zero-byte `src/fbt_prefill.metal.bak`, and these unreferenced stale mutators:
`debug_mps.py`, `debug_mps2.py`, `fix_metal_warning.py`, `fix_mps.py`,
`fix_print.py`, `fix_salt.py`, `patch_prefill.py`, and `print_half.py`.
The retained `bench_gemm_layouts` diagnostic covers all active exact shapes with nonzero
operands, correctness checks, rotated variants and GPU timing.

The current runner writes `study.toml`, `source.txt`, `run.log`, `result.toml`
and `exit.txt` beneath a unique run directory. `ConfigOverrides` is the single
schema for both TuRBO-ENN optimizer overrides and round-study controls; there is
no separate run configuration type. The normal test suite includes CLI tests
and parses the actual example file through that shared schema. Standalone
FBT/BF16 integration crate roots also import the shared schema.

Historical validation for the retired launcher: all 11 schema/CLI tests passed (build
`c304f66a-89a6-4b8d-83ab-153de7528105`), the CPU preset geometry check passed
(`063192aa-e173-4aae-a1a9-3fd49375ac1f`), all five retained GPU checks passed
(`edffb396-a760-4eb7-bcdd-ae2865f01dc4`), and the native runner build passed
(`b0a96edf-d400-418e-8a60-6849e2019e25`). Both standalone integration targets
passed, totaling 50 tests (`ceb24a55-d32f-4037-a090-396423f46768`). These build
identifiers do not verify the current artifact contract.

After the integration import fix, `./ennx dev` passed formatting, wheel checks
and Python tests for 3.12/3.14/3.13, and all 31 Rust/kernel test targets.
No complete full-size before/after or configured-versus-legacy run was executed
in the 4 GiB session; the recorded full-size allocation exceeds 16 GiB.

A later 20 GiB session completed the retired launcher's deferred full-size interface qualification
at source revision `e9dd35cac759575089c989ef02effea5c82113ee`. The typed TOML
path and legacy `--rounds 3` adapter ran sequentially for three rounds each.
Their `resolved.toml` and `metadata.toml` artifacts differ only in the required
unique output path. Both paths printed the same old/new NLL pairs in every
round—12.007460897/12.008238873, 12.036472807/12.037817242, and
12.023292716/12.020449622—the same rejection decisions, and radii 0.005, 0.005,
and 0.02. Both exited successfully. The configured loop took 192.799135 seconds
(64.266378 seconds/round mean; native build
`5f9bd935-8c45-486b-ba22-cebaf12b1583`); legacy took 186.358168 seconds
(62.119389 seconds/round mean; native build
`e407bdb4-5c92-4df2-bedb-84d101a99864`). Allocation was 16.1474-16.1552 GiB.
Local evidence is preserved in `results/bo-local-v1/` and
`results/bo-1790276402531768000-99287/`; both directories are ignored run
artifacts rather than portable source evidence.
This satisfies the plan's configured-versus-legacy acceptance check. It is
aggregate-loss and decision parity to printed precision, not bitwise tensor or
all-token parity, and it is not a comparison against a pre-cleanup executable.
The subsecond research target remains unmet.

After the artifact cleanup, the 11 schema/CLI tests passed (build
`28d69711-3cad-470c-bf99-88f153743e9e`), the standalone FBT integration target
passed 38 active tests with 11 intentional ignores (build
`ea9fdf3d-19d0-4df7-b4ae-7820d7681613`), and all five retained GPU checks
passed (build `d2e1ef29-fbff-41d6-81c3-fa430ccedc1b`). The full `./ennx dev`
gate then passed formatting, all three wheel/Python suites, and all 31
Rust/kernel targets (build `1bc813c8-3a8e-40b1-bc51-4d8acd2c93aa`).

The following FFN sections are historical evidence, not current commands or
active interfaces. Cleanup does not establish a new performance result.

## Latest result: full-round FFN epilogue fusion regresses

The retired three-round `--fused-ffn` runner used the paired TensorOps gate/up
GEMM plus SwiGLU inside the unchanged BO loop. MPS remains the default.
Canonical BF16 weights are converted directly into the paired FP16 cache on
each model revision; no extra full-model weight cache is added. The 208 MiB
gate/up activation intermediate is no longer allocated with fusion enabled.
The traced path records `gate_up_glu` and identifies paired weight packing.
Switching fusion off invalidates the layout and workspace before reuse.

Serial uninstrumented MPS / fused / MPS measurements, 2026-09-24:

| Path | Complete rounds, seconds | Loop total | Mean |
|---|---|---:|---:|
| MPS before | 63.466039, 68.745705, 67.484538 | 199.696296 | 66.565432 |
| Paired fused | 76.214695, 74.275190, 68.977331 | 219.467255 | 73.155752 |
| MPS after | 67.581552, 66.182383, 60.322149 | 194.086131 | 64.695377 |

Fusion is 11.47% slower than the combined six-round MPS mean in this sample.
It is not a demonstrated speed optimization and must not become the default.
Allocation is stable at 15.9443 GiB fused versus 16.1474 GiB MPS. All six
reported old/new mean losses match to the printed nine decimal places across
all three runs, as do all decisions (rejected). This is not an all-token,
bitwise full-size parity check. The full subsecond objective remains unmet.

All seven checks in `tools/fbt-bo --check` passed. New integration coverage
includes standard/two-pass scoring against the reference, forced accept/reject
restoration, revision-dependent traced packing, traced/untraced score equality,
and toggling the layout back to MPS and back to fusion. Trace operation counts
now account for fusion and the actual number of readout chunks instead of
assuming four chunks for every sequence length.
`./ennx test` passed all 30 Rust/kernel targets (build
`754c4b6d-4470-4ff4-a079-b781fce870cb`); `./ennx fmt --check` passed.
The broader suite log is `.cache/fbt-epilogue-suite.log`.

Builds: check `e932755a-7d0e-4648-bcf4-beca9f971cbc`; MPS before
`bde41746-9ba0-454b-9947-8a97ae7efd1e`; fused
`1f8c9e12-8ed9-473d-b745-f0456ab1f7dc`; MPS after
`8b995a05-9d93-4192-8734-ec125433af8a`.
Logs: `.cache/fbt-fused-integration-check.log`,
`.cache/fbt-bo-epilogue-{mps,fused,mps-after}.log`.
The first MPS test passed but its shell wrapper reported `--timeout: command
not found` afterward, following an in-flight help-text edit. The script passed
syntax checking; the fused and repeated MPS invocations completed cleanly
without in-flight script edits. Do not label the first wrapper invocation a
clean CLI pass.

The next performance question is matrix throughput, not whether the epilogue
removes its intermediate: that removal is demonstrated. The earlier MPS capture
has a 208-by-128 group grid for the 8192-by-13312 output, consistent with
64-by-64 outer tiles. The fused kernel uses 32-by-128 matrix tiles. Their input
reuse differs; this is a candidate explanation to investigate, not measured
attribution of the regression.

## Initial isolated FFN epilogue experiment

`gpu_scorer_ffn_fusion` in `fbt_prefill.rs`, run by
`tools/fbt-bo --check`, compares MPS packed gate/up plus the separate SwiGLU
kernel with four MPP TensorOps variants in `fbt_tensor.metal`. At this initial
measurement, the variants were test-only; full-round results are now above.

Three variants perform two inline matrix operations and consume gate/up
cooperative tensors without a device intermediate. The fourth,
`fbt_tensor_glu_paired_32_64`, uses adjacent gate/up weight columns, one 32-by-128
matrix operation and 4 KiB threadgroup storage to pair outputs by explicit
coordinates. It does not assume an undocumented cooperative-tensor lane layout.
All preserve FP16 GEMM-output rounding before FP32 activation and multiplication,
then store FP16 activations. The paired weight reorder is CPU test setup and
EXCLUDED from these isolated GPU timings. The full-round opt-in above includes
revision-aware GPU packing instead.

Build `8953d84c-f402-43e6-bd00-0e5ce28484c5`, 2026-09-24:

| Production-shape operation | Median GPU milliseconds |
|---|---:|
| MPS gate/up plus SwiGLU | 239.568 |
| Two-operation TensorOps, 64-by-32 | 258.301 |
| Two-operation TensorOps, 64-by-64 | 240.070 |
| Two-operation TensorOps, 32-by-64 | 239.338 |
| Single-operation paired TensorOps, 32-by-64 | 230.910 |

Each has 13 measured samples following three warmups, with rotating order.
MPS spans 224.016-264.666 ms; paired spans 206.018-273.175 ms. The paired
sample median is 3.6% lower, with overlapping distributions; this is not proof
of a repeatable gain. The earlier 7% two-operation median advantage did not
repeat in this run. Do not promote either result to a round-level claim.

All six GPU checks passed. All output elements at (rows, FFN width, input width)
(65, 68, 96) and (8192, 6656, 1536) were finite and numerically identical to
MPS on the deterministic fixtures, with intact output-tail canaries. This is
not universal or full-model parity. Scoped Rust formatting and CLI formatting
checks passed. No production behavior was changed by this experiment.

The historical paired round had 96 layer executions; the active
one-observation round has 48. Eliminating the 208 MiB packed gate/up
intermediate would therefore remove 19.5 GiB of logical write/read traffic per
active round. Output and down GEMM residual epilogues could each remove another
2.25 GiB. Tiled readout cross-entropy could avoid 4.59375 GiB of logits
writes/two scans, less its new partial-statistics traffic. These are
source-level accesses, not measured DRAM bytes. None of these fusions removes
the corresponding dense GEMM work.

## Latest implementation: bounded attention output

On 2026-09-24, repaired `fbt_prefill_flash_attention_half_96`: the old output
stage wrote 12,288 bytes of FP32 output through a 6,144-byte FP16 `q_tile`
allocation. The new output stage reuses each SIMD group's existing score tile
in three 32-column slices, with SIMD-group memory barriers. Normalization and
gating happen in the stores, removing twelve final diagonal matrix multiplies
per SIMD group without increasing declared threadgroup scratch. The key-tile
accumulator rescaling matrix multiplies remain unchanged.

At the attention repair, `tools/fbt-bo --check` ran five checks. The independent FP64
attention reference covers two samples, four query heads/two KV heads, dimension
96, lengths 1/31/32/33/65/4096, causal and local masks, gates, partial tiles and
output canaries. All positions are checked at short lengths; six boundary
positions per head are checked at 4K, with windows 0 and 2048. All five checks
pass. The 21 existing scorer comparisons produced identical mean losses before
and after the patch; this does not establish full-size reference parity.
`./ennx test` passed all 30 Rust/kernel targets; `./ennx fmt --check` passed.

Fresh uninstrumented legacy three-round measurements:

| Run | Complete rounds, seconds | Loop total | Mean |
|---|---|---:|---:|
| Before repair | 67.120427, 66.014191, 69.838695 | 202.973429 | 67.657810 |
| After repair | 67.303598, 62.856918, 61.021262 | 191.181817 | 63.727272 |

Allocation stayed at 16.1474 GiB and all proposals were rejected in both runs.
The 5.81% lower sample mean is not an established causal/repeatable speedup.
The six full-size mean losses differ by up to 0.000908883 despite fixed seeds;
the reason is not established. Do not claim bitwise full-size equivalence or
that the previous out-of-bounds implementation was a correctness reference.
The subsecond goal remains unmet.

Before build: `96ec498d-e8fb-48e7-92e9-9c2efd6d25b1`.
After build: `8a9b0ea7-2356-4469-8113-d675fad286b1`.
Ignored logs: `.cache/fbt-bo-{before,after}-attention-output.log` and
`.cache/fbt-attention-{before,after}-check.log`.

## Earlier measurements

User-reported post-fix three-round GPU run: 200.189331 seconds total,
66.729777 seconds/round mean; allocation 16.1474, 16.1474, 16.1512 GiB.
Build ID: `a8ca07ff-e8ab-4e81-abdd-1ca88f1c270b`, before the macOS 27 upgrade.
Do not treat capture-enabled or older hybrid timings as the current baseline.

Earlier uninstrumented macOS 27 run, build ID
`d886d4c9-7b28-4fe6-a37c-7251d05be21c`: three rounds passed in 275.324042
seconds, or 91.774681 seconds/round by loop time. Individual complete-round
times were 93.281051, 93.968275, and 88.074587 seconds. The six scoring calls
sum to 270.774612 seconds (98.35% of loop time); their twelve whole-pass Metal
GPU intervals sum to 263.429756 seconds (95.68% of loop time). These are not
per-operation timings. The loop is 37.53% slower than the earlier pre-upgrade
sample; no causal attribution is established by the two single runs.

The [compute/memory audit](bo-complexity.md) found a full-size optimized FFN
down-weight format mismatch: the fallback declares canonical BF16 bytes as
FP16. The small optimized fixture does not exercise that shape-specific branch.
The mismatch is now fixed. A production-width, short-sequence FFN regression
failed before the fix and passed afterward; all four GPU checks passed.
This is not full-model, full-context reference parity.

The [native profiling probe](native-profiling.md) now joins every logical
operation to Metal System Trace and isolates dominant scorer operations for
`gpudebug` hardware-counter replay. Gate/up, down, and QKVG use Apple's private
`mmul_kernel_a16_half_half_float` path; flash attention is a custom kernel.
Readout expands one logical GEMM into 52 private dispatches and still lacks a
trustworthy per-dispatch ranking. Read that report before claiming kernel
bottlenecks or adding instrumentation.

The exact-shape `tools/fbt-bo --gemm` probe now also compares the existing
single-output custom Metal kernels for gate/up. With identical operands,
rotated order, seven measured samples, bitwise FP16-output comparison to MPS,
and output-canary checks, median GPU times in the 2026-09-24 sample were
`238.107 ms` for MPS NN, `317.487 ms` for custom 64-by-64, and `310.790 ms` for
custom 128-by-64. Both custom implementations are rejected for production;
they were 33.3% and 30.5% slower. The same probe's MPS throughput was lower
than the earlier normal-power sample, which reinforces the unresolved
machine-state variability rather than establishing a source regression.

## Start here

- [TuRBO-ENN runbook](turbo-enn.md): commands, exact round, acceptance and tests.
- [GPU workload](m4-scoring-design.md): execution paths and arithmetic accounting.
- [Compute/memory audit](bo-complexity.md): per-operation costs and source findings.
- [Architecture contract](metal-contract.md): LocalV1 model semantics and uncertainties.
- [Measurement checklist](../kernel-architecture-plan.md): evidence required for changes.
- [Sprint requirements](full-space-bo-sprint.md): broader research requirements.

## Commands

```sh
./ennx tune --help
./ennx tune examples/tuning/turbo-enn.toml
./ennx dev
```

The CLI invokes Buck2. No caller-supplied environment variables are required.
Use JJ for repository operations. Do not invoke Git or reset unrelated changes.

## Operation trace

`--trace` is a profiling-only execution path. It preserves queue order, assigns
one command buffer to each logical GPU operation, commits without an
intermediate CPU wait, and waits once after the final operation. A two-pass
objective call records 545 scorer GPU operations plus host input upload and
loss reduction. A stale model layout adds 99 GPU preparation records: 49
weight transposes, two feedback-weight transposes, 24 QKVG packs, and 24
gate/up packs. Each record includes source operation, pass, layer, shape, CPU
encode/submit time, and Metal GPU start/end/duration. The command also emits
operation-type aggregates.

The controller profiling switch similarly separates acquisition pool scoring,
selection, and proposal materialization. `tell` separates an accepted reference
update from the history copy. Normal file-driven tuning does not use these split
command buffers.

Verified 2026-09-24 with one complete traced round:

- incumbent: 548 records (one cached-layout host record plus 547 scorer
  records), 44.343679 s wall, 42.410735 s GPU envelope;
- candidate: 646 records (99 preparation plus 547 scorer records), 41.345594 s
  wall, 41.338442 s GPU envelope;
- both interval ledgers reconcile exactly as
  `sum + gaps - overlap = envelope`;
- candidate layout preparation used 0.105454 s summed GPU time;
- ask GPU envelope: 1.932634 s, overlapped with incumbent scoring;
- rejected tell: 0.059041 s history-copy GPU time, 0.746645 s tell wall;
- complete traced round: 86.813883 s.

The raw ignored artifact is `.cache/fbt-bo-operation-trace.log` (1,279 lines).
This is a perturbative source-operation trace, not a production benchmark and
not hardware-instruction visibility inside opaque MPS GEMMs. It now attributes
all GPU work submitted by the scorer, weight-layout preparation, acquisition,
and `tell` at the logical source-operation boundary. Host bind, decision,
restore, synchronization, and accepted rebind remain enclosing phase timings,
not per-loop-iteration records.

Do not use the split trace's wall time as production performance. A complete
process-launched Metal System Trace joined all 1,199 application encoder IDs to
1,196 command buffers and 13,276 GPU activity slices, but splitting the round
into 1,193 profiling command buffers induced 864 driver `Wait for GPU` events
totalling 77.701115 seconds. The joined ignored ledger is
`.cache/fbt-bo-wire3-ledger.csv`.

An unchanged capture-enabled production path used 10 command buffers and 24
application encoders. Its 5,183 GPU activity slices had an 83.022380-second
active-interval union over a 90.907212-second GPU span, with no driver
`Wait for GPU` events. The capture process lifetime was 95.063358 seconds, so
it remains instrumentation evidence rather than an uninstrumented baseline.

A compact `gpucapture` of one production conversion buffer contains 99 named
dispatches. `gpudebug` replay produced a 70.28 ms profile and nonempty shader
cost rankings: gate/up packing 48.21% (24 invocations), transpose 41.59% (51),
and QKVG packing 10.20% (24). Compiler and executed-instruction counters plus
occupancy and limiter data are available per command.

The targeted scorer method is now implemented as `--capture-op NAME` with
optional `--capture-occurrence N`. It runs the exhaustive trace and pauses
immediately before the selected command buffer so `gpucapture` cannot race to
a neighboring operation. It does not alter normal benchmark execution.

Isolated replay established the following hardware facts:

- gate/up: private MPS kernel, 27.08% occupancy, 95.55% F32 limiter, 2.94 GiB
  device reads, 41.99% LLC miss rate;
- down: private MPS kernel, 21.41% occupancy, 82.21% F32 limiter, 3.25 GiB
  device reads, 63.70% LLC miss rate; profile consistency passed;
- QKVG: private MPS kernel, 21.59% occupancy, 97.45% launch limiter, 1.05 GiB
  device reads, 28.38% LLC miss rate; profile consistency passed;
- flash attention: custom kernel, 15.08% occupancy, 95.31% launch limiter,
  zero half-ALU share, and 93.60% buffer-L1 miss rate.

Readout is one logical MPS encoder but 52 actual private dispatches: 13 groups
of three identity operations plus one matrix multiply. Both replay modes were
inconsistent and produced false single-dispatch attribution, so no readout
kernel ranking is accepted. The full details, instruction counts, grids, and
measurement limits are in [native-profiling.md](native-profiling.md).

## Verified before this documentation cleanup

- After rejecting and removing a profiler-label experiment, four consecutive
  `tools/fbt-bo --check` invocations passed all four checks. The clean
  three-round benchmark above also passed.
- After adding the operation-selective capture window, `tools/fbt-bo --check`
  passed all four GPU checks. A targeted gate/up trace also completed its full
  one-round test successfully.
- `./ennx dev`: passed. Python 3.12/3.13: 908 passed each; Python 3.14:
  909 passed. Rust/kernel suite: 30 targets passed.
- Local verification log: `.cache/gpu-only-dev.log`. It is ignored, not a
  portable committed artifact.
- Small-fixture parity is not full-size parity or performance evidence.

## Source map

| Responsibility | Source |
|---|---|
| Typed CLI and diagnostic adapter | [turbo_enn.rs](../../rust/crates/ennx/src/turbo_enn.rs), [tools/fbt-bo](../../tools/fbt-bo) |
| Complete round and GPU parity tests | [fbt_round.rs](../../rust/crates/ennx/src/fbt_round.rs), [fbt_modeltests.rs](../../rust/crates/ennx/src/fbt_modeltests.rs) |
| GPU search controller | [bf16_metal.rs](../../rust/crates/ennx/src/bf16_metal.rs) |
| Perturbation and acquisition kernels | [bf16_search.metal](../../rust/crates/ennx/src/bf16_search.metal) |
| Model and weight ownership | [fbt_model.rs](../../rust/crates/ennx/src/fbt_model.rs) |
| Batched scorer | [fbt_prefill.rs](../../rust/crates/ennx/src/fbt_prefill.rs) |
| GPU kernels and MPS descriptors | [fbt_prefill.metal](../../rust/crates/ennx/src/fbt_prefill.metal), [fbt_mps.rs](../../rust/crates/ennx/src/fbt_mps.rs) |

## Known benchmark limitations

The harness initializes search from the first measured incumbent loss and its
variance. Later rounds start acquisition before rescoring the incumbent.
Synthetic examples change each round while historical candidate scores are
retained, and radius adaptation compares absolute rewards across those batches.
The explicit failure budget does not establish that this history predicts
improvement or that the controller is calibrated for minibatch variation.

The benchmark has no real training dataset or pretrained checkpoint. It is not
a pretraining or post-training quality evaluation.

## Removed implementation

ANE bridge, attention/FFN offload, timing fields, probe command and associated
patch scripts were removed at the user's request. Backend-selection flags are
not supported. Historical documents before cleanup are available through
`jj file show -r 5a10dbc1 docs/handoff.md`; they are not current instructions.
