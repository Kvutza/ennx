> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# BO Compute and Memory Audit

Source inspection: 2026-09-23, JJ snapshot `c176d9e8`. No timing experiment.
Scope: `./ennx tune examples/tuning/turbo-enn.toml`, optimized GPU prefill.
Counts describe executed source paths; they do not certify the output or
predict wall time. Big-O denotes total work, not GPU elapsed time. MPS internal
tiling, scratch and instruction counts are opaque to this repository.

## Notation

| Symbol | Meaning | Current value |
|---|---|---:|
| P | Independent learned coordinates | 1,065,494,016 |
| S | Parameter tensors | 243 |
| B | Examples in each scoring call | 2 |
| T | Tokens per example | 4096 |
| N | B * T | 8192 |
| L | Layers | 24 |
| d | Hidden width | 1536 |
| f | FFN width | 6656 |
| V | Vocabulary size | 100352 |
| h | Query heads | 16 |
| k | Total KV width: KV heads * head dimension | 768 |
| a | Head dimension: d / h | 96 |
| W | Local attention window | 2048 |
| F, E | Passes per score, scores per timed round | 2, 1 |
| C | Candidate pool size | 4, hard-coded |
| H | Active history rows | 1 initially, at most 2 here |
| J | Sum over tensors of ceil(elements / 65536) | 16,321 |
| R | Completed rounds | Grows throughout a run |
| c | Optimized readout row chunk | 2048 |

P = V*d + 2*d*d + d + L*(2*d + 2*d*d + 2*k*d + h*d + 3*f*d).
This counts the tied embedding/readout once. Source:
[ModelConfig::memory and parameter construction](../../rust/crates/ennx/src/fbt_model.rs).

## Setup, outside the round timer

| Computation | Work | Storage/lifetime |
|---|---|---|
| Model validation and layout | O(S) | O(S) metadata |
| Random BF16 parameter initialization | O(P) CPU hash/convert/write | 2P bytes canonical weights |
| Reference scorer caches/workspaces | Allocation sized by O(L*T*k + L*chunk*d + T*d + chunk*(d+f+V)) | Remain allocated with batched prefill |
| Per-tensor RMS and flattened initial search row | O(P) CPU, FP64 sum of squares | Temporary 2P-byte host row; O(S) scales |
| Search allocation and initial history copy | O(P) initialization/copy; capacity-sized allocations | Base, proposal, history; see memory section |
| GPU prefill allocation | O(number of buffers) host calls; requested sizes below | Persistent workspace |
| Initial weight transpose/conversion/packing | O(P) upper bound, actual packed subset | Persistent FP16 weight layouts |
| Pipeline/library/MPS compilation and object creation | Runtime-dependent; no repo-derived asymptotic compiler model | Cached runtime objects |
| Initial incumbent objective | One score: B=2, F=2, 48 layer executions | Reported as `initial_objective_ms`; excluded from timed rounds |

`correlate(42)` does not initialize the full reference row immediately.
The first `begin_ask` allocates/generates that row and reduces its per-tensor RMS:
O(P + J + S) work, 2P additional resident bytes and a completion wait. This
first-use cost is INSIDE the first round timer.

Sources: [model initialization](../../rust/crates/ennx/src/fbt_model.rs),
[`model_search`](../../rust/crates/ennx/src/fbt_modeltests.rs),
[`SearchState::new` and `init_reference`](../../rust/crates/ennx/src/bf16_metal.rs).

## Per-round controller work

| Computation | Work | Extra storage and traffic |
|---|---|---|
| Generate synthetic tokens and targets | O(N) CPU | O(N) integers |
| Generate candidate noise, round weights and accumulate distances | O(C*H*P), C=4 fused in one scan per history pair | Reads base/reference/history; candidates stay in registers, not four full rows |
| Accumulate pool diagnostics | O(C*C*P*ceil(H/2)) in the generalized count; C fixed here | Norms, six pairwise dots, four reference dots, changed counts; O(J) geometry |
| Reduce tile distances and choose candidate | O(C*H*J + C*H*H + C*H) | **One GPU thread** sums tiles and insertion-sorts neighbors |
| Materialize selected BF16 candidate | O(P) GPU | Regenerates selected noise, reads base/reference, writes 2P bytes |
| Publish acquisition result | O(J + C*H + S) CPU for fixed C | Copies shared-buffer geometry/partials into Vecs, validates and reduces them |
| Bind candidate parameter views | O(S) CPU | O(S) buffer objects; no full weight memcpy |
| Noisy decision from candidate mean/variance and stored incumbent | O(B), currently constant-sized | O(B) scalars |
| Restore rejected parameter handles | O(S) CPU | No full weight memcpy; increments revision |
| Store candidate in retained history | O(P) copy every round | 2P bytes copied; approximately 4P logical read+write bytes |
| Accept: update stored direction and RMS | O(P + J + S) | Regenerates direction, BF16 write and reductions |
| Accept: copy candidate to base and bind it | O(P) copy + O(S) handles | Another 2P bytes copied; increments model revision |
| Rotate history metadata | O(H) | Buffer handles, not H full weight copies |
| Trust-region update | O(R) CPU per round | Keeps O(R) scalar outcomes and scans previous min/max; O(R^2) cumulative work |
| Trust-region restart, when triggered | O(P) copy | Copies base into first history slot |
| Logging and timing extraction | O(B+F) current result payload | I/O latency is external; included in loop wall time |

The pool generator also performs log, sqrt and sincos for seeded noise, not
just multiply-adds. This table does not fold transcendental operations into a
single FLOP estimate. History pairing means more history repeats generation
and diagnostic work; current H <= 2 needs only one pair scan.

Sources: [`begin_ask`, `publish_start`, `tell_noisy`, `encode_pool`](../../rust/crates/ennx/src/bf16_metal.rs),
[`bf16_propose_pool`, `bf16_select`, `bf16_materialize`](../../rust/crates/ennx/src/bf16_search.metal),
[`TurboTrustRegion::update`](../../rust/crates/ennx/src/trust_region.rs).

The configured study uses two 4,096-token examples that change each round.
After one initialization score outside the timed loop, only the candidate is
scored each round, with two full-sequence passes. The minibatch-reuse and
single-pass configuration options have been removed; their historical
measurements remain below.

## Scoring work: execute this once per timed round

Costs below are per scoring call unless marked per layer/pass. Scratch columns
describe logical outputs; actual retained allocations are listed separately.

| Computation | Work | Logical scratch/output |
|---|---|---|
| Validate token IDs and copy tokens/targets | O(N) CPU | O(N) |
| Refresh packed weights when revision changes | O(P) upper bound | O(P) persistent layouts; reads BF16, writes FP16 |
| Embedding lookup, each pass | O(N*d), F times | O(N*d) |
| Second-pass feedback shift and mask | O(N*d) | O(N*d) + O(N) |
| Normalize feedback token input | O(N*d) | O(N*d) |
| Feedback state projection | O(N*d*d) | O(N*d) |
| Feedback gate projection | O(N*d*d) | O(N*d) |
| Feedback sigmoid, combine and normalization | O(N*d) | O(N*d) |
| Pre-attention learned RMSNorm, each layer/pass | O(N*d) | O(N*d) |
| Packed Q/K/V/head-gate projection, each layer/pass | O(N*d*(d+2*k+h)) | O(N*(d+2*k+h)) |
| QK RMSNorm, RoPE, head-major packing | O(N*(d+k)) | O(N*(d+2*k+h)); includes exp2/sincos |
| Attention QK and PV | O(B*h*a*A_layer) | No global T-by-T score matrix on active flash path |
| Online masked softmax | O(B*h*A_layer) | Tile scores/probabilities; running row maxima/sums |
| Attention accumulator rescaling | O(B*h*tile_visits*tile_q*a) mathematically; actual matrix work below | Register accumulators |
| Head sigmoid, output normalization/cast | O(N*d) | O(N*d) |
| Attention output projection | O(N*d*d) | O(N*d) |
| Attention residual addition | O(N*d) | Updates main state |
| Pre-FFN learned RMSNorm | O(N*d) | O(N*d) |
| Packed FFN gate/up projection | O(N*d*2*f) | O(2*N*f) |
| SiLU and elementwise gate/up product | O(N*f) | O(N*f); includes exp |
| FFN down projection | O(N*f*d) | O(N*d) |
| FFN residual addition | O(N*d) | Updates main state |
| Final learned RMSNorm, each pass | O(N*d) | First pass writes feedback history; second writes readout input |
| Final-pass tied vocabulary projection | O(N*d*V) | O(c*V) FP16 logits, stored in 8192-vocabulary-column blocks per row chunk |
| Cross entropy | O(N*V) | Two blocked-layout logit scans: maximum, then exp/sum; O(N) losses |
| CPU finite check and example means | O(N) | O(B) result means |
| MPS encode/object lookup | Expected O(1) per GEMM cache lookup | Objects cached by buffer identity/layout and matrix shape |

Source: [prefill allocation, `layer`, `start_pass`, `finish_pass`, `score`](../../rust/crates/ennx/src/fbt_prefill.rs),
[Metal kernels](../../rust/crates/ennx/src/fbt_prefill.metal),
[MPS bridge](../../rust/crates/ennx/src/fbt_mps.rs).

For the full-size route there are four layer GEMMs: QKVG, output, gate/up,
down. Across one score and two passes this is 192 layer GEMMs. The second pass
also has two feedback GEMMs and, after readout blocking, 52 readout GEMMs
because each of the four row chunks is split into 13 vocabulary blocks:
**246 MPS GEMM encodes per timed round**. There are 48 flash-attention
dispatches.
MPS may launch multiple internal kernels per encode; these counts are not
hardware kernel counts.

In the now-retired `--fused-ffn` experiment, 96 gate/up MPS encodes and 96
separate SwiGLU dispatches became 96 paired TensorOps dispatches (300 MPS
encodes remained). This is historical accounting; the current runner uses MPS.
The dense arithmetic count is unchanged. The 208 MiB gate/up intermediate is
removed, avoiding 39 GiB of logical writes/reads per round; paired weight
packing replaces, rather than adds to, the existing revision-dependent pack.
These are source-level accesses, not measured DRAM bytes. Full-round testing
found a regression despite the memory reduction; see [handoff](handoff.md).

## Attention: actual tiles versus useful pairs

A_layer is the number of unmasked causal pairs per sequence/head:

```text
full:  T*(T+1)/2
local: sum over positions t=1..T of min(t,W)
       = W*T - W*(W-1)/2 when T >= W
```

At T=4096: full=8,390,656; local=6,292,480. Four layers are full and twenty
local. Useful QK/PV arithmetic across a round is 7.835430 trillion operations.

The active 96-wide kernel processes 32 query rows by 32 key rows. Its loops
visit 8,256 tiles per full sequence/head and 6,240 per local sequence/head.
Whole tiles are multiplied before masking invalid elements. Thus actual
source-level QK/PV matrix work is 7.943542 trillion operations, not 7.835430.

The current kernel rescales each output accumulator with scalar multiplies
through threadgroup scratch, once per visited key tile. The previous diagonal
matrix-multiply implementation's 0.992943 trillion operation count no longer
describes this path. Current rescaling performs 0.062059 trillion scalar
multiplies per round, with substantial scratch traffic detailed below.
The 2026-09-24 output-stage repair removes the
final normalization matrix multiplies: scalar normalization/gating now happens
while storing three bounded slices through the existing score scratch. It also
removes the old 12,288-byte float write through the 6,144-byte half Q tile.
These are source-level matrix-operation counts, not measured device instructions.
Softmax and special functions remain additional work.

Threadgroup scratch for the 96-wide kernel is fixed by its 32-by-32 tiles:
Q/K/V tiles, padded score/probability tiles, diagonal scaling matrices and row
statistics. Register accumulators also consume on-chip storage. No persistent
O(B*h*T*T) attention matrix is allocated on this optimized path.
The active 96-wide kernel now starts each local-window query block at the first
potentially visible key tile, so the old from-zero skipped-tile control loop is
not present on the full-size optimized path. The older 128-wide kernels still
contain that branch pattern. Skipped old tiles did not load K/V or multiply, so
this removes control overhead rather than dense attention arithmetic.

## Dense arithmetic subtotal

| Component | Trillion operations per round |
|---|---:|
| FFN three projections | 24.120537 |
| QKVG and attention output projections | 5.585605 |
| Useful attention QK/PV | 3.917715 |
| Vocabulary projection | 2.525441 |
| Feedback projections, all dispatched rows | 0.077309 |

The active one-observation timed round has a dense subtotal of approximately
36.226607 trillion operations before attention tile overhead. The historical
paired round's corresponding subtotal was 72.453213 trillion.
Use the preceding section to add attention tile and accumulator work. Neither
subtotal includes controller work, scalar kernels, memory movement or waits.

Overall leading work with fixed pool/history and fixed tile sizes is:

```text
O(P + E*F*L*N*(d*d + d*k + d*f + d*h)
    + E*F*B*d*(L_full*T*T + L_local*T*min(T,W))
    + E*N*d*V + E*(F-1)*N*d*d + R)
```

This expresses dominant dense work, not the extra local-attention tile-control
term above. Full attention remains quadratic in T; FFNs and projections are
linear in T at fixed widths. Readout scales with V. Pool/history counts and
nonlinear scalar work are expanded in the tables rather than hidden as speed.

## Subsecond Capacity Ledger

The dense subtotal above describes the active changing-minibatch round: one
candidate objective call with two full-sequence passes. The table retains the
paired and one-pass runs as historical comparisons
(not an algorithm-independent arithmetic lower bound):

| Workload | Counted dense trillion operations | Representative measured wall | Effective dense TFLOP/s | TFLOP/s needed for 1 second |
|---|---:|---:|---:|---:|
| Historical paired round, two fused objective calls | 72.453213 | 27.747765 s | 2.611 | 72.453 |
| Active one-observation round | 36.226607 | 34.641166 s | 1.046 | 36.227 |
| Fixed-minibatch steady round, one standard candidate | 19.337370 | 7.448804 s | 2.596 | 19.337 |

The measured walls are from:

- `results/turbo-enn/run-1790456762994-24664-0` for the default three-round
  fused result;
- `results/turbo-enn/run-1790613032204-30491-0` for the active
  one-observation result;
- `results/turbo-enn-standard-reuse-trace/run-1790457376068-27248-0`, round 2
  candidate scorer, for the one standard candidate result.

These are counted dense source operations, not exact hardware instructions.
They also exclude scalar kernels, softmax exponentials, memory traffic,
command-buffer gaps and controller work. The conclusion is therefore
conservative: epilogue fusion and launch cleanup can improve constants, but the
current dense full-4K scorer shape cannot reach one second unless the effective
dense throughput rises by roughly 7x for the historical one-pass ablation or
34.6x for the measured active two-pass candidate. The subsecond
path needs a quantified combination of reduced computation and increased
throughput. These observations do not establish a hardware ceiling or rule
out a different exact evaluation algorithm.

### Active-round feasibility reconciliation

Recounted directly from `fbt_round.rs::run_round_study` and
`fbt_prefill.rs::{score,layer,start_pass,finish_pass}`. There are two batched
transformer passes with 8192 token rows each, giving 48 layer executions per
timed round. The separate initialization objective has the same scorer cost but
is reported outside the round loop.
Multiply-add is counted as two operations throughout.

| Operation | Dispatches per round | M,N,K per GEMM | Trillion operations |
|---|---:|---|---:|
| Packed gate/up | 48 | 8192,13312,1536 | 16.080358 |
| Down | 48 | 8192,1536,6656 | 8.040179 |
| Packed QKVG | 48 | 8192,3088,1536 | Included below |
| Attention output | 48 | 8192,1536,1536 | QKVG + output: 5.585605 |
| Readout | 52 | 2048,8192,1536; final vocabulary block 2048 | 2.525441 |
| Feedback | 2 | 8192,1536,1536 | 0.077309 |
| All GEMMs above | 246 | | 32.308892 |

Add 3.917715 trillion useful attention QK/PV operations to obtain
36.226607 trillion. Actual tiled attention QK/PV counts 3.971771 trillion;
rescaling, softmax, norms, activation, loss, controller and transfers remain
additional costs. Thus the useful-attention subtotal is not a full instruction
count, nor does it prove a lower bound for every possible algorithm.

The fresh exact-shape MPS probe sustained about 3.5 TFLOP/s (3.528 gate/up,
3.540 down, 3.495 QKVG, 3.545 output). Using 3.54 for every remaining GEMM,
including readout, gives this explicitly conditional budget:

| Assumed GEMM rate | Time for 32.308892 trillion operations alone |
|---|---:|
| 3.54 TFLOP/s | 9.13 seconds |
| 7.08 TFLOP/s (2x) | 4.56 seconds |
| 14.16 TFLOP/s (4x) | 2.28 seconds |
| 28.32 TFLOP/s (8x) | 1.14 seconds |
| 32.31 TFLOP/s | 1.00 second, no budget for anything else |

These are sensitivity scenarios, not forecasts or theoretical device limits.
Even granting free attention, all scalar operations, controller work and
memory movement does not close the gap at the observed GEMM rate. FFN alone
would take about 6.81 seconds at that rate. Multiplying the fresh isolated MPS
timings by dispatch counts gives about 9.21 seconds for layer GEMMs and readout,
before feedback or other work; this is not an end-to-end measurement.

Source audit of candidate removals:

- Readout is already absent from pass one: `finish_pass(score=false)` only
  normalizes the state for feedback. Removing another readout is not available.
- Gate/up and QKVG are already combined into packed GEMMs. Combining their
  dispatches again cannot remove their dot products.
- `synthetic_token` depends on the round index. Reusing the previous round's
  incumbent loss would score different data and change the experiment.
- Pass two consumes feedback-conditioned input; identical weights do not
  make intermediate activations identical.
- Rounded proposals change 63-90% of coordinates in the measured initial
  pool, with no unchanged rows or aligned 64-element blocks. That evidence
  does not support a block-sparse incremental evaluator.
- Fusion can remove materialization, but it does not by itself remove the
  counted projection products. Counting both a whole GEMM's time and all its
  memory traffic as independent savings would double-count overlapping work.

Decision: no evidence-backed subsecond execution design has emerged under
the unchanged workload. The measured causal-attention lead is approximately
0.58 seconds of historical GPU intervals, not the missing tens of seconds.
A proposed route must now identify which of the remaining dense operations
it avoids exactly, or substantiate the necessary throughput increase on this
device. This is an explicit unsolved gap, not a proof of physical impossibility.

## Resident memory and traffic

All figures below are requested buffer bytes, not measured process RSS, peak
allocation or memory-controller traffic. MiB/GiB use powers of two.

- One BF16 model row: 2P bytes = **1.984637 GiB**.
- Search after first reference initialization: base + proposal + two history
  rows + reference = five rows = **9.923186 GiB**, plus metadata/partials.
  Anchor/rejected handles alias history buffers; do not count them again.
- Initial model parameter allocations are another row, at least until accepted
  weights replace them and no retained object references them. Views into the
  proposal/base are aliases, not additional full rows.
- Current packed FP16 model layouts request **1.984497 GiB**. This includes
  0.457031 GiB for the 24 separately transposed down-projection matrices.
- Prefill's explicit buffers and its Feedback scratch total **1.758453 GiB**
  for this configuration, excluding pipelines, MPS scratch and allocator overhead.
- Reference Attention objects still allocate **288 MiB** of per-layer K/V
  caches, although the batched prefill route does not consume them. Other
  reference model workspaces and histories also remain allocated.

Important prefill allocations:

| Allocation | Requested size | Active optimized full-size use |
|---|---:|---|
| Six FP32 full-width state buffers | 288 MiB | Some are reference-only |
| Feedback object's scratch and scales | 48.03125 MiB | Not used by optimized feedback path |
| Head-major Q/K/V storage | 72 MiB | K/V each allocated at d, but packed at k; 24 MiB excess combined |
| Separate half Q/K/V buffers | 48 MiB | Packed QKVG bypasses these in layers; Q buffer reused by feedback |
| Half FFN gate and up buffers | 208 MiB | Up is used; separate gate's 104 MiB is bypassed |
| Packed half gate/up activations | 208 MiB | Used before SiLU-GLU |
| FP32 `logits` partial buffer | 392 MiB | Not used by optimized `finish_pass` |
| FP16 `logits_half` | 392 MiB | Used as blocked 2048-row by 8192-vocabulary-column readout scratch |
| Packed QKVG activations | 48.25 MiB | Used |

Source allocation sizes include O(N*V/32) unused partial storage even though
active readout uses O(c*V) scratch. Thus the allocated space complexity is not
just the active algorithm's minimal scratch complexity.

Candidate history storage copies 1.984637 GiB each round; acceptance adds the
same base copy and a full reference rewrite. Copies require reads and writes.
Vocabulary projection writes 3.0625 GiB of FP16 logits over a round, and the
two cross-entropy scans logically read another 6.125 GiB. Cache behavior can
change DRAM traffic; these are source-level accesses, not bandwidth counters.

The MPS matrix-object cache has no eviction and keys on buffer identity.
Newly bound raw parameter views can add retained objects across rounds; this
needs a lifetime audit before asserting a round-independent memory bound.
Scalar trust-region observation history separately grows as O(R).

## Revisions, submission and waits

- Candidate binding increments revision; its score refreshes packed weights.
- Rejection restores handles and increments revision. Acceptance rebinds base
  and increments revision. The next candidate binding and score refresh the
  one packed-layout set. There is one O(P)-bounded packed-layout refresh per
  normal candidate score.
- Search and model obtain the shared runtime queue; acquisition and candidate
  scoring execute serially in the active synchronous ordering.
- Current `score` submits **one command buffer per pass**, then waits for the
  last. It does not wait after every layer. One score yields two pass buffers.
- A normal round also submits acquisition, one weight-refresh buffer and tell:
  five command buffers in this explicit path, excluding first-use reference
  initialization, internal MPS kernels and occasional restart work.
- Blocking API calls include one refresh wait, one score wait, acquisition
  finish and tell completion. A wait may return immediately if work is done;
  counting waits does not assign them wall time.
- `gpu_seconds` currently contains pass intervals. Its comment and
  `score_examples` consumer still expect groups of L+2 intervals. That consumer
  therefore emits no layer breakdown for the current two-entry result.

Sources: [revision binding](../../rust/crates/ennx/src/fbt_model.rs),
[score scheduling](../../rust/crates/ennx/src/fbt_prefill.rs),
[logging](../../rust/crates/ennx/src/fbt_modeltests.rs),
[shared runtime](../../rust/crates/ennx/src/apple_gpu.rs).

## FP16 layout check

At width 1536 and FFN width 6656, `ensure_transposed_weights` skips Q/K/V/head
gate and FFN gate/up because packed FP16 replacements are built for them. It
does not skip the down projection. Each layer's `[1536, 6656]` down matrix gets
its own transposed FP16 buffer, and `linear_half` rejects a missing buffer.
An earlier version of this audit incorrectly claimed that down was skipped and
BF16 bytes were interpreted as FP16; current source contradicts that claim.

`gpu_scorer_full_ffn` sets the optimized fixture's FFN width to 6656 and checks
the complete scorer against the reference at short sequence length. It covers
the production FFN shape, but not full-model, 4096-token numerical parity.

## Reduction opportunities, not measured speedups

1. Retain production-width FFN coverage and establish full-size numerical
   parity before treating a timing result as a correctness baseline.
2. Remove proven-unused buffers or allocate reference/optimized workspaces on
   demand; this reduces requested memory, not automatically arithmetic.
3. Inspect single-thread acquisition reduction and host geometry publication.
   Parallel reduction changes summation order and needs parity checks.
4. Examine packed-layout lifetime across accept/reject; zero-copy binding does
   not eliminate format refresh. Extra caches trade memory for refresh work.
5. Examine scratch-staged accumulator rescaling and attention tile traversal without
   changing masks, normalization or synchronization correctness.
6. Examine readout projection/reduction fusion to avoid materializing logits;
   all vocabulary terms must still contribute to the stated objective.
7. Replace repeated scalar history extrema scans with equivalent maintained
   state only after matching restart and numerical semantics.

## Implementation traffic audit (2026-09-26)

Scope: active 4096-token, batch-two workload; one objective and two transformer
passes per timed round. These source counts identify optimization
targets, not measured DRAM traffic or promised speedups.

| Surface | Source-level cost per complete round | Candidate change |
|---|---|---|
| Attention accumulator rescale | 462.375 GiB threadgroup scratch accesses; 969,670,656 SIMD-group barrier executions | Register-resident row rescaling, if supported fragment access permits it |
| FFN gate/up materialization | 19.5 GiB logical buffer writes/reads | Fuse activation into a competitive GEMM epilogue |
| FFN activated intermediate | 9.75 GiB logical writes/minimum reads | Tiled producer/consumer fusion, subject to register and reuse costs |
| Readout logits | 3.0625 GiB logical writes/reads | GEMM/loss reduction fusion retaining the full vocabulary normalization |

Attention derivation (`fbt_prefill.metal`, `fbt_prefill_flash_attention_half_96`):
there are `2 * 32 * (4 * 8256 + 20 * 6240) = 10,100,736` visited tiles.
Each rescales 32-by-96 floats by matrix store, scalar read/write, and matrix
load: 16 logical bytes per float. Each of four SIMD groups executes two
barriers for each of twelve accumulator matrices. These are source execution
counts; generated instructions and physical memory transactions can differ.
This is on-chip scratch traffic, not 462.375 GiB of device-memory traffic.

FFN derivation (`fbt_prefill.rs`, optimized layer path): 48 layer executions,
each writing then reading a 208 MiB packed gate/up tensor and a 104 MiB
activated tensor. A previous fused implementation regressed; eliminating
materialization alone does not establish a faster implementation.

The single packed-weight cache is invalidated by parameter revision changes.
Candidate preparation in the existing one-round trace totals 0.126543 GPU
seconds across gate/up packing, QKVG packing, and transposes. Restoring an
unchanged incumbent can require repacking after the candidate overwrites that
cache. Retaining both layouts costs memory; content-equivalent reuse requires
explicit lifetime correctness. Parameter binding itself uses no-copy views.

Controller history insertion (`bf16_metal.rs`) copies a 1.984637 GiB BF16
candidate payload, even on rejection. History needs those values; ownership
rotation rather than deleting history is the potential copy-elimination path.

Normal scoring does not wait after every layer: it submits each pass and
waits for the final command buffer. Operation-level profiling changes those
boundaries. The latest ten-round allocation count is flat, so this audit does
not establish a current allocation leak. The corrected operation timing table
is in [native profiling](native-profiling.md); do not sum raw timings and their
group summaries together.

### Bounded rescale barrier experiment

Tested 2026-09-26, then reverted: stage four 8-by-8 output accumulators
side by side in the existing 8-by-33 score scratch, rescale all eight rows,
then reload and perform the unchanged PV updates. Three slices replace twelve
individual stages. This reduces rescale barriers from 24 to 6 per SIMD group
per visited tile, without reducing scratch traffic or increasing allocation.

All 30 `./ennx test` targets and all five `tools/fbt-bo --check` diagnostics
passed. The 4096-token causal/local diagnostic checked selected positions
against the numerical oracle, not every output element. Printed full-round
losses, decisions and radii matched between the candidate and restored baseline.

| Run | Round seconds | Mean seconds |
|---|---|---:|
| Candidate, three rounds | 26.077391, 26.476630, 26.904716 | 26.486246 |
| Restored baseline, three rounds | 28.206406, 27.249378, 27.038006 | 27.497930 |

Artifacts under `results/turbo-enn/`:
`run-1790476303540-59392-0` (candidate) and
`run-1790476766253-59833-0` (restored baseline). Both use unchanged 4096-token,
batch-two full rounds, with 16.0615 GiB reported allocation throughout.
Mean incumbent-plus-candidate scorer time was 26.284652 versus 26.640833
seconds, approximately 1.34% lower; most of the 3.68% round-time difference
was outside the changed scorer. This sequential comparison does not establish
a repeatable kernel speedup. Earlier single-round baseline/candidate times
were 27.168287/28.365533 seconds, with matching printed losses.

Decision: do not retain the runtime change. Barrier batching alone did not
demonstrate a substantial win. Scratch traffic remains unchanged, so this
does not test or rule out register-resident rescaling. `jj diff` verified the
kernel was restored exactly to its pre-experiment state.

### Existing MLX comparison (2026-09-26)

Read-only probe of installed MLX 0.32.1 on Apple M4 (`applegpu_g16g`,
24 GiB), followed serially by the retained `tools/fbt-bo --gemm` probe
(Build ID `8805e852-bd9b-4ab1-9bbf-50a1d535dbf9`). No production changes.
MLX used seed 42, nonzero random FP16 operands, two warmups and seven timed
observations with fresh operations, `mx.eval` and `mx.synchronize` each time.
Inputs, contiguous/transposed weight layouts, and masks were prepared outside
timing. Outputs were checked finite, not compared numerically with ENNX.

| GEMM | Shape M,N,K | MPS NN wall ms | MLX NN median wall ms | MLX NT median wall ms |
|---|---|---:|---:|---:|
| Gate/up | 8192,13312,1536 | 95.334 | 103.570 | 105.215 |
| Down | 8192,1536,6656 | 47.631 | 53.844 | 53.719 |
| Attention output | 8192,1536,1536 | 11.255 | 12.471 | 12.443 |

MPS and MLX inputs differ and runs were not interleaved. MLX times include
Python dispatch/allocation; MPS wall includes encode/submit/wait. These are
screening results, not a controlled framework ranking or a hardware ceiling.
They do not reveal a faster off-the-shelf replacement for these dense GEMMs.

Attention used Q=[2,16,4096,96], K/V=[2,8,4096,96], FP16, scale=96**-0.5.
MLX causal mask was the string `causal`; local mask was a precomputed boolean
4096-by-4096 array allowing causal distances below 2048. Additional timings
included FP32 sigmoid-gate multiplication, FP16 output and contiguous
batch/token/head/channel layout. Gate sigmoid was prepared outside timing,
whereas ENNX computes sigmoid inside attention. MLX rounds its attention
output to FP16 before the gate, so numerical equivalence is not established.

| Attention | MLX core median ms | MLX with gate/layout ms | Historical ENNX GPU mean ms |
|---|---:|---:|---:|
| Full causal | 36.875 | 40.254 | 76.467 |
| Causal local-2048 | 77.930 | 81.439 | 57.512 |

ENNX comparison is the existing trace `run-1790455593754-23210-0`, not a
fresh same-input isolated run: 16 full and 80 local dispatches per round.
Multiplying these counts by MLX gate/layout medians gives a screening estimate
of 7.159 seconds, versus the historical 5.824 seconds for our attention.
Replacing only full-causal dispatches suggests 0.579 seconds of interval
savings, not a measured complete-round improvement. Mask-dependent dispatch
must be inspected in the installed version before attributing the difference.
Readout-block MLX NN measured 16.482 ms for 2048,8192,1536; the MPS probe
times the entire vocabulary, so these are not directly comparable cells.

Conclusion: no blanket MLX port is justified by this probe. The full-causal
path is a specific implementation lead; the local-mask path and dense GEMMs
do not show the same benefit. No subsecond result or prediction follows.

### Paired local-bound audit (2026-09-28)

The ignored `fbt_model::prefill::boundtests::paired_bound_audit` runs the
full seed-42 LocalV1 model at batch two, length 4096, both feedback passes,
with incumbent and initial candidate 0 (proposal seed 123, radius 0.005).
It snapshots real activations after each layer. All layers and all 16 heads
are covered, but only the final query/token of each sequence is inspected.
This is 1536 attention comparisons and 102 unit-RMS comparisons, not an
all-token certificate. Log: `.cache/paired-bound-audit.log`.

This is an optimistic post-hoc audit: input differences, logit differences
and value differences are measured from both completed forwards. No incoming
uncertainty is propagated. RMS and attention are recomputed in f64 from the
captured inputs, without certifying rounding of the production Metal kernels.
RMS checks exclude learned gamma; attention checks exclude output gating and
projection. Feedback checks unit-RMS on the reconstructed gated vector, not
the uncertainty introduced by the feedback projections or gate bound.

| Local operation | Comparisons | Bound/error median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Unit-RMS, pass 1 boundaries | 50 | 1.0483 | 1.0065 | 1.0766 |
| Unit-RMS, pass 2 boundaries | 50 | 1.0803 | 1.0716 | 1.0901 |
| Feedback gated-vector unit-RMS | 2 | 1.0759 | 1.0714 | 1.0804 |
| Attention, pass 1 | 768 | 112.3093 | 78.6765 | 182.0223 |
| Attention, pass 2 | 768 | 114.2093 | 77.7274 | 177.0549 |

All sampled inequalities passed within the diagnostic's f64 numerical
tolerance; this is numerical validation, not a formal proof. The attention
bound uses `max_i ||delta_v_i|| + 2*value_radius*tanh(logit_range/4)`;
the value radius is measured around the incumbent value mean, providing a
valid upper bound on half the diameter. Using that upper bound, rather than
an exact diameter, can itself introduce slack. Masks match the permitted
keys for the last token: full causal or local window 2048.

The first checked attention head already gives actual error 0.006308403
versus bound 0.731918386 (116.02x). Its total-variation bound is 0.024676
versus observed TV 0.005211. Hence this norm-envelope approach loses substantial
tightness before cross-layer propagation. These observations do not rule out
tighter attention-distribution-dependent or relational certificates.

The completed GPU scores were 12.007460902 (incumbent) and 12.007834102
(candidate). No loss-error certificate or BO-decision certificate was derived.
No latency claim follows: diagnostic synchronization, snapshots and f64
replay are intentionally outside the production path.

Decision: the unit-RMS local formula merits retention as a building block,
but the tested attention envelope does not demonstrate a tight two-pass
certificate. Do not implement an accelerated evaluator based on these bounds.
The next mathematical target is preserving cancellations and probability/value
dependence in attention, rather than multiplying these scalar bounds onward.

```sh
./buck2w test //rust/crates/ennx:ennx-unit --local-only -j 1 -- --timeout 600 --test-arg=--ignored --test-arg=--nocapture --test-arg=--test-threads=1 --test-arg=paired_bound_audit
```

### Cancellation-preserving attention bound (2026-09-28)

The same `paired_bound_audit` now also tests first- and second-order signed
exponential reweighting. The production path is unchanged. The scope remains
candidate 0, radius 0.005, one seed, batch two, context 4096, all 24 layers
in both passes, all heads, final query only per sequence. Log:
`.cache/paired-relational-bound-audit.log`.

For a permitted key set, let p be the incumbent softmax probabilities, A
its attention output, delta the candidate-minus-incumbent logits, and v'
the candidate values. Define c=sum(p*delta), t=delta-c, w=v'-A. Exactly in
real arithmetic, the attention change is E_p[exp(t)*w]/E_p[exp(t)].
Common shifts cancel and signed vector contributions remain intact.

For polynomial P of degree j=1 or 2, form:

```text
Zhat = sum_i p_i P(t_i)
Dhat = sum_i p_i P(t_i) w_i / Zhat
r_i = exp(max(t_i,0)) * abs(t_i)^(j+1) / (j+1)!
L = max(exp(min_i t_i), Zhat - sum_i p_i r_i)
error_bound = sum_i p_i r_i ||w_i - Dhat||_2 / L
```

Require Zhat>0 and L>0. Taylor's theorem bounds |exp(t_i)-P(t_i)| by r_i.
The identity sum_i p_i P(t_i)(w_i-Dhat)=0 removes the polynomial part from
the residual exactly. Consequently ||(A'-A)-Dhat||_2 <= error_bound in real
arithmetic. The resulting upper bound on ||A'-A|| is ||Dhat||+error_bound.
The implementation checks both inequalities numerically in f64, with a small
comparison tolerance, not directed rounding or a machine-level certificate.

| Order | Pass | Median upper bound / observed change | Worst ratio | Median remainder / observed change | Worst relative remainder |
|---|---|---:|---:|---:|---:|
| 1 | 1 | 1.7068 | 2.5806 | 0.7030 | 1.5830 |
| 1 | 2 | 2.1965 | 2.9001 | 1.1970 | 1.9074 |
| 2 | 1 | 1.0233 | 1.0895 | 0.0245 | 0.0917 |
| 2 | 2 | 1.0625 | 1.1212 | 0.0656 | 0.1253 |

All 3072 relational checks passed (768 per order/pass). The second-order
local bound is substantially tighter than the 112-114x median scalar envelope.
This demonstrates local tightness for the sampled paired states, not
end-to-end propagation: exact candidate Q/K/V and incumbent probabilities
were obtained from completed forwards. Computing the diagnostic also scans
every allowed key for each sampled query and retains baseline snapshots.
It therefore demonstrates no avoided GEMM, forward pass, or latency saving.

Next unresolved obligation: replace those exact candidate quantities by
cheaper computable approximations with certified uncertainty, including
projection, FFN, gate, gain, feedback and floating-point errors, without losing
the cancellation responsible for these results. No BO loss or decision
certificate is yet available. The useful result is a locally tight attention
building block; the subsecond feasibility gap is unchanged.

### Realized BF16 proposal structure (2026-09-26)

The saved round artifacts contain scores and configuration, not weight
snapshots. The ignored `fbt_model::tests::proposal_structure` diagnostic
therefore reconstructs the full LocalV1 model with seed 42 and the production
model-search setup (reference seed 42, initial length 0.01). It calls the
existing GPU `test_candidate` helper at proposal root seed 123 for all four
pool entries. It performs no forward passes and no accept/reject updates.
The initial observation supplied to search is a placeholder; candidate
generation here does not perform acquisition selection or use that score.

All 1,065,494,016 BF16 coordinates are compared, not a statistical sample.
Log: `.cache/proposal-structure.log`.

| Candidate | Direction family | Radius | Changed coordinates | Changed percent |
|---|---|---:|---:|---:|
| 0 | Reference-correlated | 0.005 | 673,330,105 | 63.1942 |
| 1 | Reference-correlated | 0.020 | 958,691,150 | 89.9762 |
| 2 | Fresh noise | 0.005 | 673,313,933 | 63.1927 |
| 3 | Fresh noise | 0.020 | 958,670,997 | 89.9743 |

Every candidate has zero wholly unchanged last-dimension rows and zero
wholly unchanged aligned contiguous 64-element blocks, across all tensors.
Each candidate covers 16,648,344 such blocks. Norm vectors count as one row;
these are contiguous blocks, not a claim about every possible 2D tile layout.
All proposed values were finite. Counts compare exact BF16 bit patterns.

The embedding/tied readout and every other parameter tensor change. For
example, candidate 0 changes 97,407,537 of 154,140,672 embedding coordinates,
with no unchanged embedding row. The first layer's FFN gate changes
6,462,481 of 10,223,616 coordinates, also with no unchanged row or 64-block.

Conclusion: initial BF16 rounding does not yield a mostly unchanged or
block-sparse update under these seeds and radii. Individual unchanged weights
do not establish cheap incremental inference: downstream inputs change too.
This diagnostic does not measure exact/approximate rank, activation reuse,
trained-checkpoint behavior, smaller future radii, or later accepted states.
Do not generalize the counts into an impossibility result for all reuse.

Reproduce after `./ennx fmt --check`:

```sh
./buck2w test //rust/crates/ennx:ennx-unit --local-only -j 1 -- --timeout 600 --test-arg=--ignored --test-arg=--nocapture --test-arg=--test-threads=1 --test-arg=proposal_structure
```

None of these source observations establishes its wall-time contribution or a
subsecond result. The audit provides the work/memory inventory for that next
measurement and implementation stage.
