# Current research state

Checked against source and artifacts on 2026-10-04. The active machine is an
Apple M4 with 24 GB unified memory. This document records current behavior and
open falsifiers; it is not a chronological experiment log.

The target workload is one million **newly generated tokens per candidate**, with
their complete causal history retained, in a complete BO round under one second.
The tighter ambition remains 200 ms. A million-token input with a 4K continuation
does not meet this workload definition.

## Active workload

[diffusion.toml](../examples/tuning/diffusion.toml) now runs learned draft proposals
through a causal target before scoring or committing tokens. The model remains
`fbt-pisa1-diffusion-mhc4-v1`: five physical layers, seven target layer executions,
four mHC streams, 625 routed experts plus a shared expert, and top-three routing.
Its 1,047,704,344 FP16 coordinates include the learned mask and index projection.
The strict checkpoint format remains `ennx.fbt-pisa1-diffusion-mhc4-rope.v1`.

Each candidate round performs acquisition and full-coordinate perturbation,
fresh diffusion drafting, target verification, generated-output scoring, and
resident ENN/Pareto/TuRBO update. The draft shares candidate weights; its block
visibility never becomes target visibility. The first target pass rebuilds all
readable KV causally. Candidate changes do not reuse incumbent KV.

The example runs 40 rounds, reaching guided acquisition after 31 initialization
rounds with 32-neighbor ENN history. It uses three independently reported objectives:

- Generated code: clipped non-whitespace byte n-gram overlap with the corpus
  continuation, minus overlap with an unrelated continuation.
- Draft agreement: the fraction of initial draft tokens matching the final target
  completion. This provides a signal for draft-only parameters even when exact
  verification leaves final text unchanged.
- Work: negative total evaluated positions per committed output token, counting
  both draft and target. Higher means less repeated position work.

Pareto acquisition receives the whole vector with explicit scales. It does not
promise that each objective improves on every accepted step. Agreement after the
first mismatch is a training proxy; only the accepted prefix establishes causal
acceptance. Position work is not a FLOP count or a latency guarantee.

The former diffusion-only generator and denoising-only BO reward were removed
from the active generation path. Public configuration uses `[generation.draft]`;
`generation.diffusion` is not a compatibility alias. Supervised probe loss is
not the current training objective. Text overlap does not certify coherence,
functional correctness, or held-out generalization.

The current verifier still uses exact repair waves when the draft fails. This
is a correctness baseline, not the intended cheap drafter or near-1x execution.
Draft and target cost, accepted prefix, all-position agreement, amplification,
scoring, proposals, and ENN/tell timing are recorded separately. The round timer
includes completion materialization and controller writes. Setup, initial
incumbent, curve writes, console display, held-out validation, and final checkpoint
are reported or performed outside that timing boundary.

The success flag requires every measured candidate to generate the requested
number of new tokens, actually perturb weights, and fit the maximum round latency
budget. Prompt length, retained context positions, output length and allocated
cache capacity remain distinct. The default budget is 1,000 ms; the 200 ms and
million-output goals remain open.

The optional polynomial-threshold proposal model retains all 148 spectral
features before posterior sampling. Its history is separate from the ENN's
128 logical observations and two resident weight rows. The example uses
Rademacher directions. Nonresident full-vector distances remain approximate.

The 128-token Metal integration test compares target commits against direct
causal sampling at temperature 0.8 and verifies that changing clean labels does
not change generation. CUDA source remains an unexecuted diffusion proposal
implementation; it does not yet implement this complete BO path. The NVIDIA host
is disconnected. TVM remains unexecuted.

## Complete draft/target BO measurement

The coupled-sampling 4K run completed 40 candidate rounds: 31 initialization
rounds and nine guided ENN/Pareto rounds, with 12 acceptances. Every candidate
generated and committed 4,096 new tokens after the 128-token prompt. Median
complete-round latency was 3,540.663 ms; maximum was 3,881.832 ms. The goal is unmet.

Mean measured costs were 149.768 ms for proposals, 428.423 ms for draft GPU work,
2,896.087 ms for target GPU work, 3.168 ms for scoring, and 23.298 ms for tell.
These are overlapping wall/device accounting views, not additive wall phases.
Combined position amplification averaged 5.8703x and target correction averaged
89.85 waves. Draft/target agreement averaged 68.50%, but the initial accepted
prefix was one token in every round. Dense agreement is not causal acceptance.

Before coupling position-indexed sampling draws, an eight-round initialization
run took 3,669.454 ms median and matched zero draft tokens. Equal zero-logit draft
and target distributions now accept all 128 tokens in the integration test.
The initial target completion and first perturbed candidate completion remain
exactly equal across the two runs. Later vector decisions can differ because
agreement and measured work are optimization objectives. The sequential runs do
not establish a controlled kernel speedup or learned draft quality.

Artifacts: `.cache/ennx/runs/context-loop/run-1791144172729-368` (40 rounds),
`.cache/ennx/runs/context-loop/run-1791143825181-98623` (uncoupled baseline), and
`.cache/ennx/measurements/draft-target-20261004` (logs and changed-source snapshots).
The 64K probe generated and committed 65,536 new tokens in one complete
initialization candidate round at 66,346.802 ms. Draft GPU work took 9,113.465 ms,
target GPU work 55,701.967 ms, scoring 33.131 ms, proposals 135.737 ms and tell
75.475 ms. Agreement was 76.19%, the initial accepted prefix was one token,
and 1,579 correction waves produced 7.0254x combined position amplification.
Artifact: `.cache/ennx/runs/context-loop/run-1791144461997-806`.
One round does not estimate a latency distribution or demonstrate guided BO.

No held-out quality result was collected. CUDA and the million-output workload
remain unmeasured for this corrected engine. Runtime serial auditing remains
unsupported for compact drafting; the integration test performs its direct
comparison with a separate noncompact decoder. The public validator preserves
that restriction.

## Historical diffusion-only measurements

These measurements included BO proposal/scoring/tell, but diffusion itself
produced the final tokens and a clean-data reconstruction probe supplied the
reward. They do not validate the current draft/target engine or its objectives.
The historical source, resolved configurations, output and plots remain intact.

Measured on 2026-10-04 using the 8.192M-token corpus
`.cache/ennx/corpora/202b69a6d707ceb1290d/train.ennxptn`. All three workloads
share the prompt, initial model, proposal seeds, sampling seed, corruption,
and reconstruction probe. The M4 was on AC power; clocks and temperature were
uncontrolled. Model initialization and the initial incumbent rollout are outside
complete candidate-round timing.

| Newly generated tokens | Candidate rounds | Before | Optimized median | Generation GPU | Cache refresh GPU | Objective GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4 | 1.395 s | 1.292 s | 0.386 s | 0.018 s | 0.730 s |
| 65,536 | 1 | 12.709 s | 9.896 s | 6.081 s | 2.897 s | 0.724 s |
| 1,048,576 | 1 | 179.253 s | 155.847 s | 99.925 s | 52.030 s | 0.743 s |

GPU phase values are means across candidate rounds; the complete-round column
uses the conventional median. The current 4K range is 1.256–1.305 s, versus
1.314–1.447 s before. The two
larger workloads each have one candidate measurement, not an estimated latency
distribution. The current million-output initial rollout took 140.835 s wall,
versus 156.131 s before. Every initial and candidate rollout at all three lengths
matches the earlier generated tokens, reconstruction rewards, index metrics,
tensor updates and controller decisions exactly. The observed reductions are
7.4%, 22.1% and 13.1%; these historical/subsequent comparisons do not isolate
uncontrolled clocks or temperature.

The vector mHC kernel reuses source loads across all four output streams. In the
64K stage profiles, residual transport averaged 9.808 ms per 4K generation pass,
versus 21.810 ms before. Sampling reduction averaged 0.611 ms; expert projections
and selected attention remain larger costs. The final cache pass also omits work
that cannot affect later KV. Profiling is disabled in the complete-round table.

Every rollout produced its requested number of tokens with zero autoregressive
repair waves. At 1M, 2,097,152 generation positions, 1,044,608 cache positions,
and 16,640 objective positions give 3.0121x model-position amplification.
Generation and cache refresh dominate latency. The subsecond and 200 ms goals
remain unmet at every measured length.

Initial fixed-probe NLL was 9.090727 at all three lengths; the first candidate's
NLL was 9.090835 and it was rejected at each length. Over four 4K rounds, two
later candidates were accepted and incumbent NLL reached 9.089439. There was no
held-out validation in these systems probes. Neither these small fixed-probe
changes nor the full-length generation establish coherent text or generalization.
The probe remains worse than an 8,192-way uniform distribution's NLL of 9.010913.

Historical artifacts:

- 4K: `.cache/ennx/runs/context-loop/run-1791140730481-92734`.
- 64K: `.cache/ennx/runs/context-loop/run-1791140855731-92907`.
- 1M: `.cache/ennx/runs/context-loop/run-1791141031361-93164`.
- Combined data and reproducible Node report:
  `.cache/ennx/measurements/kernel-20261004/summary.json` and `report.cjs`.
- Plot: `.cache/ennx/measurements/kernel-20261004/latency.svg` and its PNG render.
- Earlier measurements remain in `.cache/ennx/measurements/diffusion-20261004`.

The earlier 4K/64K runs used an 81,920-token fixture and are retained separately.
The first 1M attempt failed for insufficient corpus positions; the next failed
because cache allocation reserved eleven executions. The matched successful run
uses the larger corpus and seven caches. No context truncation or corpus repetition
was introduced. No generated checkpoint was saved by these probes.

## Earlier autoregressive experiment

`examples/tuning/residual-looped-mhc4-8m.toml` runs 1,000 rounds with 4,096
newly generated tokens per candidate, temperature 0.8, and held-out validation
every 25 rounds. The active run is
`.cache/ennx/runs/pretrain/571e8175b6193794a8eb/run-1791088950674-17284-0`.
The run completed 1,000 rounds, generated 4,096,000 candidate tokens, and
accepted 16 candidates. Median complete-round wall time was 2.267 seconds;
the maximum was 4.611 seconds. Initial held-out rewards were
`[0.18494862, 0.18434012]`; final rewards were `[0.18498582, 0.18390691]`.
These results do not establish improved generalization. Generated output
remains incoherent and the 200 ms target is unmet. Generated candidate tokens
are evaluation work, not a conventional corpus-training token count.

## CUDA recurrent execution

The native T4 executor loads the looped mHC Safetensors checkpoint, retains
weights and scratch buffers on-device, runs selective recurrence, and performs
tied readout and sampling. It includes RoPE, 625-expert routing, grouped and
shared MoE branches, and four-stream mHC transitions. Stable route packing uses
1,024-assignment chunk prefixes instead of scanning the entire earlier route
history for each assignment.

The analytical 4K fixture passed at one, two, and four recurrent visits.
Seven layer visits measured 496.120 ms of device time and 499.806 ms wall time.
This fixture has uniform weights and zero MoE branches; it does not establish
checkpoint agreement between Metal and CUDA or full optimization-loop latency.
Sampled generation at temperature 0.8 also passed: 3,968 tokens after a
128-token prompt were identical for one- and two-pass repair submissions and
matched the analytical fixture's serial sampling semantics.

The existing 36 CUDA parity cases pass. Cooperative warp key reads reduced the
4K attention slice's measured median device time from 59.589 to 50.847 ms
(14.7%); median wall time was 51.876 ms. FP32 summation order changed, so exact
bitwise agreement on arbitrary attention inputs is not claimed.

TVM Metal/CUDA contraction schedules remain compilation plans. No TVM compiler
or runtime has executed these experiments. Native CUDA generation is distinct
from the Metal `./ennx tune` optimization loop; CUDA optimizer integration and
cross-backend checkpoint parity remain unfinished.

## Million-token attention

The current PISA hierarchy now has a persistent KV cache on Metal and CUDA
with compact query buffers, parallel tree construction, and incremental repair.
The numerical fixture passed at 4K, 64K, and 1,048,576 context tokens on the M4
and T4. It checks prefix append, selected block IDs, causal masking, attention
output against a Rust reference, future-content invariance, and tree repair.
CUDA memcheck reports zero errors at 1M. The 32 Rust test targets pass.

Median device time for 128 queries, seven repetitions after numerical checks:

| Available context | M4 Metal | T4 CUDA |
| --- | ---: | ---: |
| 4,096 | 0.410 ms | 2.458 ms |
| 65,536 | 0.440 ms | 2.616 ms |
| 1,048,576 | 0.464 ms | 2.779 ms |

At 1M, query wall time was 0.964 ms on Metal and 2.906 ms on CUDA. The two
prefix writes that construct the tree used 3.571 ms and 1.402 ms of device time,
respectively; a four-token repair across a leaf boundary used 0.078 ms and
0.123 ms. Each layer visit uses 256 MiB of FP16 KV and approximately 4 MiB of
tree summaries. Both backends' maximum absolute error against the selected-
support reference was `0.00010019541` (limit `0.003`). Clocks were not controlled;
these are fixture measurements, not hardware-normalized comparisons.

Artifacts are in `.cache/ennx/runs/context/20261004`. Commands and the reviewed
2026 research are in [the context design](kernel-architecture-plan.md#million-token-context).
The cache fixture omits projection, RoPE, MoE, readout, generation, and BO
proposals. Metal now wires persistent KV into chunked execution of those model
operations and accepted-prefix generation. It retains one KV/tree pair per
executed layer visit, rebuilds the readable prefix for each candidate, and uses
bounded scratch for 128..4096-row chunks. The full CUDA checkpoint executor
remains capped at 64K; its persistent-cache fixture is a separate path.

The Metal chunked model passed a comparison with the existing full scorer:
sampled token IDs and FP16 trees agree exactly; hidden states and generated-prefix
target losses agree within 0.003. All 32 Rust test targets pass. This checks the
executor, not learned long-context quality or agreement with dense attention.

A complete Metal BO round at 1,048,576 evaluated context positions generated
only 4,096 tokens and took 26.328 seconds, including 25.813 seconds of rollout GPU
time and a 201.689 ms proposal. Its artifact is
`.cache/ennx/runs/context-loop/run-1791095510495-36080`. This is a long-prefix
measurement, not the million-generated-token target.

`./ennx model context-loop CONFIG DATASET --generated 1048576 --prompt 128 --rounds 1`
declares the new-token workload explicitly. The dataset supplies distinct prompt
and scoring-reference positions; model inputs after the prompt are generated
tokens. A 128-token prompt plus 1,048,576 new tokens needs 1,048,703 evaluated
positions, with a 2,097,152-slot cache. No sliding-window truncation is applied.
Run settings and provenance are retained in `generation.json`.

The initial latency probe uses native token-accuracy scoring over the entire
generated trajectory. Its TOML is
`.cache/ennx/measurements/million-generation-20261004/latency.toml`.
This objective is explicitly different from the code-objective reconstruction
probe. Whole-output exact byte edit distance has quadratic length cost and must
not be described as constant-time feedback. Timing with token accuracy does not
establish code quality, coherence, or pretraining progress.

The million-output probe completed on 2026-10-04. Artifact:
`.cache/ennx/runs/context-loop/run-1791096315540-37944`.

| Work | Measured result |
| --- | ---: |
| Newly generated tokens per rollout | 1,048,576 |
| Retained logical context positions | 1,048,703 |
| Initial generation wall time | 642.628 s |
| Complete candidate BO round | 599.044 s |
| Candidate rollout GPU time | 587.598 s |
| Candidate correction waves / repair submissions | 24,847 / 17,626 |
| Candidate evaluated model positions | 4,670,720 |
| Proposal wall time | 157.083 ms |
| Native token-accuracy scoring | 0.892 ms |
| Numerically changed weights | 889,707,164 |

The candidate generated at approximately 1,750 tokens/s over the complete BO
round. The subsecond target requires more than 1,048,576 tokens/s. Both rollouts
finished by length; neither stopped at EOS. The candidate differs from the initial
output at 43,855 token positions. Both token-accuracy rewards were `0.00013923645`,
so the candidate was rejected. This probe establishes full-length execution,
not improved pretraining or coherent output.

`./ennx model scale` turns a measured trajectory into the relevant work ratios.
For this run, use:

```sh
./ennx model scale --generated 1048576 --prompt 128 \
  --evaluated 4670720 --round-ms 599044 --target-ms 1000
```

The current verifier amplifies model work by 4.454x. Its dense contractions total
185.454 trillion FLOPs, requiring 185.454 effective TFLOP/s to finish in one
second. A single broad pass is 41.639 trillion FLOPs and needs 41.639 effective
TFLOP/s. The distinction is decisive: removing every repair cuts 77.5% of the
counted contraction work, but still leaves a 42.6-second measured broad pass.
The model executes 19,852,800 contraction coefficients per evaluated position,
1.895% of the 1,047,699,736-coordinate arena. The full arena is perturbed once
per candidate, so longer outputs amortize proposal cost while forward work grows
with evaluated positions.

An 8,192-row context chunk was tested against the 4,096-row production chunk on
the same deterministic 16,384-token workload. It changed the generated sequence
and repair trajectory and increased candidate-round wall time from 12.837 seconds
to 27.655 seconds. It was rejected and reverted. The retained baseline artifact
is `.cache/ennx/runs/context-loop/run-1791132185980-65406`.

Progress counters separate repair GPU time from the broad verification pass:
the initial rollout used approximately 22.706 s for its broad pass and 600.193 s
for repairs; the candidate used 42.632 s and 544.965 s, respectively. These are
derived from total device time and the final repair counter, not per-kernel traces.
The run used battery power; CPU compilation overlapped part of initialization.
No other GPU benchmark ran concurrently. These timings are observations rather
than a controlled hardware ceiling.

The artifact's `controller_seconds` includes candidate observation because its
timer began before generation. The source now starts that timer after observation.
The complete-round timer was independent and remains valid. Raw artifacts retain
the original field; it must not be read as controller-only cost.

## iPhone browser compute source

The no-install iPhone path uses Safari 26 WebGPU over private Tailscale HTTPS.
`./ennx iphone web` serves a one-shot WGSL worker; the browser must remain in the
foreground. The native iOS application is not required for this path.

The proposal probe changed 16,777,216 resident FP32 coordinates in a 6.0 ms
median over seven repetitions and reported 22.37 effective GB/s. The physical
iPhone exposed a 256 MiB maximum buffer, a 128 MiB maximum storage binding,
shader FP16, and timestamp queries.

A shape-faithful FP16 readout probe used 128 rows, width 512, vocabulary 8,192,
and FP32 accumulation. It passed the expected all-ones output of 512 and measured
11.613 ms median GPU time, 13 ms submission-to-completion wall, and 0.092
TFLOP/s with the portable 16-by-16 WGSL kernel. Thirty-two such batches cover a
4,096-token readout and extrapolate to 372 ms before attention, MoE, routing, or
verification. The current portable browser kernel therefore does not justify a
full FBT evaluator port as a latency optimization.

The first candidate-objective check replayed the completed million-token
candidate from `run-1791096315540-37944`. WebGPU found 146 matching positions
out of 1,048,576 and returned token accuracy `0.0001392364501953125`, exactly
equal to the independently computed Rust value. Device completion took 8 ms;
the first browser evaluator invocation took 170 ms after payload receipt. A
timestamp-qualified warm replay measured 1.416 ms in the objective kernel,
5 ms from submission through device completion, and 31 ms for the evaluator.
This establishes real objective execution and cross-backend parity. It does not
establish an iPhone FBT forward pass: model prefill and decode remain on Metal/CUDA until the
weight arena and model kernels are lowered to WGSL. Shipping candidate weights
per round is rejected; a useful forward evaluator must load sharded weights once
and regenerate candidates from seed and radius on the phone.

### Combined Mac and iPhone schedule

The measured phone path does not improve the current objective or proposal
stages. Token accuracy costs 1.416 ms on the phone but less than 1 ms natively
on the Mac, before transport. Extrapolating the phone's 22.37 GB/s proposal
probe across the full FP16 arena gives about 179 ms for one read/write pass,
compared with the measured 88.907 ms Mac proposal. Offloading either stage adds
a measured 59 ms Tailscale round trip.

The viable subsecond split is a resident compact block editor on the iPhone and
exact candidate verification on the Mac. The conservative critical-path budget
is 88.907 ms Mac proposal, 59 ms one-shot transport, 371.609 ms for 32 measured
phone readout batches, about 95 ms Mac verification, and 95.992 ms of remaining
measured round wall. This totals 710.508 ms before the phone editor core, leaving
289.492 ms under one second. The editor must return a complete 4,096-token block
in one exchange and attain enough acceptance to avoid another network round.

The current portable WGSL contraction rate gives a roughly 1.84 second
arithmetic floor for the active model's 170.5 billion 4K contraction operations,
before routing, attention, proposal, or transport. A second full candidate lane
therefore requires at least a 2.3-fold contraction improvement plus complete
resident weight and kernel support before it can enter a subsecond schedule.
Layer-by-layer partitioning, phone-only objective scoring, and phone-only
proposal generation are rejected for the latency path.

## Apple Neural Engine probe

`./ennx ane probe` generates and compiles a Core ML FP16 readout without Python.
The measured shape is the production readout: width 512 and vocabulary 8,192.
Core ML's compute plan reports whether the layer prefers CPU, GPU, or ANE.

With 4,096 rows and fixed compiled weights, ANE took 6.133 ms median over 15
timed predictions and delivered 5.60 TFLOP/s. The BO-relevant probe supplies the
perturbed 512-by-8,192 matrix at runtime. It still preferred ANE and took 10.079
ms median over nine predictions, versus 60.628 ms on CPU. Automatic placement
also preferred ANE and took 10.582 ms. The runtime-input result needs no model
recompilation between candidates.

ANE is useful as a concurrent matrix lane, not as a demonstrated reduction in
isolated readout latency. The production Metal readout is already roughly 10 ms
for 4,096 rows. Moving eligible dense work to ANE can free the GPU for proposal,
attention, routing, and exact verification. The next integration must measure a
complete round with Core ML input/output transfer and Metal running concurrently;
the isolated probe does not establish that overlap.

The shared probe also builds into the native iPhone worker and is invoked with
`./ennx iphone ane HOST --rows 4096`. It has not run on the physical phone: Xcode
currently reports both paired phones unavailable, the machine has no signing
identity, and Safari cannot access ANE. The unsigned iPhone build succeeds.

The 200 ms ambition cannot use the current remote phone boundary: proposal,
one exact verifier pass, and one 59 ms network round already sum to about 243 ms.
That target remains a Mac-resident kernel and verifier problem. The phone can
increase candidate throughput asynchronously after the compact editor is
trained; it cannot shorten the 200 ms serial critical path over the measured
network.

## What is established

### Complete-round latency

One complete free-generation BO round changed all 1,047,732,224 coordinates,
generated and scored all 4,096 positions, and completed in 500.886 ms. The
proposal took 88.907 ms and reported GPU rollout time was 315.987 ms. The
candidate was accepted.

This proves one genuine subsecond round on this machine. It does not establish
sustained subsecond throughput or the 200 ms target. A single exact 4,096-row
verification pass is about 95 ms and recent full-coordinate proposals are about
84-117 ms. The remaining path to 200 ms requires avoiding repair passes and host
synchronization without reducing the declared tokens, coordinates, or model.

Exact greedy execution is now an explicit `[generation.verify]` policy. Serial
decoding and accepted-prefix verification have identical token semantics;
`window` and `max_window` bound power-of-two verifier batches, while `passes`
controls broad fixed-point passes. `unroll` batches stalled repair waves into
one Metal command buffer; it changes scheduling, not the exact-token contract.
Omitted policy preserves automatic selection, fixed 128-row windows, two broad
passes, and one repair wave per submission.

On the deterministic 4,096-token smoke candidate, repair unrolling
preserved completion SHA-256
`9b512fe89e5f8aa846d32352f23a3377ef12ed0488fc7c405848c93708c30ac1`.
With AC Low Power Mode enabled, two-way unrolling reduced complete-round wall
time from 1,865.664 ms to 1,640.579 ms and rollout GPU time from 1,622.785 ms
to 1,450.035 ms. Four-way unrolling reached 1,490.106 ms GPU and eight-way
unrolling regressed to 1,696.145 ms, so the retained configurations use two.
These throttled observations establish the synchronization effect, not an
unthrottled latency baseline.

Suffix verification now keeps one PISA pyramid per model layer and visit. A
repair recomputes only the 64-token leaves intersecting its dirty row range and
their ancestor closure. A 128-row repair therefore rebuilds two or three leaves,
not all 64 leaves at 4K or all 128 leaves at 8K. Each rebuilt node retains the
same FP32 accumulation order and FP16 store boundary as the full rebuild. This
is a structural work reduction; no new latency result is claimed until a paired
exact-token run measures it.

The accepted-prefix path slides each verifier window on the four-row PISA
query-tile boundary rather than recomputing from the prior 128-row boundary.
For the deterministic 4,096-token smoke candidate, this preserved the completion
hash while reducing verifier work from 19,968 to 17,024 evaluated positions and
paired rollout GPU time from 1,648.163 ms to 1,359.942 ms. The current clean run
with a 1,024-row adaptive cap completed in 934.092 ms with a 108.784 ms proposal,
19,328 evaluated positions, 80 correction waves, and all 4,096 tokens generated
in-loop on an Apple M4 with Low Power Mode disabled. An earlier 4,096-row cap
required 20,096 evaluated positions and 1,242.413 ms, so that aggressive cap is
rejected. The 934 ms observation re-establishes one subsecond round after the
change, but it is not a paired attribution result and the 200 ms target remains
unmet.

Two cheap-draft hypotheses were rejected on the same deterministic candidate.
A stale-hidden candidate-readout draft preserved only 30 accepted tokens and
regressed the round to 1,850.861 ms. A one-feedback-pass candidate-conditioned
self-draft accepted zero tokens and regressed it to 2,112.710 ms. Their final
token arrays matched exactly after target verification, so correctness held
while both performance hypotheses failed. Both runtime paths were removed. The
surviving change only corrects the dormant five-feature capture's context stride
and suffix row addressing. The next decoder must be a trained block editor with
the existing target verifier left unchanged.

### mHC hybrid pretraining round

The corrected one-round mHC smoke is retained at
`.cache/ennx/runs/pretrain/hybrid-smoke-20261003/run-1791033933034-64285-0`.
The candidate and incumbent were evaluated on the same 8,190 causal targets.
NLL moved from 9.038374 to 9.038240, a paired improvement of 0.000133 with
estimated variance `5.08e-9`, and the candidate was accepted. The candidate
also generated all 4,096 tokens inside the round. That smoke used temperature
zero, so its single-token output measures greedy argmax collapse rather than
the diversity or coherence of sampled generation.

The measured round took 1,597.272 ms. The paired causal scorer used 1,085.976 ms,
the proposal used 141.719 ms, and the free-running rollout used 302.830 ms of
reported GPU time. This validates the learning/generation loop and misses the
200 ms target. Initial and final held-out generation rewards were
`[0.16043526, 0.099351406]` and `[0.16043526, 0.09939170]`.

### Earlier free-running-only mHC round

A one-round mHC smoke run used the production free-running path with Megatron
initialization. The candidate changed 889,707,164 of 1,047,699,736 weights,
generated all 4,096 tokens, evaluated 4,224 positions in one broad verifier
pass, and completed the measured BO round in 1,217.510 ms. Proposal time was
735.425 ms and rollout GPU time was 337.341 ms. Low Power Mode was enabled.

The objective vector was -9.034104 mean target NLL, -9.172988 worst-window
target NLL, and 0.177063 byte reconstruction. The output contained one unique
token repeated 4,096 times. This run verifies generation, reward, MORBO
feedback, checkpointing, and mHC execution together. The current looped experiment uses the free-running objective vector; this
older collapsed run remains a falsifier of target NLL alone. It does not establish
coding ability or meet the 200 ms target.

### Free-running pretraining falsifier

A complete 100-round run of `code-pretrain-generated.toml` is retained at
`.cache/ennx/runs/pretrain/05d81d8cc8427c8bf913/run-1790901977084-13432-0`.
It perturbed all 1,047,732,224 weights, generated and scored 4,096 tokens every
round, accepted six candidates, and wrote a 2.0 GiB checkpoint. Incumbent reward
improved from -9.013134 to -9.010755, but generated-text quality did not improve:
unique tokens fell from 11 to 9, the longest identical-token run grew from 822
to 926, and repeated four-grams rose from 96.41% to 98.36%. Free-running target
NLL therefore rewards this collapse and is rejected as a standalone pretraining
objective.

The run also falsifies sustained subsecond optimization with the current ENN
proposal path. Median complete-round latency was 5.841 seconds and the maximum
was 10.701 seconds. Mean proposal time grew from 332 ms over rounds 1-10 to
8.133 seconds over rounds 91-100, while reward evaluation itself stayed below a
millisecond. The immediate systems target is the history-dependent proposal
path, not reward computation. The run used an adaptive verification range of
128-1,024 rows; the 200 ms target was not met.

### FineWeb selection pilot

The matched 12-round FineWeb arms completed with median candidate-round walls of
496.070 ms for ENN and 495.023 ms for random selection. Initial held-out NLL was
9.013232470; final NLL was 9.012842178 for ENN and 9.013103008 for random.

This is a corpus-prefix, teacher-forced control with one paired repetition and
only three guided rounds. It does not establish an optimizer advantage and must
not be represented as the free-generation workload.

## What failed

### Code-contrast generation pilot

The prepared 512-round ENN and random-selection studies are not valid training
experiments in their current form. Their initial runs exposed severe collapse:

| Measurement | Generated candidates | Corpus targets |
| --- | ---: | ---: |
| Unique tokens in 4,096 | 4-7 | 247-671 |
| Repeated four-gram fraction | 0.979-0.995 | 0.309-0.764 |
| Longest identical-token run | up to 3,629 | up to 10 |

The live reward is count-clipped byte n-gram overlap with the target minus the
strongest decoy overlap. Repetition statistics are logged but are not part of
the reward. The startup control checks only constant single-byte strings, so it
does not catch repeated tokens or short multi-token cycles. The reward-dispersion
gate also accepts differently collapsed strings.

The ENN arm stopped after round 20 and the random arm after round 33. Each had
one accepted candidate, only 4-7 unique generated tokens, and sustained timing
above the target. These runs are falsification evidence, not learning results.
Do not resume the 512-round studies. The untrained starting policy has not
passed the base-policy promotion gate in the
[coding-agent contract](coding-agent-contract.md); changing a surface reward
cannot supply the missing coding capability.

The residual architecture campaign now uses paired causal corpus loss for
selection and retains a complete generated continuation as the per-round
emergence probe. Held-out causal NLL and held-out generation remain outside
optimization. Contrastive and reconstruction measurements remain diagnostics.

## Corpus state

The code-generation corpus uses repository-disjoint splits. The v2 episode
artifact selects at most one 4,224-token window per source file: 16 training,
three validation, and three untouched test episodes. Validation no longer
borrows examples from test. The prepared artifact contains 12, two, and three
repositories respectively, with zero repository overlap between every pair of
splits. All episodes retain repository, commit, path, content ID, source split,
and immutable token-pool offset.

The expensive Stack scan is cached in
`.cache/ennx/token-pools/f2fffb7dc5e78f1d47ff`. Preserve it. The resolved
generation corpus is `.cache/ennx/corpora/95bda3b17e0602eb02ac`; resolving either
matched arm should hit and fully hash-validate this cache. Its episode SHA256 is
bound by the corpus manifest.

## Immediate priorities

1. Produce a coherent coding checkpoint through causal training and generate
   its `ennx.base_policy.v1` qualification record.
2. Complete incremental `ennx.agent.v1` stream conformance and implement an
   executable immediate objective as specified by the coding-agent contract.
3. Run a short matched falsification experiment before any long learning campaign.
4. Demonstrate sustained subsecond complete rounds with 1,048,576 newly generated
   tokens per candidate and full history, then close the 200 ms gap.
5. Establish ENN selection benefit against random selection under matched
   objective budgets and independent repetitions.
6. Make 128-point history useful without storing 128 billion-coordinate rows.

No held-out coding-quality, global-optimality, sustained-subsecond, or 200 ms
claim follows from the current evidence.

## Source map

| Responsibility | Source |
| --- | --- |
| Experiment schema | [`config.rs`](../rust/crates/ennx/src/config.rs) |
| Generation loop | [`fbt_generation.rs`](../rust/crates/ennx/src/fbt_generation.rs) |
| Generation reward | [`reward.rs`](../rust/crates/ennx/src/fbt_generation/reward.rs), [`fbt_contrast.rs`](../rust/crates/ennx/src/fbt_contrast.rs) |
| Agent transcript | [`agent_contract.rs`](../rust/crates/ennx/src/agent_contract.rs) |
| Coding outcomes | [`coding_outcome.rs`](../rust/crates/ennx/src/coding_outcome.rs) |
| Base-policy gate | [`base_policy.rs`](../rust/crates/ennx/src/base_policy.rs) |
| Active model | [`fbt_moe.rs`](../rust/crates/ennx/src/fbt_moe.rs) |
| Production scorer | [`fbt_scorer.rs`](../rust/crates/ennx/src/fbt_scorer.rs) |
| MoE routing | [`fbt_moe_routing.rs`](../rust/crates/ennx/src/fbt_moe_routing.rs) |
| History and acquisition | [`bf16_metal.rs`](../rust/crates/ennx/src/bf16_metal.rs) |
| Proposal kernels | [`bf16_search.metal`](../rust/crates/ennx/src/bf16_search.metal) |
| Corpus preparation | [`pretrain.py`](../ops/pretrain.py) |
