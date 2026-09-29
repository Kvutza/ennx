# GPU execution and measurements

Updated: 2026-10-04. The active engine runs diffusion as a drafter and commits
causal target tokens. CUDA and TVM have not executed this complete engine.

## Learned draft and causal target

The complete round is acquisition → full-coordinate perturbation → learned
block draft → causal target verification → generated-output scoring → ENN/Pareto
and TuRBO update. [The example](../examples/tuning/diffusion.toml) uses
`[generation.draft]`, temperature 0.8, two denoising steps, 128-token visibility
blocks, and one or two recurrent visits in the draft. The target always executes
its seven-layer schedule with causal visibility.

The draft uses the candidate's learned mask embedding and index projection.
A query reads preceding blocks and its own block. Final draft tokens are proposals,
not commits. Target verification starts at the prompt, rebuilding every readable
cache entry under causal attention before accepting a prefix. Incumbent tokens
and block-visible draft KV cannot bypass this rebuild.

The drafter still executes full-width model layers. Low draft acceptance still
invokes target repair waves. This establishes the correct composition, not a
speedup. A smaller learned draft and efficient correction remain necessary before
claiming near-1x amplification.

`drafted-code` emits three objectives: generated-code contrastive overlap,
draft/target token agreement, and negative combined position amplification. All
three enter native vector acquisition. None uses a teacher-forced model forward.
Agreement is a draft training proxy; overlap is a text proxy; measured complete
round latency is the budget test. Pareto selection allows tradeoffs.

Draft metrics record initial proposal count, evaluated positions, device time,
wall time, initial accepted prefix and all-position agreement. Target counters
exclude draft work; total rollout counters include both. Main round timing
includes proposals, generation, scoring, tell, completion decoding/writing and
controller writing. Initial model setup and initial incumbent are outside it;
curve writes, console display, held-out validation and checkpoint export are
explicitly excluded.

Metal's vector mHC residual kernel preserves ordered FMAs and FP16 stores while
reusing source loads across four destinations. Its signed 129-row GPU comparison
is bit-exact. Draft cache refresh omits final-layer work that cannot change KV.
`ENNX_DIFFUSION_TRACE=1` reports encoder stages; disable it for latency runs.

The 128-token integration test checks exact final token agreement with direct
causal sampling at temperature 0.8 and invariance to changed reference labels.
Separate checks cover masks, confidence, visibility, selected attention support,
index drift and cache-only execution.

For context and newly generated output measurements:

```sh
./ennx model context-loop examples/tuning/diffusion.toml DATASET \
  --generated 4096 --prompt 128 --rounds 2
```

`--generated` counts new target tokens. At 1,048,576 new tokens and a 128-token
prompt, causal target evaluation uses 1,048,703 positions; draft generation uses
1,048,704. Both require the hierarchy's 2,097,152-slot allocation. Target execution
always has seven caches even if the draft uses fewer visits.

[Historical measurements](handoff.md#historical-diffusion-only-measurements)
reached 1.292 s at 4K, 9.896 s at 64K and 155.847 s at 1M. Those rounds scored
reconstruction and generated diffusion-only final output. They are not latency
results for this engine. Their 3.0121x position count and 112.577 trillion counted
contraction FLOPs likewise describe that historical workload.

Selective recurrence and block proposals draw on
[LoopMDM](https://arxiv.org/abs/2605.26106), soft mask/prediction inputs on
[DMax](https://arxiv.org/abs/2604.08302), and grouped/refined selection on
[PIVOT](https://arxiv.org/abs/2607.24593). Their trained-model speedups are not
ENNX measurements. Current draft support reuse remains an approximation within a
physical layer. TVM is a candidate compiler for stateless chunk contractions;
it does not currently own or execute the cache, target, or BO update.

## Kernel-agent harness

`./ennx tune examples/tuning/kernel-search.toml --prepare` saves the baseline,
candidate shader snapshots, hypotheses, numerical/performance contract and recent
search outcomes in `context.json`. Omit `--prepare` to execute the campaign.
Paths in the manifest are relative to that file; no experiment path is embedded in Rust.

The coding agent writes candidates. The CLI does not invoke a model API or spawn
coding agents. Each candidate supplies an operator, optional source file, optional
preprocessor defines, branch parent, measured bottleneck/evidence, hypothesis,
falsification criterion and expected milliseconds saved. Parent is provenance,
not implicit source inheritance. A source omitted from the manifest means the
production source embedded in the current executable.

Supported replacement boundaries are PISA's production four-query attention
entry point and the production MoE TensorOps source. Launch dimensions and buffer
ABI remain the host implementation's responsibility. This is a trusted-code
research harness, not a sandbox or proof that arbitrary shader code is correct.

Execution builds and snapshots one worker, copies the resolved dataset, and runs
branches serially. Search pairs share generated proposal/acquisition seeds and
alternate baseline/candidate order. The baseline's complete round count and all
model/objective/optimizer settings are retained. Diagnostics in the baseline are
rejected, not silently removed. Compilation, model initialization and frozen
anchor setup are outside loop timing, as in the existing production run.

Every measured trial retains all 8,192 token losses per round, two sequence losses,
and controller choices. Numerical capture occurs after the per-round timer but is
included in complete-loop time for both arms. Promotion uses paired **complete-loop
time divided by round count**, not a scorer-only win or the fastest round. The
configured numerical limits allow bounded differences; a changed proposal or
acceptance trajectory is classified as incomparable, not a numerical equivalence
success. Gate thresholds are policy choices, not statistical confidence intervals.

Only the strongest passing branch receives final validation, with new seeds drawn
after source selection is frozen. These are held-out perturbations on the same
model and corpus, not an unseen model/shape benchmark. Validation must pass the
same numerical and whole-loop timing gates. The 200 ms target is reported separately
from a relative improvement using complete-round maximum and loop-average time;
eligibility never rewrites production files.

Artifacts include `context.json`, `feedback.json`, per-pair `experiment.toml`, raw worker
logs, controller records, environment records, candidate sources, and the executable.
Compilation/numerical failures and slower or inconclusive branches persist as
feedback for subsequent campaigns. Interrupted campaigns cannot promote a branch.
Worker timeouts terminate only that worker and abort the campaign. A workspace
lock serializes this harness's campaigns; it cannot exclude unrelated GPU users.
Low Power Mode invalidates a trial. Neither that guard nor order alternation
guarantees thermal stability or exclusive GPU access.

This implements the execution/feedback and persistent-evidence ideas discussed
in [Metal-Sci](https://arxiv.org/abs/2605.09708),
[KernelOPT](https://arxiv.org/abs/2609.30059), and
[Harness Engineering](https://arxiv.org/abs/2607.17979).
It does not claim their reported CUDA speedups transfer to this workload.

## Fixed workload

Use the [TuRBO-ENN runbook](turbo-enn.md). Preserve full parameter coverage,
two 4K examples, one selected-candidate objective per timed round, both FBT
passes and the active model-aware acceptance rule unless a change is explicitly
authorized. Initial incumbent scoring is separate. GPU-only execution is the
current requirement; do not substitute the legacy paired dense workload.

## Before claiming an improvement

- Record a correctness-qualified complete-round baseline over multiple rounds.
- Identify exact source functions and the work the change removes or overlaps.
- Account for packing, weight refresh, allocation lifetime, command submission,
  completion waits and readback.
- For shared computation, identify unchanged inputs and prove cache validity
  across weight revisions, examples and feedback passes.
- For attention changes, preserve the selected preset's causal masks, routing,
  grouping, normalization, positional encoding and gating. Do not transfer
  LocalV1 assumptions to PISA/MoE without checking the active implementation.
- For precision changes, name operand/accumulator formats and report token-loss,
  mean-loss and decision differences. Do not label a tolerance check exact.
- Run the relevant `./ennx` check after each edit and `./ennx dev` for integrated
  changes. Report complete-round time, not only an isolated kernel improvement.

## Measured budget and falsifiers

### 4,096-token generated-reward path

The current generation architecture is a separate workload from the older
two-example pretraining measurements below. It has 1,047,732,224 independent
FP16 coordinates, five physical layers with two feedback visits, 625 routed
experts plus one shared expert, and top-three routing.

| Dependent component | Measured wall ms | Correctness gate |
| --- | ---: | --- |
| Selected full-coordinate Rademacher proposal | 83.758 | 1,047,732,224/1,047,732,224 FP16 values changed |
| Selected full-coordinate Ziggurat Gaussian proposal | 151.000 | All coordinates sampled; 547,552,769 FP16 values changed |
| One exact 4,096-position fixed-point verification | about 95-96 | Exact rollout equality on the verifier test |
| Exact 2,048-position suffix verification | 49.450 | Bit-identical suffix versus the full verifier |
| Controlled Rademacher full BO round | 5,967.550 | 4,096 generated tokens; 99,200 evaluated positions |
| Controlled Gaussian full BO round | 7,100.504 | 4,096 generated tokens; 117,120 evaluated positions |

Exact tile-Jacobi repair reuses all 128 proposals from every aligned suffix
pass. Matching rows advance the causal frontier, the first mismatch is committed
because its prefix is exact, and later rows seed the next pass. An older
scalar-commit run measured 31,187.848 ms and a later favorable tile-Jacobi run
measured 734.862 ms, but their independently randomized initializations make
that magnitude non-causal. The
first automatically seed-matched comparison measured 5,967.550 ms for
Rademacher and 7,100.504 ms for Gaussian. It evaluated 99,200 and 117,120
positions respectively. The authoritative controlled artifacts are
`.cache/ennx/runs/generation-4096/run-1790815988746-59739-0` and
`.cache/ennx/runs/generation-4096-gaussian/run-1790816007822-60088-0`.

The repair window now starts at the nearest preceding four-row PISA query-tile
boundary instead of the preceding 128-row boundary. On the same deterministic
4,096-token candidate this reduced evaluated positions from 19,968 to 17,024
and rollout GPU time from 1,648.163 ms to 1,359.942 ms while preserving the
completion SHA-256
`9b512fe89e5f8aa846d32352f23a3377ef12ed0488fc7c405848c93708c30ac1`.
The paired complete-round walls were 1,865.865 ms and 1,607.394 ms. These runs
were taken with AC Low Power Mode enabled and therefore establish the causal
windowing improvement, not a replacement for the normal-power latency record.
A subsequent clean run reported 1,595.796 ms complete wall, 137.981 ms proposal,
1,390.939 ms rollout GPU, and the same token hash.

Two parameter-free speculative drafts were falsified on the same deterministic
candidate under AC Low Power Mode. Re-reading prior target hidden states with
the candidate final normalization and readout left accepted draft tokens at 30,
increased evaluated positions to 17,536, and raised the complete round to
1,850.861 ms. A candidate-conditioned one-feedback-pass self-draft accepted
zero tokens, increased evaluated positions to 25,216, and raised the round to
2,112.710 ms. Both produced the same final token array after exact verification.
Do not restore either shortcut: full-coordinate perturbations require a trained
candidate-conditioned editor, not a stale readout or an early recurrent exit.

The dormant five-tap draft capture now uses the allocation context as its slot
stride and the active suffix row as its source and destination offset. The old
active-row stride would alias feature slots and misplace suffix captures.
Capture remains outside the production hot path.

The 200 ms target remains unmet. Rademacher proposal plus one verifier consumes
roughly 179 ms before host overhead; Gaussian consumes roughly 246 ms. A valid
next result must lower those baseline kernels and demonstrate useful
candidate-conditioned draft acceptance on a trained checkpoint. Both current
artifacts use untrained initialization and constant zero reward, so longer runs
would test execution stability but not BO learning.

The unchanged Gaussian production experiment completed 12 rounds at 945.512 ms
median wall and 745.128 ms median scorer GPU time. Artifact:
`.cache/ennx/runs/pretrain/6fcb4cb43872c211c4c8/run-1790785007632-68966-0`.
These are separately reported medians, not additive stage measurements.

The following existing `scorer-stages.toml` experiment measured three complete
profiled scores at 734.412 ms median, with zero observed sequence-NLL error
against the unsplit scorer. Its following production rounds had 949 ms median
wall and the same printed NLLs and acceptance decisions. Artifact:
`.cache/ennx/runs/pretrain/d123b1cb1e828162fa84/run-1790785298873-69532-0`.
Profiling splits encoder boundaries; do not present it as unperturbed dispatch
timing or compare these two studies as a kernel improvement.

| Stage | Profiled median ms | Logical matrix operations, billions |
| --- | ---: | ---: |
| Routed and shared gate/up | 215.775 | 695.785 |
| PISA selection and attention | 174.774 | 364.401 |
| Routed and shared down | 104.895 | 347.892 |
| QKV | 72.818 | 257.698 |
| Output projection | 59.419 | 206.158 |
| Routing and packing | 36.247 | 51.540 |
| Combined MoE/residual/RMS | 28.162 | — |
| Readout | 19.733 | 68.719 |
| Pre-FFN | 14.798 | — |
| Feedback | 2.757 | 8.590 |
| PISA pyramid | 1.176 | — |
| Embed | 0.682 | — |

Count multiply and add separately. Matrix work totals 2,000,783,671,296
operations for 8,192 rows and 48 layer executions. Gate/up and down include
three routed experts plus the shared expert at width 216. QKV width is 640;
model width is 512. Attention counts, per sequence position, 64 keys for each
of up to seven preceding selected blocks and the valid prefix of the current
block. There are 1,853,440 valid query/key pairs per sequence across all
positions. The executed eight-key tile coverage instead counts 1,867,776,
raising QK/PV matrix work to 367.220 billion. This still excludes online
rescale matrices, expert tile padding, routing selection, Gaussian generation,
activations, normalization and softmax. It is not an instruction count.

Three conclusions and their limits:

- **Proposal-only work cannot achieve 200 ms:** the measured scorer alone is
  over 700 ms. Overlapping CPU bookkeeping does not remove dependent GPU work.
- **Gate/up plus attention alone cannot achieve 200 ms:** even subtracting
  both measured stages entirely leaves about 344 ms of profiled scorer work.
  This is a budget calculation, not a prediction of rescheduled execution.
- **Hardware feasibility remains unproven:** a 200 ms round needs over 10.0
  trillion useful matrix operations/s even with all other work free. Gate/up,
  down, QKV and output projection currently deliver about 3.33 trillion/s
  together. That is achieved throughput, not a measured silicon ceiling.

The next throughput hypothesis must distinguish insufficient matrix issue
rate from cache traffic, spills and synchronization at the actual production
shapes. Source-level operand loads are not measured DRAM bytes. Any proposed
schedule must count both activation and weight reuse, live accumulators and
scratch; reducing one operand's reads is insufficient. For example, the prior
32-row joint gate doubles logical weight operands relative to 64-row production
while reducing input operands; its seven-pair scorer comparison lost by
46.728 ms median. The 16-row cached-input experiment produced NaNs and has no
valid speed result. Loop unrolling won only four of seven pairs at a
0.832 ms median gain: inconclusive, not a production improvement.

Before the next kernel edit, state the removed work, an expected stage and
whole-round saving, and a rejection criterion. Use alternating complete-scorer
pairs to challenge the claim, check objective differences explicitly, then
validate any retained change through full production rounds. A counter or
same-shape throughput probe that cannot expose the proposed bottleneck leaves
the claim unresolved. The retrospective Metal metrics collection for the
production run returned `No session found`; it supplied no bandwidth,
occupancy or spill evidence. Do not repeat the known heavyweight capture
failures merely to obtain an empty counter report.

## CUDA-to-Metal transfer, 2026-09-30

The current [CUTLASS/CuTe DSL stack](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
exposes layout, copy, MMA and pipeline primitives. Its
[Blackwell producer/MMA/epilogue example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_3_1.py)
depends on TMA and tensor memory. Transfer the data ownership and lifetime
decisions, not those NVIDIA instructions or their resource budgets.

| CUDA mechanism | Metal experiment or boundary |
| --- | --- |
| Interleaved gate/up and paired activation epilogue | One M64/N128/K512 operation per 64-channel pair; N48 for the final 24 channels |
| Register-resident epilogue | Cooperative gate fragment retained; only up values exchanged through 8 KiB threadgroup memory |
| Producer/MMA/epilogue warp specialization | Not a mechanical port: Metal's synchronous TensorOps scope and cache hierarchy need their own schedule |

The layout experiment comes from the dataflow in the
[TIRx/cuDNN persistent SwiGLU kernel](https://github.com/mlc-ai/TIRx-kernels/blob/main/tirx_kernels/ported/cudnn/swiglu/dense_gemm_persistent_swiglu.py),
not a copied PTX implementation. It keeps canonical model weights and packs a
1,369,571,328-byte FP16 buffer once per scored candidate, reused across both
feedback passes. Shared expert zero keeps its original layout. Packing reads
and writes 2,739,142,656 logical bytes per score and is inside comparison timing.
No claim that halving source-level input operands halves DRAM traffic.

[Apple's inline TensorOps guidance](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
explicitly says its on-chip cache hierarchy avoids the need for threadgroup
staging of matrix inputs. It recommends cooperative register results and
restricts the per-core neural-accelerator claim to Apple GPU family 10 and
later; API availability alone does not establish that hardware on this M4.

The first interleaved version exchanged both branches through 16 KiB scratch.
It matched all measured token losses but lost all seven scorer pairs by
29.650 ms median, including packing. Keep it as a control, not production.
Artifact: `.cache/ennx/runs/pretrain/dd4a590e067ad6db80c1/run-1790788088377-76775-0`.

The register epilogue initially changed maximum token NLL by 0.003046036
while sequence NLL changed by 0.000000954. An explicit volatile-half gate
boundary restored zero observed differences without relaxing the thresholds.
This isolates a sensitive rounding boundary, not a proof of identical machine
instructions. Its first timing run was unstable: baseline scorer median
1,717.324 ms versus the preceding 745.203 ms control. Do not use that run to
claim a clean speed delta. Artifact:
`.cache/ennx/runs/pretrain/f0f4588520e0bd39f612/run-1790788369489-78493-0`.

The instrumented repeat measured 45.019 ms of packing and 511.300 ms of
gate/up, versus 478.804 ms baseline gate/up. These profiles are sequential,
not clock-normalized. Its alternating complete-scorer comparison lost six of
seven pairs by 71.039 ms median with zero observed token/sequence-NLL error.
The earlier repeat lost all seven pairs by 50.360 ms, then failed profiling
because the counter allocation omitted the optional preparation stage. That
capacity error is fixed; no invalid profile is used for the attribution.

After the repeat, `pmset -g batt` reported AC power, and `pmset -g custom`
reported `lowpowermode 1` for AC but `0` for battery. The earlier power state
was not recorded, so this is a confound, not proof of the cause of every
timing change. Do not compare those regimes as a kernel effect. Power settings
were left unchanged pending the user's choice. No interleaved kernel is
promoted to production.

Low-power instrumented artifact:
`.cache/ennx/runs/pretrain/a16eaf1e175dd269e684/run-1790788664693-80063-0`.

On battery with Low Power Mode verified off, the interleaved register version
still lost all seven pairs: +31.150 ms median (baseline 776.989 ms, candidate
807.885 ms). Token and sequence losses matched. Packing alone measured
36.982 ms in the separate candidate stage profile. This rejects the complete
repacking path, not every possible canonical-layout change. Artifact:
`.cache/ennx/runs/pretrain/cd53a8e1a5f759b2b9df/run-1790790032972-4148-0`.

## PISA fragment lowering, 2026-09-30

Emitting AIR exposed two private arrays in the active Q4 kernel: eight FP32
result fragments and eight FP16 query fragments, accessed with dynamic
indices. An unroll pragma only added loop metadata. Explicit compile-time
indices removed both arrays and their dynamic addressing from AIR. The
loop-control function's AIR matched the pre-edit function; the offline
default-language and Metal 4 control functions also matched. This does not
prove the original final GPU binary spilled registers.

The removed PISA experiment compared literal indexing and direct register
rescaling with the existing identity skip. Neither variant became production.

The first valid literal-index comparison matched every measured token loss
but won only four of seven pairs, at -0.700 ms median. This is inconclusive,
not an optimization win. The following unchanged-production rounds completed
at 973 ms median with two acceptances. Artifact:
`.cache/ennx/runs/pretrain/1547427656a88ea0811b/run-1790790438824-5712-0`.

During the combined rescale experiment the machine was found back on AC with Low
Power Mode enabled; baseline scorer time rose to about 2.7 seconds. That run
was interrupted, and its apparent literal-index speedup is not used. The
rescale candidate has offline compilation evidence but no completed numerical
or speed validation yet. Interrupted artifact:
`.cache/ennx/runs/pretrain/bea4db5bf50b7e9c9818/run-1790790545388-6209-0`.

Full-scorer comparisons and stage profiles now read the public
`NSProcessInfo.isLowPowerModeEnabled` property before and after GPU scores,
and reject low-power measurements. This does not change power settings or
add checks to normal production rounds. It does not detect every thermal or
contention change. The CLI's low-power rejection was verified at
`.cache/ennx/runs/pretrain/3858bc09632e89d1b243/run-1790790738342-7169-0`.

Reproduce the candidate AIR inspection (this is not GPU machine assembly):

```sh
xcrun metal -std=metal4.0 -O3 -ffast-math -DPISA_SKIP_IDENTITY_RESCALE -DPISA_UNROLL_FRAGMENTS -DPISA_DIRECT_RESCALE -S -emit-llvm rust/crates/ennx/src/fbt_pisa1.metal -o /tmp/ennx-pisa-rescale.ll
```

## PISA scratch lifetime reuse, 2026-09-30

`PISA_REUSE_SCORES` is an opt-in kernel-search branch. Each head reads its
64 FP32 scores before a SIMD-group barrier permits overwriting their first
128 bytes with FP16 probabilities. The byte stride between heads is preserved;
PV loads use twice the original stride in half elements. After routing, its
64-float query scratch holds the rescaling diagonal instead of overwriting
live probabilities. The quad-softmax variant is rejected at compile time
because it has a different lifetime schedule. Default production is unchanged.

On-device pipeline metadata confirms 14,016 -> 9,856 bytes of static
threadgroup storage (4,160 bytes removed, 29.7%). A standalone full-shape
correctness probe compared production against scratch reuse alone and reuse
with literal fragments/direct rescaling. For each variant, all 4,194,304 FP16
attention outputs and 65,536 selected blocks matched bit-for-bit at input
scales 0, 0.05, 0.5 and 2; no nonfinite outputs. Generated seed:
11497193432086626079. Probe, libraries and executable:
`/tmp/ennx-pisa-scratch.S7nJmD/` (temporary local artifacts).

This was correctness-only, including zero/tied inputs and random inputs at
three magnitudes. It is not a proof for arbitrary inputs or a complete-model
loss/controller check. AC Low Power Mode was still enabled, so no performance
claim or promotion is made. `kernel-search.toml` includes both branches for
paired complete-loop evaluation. Extra softmax barriers and doubled half
strides may erase an occupancy benefit. Manifest savings are hypotheses,
not measured results.

## Relevant commands

```sh
./ennx tune examples/tuning/code-pretrain-kernel.toml
./ennx tune examples/tuning/scorer-stages.toml
./ennx tune examples/tuning/kernel-search.toml
```

The first command is production optimization; the second explicitly adds
stage attribution before its production rounds. Neither is a unit check.
Use only the intended configuration and authorized budget. Source changes
need relevant validation; evidence-only documentation does not require the
full test suite. Legacy tools/fbt-bo
diagnostics cover the dense scorer and do not certify the active MoE path.
See [handoff](handoff.md) for dated results and remaining evidence gaps.

## Million-token context

The target is 1,048,576 newly generated tokens per candidate in a complete BO
round under one second. Retain the prompt and full generated history. Retain the
five physical layers, selective recurrence, four mHC streams, and 625-expert
bank. Store one KV head per execution of a layer; process queries and MoE
activations in chunks. Readout storage must scale with the query chunk.
Measure initialization, candidate generation, scoring, and complete BO wall time
separately. A short continuation after a million-token input measures a different
workload.

The first implementation is a persistent PISA attention cache on Metal and
CUDA-Oxide. It accepts contiguous prefix writes and repairs, rebuilds affected
leaves and their ancestor closure, and evaluates compact queries against the
cache. Metal reuses the production SIMD-matrix attention kernel; CUDA reuses
its cooperative selected-block attention kernel. Parent summaries are built
in parallel per level. Selection retains the first, previous, and current
blocks and chooses up to eight 64-token blocks in total. The hierarchy has a
logarithmic number of selection levels; selected-token attention remains
bounded at 512 tokens per query.

At the target length, each execution needs 268,435,456 bytes of FP16 KV storage
and 4,194,176 bytes of key summaries. Seven executions require 1.75 GiB of KV
plus approximately 28 MiB of summaries, excluding weights, chunk scratch,
routing, logits, and allocator overhead. A 128-query attention chunk uses
266,240 bytes for queries, selected blocks, and outputs. These are calculated
buffer sizes, not a measured full-model memory footprint.

A 128-token prompt plus 1,048,576 new tokens requires 1,048,703 evaluated
positions. The power-of-two hierarchy allocates 2,097,152 slots; seven visits
therefore allocate 3.5 GiB of KV and approximately 56 MiB of summaries.

```sh
./ennx model context --tokens 1048576 --queries 128 --repeats 7
./ennx cuda context 1048576 128 7
./ennx cuda sanitize context 1048576 128 3
```

The shared Rust fixture checks prefix initialization, append, FP16 tree
agreement, exact selected block IDs, bounded output error, early/middle/final
positions, future-content invariance, and a four-token repair crossing a leaf
boundary. Records identify attention execution without claiming full-model
generation or learned capability. The fixture supplies already projected
queries and keys; it does not test RoPE or QKV projection. Artifacts are retained
under `.cache/ennx/runs/context/20261004`.

### Research reviewed on 2026-10-04

| Work | Implication for ENNX |
| --- | --- |
| [NAMOH, September 30, 2026](https://arxiv.org/abs/2609.38832) | Couples active attention parameters to routed token histories. Reported long-context quality comparisons reach 32K. Head-relative positions preserve order but discard original distances. Keep this as an architectural comparison; ENNX shares one KV head and has no head router. |
| [MiniMax Sparse Attention, June 2026](https://arxiv.org/abs/2606.13392) | Selects blocks before exact attention on that support. Its full index branch still contributes a quadratic prefill term. Retain hierarchical selection and evaluate KV reuse across queries; H800 speedups do not establish T4 or M4 performance. |
| [OctoLong, August 2026](https://arxiv.org/abs/2608.05141) | Constructs dependency-rich code contexts using ASTs, language servers, and package resolution. Its data reaches millions of tokens; reported context-extension training is capped at 128K. Apply the construction principle to Rust repositories. |
| [SPIN, April 2026](https://arxiv.org/abs/2604.26837) | Aligns sparse selection with paged KV storage. Start with resident FP16 KV on M4/T4; add paging after measuring capacity and transfer costs. |
| [Frayed RoPE, ICLR 2026](https://arxiv.org/abs/2603.18017) | Studies positional extrapolation failures and proposes partial-channel RoPE-ID. Compare explicit positional variants on a learned checkpoint. |
| [LongCodeBench](https://arxiv.org/abs/2505.07897) | Uses code comprehension and repair at million-token windows. Functional coding outcomes should accompany retrieval probes. |

### Complete generation execution

Metal now chunks the complete recurrent forward pass, including RoPE, routing,
mHC, and sampled readout, with a separate persistent cache for each layer visit.
Each BO candidate rebuilds its readable history under its own weights. The
incumbent supplies draft tokens; its KV state is not reused across weight changes.
Numerical comparison with the packed scorer checks sampled token IDs, summary
trees, hidden states, and target losses. CUDA now has a streamed executor with
separate recurrent-visit KV. Its 4K and 64K generation comparisons match packed
token IDs and accepted frontiers; see `cuda/README.md` for measured artifacts.

```sh
./ennx model context-loop CONFIG DATASET --generated 1048576 --prompt 128 --rounds 1
```

The current dense contractions require at least 39,705,600 operations per
generated token: seven visits through QKV, attention-output projection,
625-expert routing, and the active 864-wide SwiGLU branches, followed by
8192-token vocabulary readout. Counting multiply and add separately gives
41.634 trillion operations per million-token pass. Attention, mHC, elementwise
work, padding, and verification retries add to this number. This is an arithmetic
work count, not a measured runtime or a bound for alternative architectures.

NVIDIA specifies a [65 TFLOP/s mixed-precision peak for T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf).
Dividing the contraction count by that peak gives an ideal 640.5 ms per pass,
before attention, transfers, proposal, scoring, and repair. Two complete passes
already exceed one second on this arithmetic model. The 200 ms ambition requires
reducing that work, changing precision or execution semantics, or using more
compute; compiler scheduling alone does not remove the operations.

Compare 4K, 64K, 256K, and 1M contexts at fixed active expert
width, recurrence, and training exposure. Vary active expert computation and
data exposure separately to map equal-capability curves. Measure functional
correctness on held-out repositories with distant dependencies, completed
generation latency, prefill latency, and peak memory. Keep generation objectives
as the experiment objective; numerical attention checks do not introduce a
teacher-forced training objective.

TVM remains a candidate compiler for stateless chunk contractions. It does not
own the cache, execute PISA selection, or compile these new kernels today.

### Execution and architecture priorities, 2026-10-05

The current goal is a million-token context window with a declared output budget,
not a mandatory million new tokens per second. The initial engineering workload
uses 1,024 new tokens. Full-space weight changes invalidate the context state;
fresh prefill belongs in every candidate's measured BO round.

The first real-corpus 1M CUDA run completed on T4: 1,047,552 prompt tokens and
1,024 new tokens, with identical outputs and frontiers at chunk sizes 512 and
1,024. At chunk 512, generation took 60.484 s wall / 59.218 s device. The first
full forward wave took 58.401 s, versus 0.817 s for subsequent verification.
The first wave includes initial suffix predictions, so this is not a pure
prefill measurement. It nevertheless establishes where this workload spends
its time: removing later repair waves alone cannot deliver the target. A
chunk-1,024 run took 54.885 s device. Output remains incoherent; these runs do
not establish learned long-context use or include the BO update. Evidence:
`results/cuda/generation-t4-corpus-1m/` and its adjacent `-source/` directory.

A follow-up run sampled stage events at three first-wave chunks. At position
524,288 with 1,024 rows, MoE took 20.687 ms (37.2%), selected attention
17.617 ms (31.7%), and mHC 8.976 ms (16.1%) of 55.585 ms. Hierarchy/index work
took 3.775 ms (6.8%). These are sampled-chunk measurements, not whole-context
stage totals. The profiled run reproduced every baseline token and frontier.
Evidence: `results/cuda/generation-t4-corpus-1m-profile/`. Prioritize expert
batching and selected-attention arithmetic; faster tree lookup alone cannot
remove the dominant costs.

The 4,096-row follow-up preserved every token and frontier while lowering device
time to 52.025 s versus a paired 512-row reference's 60.956 s. Constructor
scratch is now reused when its shape already matches; KV/tree allocation remains
fresh per invocation. Generation wall time was 52.042 s, first-wave time
49.367 s, and subsequent verification 2.659 s. The larger chunks improved
prefill throughput but increased repair work to 1,101,824 evaluated positions.
This motivated separate prefill/repair chunk scheduling, now measured below,
and selected-attention/expert kernel optimization. Neither changes the requirement
to rebuild context states after full-space weight perturbations. Evidence:
`results/cuda/generation-t4-corpus-1m-chunk4096/`.

### Hardware measurements, 2026-10-05

All workloads below generate 1,024 tokens. Metal ran on a 10-GPU-core M4 with
24 GB unified memory and **Low Power Mode enabled**. CUDA ran on Modal T4
with 16 GB GDDR6. These are different execution scopes and sampling paths;
the table is not a backend speed comparison. Peak memory was not measured.

| Backend and scope | Context window | Wall time | Evidence |
| --- | ---: | ---: | --- |
| Metal BO initialization round | 4,096 | 0.773 s | One observation; guided acquisition absent |
| Metal BO initialization round | 65,536 | 3.692 s | One observation; guided acquisition absent |
| Metal BO initialization round | 1,048,576 | 52.175 s | One observation; guided acquisition absent |
| Metal guided BO rounds | 4,096 | 0.583 s median; 0.478–0.774 s | Five sequential guided rounds after 31 initialization rounds |
| CUDA generation, 4096/512 scheduling | 65,536 | 3.797 s | One observation; BO stages excluded |
| CUDA generation, 4096/512 scheduling | 1,048,576 | 51.338 s | One observation; BO stages excluded |

Metal round timing includes proposal, perturbation, generation, scoring, tell,
completion writes and controller writes. It excludes setup, initial incumbent,
curve writing, console display, held-out validation and checkpoint export.
At 1M the full invocation took 165.849 s, including setup and initial incumbent.
CUDA timing includes fresh cache allocation and generation; checkpoint load,
BO stages and artifact serialization are outside it. No timing establishes
useful text quality: completions remain incoherent. Five observed 4K guided
rounds satisfy one second but miss the 200 ms target; larger contexts remain
well above one second even before guided acquisition is measured.

The CUDA scheduler now uses separate prefill and repair chunks. The 1M first
wave took 50.369 s, followed by 0.850 s of verification. The paired all-512
reference took 61.679 s device versus 51.220 s for 4096/512, with exact tokens
and frontiers. Numerical gates also passed against packed 4K and 64K execution.

Raw records are in `results/metal/bo-*-20261005/` and
`results/cuda/generation-t4-schedule-*/`. Publication-format PDF/SVG/PNG figures,
CSV tables, plotting sources and detailed captions are under
`results/reports/20261005-context/`. Figures state the sample counts and power
condition; sequential rounds are not independent repetitions. Normal-power
Metal measurements, repeated matched workloads and a complete CUDA BO path
remain necessary for a backend performance comparison.

### Remaining execution priorities

1. Retain accepted-prefix KV within generation and bound activation scratch.
   CUDA now implements this. Prefix readout before the final prompt position is
   omitted; sampling retains absolute-position seeds. Tree ancestors update in
   one dispatch, and small routed batches use 16-row tensor-core tiles.
2. Optimize selected-support attention and routing. Current PISA attention uses
   scalar FP32 dot products. Routing counts scan the assignment array once per
   expert. These are kernel targets with unchanged model semantics.
3. Train a cheaper context encoder and shared generator memory, following
   [YOCO](https://arxiv.org/abs/2405.05254) and
   [DeepSeek V4.1 Flash](https://arxiv.org/abs/2609.19969). This changes the
   architecture and requires training; existing checkpoint KV cannot be shared
   arbitrarily between visits.
4. Evaluate learned sequence compression before expensive recurrent/MoE work,
   using [H-Net](https://arxiv.org/abs/2507.07955) as a mechanism reference.
   Retain fine-detail retrieval through PISA. H-Net's gradient-trained results
   do not establish learnability with our full-space BO procedure.
5. Train a target-conditioned block drafter and acceptance scheduler using
   [DFlash](https://arxiv.org/abs/2602.06036) and
   [DSpark](https://arxiv.org/abs/2607.05147). No trained diffusion drafter is
   currently available. Judge accepted tokens per total draft/verification time.

Independent architectural comparisons are
[LatentMoE](https://arxiv.org/abs/2601.18089), which reduces routed dimension,
and [DeaMoE](https://arxiv.org/abs/2608.14385), which shares expert-group weights.
Single-Pass mHC changes the recurrence to enable residual fusion. Each needs a
trained comparison before replacing the current model.

For the present seven-visit shapes, an estimated fresh 1,048,576-token prefill
costs 41.504 trillion operations: 19.482T expert, 14.799T projection/router/mHC
predictor, and 7.223T selected attention, excluding prompt vocabulary readout.
This assumes 480.5 visible selected tokens per query and omits elementwise work,
padding and search overhead. At a hypothetical sustained 20 TFLOP/s, a 400 ms
prefill budget allows 8T operations, requiring about 5.2x less arithmetic.
This is a design budget, not measured throughput or an achieved BO latency.
