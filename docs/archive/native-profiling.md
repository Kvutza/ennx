> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# Native GPU profiling probe

Verified 2026-09-24 on Apple M4, macOS 27.0 (26A428), Xcode 26.1.1.
System tools: `/usr/bin/gpucapture`, `gpudebug`, `metalperftrace`.

## Reproduction

```sh
tools/fbt-bo --check
gpucapture list
gpucapture boundaries --pid PID
gpucapture start --pid PID --boundary QUEUE_ID --count 1 \
  --disable-unused-recording --disable-backtrace-recording \
  --output .cache/bo-native-probe.gputrace
gpudebug -t .cache/bo-native-probe.gputrace -c 'status'
```

The full-space experiment's exact-shape feasibility probe currently rejects its
fused gate/up candidate before a full model run, so it is not a profiling
workload. The observations below describe an earlier full-model capture.

Use the actual PID and queue ID returned by the tools. Logical-operation and GPU
capture instrumentation remains outside the acceptance TOML because it perturbs
the timing under test. An arbitrary attachment point does not guarantee which
command buffers are captured. Do not use traced or capture-enabled timing as an
uninstrumented baseline.

## Observed capture

- Buck2 build ID: `d62c8dc4-c237-4d56-b183-9da44a024f71`.
- Three-round test passed; instrumented loop time: 337.747752 seconds.
- Capture took 182.5 seconds, exported 993 resources / 23494 MiB of GPU data,
  and occupies approximately 18 GiB on disk.
- Two command buffers: 99 conversion/packing encoders, then 266 encoders in
  the first transformer pass. Total: 365 compute dispatches.
- First buffer begins with `fbt_transpose_half`. Second begins with `fbt_lookup`,
  contains 24 `fbt_prefill_flash_attention_half_96` dispatches, and ends with
  `fbt_prefill_rms_half`.
- MPS dispatch internals are visible: the gate/up dispatch exposes
  `mmul_kernel_a16_half_half_float (2)`, grid 208x128x1, threadgroup 128x1x1.
- This capture does NOT cover the complete BO round, second-pass feedback,
  vocabulary readout, cross entropy, or controller execution.

## Profiling result and blocker

Both native replay profiles completed and were embedded in the trace:

```text
profile run --gpu-state default --exec overlapping --embed
profile run --gpu-state medium --exec serial --embed
```

The debugger exposes 30 hardware-counter groups, including occupancy,
instruction throughput, FP16/FP32, bandwidth, cache, and SIMD-inflight data.
Counter summaries are populated. This is richer than the timestamp-only
application-facing Metal counter API queried before the OS upgrade.

However, command/encoder/shader COST RANKINGS are empty in both profiles.
The first profile's encoder timeline assigns many encoders the same 0-to-
23985.8597 ms interval. Do not sum these intervals or rank kernels from them.
The cause is not established. Tool/runtime compatibility and capture/profile
configuration need investigation before adding more application instrumentation.
Serial replay is diagnostic, not normal execution timing.

Ignored local artifacts:

- `.cache/bo-native-probe.gputrace`: capture with embedded profiles.
- `.cache/bo-native-profile.jsonl`: exported profile inventory and timeline.
- `.cache/bo-capture-sample.txt`: CPU stack sample during the capture-enabled run.

No per-instruction attribution or reliable per-dispatch duration is established.

## Attribution investigation, 2026-09-24

Reloading embedded profile 1 in a fresh debugger session reproduces empty cost
rankings. Encoder inspection reports `Unknown` type and zero GPU time despite
frame-wide timeline intervals. This is not fixed by recreating the CLI session.

Read-only inspection of the `streamData` binary property list (NSKeyedArchiver)
found empty `shaderProfilerData`, `gpuTimelineData`, `APSData`, and
`APSCounterData` arrays. `APSTimelineData` has 704 entries. Encoder metadata has
365 records (14600 bytes / recorded 40-byte size), and command metadata has
365 records (11680 bytes / recorded 32-byte size). Thus metadata and sampled
counter data exist, but shader attribution is absent in these named arrays.
The private data format was not reverse-engineered; absence here does not prove
that no other representation exists. Do not manufacture timings from metadata.

### Independent Instruments probe

A five-second Metal System Trace capability test succeeded. Its exported GPU
table contains distinct start/duration values and encoder IDs. The default
template explicitly has shader timeline disabled and counter profile zero.

A subsequent 240-second all-process recording accompanied the unchanged legacy
three-round runner (capture disabled), build ID
`86f16fcd-3417-484f-8887-7a7df85cb412`.

- Instruments 26.0 (17B100), supplied by Xcode 26.1.1 on macOS 27, emitted
  overlapping-dylib mapping warnings during processing and export.
- Processing consumed substantial memory: a `sample` report recorded 30.6G
  physical footprint and 66.4G peak. This broad capture is unsuitable for
  routine measurements on the 24 GB machine.
- The BO process was deliberately interrupted after two complete rounds and
  the third incumbent score, to relieve memory pressure. The test therefore
  failed by interruption, not an observed assertion failure. It is NOT a
  successful three-round benchmark or a valid speed comparison.
- A stop signal was later sent to xctrace; it saved the recording and exited
  successfully. No profiling or BO process from this probe was left running.
- Exported BO GPU table: 30405 execution slices, 18 command-buffer IDs, 46
  encoder IDs. CPU encoder table: 47 BO encoder records across 18 buffers.
  Shader-profiler table: zero rows. These records are not a one-to-one inventory
  of the hundreds of application dispatches; they do not resolve kernel cost.
- A structured join on encoder ID matches 46 of the 47 CPU encoder records to
  the GPU table. The 30405 GPU rows repeatedly reference those same 46 IDs
  (one ID occurs in 2306 rows spanning about 12 seconds). They are GPU activity
  slices, not 30405 independently timed kernels. Summing or ranking those rows
  as dispatch costs would be invalid. The unmatched CPU encoder and generic
  `Compute Command` labels also prevent a complete named-operation ledger.
- Taking the union of all 30405 `Active` intervals assigned to this BO process
  gives 173.237 seconds of GPU activity within the 205.068-second span from
  its first to last recorded GPU interval (84.48%). This is trace coverage,
  not a clean-round utilization or throughput measurement: the recording was
  instrumented, all-process, and interrupted before three rounds completed.

Artifacts in `.cache`: `metal-system-capability.trace`,
`bo-metal-system.trace`, `bo-metal-gpu.xml`, `bo-metal-encoders.xml`,
`bo-metal-shaders.xml`, and `xctrace-processing-sample.txt`.

### Bounded shader-timeline probe

The installed Metal System Trace template has `shaderprofiler` and
`shaderprofilerinternal` disabled. A local copy in `.cache` enabled both fields;
the Instruments export confirmed `Shader Timeline: Enabled` and
`shader-profiler="1"`. No repository code or model path was changed.

A 20-second all-process trace covered `tools/fbt-bo --check` (build ID
`204d79f7-8bdb-4dab-9cd2-21d8746578fa`); all four scorer checks passed.
The exported GPU interval table has 8918 rows, including 402 assigned to the
`ennx` test process (PID 6815), so the trace did catch its GPU activity.
Of those 402, 360 active GPU intervals have distinct encoder IDs that match all
360 CPU encoder records for that process. Their recorded interval durations
range from 2.167 microseconds to 101.652 milliseconds (median 3.663 ms).
This establishes useful encoder-level GPU timing for the small scorer checks;
it does not establish named kernel costs or timings for the full BO workload.
The encoders are generically labeled `Compute Command`, and the other 42 GPU
rows are not additional distinct encoder dispatches.

Separately, `gpu-shader-profiler-sample` has zero rows. Instruments reported
`GPU Service reported error: Selected counter profile is not supported on
target device`. Thus enabling the template switch did not yield shader samples
or shader counter attribution on this M4/Xcode 26.1.1/macOS 27 combination.
The warning identifies an unsupported selected profile; it does not establish
whether the hardware, tool version, or profile configuration is responsible.

Ignored artifacts: `.cache/Metal Shader Timeline.tracetemplate`,
`shader-bo-check.trace`, `shader-bo-samples.xml`, and `shader-bo-gpu.xml`.
Do not repeat a long all-process capture with this configuration. Encoder
timeline timing and shader counter sampling are distinct capabilities; an
unsupported counter profile does not invalidate the 360 measured encoder
intervals. A named-operation mapping and a trace that covers the actual BO
dispatches are required before claiming a full-round bottleneck ranking.
No model changes or additional runtime instrumentation were made. The
per-operation timing blocker remains unresolved.

### Rejected encoder-label experiment

A temporary experiment labeled custom Metal pipelines/encoders with function
names and labeled cached MPS matrix kernels by shape. It preserved the existing
command-buffer and wait structure. A short Metal System Trace captured all four
small scorer checks and recorded ordered custom-kernel sequences such as
`fbt_prefill_rms_half`, `fbt_prefill_prepare_half_fused`,
`fbt_prefill_flash_attention_half_96`, and the FFN kernels.

This did not provide operation timings. Instruments concatenated the custom
kernel names into one encoder label for a whole command sequence. The MPS GEMM
shape labels were absent from the exported encoder, driver, GPU interval, and
MPS hardware tables. The execution-point table contained submission markers,
not per-operation durations.

More importantly, the labeled build was not numerically trustworthy: two of
six `tools/fbt-bo --check` invocations failed parity. One failure reported a
0.158112 maximum token-NLL error; another invocation failed two scorer cases
at 0.096912 and 0.188381. The other four labeled invocations passed. After all
labeling code was removed, four consecutive checks passed. This sample does
not prove the labels caused the intermittent errors or that the restored path
can never fail, but it is sufficient to reject measurements from the labeled
build. No labeling code remains in the runtime.

Ignored artifacts: `.cache/labeled-bo-check.trace`, `labeled-bo-encoders.xml`,
`labeled-bo-gpu.xml`, `labeled-bo-points.xml`, `labeled-bo-driver.xml`, and
`labeled-bo-mps.xml`.

### Explicit source-operation trace

The Instruments naming blocker is bypassed by the opt-in TuRBO-ENN config field
`trace = true`; the retained one-round runbook is
`examples/tuning/turbo-enn-trace.toml`. The normal config leaves this disabled.
The trace does not label or alter the normal scorer path. Instead, each logical
operation receives one ordered command buffer on the same queue. All commands
are submitted without intermediate waits and the CPU waits once on the final
command. Completed public Metal command-buffer timestamps supply each
operation's GPU interval.

For the fixed LocalV1 fused scorer, one objective call contains 545 scorer GPU
operations:

- pass 1: embedding lookup, 24 layers times 11 operations, final RMS = 266;
- pass 2: lookup, five feedback operations, 24 layers times 11 operations,
  final RMS, four readout GEMMs and four cross-entropies = 279.

Host input upload and loss reduction bring the emitted scorer ledger to 547
records. Every record contains domain (`host`, custom `metal`, or opaque
`mps`), pass, layer, logical operation, tensor shape, CPU encode/submit time,
and GPU start/end/duration. The command asserts the record count and timeline
reconciliation. Because Metal command-buffer intervals can overlap, the exact
identity is `GPU sum + gaps - overlap = GPU envelope`.

When model weights have changed, the same timeline starts with 99 GPU layout
operations: 49 generic weight transposes, two feedback-weight transposes, 24
QKVG packs, and 24 gate/up packs. A cached layout emits one host reuse record
instead. Preparation and scorer counts are reported and asserted separately.

The verified 2026-09-24 trace produced exact reconciliation for both objective
calls. The cached-layout incumbent emitted 548 records and took 44.343679
seconds wall with a 42.410735-second GPU envelope. The refreshed candidate
emitted 646 records (99 preparation plus 547 scorer), took 41.345594 seconds
wall, and had a 41.338442-second GPU envelope. Candidate layout preparation
used 0.105454 seconds summed GPU time. Incumbent aggregate GPU sums included
16.115497 seconds for 48 gate/up GEMMs, 8.511996 seconds for 48 down GEMMs,
7.721045 seconds for 48 flash-attention dispatches, 4.082449 seconds for 48
QKVG GEMMs, and 2.887037 seconds for four readout GEMMs. These sums are
descriptive trace evidence; they are not interchangeable with the GPU envelope
when intervals overlap.

Controller trace mode separately measured acquisition pool scoring, selection,
and materialization at 0.888629, 0.045988, and 0.109733 seconds, with a
1.932634-second GPU envelope overlapping incumbent scoring. Rejected `tell`
measured a 0.059041-second history-copy GPU interval and 0.746645 seconds wall.
The complete perturbative round took 86.813883 seconds.

Raw ignored artifact: `.cache/fbt-bo-operation-trace.log`. This provides
per-logical-operation attribution for scorer, weight preparation, acquisition,
and `tell` GPU submissions. Host bind, decision, restore, synchronization, and
accepted rebind are timed as complete phases. Opaque MPS GEMM implementation
details and hardware instructions remain below the visibility of this ledger.

The 2026-09-26 public trace config
`examples/tuning/turbo-enn-trace.toml` completed one rejected round
successfully at `results/turbo-enn-trace/run-1790446557798-7669-0`. The
result reported 63.988838 seconds for the round, 16.4052 GiB allocated,
`old_nll = 12.007460897`, `new_nll = 12.008238873`, `accepted = false`,
and `goal_met = false`. This was an attribution run, not a production timing
baseline.

Across the incumbent and candidate scorer traces, the largest operation groups
by summed GPU interval were:

| Operation group | Count | GPU seconds |
|---|---:|---:|
| MPS `gate_up_gemm` | 96 | 23.352882000 |
| Metal `flash_attention` | 96 | 12.195547500 |
| MPS `down_gemm` | 96 | 11.927438083 |
| MPS `qkvg_gemm` | 96 | 5.555670458 |
| MPS `readout_gemm` | 8 | 4.354876708 |
| MPS `attention_output_gemm` | 96 | 1.937206708 |

Controller trace intervals were 0.524236633 seconds for pool scoring,
0.027870125 seconds for selection, 0.295810883 seconds for materialization,
and a 0.847919983-second acquisition GPU envelope. Rejected `tell` spent
0.053198002 seconds in the history copy but 0.606576042 seconds wall. The
scorer remains the dominant measured surface; gate/up, flash attention, down
projection and QKVG are the immediate optimization order from this run.

The next gate/up diagnostic extended `tools/fbt-bo --gemm` with a split-MPS
variant that runs two `8192 x 6656` MPS GEMMs against separate gate and up
weights. It passed correctness and canary checks on 2026-09-26, but did not
beat the existing packed `8192 x 13312` MPS route:

| Gate/up variant | Median GPU seconds |
|---|---:|
| MPS packed NN | 0.227162 |
| MPS packed NT | 0.250308 |
| MPS split NN x2 | 0.236952 |
| Custom 64x64 | 0.304312 |
| Custom 128x64 | 0.308488 |

This rejects the split-MPS gate/up route for now. The bottleneck remains real,
but evidence points away from two separate MPS multiplies or the current custom
single-output kernels as the next production change.

The 96-wide flash-attention kernel was then changed to start each local-window
query block at the first potentially visible key tile instead of looping from
key block zero and branching over wholly invisible old tiles. The focused
`gpu_scorer_attention96` fixture passed for lengths 1/31/32/33/65/4096, full
causal and local masks; the 4K local case reported max error 0.000329564 and
0.014004 GPU seconds. A follow-up one-round public trace completed at
`results/turbo-enn-trace/run-1790447357485-10392-0` with the same printed
losses, decision and radius as the previous trace. The traced round changed
from 63.988838 to 61.390913 seconds. Summed `flash_attention` GPU interval
changed from 12.195547500 to 11.827949708 seconds. Treat this as a small
attribution-guided improvement, not a stable production benchmark distribution.

The matching one-round uninstrumented config
`examples/tuning/turbo-enn-one-round.toml` completed at
`results/turbo-enn-one-round/run-1790447634868-11743-0` with a 61.182787-second
round. Its two production scorer calls reported 29.822954 and 28.887949
seconds. The run exited successfully with `goal_met = false`; it confirms the
normal path still runs after the attention edit but does not establish a stable
speed distribution.

A follow-up normal run at
`results/turbo-enn-one-round/run-1790447807157-13452-0` completed with the same
printed losses, decision and radius but measured 77.249205 seconds. The newly
visible production phase line reported `restore_seconds = 0.002904875` and
`tell_seconds = 1.431390041`; the incumbent scorer itself took 42.797854
seconds. The non-scorer phase can matter, but scorer variability remains the
dominant uncontrolled effect in this pair.

Rejected `tell_paired` then stopped calling `check_reference()` after the
proposal-to-history copy, because rejection does not mutate the reference row
or reference RMS scales. The absolute FIFO and restart parity controller tests
passed. A subsequent production run at
`results/turbo-enn-one-round/run-1790450248171-16385-0` preserved printed
losses, decision and radius, and reported `tell_seconds = 0.393944250` with
`restore_seconds = 0.010624416`. Both scorer calls were slower than the prior
run, so the total round rose to 81.136647 seconds; only the phase split is
useful evidence here.

The readout blocking change replaced the optimized scorer's single
full-vocabulary readout GEMM per row chunk with 8192-vocabulary-column MPS
blocks plus a blocked-layout cross-entropy kernel. The first attempted wiring
fed BF16 parameter storage directly to MPS as FP16 and failed
`gpu_scorer_optimized` with token-NLL errors above 300. The accepted wiring
uses the existing FP16 transposed embedding cache with MPS column origins.
`gpu_scorer_optimized` then passed with maximum token-loss errors below
0.006, and `tools/fbt-bo --gemm` passed its exact-shape layout checks.

The matching one-round production run
`results/turbo-enn-one-round/run-1790455459958-22814-0` completed with the same
printed losses, decision and radius as the earlier one-round samples. It
reported a 28.365439-second round, 16.0615 GiB allocated, and scorer calls of
14.044583 and 12.946897 seconds. `goal_met` remains false for the 1000 ms
target. The paired trace run
`results/turbo-enn-trace/run-1790455593754-23210-0` reported a 28.190632-second
round. Its largest operation groups by summed GPU interval were:

| Operation group | Count | GPU seconds |
|---|---:|---:|
| MPS `gate_up_gemm` | 96 | 9.308923625 |
| Metal `flash_attention` | 96 | 5.824452333 |
| MPS `down_gemm` | 96 | 4.676166750 |
| MPS `qkvg_gemm` | 96 | 2.173863167 |
| MPS `readout_gemm` | 104 | 1.474680208 |
| MPS `attention_output_gemm` | 96 | 1.073910833 |
| Metal `cross_entropy` | 8 | 0.064890625 |

Corrected 2026-09-26: sum only `FBT_OP_GROUP` records, including their `count`
fields. The previous table counted both raw operations and their summaries,
doubling GPU times and adding two summary records to each operation count.
These instrumented GPU intervals are not uninstrumented round wall times.

This is a successful implementation-efficiency improvement, not a subsecond
result. Gate/up, flash attention, down projection and QKVG remain the measured
dominant surfaces.

The three-round configured run
`results/turbo-enn/run-1790456762994-24664-0` then completed with mean
27.747765 seconds and max 28.297495 seconds. Round times were 28.297495,
27.888071 and 27.057729 seconds; all three printed `accepted = false` and
`goal_met = false`. The scorer calls were 12.926092/12.926073 seconds in round
1, 14.618022/13.188155 seconds in round 2, and 13.587794/13.386063 seconds in
round 3. Rejected `tell` was 2.241657 seconds in round 1 but only 0.078848 and
0.081216 seconds in rounds 2 and 3, so steady-state wall time is scorer-bound
again after the first round's controller/history overhead.

The following minibatch-reuse and one-pass runs are historical ablations.
Their configs and runner options have been removed. The current study always
uses 4,096-token examples, two passes, and two objective calls per round.

The historical minibatch-reuse ablation
`results/turbo-enn-reuse-minibatch/run-1790457009127-25442-0` set
`minibatch_refresh = 3` while the default config was `1`. Round 1 took
28.716530 seconds with `incumbent_reused = false` and two objective calls.
Rounds 2 and 3 reported `incumbent_reused = true`, one objective call, and
14.257172 and 13.864263 seconds. The incumbent interval collapsed to
0.000342 and 0.000047 seconds, while the candidate scorer remained
13.317286 and 13.428642 seconds. This isolates the cost of incumbent rescoring
but still leaves the candidate scorer far above the subsecond goal.

The paired trace run
`results/turbo-enn-reuse-trace/run-1790457146535-26242-0` confirmed the same
fixed-minibatch shape with per-operation attribution. Rounds 2 and 3 reported
`incumbent_reused = true` and candidate scorer walls of 13.386341 and
13.395416 seconds. Summed across those two candidate scorers, the largest GPU
operation groups were:

| Operation group | Count | GPU seconds |
|---|---:|---:|
| MPS `gate_up_gemm` | 96 | 9.711699625 |
| Metal `flash_attention` | 96 | 5.831277126 |
| MPS `down_gemm` | 96 | 4.861501627 |
| MPS `qkvg_gemm` | 96 | 2.266551834 |
| MPS `readout_gemm` | 104 | 1.544039291 |
| MPS `attention_output_gemm` | 96 | 1.114871542 |

For one candidate scorer, round 2 alone spent 4.804487250 seconds in gate/up,
2.904708292 seconds in flash attention, 2.409961587 seconds in down projection,
1.123037920 seconds in QKVG, 0.763067375 seconds in readout, and
0.554467999 seconds in attention output. This is now the most relevant
single-objective-call ledger.

The one-pass ablation
`results/turbo-enn-standard-reuse/run-1790457331619-26875-0` set
`score_mode = "standard"` and kept `minibatch_refresh = 3`. This changes the
objective and is not the default FBT result. It measured 14.228903 seconds in
round 1 with two objective calls, then 7.693430 and 7.398349 seconds in rounds
2 and 3 with incumbent reuse and one objective call. The paired trace
`results/turbo-enn-standard-reuse-trace/run-1790457376068-27248-0` measured
8.204646 and 7.517054 seconds in the reused-incumbent rounds. Round 2's
one-pass candidate scorer spent 2.420446666 seconds in gate/up, 1.456713292
seconds in flash attention, 1.215010834 seconds in down projection,
0.769997082 seconds in readout, 0.570047916 seconds in QKVG, and 0.278694459
seconds in attention output. Thus the second feedback-conditioned pass accounts
for roughly half the fused scorer wall, but the single-pass full-context dense
candidate scorer remains far above one second.

External hardware context is consistent with the local GEMM probes. Public
M4 GPU references report roughly 3--4 TFLOP/s FP32-class peak/measured GPU
throughput, while this repo's exact-shape MPS GEMM probes report about
3.2--3.5 effective dense TFLOP/s for the dominant matrices. The compute audit's
capacity ledger therefore treats the current 2.6--2.7 effective dense TFLOP/s
as an implementation result in the same order of magnitude as the available GPU
surface, not as a 30x-missed epilogue-fusion opportunity. The subsecond path
must reduce or share arithmetic in the scorer.

### Joined wire-level traces

The source-operation trace is an outer naming map, not a production wall-time
measurement. A race-free Metal System Trace launched the already-built test
binary so recording was active before process creation. The target exited zero.
The trace contains 1,196 command-buffer submissions and 1,199 application
encoders: six setup encoders, three acquisition encoders, 545 incumbent scorer
encoders, 644 candidate preparation/scorer encoders, and one rejected-history
copy. Every application command-buffer and encoder ID appears in the hardware
GPU table.

Those encoders expand to 13,276 GPU activity slices. Their union is 89.623282
seconds across a 98.687888-second first-to-last GPU span. The ignored joined
artifact `.cache/fbt-bo-wire3-ledger.csv` has one row per application encoder:
source operation and shape, command-buffer and encoder IDs, CPU encode and
submission intervals, submission-to-GPU latency, GPU first/last time, span,
union, sum, and activity-slice count.

This exhaustive trace materially perturbs execution. One logical operation per
command buffer produced 1,193 round command buffers and 864 driver `Wait for
GPU` intervals totalling 77.701115 seconds. Per-operation GPU intervals remain
useful attribution, but its wall time and queue latency are not production
performance evidence.

A separate unchanged production-path Metal System Trace required
`MTL_CAPTURE_ENABLED=1` to produce nonempty tables. It exited zero and recorded:

- 10 command buffers and 24 application encoders;
- 5,183 GPU activity slices;
- 83.022380 seconds of unioned GPU-active time over a 90.907212-second GPU span;
- no driver `Wait for GPU` events;
- 892 `Wire Memory` intervals totalling 3.285977 seconds;
- 95.063358 seconds from process launch through exit.

That run is still capture-perturbed. A subsequent ordinary legacy one-round
run took 96.022929 seconds, with 48.741845- and 45.841844-second
objective calls. Earlier ordinary rounds in the same repository were about
60--70 seconds. The cause of this run-to-run state difference is not established;
do not compare captures or optimization changes without repeated controlled
baselines.

### Replay counters and instructions

A bounded production `gpucapture` used one queue boundary with unused-resource
and backtrace recording disabled. It still occupied 4.5 GiB because the
captured conversion buffer referenced 293 resources / 4,064 MiB. The buffer
contains exactly 99 encoders and 99 named dispatches:

- 51 `fbt_transpose_half` dispatches;
- 24 `fbt_pack_qkvg` dispatches;
- 24 `fbt_pack_gate_up` dispatches.

`gpudebug` overlapping replay succeeded in 36.6 seconds and embedded a 70.28 ms
profile. Unlike the earlier 18 GiB capture, this profile has populated cost
rankings:

| Shader | Cost | Invocations | Active time | Compiler instructions |
|---|---:|---:|---:|---:|
| `fbt_pack_gate_up` | 48.21% | 24 | 58.314 ms | 30 |
| `fbt_transpose_half` | 41.59% | 51 | 57.310 ms | 44 |
| `fbt_pack_qkvg` | 10.20% | 24 | 14.271 ms | 53 |

The largest transpose dispatch reports 677,630,991 executed instructions,
338,815,500 ALU instructions, 80.80% kernel occupancy, a 54.76% instruction
throughput limiter, and a 32.18% last-level-cache limiter. A representative
gate/up pack reports 61,102,080 executed instructions, 22,404,100 ALU
instructions, 91.39% occupancy, a 68.99% instruction-throughput limiter, an
80.56% integer/complex limiter, and a 36.52% last-level-cache limiter.

This proves instruction and hardware-counter visibility on the current M4.

### Targeted scorer captures

A production candidate capture established the private-dispatch inventory
before targeted replay: three command buffers, 644 application encoders, and
848 actual dispatches. The buffers contain 99 weight-layout encoders, 266
first-pass scorer encoders, and 279 feedback-conditioned second-pass encoders.
Thus MPS expands the 644 logical encoders into 204 additional private
dispatches. Pass-wide counter replays were rejected: they reported
`Consistent state: no`, assigned most encoders the full frame interval, and
left cost rankings empty.

`--capture-op` then isolated one command buffer for each dominant operation.
Normal scoring does not use the trace path or its attachment delay. The
following are replay measurements, not uninstrumented wall times:

| Logical operation | Actual shader | Grid | Replay GPU time | Executed instructions | Occupancy |
|---|---|---:|---:|---:|---:|
| `gate_up_gemm` | `mmul_kernel_a16_half_half_float (2)` | 208x128, 128 threads | 249.264 ms | 1,122,435,704 | 27.08% |
| `down_gemm` | `mmul_kernel_a16_half_half_float (2)` | 24x128, 128 threads | 65.330 ms | 110,231,504 | 21.41% |
| `qkvg_gemm` | `mmul_kernel_a16_half_half_float` | 49x128, 128 threads | 30.536 ms | 9,857,846 | 21.59% |
| `flash_attention` | `fbt_prefill_flash_attention_half_96` | 128x32, 128 threads | 77.521 ms | 64,944,256 | 15.08% |

The down and QKVG profiles report `Consistent state: yes`. Gate/up and flash
report `no`, so their complete-frame times and counters are retained but not
used as comparative rankings.

Additional counters identify different limiting behavior:

| Operation | Principal limiters | Device traffic | Cache misses | ALU mix |
|---|---|---:|---:|---:|
| gate/up | F32 95.55%, launch 94.17%, instruction 88.81% | 2.94 GiB read, 286.08 MiB write | LLC 41.99%, buffer L1 23.98% | 97.81% float |
| down | F32 82.21%, instruction 69.95%, launch 55.83% | 3.25 GiB read, 55.85 MiB write | LLC 63.70%, buffer L1 24.91% | 92.45% float |
| QKVG | launch 97.45%, F32 82.85%, instruction 73.71% | 1.05 GiB read, 71.77 MiB write | LLC 28.38%, buffer L1 30.24% | 89.63% float |
| flash attention | launch 95.31%, F32 63.07%, instruction 52.37% | 830.08 MiB read, 20.61 MiB write | LLC 32.56%, buffer L1 93.60% | 72.48% float, 0% half |

One logical readout GEMM (`m=2048`, `n=100352`, `k=1536`) expands into 52
private dispatches: 13 repetitions of three `ndArrayIdentity` dispatches and
one `mmul_kernel_a16_half_half_float` dispatch. Twelve groups use 256x48,
256x64, 128x32, and 256x64 grids; the final group uses 64x48, 64x64, 32x32,
and 64x64. Its default replay took 3.252 seconds, but both default and fixed
high-state profiles were inconsistent and attributed all cost to one identity
dispatch. No per-dispatch readout ranking is claimed. `gpucapture boundaries`
exposes only device and queue boundaries on this system, so it cannot isolate a
private dispatch inside that MPS encoder.

The targeted gate/up trace completed its full one-round test successfully.
After adding the capture selector, `tools/fbt-bo --check` passed all four GPU
checks. Useful ignored artifacts retained locally include
`.cache/fbt-bo-wire3.trace`, `.cache/fbt-bo-production-wire2.trace`,
`.cache/fbt-bo-compact.gputrace`, and `.cache/fbt-bo-scorer.gputrace`. The large
targeted capture packages were deleted after their counters were recorded to
conserve disk space; the capture commands and run logs remain reproducible.
Capture/profile timing must never replace the ordinary CLI wall-time baseline.
