> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# GPU Scoring Workload

Updated: 2026-09-24. Scope: current GPU-only FBT BO path.

## Execution

[Model](../../rust/crates/ennx/src/fbt_model.rs) owns canonical BF16 parameters.
[Prefill](../../rust/crates/ennx/src/fbt_prefill.rs) scores equal-length examples
layer-major using Metal kernels and MPS matrix operations.
[fbt_prefill.metal](../../rust/crates/ennx/src/fbt_prefill.metal) contains attention,
normalization, packing, FFN and readout kernels.
[fbt_mps.rs](../../rust/crates/ennx/src/fbt_mps.rs) describes matrix layouts.

The reference and optimized GPU routes are not identical precision paths.
The optimized route uses half-precision intermediates and packed weights in
supported shapes. Weight representations are refreshed against model revisions.
The CLI defaults to optimized batched prefill. See the
[runbook](turbo-enn.md) for the complete workload and checks.

## Arithmetic accounting

Count multiply-add as two operations. The following is conventional dense
arithmetic for the specified workload, not a hardware measurement or a lower
bound on every possible algorithm.

| Component per complete round | Trillion floating-point operations |
|---|---:|
| FFNs | 48.241073 |
| Other layer projections | 11.171211 |
| QK and probability-times-V | 7.835430 |
| Vocabulary projection | 5.050882 |
| Feedback projections | 0.154581 |
| Total | 72.453175 |

Reproduction inputs:

- Each layer has 37,773,312 projection/FFN matrix elements, including
  `3 * 1536 * 6656` FFN elements.
- Layer matrix work: matrix elements times 24 layers times 4096 positions
  times eight sequence passes times two operations.
- Readout: `1536 * 100352` weights times 4096 positions times four scored
  sequence passes times two operations.
- Full causal attention has 8,390,656 valid pairs per sequence/head.
  A 2048-token local window has 6,292,480.
- Combine four full and twenty local layers, sixteen heads, head dimension
  96, eight sequence passes and four operations per pair per head dimension.
- Feedback: two 1536-square matrices, 4095 noninitial positions, four
  feedback sequence passes and two operations per multiply-add.

Excluded: masked/padded extra work, softmax, normalization, activations,
conversion, acquisition, memory movement and synchronization.
The components are rounded independently in the table.

The [source-level audit](bo-complexity.md) additionally counts masked feedback
rows still dispatched, attention tile padding, and diagonal matrix rescaling
of attention accumulators. The table above counts useful dense work, not all
matrix operations executed by the current implementation.

Unchanged execution of this counted work in one second would require 72.45
trillion useful floating-point operations per second, plus excluded work.
This arithmetic does not establish either attainability or impossibility on
the machine, and is not a measured bottleneck breakdown.

## Evidence status

The inspected machine reports M4, a 10-core GPU and 24 GB unified memory.
An uninstrumented three-round run on macOS 27 passed in 275.324042 seconds,
91.774681 seconds per round by loop time. Whole scoring calls account for
98.35% of that loop and whole-pass Metal GPU intervals for 95.68%. These
measurements localize time to scoring/GPU passes, not to individual kernels.
See [handoff](handoff.md) for the build ID and round-level results.

Previous documents mixed different implementations, precisions and timing
boundaries. Older results must not be presented as the current GPU baseline.
The previous document is available with
`jj file show -r 5a10dbc1 docs/m4-scoring-design.md`.

## Performance changes

Keep model dimensions, examples, feedback passes, objective and parameter
coverage explicit. Report any numerical-contract change separately. Scheduling,
fusion, layout reuse and arithmetic reuse are distinct mechanisms; each needs
a source change, correctness checks and a complete-round measurement before
a speedup can be attributed to it.

See the [measurement checklist](../kernel-architecture-plan.md). No accelerator
peak, decode-throughput result, or unmeasured overlap schedule establishes a
subsecond BO round.
