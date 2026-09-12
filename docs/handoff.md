# Current pretraining state

Checked against source: 2026-09-29. Native Rust/Metal on Apple M4,
24 GB unified memory.

## Implemented

- Preset: FBT/PISA/fine-grained MoE, 1,047,650,816 independent FP16 weights in
  6,316 tensor blocks.
- Each timed round scores one selected candidate on two 4,096-token examples,
  with two feedback passes. Initial incumbent scoring is outside the round loop.
- Four proposals: two independently seeded full-coordinate directions, each
  at half and twice the trust length, clipped to bounds.
- Initial tensor RMS scales, with a positive floor. The active configuration
  learns four bounded family sensitivities from ENN leave-one-out likelihood;
  fixed family multipliers remain available as a baseline. Gaussian and
  Rademacher are separate supported laws.
- Up to 128 logical observations; only two historical weight rows resident.
  Other history distances are approximations, not exact full-vector distances.
- ENN fitting uses the stored distance matrix and row-ID LOOCV. Guided selection
  and model-aware acceptance use the surrogate; initialization has its own rule.
- Pretraining compares candidate and incumbent on the same token batch. The ENN
  target accumulates paired improvement from zero, and uncertainty uses
  contiguous 128-token loss-difference blocks. Configured neighbor count is a
  fitted upper bound in the checked-in experiments, not a fixed posterior width.
- At 128 observations, history compacts to the incumbent and initialization
  resumes until the configured neighbor count is reached.
- Selected weights bind directly into the scorer; per-tensor realized-change
  records describe attempted proposals, including rejected ones.

See [run semantics](turbo-enn.md) and [history limitations](history.md).

## Evidence, not defaults

| Experiment | Observed result | Limit |
| --- | --- | --- |
| Recorded 100-round full-weight pretraining run | Median wall 1.266180 s, four acceptances | Older configuration; not demonstrated pretraining quality or subsecond execution |
| Three-seed, 32-round ENN-vs-random selection audit | Random finished slightly better in all three pairs; mean NLL difference 0.00004677 | Shared ENN acceptance, one model initialization, small validation set |
| Exact-distance synthetic audit | 107/270 winners changed; million-coordinate score differences were tiny | Not a billion-weight training-harm estimate |

The 100-round artifact is
`.cache/ennx/runs/pretrain/e9ebf5ccdffe38d3217f/run-1790654604391-26725-0`.
Its resolved settings differ from the current example. Full protocols,
test counts, and artifact references are in [the selection audit](bo-audit.md).
Historical timing attribution is in [the archived handoff](archive/handoff.md).

## Open problems

1. Make 128-point history useful within memory and latency budgets. Exact seeded
   replay is a proposed storage strategy, not an implemented fast distance oracle.
2. Establish that acquisition improves fixed validation outcomes over random
   selection under matched objective budgets.
3. Assess uncertainty and acceptance under changing minibatches. Two examples
   and a two-standard-error threshold do not provide calibrated confidence.
4. Reach a subsecond complete round without changing the declared workload.
5. Keep metadata honest: the existing `full_realized_weights` label does not
   mean that every historical distance is exact.

No held-out generation-quality or global-optimality claim follows from these runs.

## Granular MoE scorer

The candidate scorer now uses the fine-grained MoE path implemented in
`fbt_moe_routing.rs`, `fbt_moe_routing.metal`, and
`fbt_moe_routing_tensorops.metal` for the agreed architecture:
128 token-choice routed experts plus one always-on shared expert, all width
216; each token selects three routed experts by router logit and combines
their sigmoid scores after normalization. The shared branch plus three routed
branches gives active hidden width 864. The expert FFN parameter count is
0.78% above the 32x864 baseline; including the wider router, the MoE parameter
count is about 0.90% higher per layer. The complete candidate inventory is
1,047,650,816 FP16 parameters and is derived from tensor shapes in code.

The implemented stage computes all router logits with a Metal 4 MPP TensorOps
projection, chooses ties by lower expert id, records the top-3/top-4 logit
margin, computes exact expert loads, and packs all `8192 * 3` assignments into
stable variable-count expert segments. It is token-choice and dropless: there
is no fixed per-expert capacity, overflow drop, or batch-dependent
expert-choice route. The packed input, token id, expert id, route weight,
segment offset, and load buffers feed variable-load Metal 4 gate/up and down
projections. The always-on shared branch runs beside the routed assignments;
one gather per token and output column adds the shared result and the three
weighted routed results without atomics. Dynamic row extents cover partially
filled expert tiles, and dynamic K preserves the exact width-216 down
projection instead of padding it to a hardware multiple.

The normal test suite compiles and executes the complete Metal 4 path on a
four-token deterministic tie case. It verifies lower-ID top-3 selection,
normalized equal weights, exact dropless loads and offsets, a complete packed
row permutation, packed metadata, shared/routed expert evaluation, and final
numeric recombination. This is a correctness test, not evidence for
production-shape latency. No end-to-end performance workload has run since
the scorer was rewired.

## Source map

| Responsibility | Source |
| --- | --- |
| Example and schema | [code-pretrain.toml](../examples/tuning/code-pretrain.toml), [config.rs](../rust/crates/ennx/src/config.rs) |
| Active model and loop | [fbt_moe.rs](../rust/crates/ennx/src/fbt_moe.rs) |
| Fine-grained MoE routing host | [fbt_moe_routing.rs](../rust/crates/ennx/src/fbt_moe_routing.rs) |
| Fine-grained MoE routing kernels | [fbt_moe_routing.metal](../rust/crates/ennx/src/fbt_moe_routing.metal) |
| Fine-grained MoE router projection | [fbt_moe_routing_tensorops.metal](../rust/crates/ennx/src/fbt_moe_routing_tensorops.metal) |
| History, fit, acceptance | [bf16_metal.rs](../rust/crates/ennx/src/bf16_metal.rs) |
| Learned family metric | [bf16_family.rs](../rust/crates/ennx/src/bf16_family.rs) |
| Proposal and selection kernels | [bf16_search.metal](../rust/crates/ennx/src/bf16_search.metal) |
| Geometry and selection diagnostics | [bf16_audit.rs](../rust/crates/ennx/src/bf16_audit.rs), [fbt_ablation.rs](../rust/crates/ennx/src/fbt_ablation.rs) |

Use no Git commands for this work. Preserve unrelated user changes.
