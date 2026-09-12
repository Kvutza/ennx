# GPU Optimization Evidence Checklist

Updated: 2026-09-29. This replaces the previous heterogeneous execution plan.
These are investigation requirements, not implemented optimizations or
performance predictions.

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

## Current checks

```sh
./ennx test
./ennx tune examples/tuning/code-pretrain.toml
./ennx dev
```

The tune command launches actual optimization, not a unit check; run it only
with the intended configuration and authorized budget. Legacy tools/fbt-bo
diagnostics cover the dense scorer and do not certify the active MoE path.
See [handoff](handoff.md) for dated results and remaining evidence gaps.
