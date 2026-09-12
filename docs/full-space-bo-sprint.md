# Full-space research contract

Consolidated: 2026-09-29. Requirements are distinct from implemented behavior.

## Scope

Optimize every independent model weight without backpropagation or an adapter.
Dense independent coordinate innovations are required, subject to FP16 rounding.
A seed descriptor for an exactly replayable dense proposal does not reduce the
search dimension. Low-rank generating factors, sketches, or sampled-coordinate
metrics are not silently interchangeable with full realized-weight geometry.

Gaussian is the current example's law; Rademacher is an explicit supported
alternative. Neither a change of law nor a change of controller is a
semantics-preserving kernel optimization.

## Fixed boundaries

- Keep row-ID LOOCV as the surrogate-fitting objective for this sprint.
- Keep perturbation law, candidate geometry, acquisition, acceptance, and
  trust-region policy separately identified.
- Preserve tied weights, tensor coverage, fixed scales, and rounded candidate
  identity between selection and evaluation.
- Do not reuse activations or KV state across changed model weights.
- The active performance target is a complete 4K, batch-two, two-pass round
  under one second. Report 16K/32K separately when those contexts are tested;
  buffer capacity alone is not throughput or quality evidence.
- Compare against random selection and eventually forward-budget-matched
  full-space ES. Merely naming EGGROLL/ES is not a completed comparison.

## Known implementation gap

The original contract requires exact realized-vector distances to retained
observations. Current pretraining retains 128 logical observations but only two
resident historical rows and approximates other distances. This does not fulfill
the exact-history requirement. [History.md](history.md) records the memory cost
and proposed replay alternative; it does not silently relax the contract.

## Accepting a research change

State the mechanism, assumptions, changed search geometry, memory cost, and
complete-round work. Keep a reference path and reproducible numerical checks.
Match model initialization, corpus, context, objective budget, and declared
random streams in optimizer comparisons. Separate optimization measurements
from validation, and reserve a fresh test set for final claims.

Judge Bayesian selection by fixed evaluation outcomes, not acceptance count,
a tiny acquisition-score difference, or the number of knobs exposed.
An original implementation of a paper's mechanism still needs provenance and
license review before importing any external code.

See [current state](handoff.md), [measurement checklist](kernel-architecture-plan.md),
and [archived research/experiment ledger](archive/full-space-bo-sprint.md).
