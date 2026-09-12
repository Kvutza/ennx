# Documentation

## Active billion-weight pretraining

Start here, in order:

1. [Current state](handoff.md): implementation, evidence, and open problems.
2. [Runbook](turbo-enn.md): configuration and actual round semantics.
3. [History at billion dimensions](history.md): memory costs, approximation,
   and the proposed exact-replay alternative.
4. [Selection audit](bo-audit.md): measured Bayesian-selection results and limits.
5. [Optimizer evals](evals.md): paired black-box suites and scorecards.
6. [Research contract](full-space-bo-sprint.md) and
   [measurement checklist](kernel-architecture-plan.md).
7. [Perturbation lab](perturbation-lab.md): noise laws and extension boundaries.

The active model is the 1,038,508,544-weight FP16 FBT/PISA/MoE preset.
Do not substitute dense LocalV1, Qwen, or FLAME results for this workload.
The current example and a run's resolved study.toml are authoritative for
settings; dated measurements are not current defaults.

## Development and separate workloads

- [Build](buck2.md), [tests](testing.md), [API](api.md),
  [Python integrations](interop.md), [Bazel](bazel.md), [Colab](colab.md).
- [Qwen](qwen.md) and [FLAME](flame.md): separate experimental workflows.
  Their dated validation sections do not certify the active pretraining path.
- [Correlated proposals](perturbations.md): Qwen/FLAME mathematics, not the
  independent-noise pretraining default.

## Historical evidence

[Archive](archive/README.md) preserves superseded plans and benchmark narratives.
It is evidence, not an implementation checklist. Current documentation takes
precedence; archived commands and claims of "current" behavior are historical.
