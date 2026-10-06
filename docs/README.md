# Documentation

## Active billion-weight pretraining

Start here, in order:

1. [Current state](handoff.md): implementation, evidence, and open problems.
2. [Runbook](turbo-enn.md): configuration and actual round semantics.
3. [History at billion dimensions](history.md): memory costs, approximation,
   and the proposed exact-replay alternative.
4. [Selection audit](bo-audit.md): measured Bayesian-selection results and limits.
5. [Optimizer evals](evals.md): paired black-box suites and scorecards.
6. [Coding-agent contract](coding-agent-contract.md): data, reward, evaluation,
   and eventual agent-harness promotion gates.
7. [Feedback transition](feedback-transition.md): inter-pass invariants and the
   architecture decision.
8. [Manifold-constrained residual streams](manifold-hyperconnections.md):
   four-stream model, Metal contract, and matched experiment.
9. [GPU execution and measurements](kernel-architecture-plan.md).
10. [Perturbation lab](perturbation-lab.md): noise laws and extension boundaries.

The active experiment uses `fbt-pisa1-diffusion-mhc4-v1` with
1,047,704,344 perturbable FP16 coordinates. Diffusion drafts; the causal target
commits. Native vector BO optimizes generated-code overlap, draft agreement and
combined position work. See [the experiment](../examples/tuning/diffusion.toml)
and [its execution semantics](kernel-architecture-plan.md#learned-draft-and-causal-target).
The current example and a run's resolved experiment.toml are authoritative for
settings; dated measurements are not current defaults.

## Development and separate workloads

- [Build](buck2.md), [tests](testing.md), [API](api.md),
  [Python integrations](interop.md), [Bazel](bazel.md), [Colab](colab.md).
- [Upstream metric learning](metric-learning.md): reading note for the
  `dsweet/more` AUTO diagonal metric branch.
- [Experiment catalog](catalog.md): algorithm inventory and validated TOML generation.
- [Qwen](qwen.md) and [FLAME](flame.md): separate experimental workflows.
  Their dated validation sections do not certify the active pretraining path.
- [Correlated proposals](perturbations.md): Qwen/FLAME mathematics, not the
  independent-noise pretraining default.

Superseded experiment narratives were removed after their surviving decisions
and falsifiers were folded into the current-state and evaluation documents.
