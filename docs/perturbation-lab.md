# Full-Weight Perturbation Lab

Updated: 2026-09-29.

## Scope

This is the perturbation boundary for the 1,038,508,544-parameter FP16
pretraining BO loop. A perturbation law supplies one centered, standardized
coordinate to the resident Metal search. Tensor scaling, trust-region radius,
candidate pairing, realized-FP16 distance, acquisition, and accept/reject logic
remain separate. Every implemented law must cover every weight and must replay
the selected candidate bit-for-bit during materialization.

The active laws are:

| TOML value | Coordinate law | Mean | Variance | Metal generation |
| --- | --- | ---: | ---: | --- |
| `gaussian` | Standard normal | 0 | 1 | SplitMix64 and Box--Muller |
| `rademacher` | Uniform over {-1, +1} | 0 | 1 | SplitMix64 and two bit tests per coordinate pair |

For tensor block `j`, coordinate `i`, current tensor scale `s_j`, and candidate
radius `r`, both paths form the ideal update

```text
delta[j,i] = r * s[j] * z[j,i]
candidate[j,i] = fp16(incumbent[j,i] + delta[j,i])
```

Both laws therefore preserve the current first and second coordinate moments.
They are not optimization-equivalent: the Gaussian fourth moment is 3 and the
Rademacher fourth moment is 1, their tail behavior differs, and FP16 rounding
can map them to different realized neighborhoods. Rademacher is a research
alternative, not an exact implementation optimization of Gaussian search.

The four-candidate pool remains two independently seeded directions evaluated
at `0.5 * trust_length` and `2 * trust_length`, clipped to the configured trust
limits. The same direction is shared by each radius pair. The GPU computes
exact distances to resident historical FP16 rows and materializes only the
selected row. Other logical-history distances are approximated; see
[history geometry](history.md). Full-coordinate generation does not imply
exact distances to all 128 logical observations.

The perturbation distribution is independent of the tensor-family shape. In
the active learned mode, four ENN metric sensitivities adapt `s_j` by family;
their scales are normalized so the global trust-region energy remains under
the radius controller. The fixed initial-RMS/static-family mode remains the
comparison baseline.

## Configuration

New studies use the sectioned TOML form:

```toml
[perturbation]
distribution = "rademacher"
```

The legacy flat `perturbation = "rademacher"` form remains supported.
Omitting the selector selects `gaussian`, preserving prior runs. The field is
rejected for workloads that have not wired the same contract. Run artifacts
record the selected law in `study.toml`, `run.log`, and `result.toml`.

Perturbation law and acquisition are independent controls. The selected
full-weight proposal pool is scored through the shared TuRBO-ENN `Ask` contract,
whose neighbor count, Thompson/UCB choice, variance scales, output scale and
seed are resolved from the same study TOML.

## Performance hypothesis

Gaussian generation performs a logarithm, square root, and `sincos` for each
coordinate pair in each of two independent streams. Rademacher generation uses
one mixed 64-bit word per stream and extracts two signs. It removes those
transcendentals from both the four-candidate pool traversal and selected-row
materialization.

This does not predict an end-to-end factor. The controller still reads the
incumbent and history, rounds four candidates to FP16, computes full realized
resident distances and pool geometry, reduces tile statistics, and writes the
selected 2.077 GB row. The scorer is unchanged. Only complete 4K-context BO
rounds can establish whether this changes end-to-end latency. Historical
component medians are not current budgets; see [dated evidence](handoff.md).

Compare the laws with identical model, corpus, rounds, power state, and seeds.
Record complete wall time, scorer GPU time, ask time, tell time, changed-weight
fraction, realized/requested RMS ratio, acceptance, and trust radius. Treat
distribution-dependent rewards as research outcomes, not timing parity.

## Design provenance

The ENNX implementation is original Metal and Rust code. No source code,
examples, tests, or benchmark assets from the projects below were copied. Their
architectures and papers informed the separation of perturbation law,
controller, objective evaluation, and experiment logging.

| Project | Contribution acknowledged | Repository license |
| --- | --- | --- |
| [Modular CMA-ES](https://github.com/IOHprofiler/ModularCMAES) | Independently configurable sampling, adaptation, restart, and boundary modules | MIT |
| [IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter) | Separation of experiment execution from structured logging | BSD-3-Clause |
| [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust) | Native ask/tell boundary and external accelerator evaluation | MIT |
| [Nevergrad](https://github.com/facebookresearch/nevergrad) | Common derivative-free optimizer and parametrization interfaces | MIT |
| [evosax](https://github.com/RobertTLange/evosax) | Batched strategy interfaces and composable fitness/restart machinery | Apache-2.0 |
| [EvoX](https://github.com/EMI-Group/EvoX) | GPU-oriented evolutionary workflow research only; no GPL implementation code is used | GPL-3.0 |

Primary research references are de Nobel et al., [Modular CMA-ES
(2021)](https://doi.org/10.1145/3449726.3463167); de Nobel et al.,
[IOHexperimenter (2024)](https://doi.org/10.1162/evco_a_00342); Rapin and
Teytaud, [Nevergrad (2018)](https://github.com/facebookresearch/nevergrad);
Lange, [evosax (2022)](https://arxiv.org/abs/2212.04180); and Huang et al.,
[EvoX (2024)](https://doi.org/10.1109/TEVC.2024.3388550).

Permissive upstream licenses still require their notices if source is copied.
Apache-2.0 additionally requires marking modified files and retaining applicable
notices. GPL-derived implementation code is deliberately excluded from this
MIT codebase. File-level notices and bundled third-party terms must be checked
before any future source import; this project-level inventory is not a substitute
for that review.

## Extension rule

Add a new law only when all of the following land together:

1. A stable Rust enum value and Metal ABI value.
2. Pool generation and selected-row replay from the same seed mapping.
3. CPU/Metal coordinate parity across odd lengths and tile boundaries.
4. Declared moments, support, tensor scaling, and FP16 rounding behavior.
5. Artifact identity and per-tensor realized-change records.
6. Source and paper attribution with license review.
7. A complete-round comparison against the retained Gaussian baseline.

Controller policies such as mirrored sampling, antithetic selection, step-size
adaptation, and restarts are different axes. They must not be smuggled into a
law implementation because that would make optimization and kernel effects
impossible to attribute.
