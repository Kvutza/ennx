# Full-Weight Perturbation Lab

Updated: 2026-09-30.

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
| `gaussian` | Standard normal | 0 | 1 | SplitMix64 and a 256-strip Ziggurat |
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
record the selected law in `experiment.toml`, `run.log`, and `result.toml`.

Perturbation law and acquisition are independent controls. The selected
full-weight proposal pool is scored through the shared TuRBO-ENN `Ask` contract,
whose neighbor count, Thompson/UCB choice, variance scales, output scale and
seed are resolved from the same experiment TOML.

## Performance hypothesis

Gaussian generation uses a 256-strip Ziggurat with tables embedded in Metal
constant memory. The common path is bounded integer hashing, a table lookup,
one multiply, and one comparison. Rejection and tail paths retain `exp` or
`log`, but derive every retry from `(seed, tensor key, coordinate, attempt)` so
pool scoring and selected-row materialization replay exactly. Rademacher still
uses fewer instructions: one mixed word supplies two signs.

The first-round FP16 kernel consumes coordinate pairs directly. The preceding
scalar loop recomputed the same Box--Muller pair for adjacent coordinates and
discarded half of each result. Pool and materialization paths were already
pair-oriented.

This does not predict an end-to-end factor. The controller still reads the
incumbent and history, rounds four candidates to FP16, computes full realized
resident distances and pool geometry, reduces tile statistics, and writes the
selected 2.077 GB row. The scorer is unchanged. Only complete 4K-context BO
rounds can establish whether this changes end-to-end latency. Historical
component medians are not current budgets; see [dated evidence](handoff.md).

Early 4,096-token runs with independently randomized initialization took
734.862 ms for Rademacher and 526.627 ms for Ziggurat Gaussian. They are useful
favorable-point measurements, not an A/B comparison. Generation now derives
model, proposal, acquisition, and sampling seeds from the experiment configuration;
the derivation deliberately excludes perturbation law. This avoids checked-in
seed literals and gives treatment arms identical streams automatically.

The first controlled one-round comparison took 5,967.550 ms for Rademacher and
7,100.504 ms for Ziggurat Gaussian. Proposal times were 83.758 and 151.000 ms;
exact rollout times were 5,849.261 and 6,914.140 ms. Rademacher evaluated
99,200 positions and Gaussian 117,120. Both sampled all 1,047,732,224
coordinates; FP16 rounding changed every Rademacher coordinate and 547,552,769
Gaussian coordinates. These untrained, zero-reward runs demonstrate strong
seed-dependent fixed-point behavior, not search quality or a stable timing
distribution. Artifacts:
`.cache/ennx/runs/generation-4096/run-1790815988746-59739-0` and
`.cache/ennx/runs/generation-4096-gaussian/run-1790816007822-60088-0`.

Compare the laws with identical model, corpus, rounds, power state, and seeds.
Record complete wall time, scorer GPU time, ask time, tell time, changed-weight
fraction, realized/requested RMS ratio, acceptance, and trust radius. Treat
distribution-dependent rewards as research outcomes, not timing parity.

The checked-in matched comparison is:

```sh
./ennx tune examples/tuning/code-pretrain-perturbation-gaussian.toml
./ennx tune examples/tuning/code-pretrain-perturbation-rademacher.toml
```

Each arm runs two independently derived repetitions of 32 rounds. With ten
neighbors, every repetition contains ten initialization rounds followed by 22
ENN-guided selections. The two TOMLs differ only in perturbation distribution,
and the seed derivation deliberately excludes that treatment field.

## Workspace protocol

Keep the runnable experiment platform at a named JJ revision. Create clean
sibling workspaces for control and treatment from that exact revision; keep an
additional workspace at any historical source revision used for comparison.
Never run a control from an older harness than its treatment. Harness changes
land on the platform first, then both experiment workspaces advance to the same
platform revision before either arm changes.

For an A/B comparison, interleave whole control and treatment CLI runs on one
machine. Within one CLI run, repetitions execute sequentially. The repetition
scheduler derives matched proposal and acquisition streams without checked-in
seed literals and writes a separate artifact directory for every repetition.
Record the source revision and resolved experiment with the results. A historical
timing remains historical evidence unless it is rerun through the same harness,
objective, model, and system state as the current arm.

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
