# Testing policy

`main` is the source of truth. Experimental bookmarks are temporary, and work
is not considered done on `main` until it has a clear test story.

The testing strategy is Antithesis-inspired: test examples, but also test the
laws of the system under generated pressure. Every generated or adversarial test
must be reproducible from a seed, workload shape, and backend selection.

## Test tiers

| Tier | Purpose | Typical command |
| --- | --- | --- |
| Working copy | Repair and check the current JJ diff with KISS coverage. | `./ennx dev`; use `./ennx dev --full` for the current-platform full pass |
| Rust core | Targeted Buck2 checks for the dev CLI or ENNX core crate. | `./buck2w test //rust/crates/dev-cli:ennx-test`; `./buck2w test //rust/crates/ennx:ennx-unit` |
| Wheel/API | Tests that require the built wheel or native extension. | `./ennx wheel`; the GitHub Actions matrix covers every supported CPython version |
| Python optional | Tests requiring optional packages such as Torch/Gpytorch/click or source-tree extension setup. | Install the optional dependencies and built extension, then run the selected pytest group. |
| Project CI | Full source verification plus Buck2 build/test on the current platform. | `./ennx ci` |
| Wheel build | Build and verify the current platform wheel. | `./ennx wheel` |
| Hardware | Metal/OpenCL behavior and CPU parity. Run when touching accelerator, KNN, dense, BF16, or native build code. | platform-specific Rust, Bazel, or Buck2 tests |
| Benchmark | Speed and numerical-regression checks. Run before performance claims. | benchmark scripts plus platform profiler captures |

Use the smallest tier that can catch the bug while iterating. Before merging,
run every tier affected by the files changed.

Prefer `./ennx` for normal testing. It is the repo-owned entrypoint and wraps
the Buck2 workflow. Use raw `cargo test`, `pytest`, or `buck2w`
only when diagnosing one specific package, test file, or feature set.

Do not use `cargo test --manifest-path Cargo.toml --workspace` as the
canonical Rust gate. `ennx-py` is a PyO3 `cdylib`; generic Cargo test linking
can fail on Python C symbols even when the wheel/API path is healthy. Test
`ennx-py` through the Python wheel/API tier unless the PyO3 embedding link mode
is being changed directly.

Bazel stays available for the consumer/audit path, but the repo’s primary
development and CI flow is `./ennx` backed by Buck2.

Run hardware-sensitive gates serially when checking timing, profiling, or Metal
counter behavior. Correctness tests must not depend on a timing counter being
available; timing is evidence for performance, not correctness.

## Generated workloads

Generated tests should follow these rules:

- Use fixed corpus seeds by default.
- Print or encode the seed, dimensions, backend, tolerance, and failure context.
- Include adversarial shapes, not only random shapes: empty rows when valid,
  one row, one dimension, powers of two, one less/greater than vector lanes,
  duplicate vectors, tied distances, extreme but finite values, and precision
  boundary cases.
- Prefer independent oracles. For example, compare a SIMD/vectorized path to a
  simple scalar reference instead of comparing two implementations with the same
  loop structure.
- Keep hardware tests explicit. If a property depends on Metal or OpenCL, it
  must skip cleanly when the backend is unavailable and report the backend that
  was tested.
- Gate feature-specific Rust tests and examples with the same `cfg` or
  `required-features` boundary as the code under test. A no-default build should
  not panic because a hardware feature is intentionally absent.

The goal is not random noise. The goal is deterministic exploration: many cases,
clear invariants, and exact replay.

## Core invariants

KNN and distance:

- Squared distances are finite and non-negative for finite inputs.
- Self-distance is zero.
- Distances are symmetric.
- Top-k results are sorted by distance.
- Increasing `k` preserves the prefix when there are no ties, and never returns
  rows outside the index.
- CPU and accelerator backends agree within the documented tolerance.

Dense, BF16, and accelerator kernels:

- Output shapes match the requested batch, dimension, and layer layout.
- Finite inputs produce finite outputs unless the API explicitly permits
  overflow.
- CPU and accelerator outputs agree within a tolerance documented by dtype and
  kernel.
- BF16 error stays within a bound chosen from the CPU f32/f64 reference.

Optimizer and trials:

- The same seed and config produce the same choices.
- Accepted updates preserve model/config invariants.
- Invalid configs fail with structured errors rather than accidental panics.
- Replay fixtures keep routing, shape, and incumbent behavior stable.

Config and serialization:

- Parse, serialize, and parse again preserves meaning.
- Unknown or invalid fields are rejected consistently.
- Cargo, Bazel, Buck2, and Python-facing defaults do not drift apart silently.

## Current audit notes

The repo already has broad coverage across Python parity tests, Rust unit and
integration tests, Bazel consumer checks, Buck2 gates, and Zig ABI checks. The
next hardening step is to convert hand-rolled random tests into deterministic
generated workloads with replay seeds, starting from pure Rust kernels and then
moving outward to KNN backend parity, optimizer replay, and hardware paths.
