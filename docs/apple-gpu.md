# Apple GPU execution

ENNX uses one Rust-owned Apple GPU runtime for every Metal device. Python makes
one `ask` or `tell` call; it does not schedule kernels, own tensors, or enter the
numerical hot path.

## Contract

- Stable ENN, TuRBO, UCB, Thompson, Pareto, MORBO, experimental multi-trust
  region, quantized trials, and weight selection retain their existing public
  mathematics and seeded determinism.
- `IndexDriver::Metal` forces source-compiled Metal KNN.
  `IndexDriver::Agx` forces a validated device-native binary archive.
  `IndexDriver::Auto` keeps exact CPU and AGX mirrors resident, calibrates the
  first real query in each logarithmic workload bucket, checks exact
  neighbor-index agreement, and caches the faster valid route. Other platforms
  resolve `Auto` to exact CPU.
- `ComputeDevice::Metal` forces the source-compiled quantized Metal engines;
  `ComputeDevice::Agx` forces native-archive trial and weight kernels.
  `ComputeDevice::Auto` uses the available accelerator; weight selection also
  calibrates CPU against Metal and accepts Metal only when the selected index
  and score agree.
- Multi-region optimization concatenates all region candidate blocks, performs
  one posterior call, then applies segmented acquisition. Region allocations
  remain unchanged; repeated host-to-device posterior launches are removed.
- CPU remains the control plane for trust-region transitions, observation
  identity, deterministic tie handling, posterior reduction, and error
  recovery. Bulk distance, top-k neighbor search, trial scoring, and weight
  scoring are the GPU data plane.

## Runtime

The process-wide runtime owns one `MTLDevice`, one command queue, a source
pipeline cache, and a native archive cache keyed by device, source content, and
kernel name. Metal validates archive compatibility while creating the pipeline;
ENNX does not assume that every M-series generation serializes the private
archive container identically. Metal indices retain observation and scratch
buffers, grow geometrically, and update appended rows in place. Trials and
weights share the same runtime and caches.

The runtime recognizes the known Apple GPU generations behind M1 through M4 for
telemetry. Unknown future M-series devices still use portable MSL compilation;
generation recognition is not a feature gate.

## Native research boundary

On M4, the observed native compilation chain is:

```text
MSL -> AIR -> MTLBinaryArchive -> applegpu_g16g Mach-O
    -> nested __compute Mach-O -> AGX3 __text
```

The native corpus established target `applegpu_g16g`, AGX3 code, SIMD width 32,
and a 1024-thread maximum. ENNX keeps the runtime ABI at public Metal while
executing cached `applegpu` slices through `MTLBinaryArchive`. This preserves
portability across M-series chips and provides distinct CPU, source-Metal, and
native-AGX routes without placing a private driver ABI in the library.

## Generated instruction schedules

The native KNN path generates scalar, two-accumulator, and four-accumulator
distance kernels from one MSL source. The independent FMA chains expose
instruction-level parallelism without changing the distance definition. On the
first search for a vector dimension, ENNX checks every candidate against the
scalar result, measures three synchronized executions, and retains the median
winner in a process-wide dimension cache. Invalid schedules fail closed to the
scalar kernel.

This is generated AGX3 scheduling rather than private ISA injection: Apple's
compiler lowers each schedule into the cached native `applegpu` archive. Mesa's
open assembler currently describes AGX2, so ENNX does not use it to encode M4
AGX3 instructions.

## Measured M4 frontier

Exact KNN, 32 dimensions and 16 neighbors:

| Rows | Queries | CPU | Metal | Result |
|---:|---:|---:|---:|---|
| 2,048 | 256 | 2.20 ms | 4.65 ms | CPU |
| 8,192 | 1,024 | 48.72 ms | 15.85 ms | Metal, 3.07x |
| 32,768 | 2,048 | 343.16 ms | 181.08 ms | Metal, 1.90x |

Pipeline setup fell from 93.9 ms cold to 0.133 ms warm in the first measured
shape. These crossings are evidence for device-local calibration, not permanent
thresholds: `Auto` learns from the current chip and workload.

With generated schedules enabled, a 25-round steady-state M4 run at 8,192 rows
and 1,024 queries measured 44.89 ms CPU, 22.70 ms source-Metal, and 17.64 ms
AGX. The native route was 2.55x faster than CPU and 22.3% faster than
source-Metal. Setup and the one-time schedule calibration were excluded from
the search median; `Auto` still measures locally because thermal state and
query shape move the crossing.

A 9-round dimension sweep at 8,192 rows and 512 queries showed the expected
crossing:

| Dimensions | CPU | Metal | AGX |
|---:|---:|---:|---:|
| 8 | 4.82 ms | 6.58 ms | 7.08 ms |
| 16 | 7.73 ms | 7.03 ms | 6.52 ms |
| 32 | 13.50 ms | 9.95 ms | 7.09 ms |
| 64 | 37.68 ms | 11.72 ms | 7.85 ms |
| 128 | 82.06 ms | 15.47 ms | 10.79 ms |

Run the pure Rust frontier experiment with:

```sh
pixi run -e ennx -- cargo run --manifest-path Cargo.toml \
  -p ennx --example apple_gpu_frontier --features metal --release
```

## Acceptance

Apple changes must pass exact CPU/Metal parity, deterministic strategy tests,
Metal Buck2 tests, Rust formatting and linting, KISS structure checks, and the
Buck2 build. Performance claims require warm measurements after pipeline and
buffer initialization.
