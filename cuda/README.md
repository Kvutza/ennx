# ENNx CUDA-Oxide

This workspace contains ENNx CUDA kernels written in Rust with CUDA-Oxide. It
is pinned to the same compiler revision and nightly as
`ops/cuda_oxide_toolchain.py` and currently targets the T4's `sm_75` ISA.

The kernels keep packed int4 or int8 trial rows resident on the device and run
materialization, exact distance, ENN scoring, and winner selection. They
preserve ENNx's trial semantics and are tested against an independent CPU
implementation before any timing result is accepted.

On a prepared Linux CUDA host, run from this directory:

```bash
cd cuda
cargo oxide run --arch sm_75 -- parity
cargo oxide run --arch sm_75 -- resident
cargo oxide run --arch sm_75 -- bench 16777216 100 4
cargo oxide run --arch sm_75 -- \
  trial-bench 1024 32 8192 20
compute-sanitizer --tool memcheck --error-exitcode 99 \
  target/release/ennx-cuda resident
```

`trial-bench` arguments are candidates, resident history rows, packed parameter
elements, and iterations.

## T4 execution path

The `sm_75` trial scorer uses one 256-thread block per candidate. Candidate
values are decoded once into shared-memory tiles, while eight warps accumulate
distance against separate history rows. Warp reductions combine each row's
partial distance without global atomics. A second kernel launches one block per
trust region and performs deterministic argmax selection, so multi-region asks
use one scoring launch and one selection launch instead of serial host calls.

Compact trust-region center trees are resolved in the scoring kernel from a
parent-linked representation. The host validates topology and limits center
depth to eight before launch. Leaf bounds, packed-row dimensions, region
layouts, and buffer capacities are also checked on the host; the resulting
launch contracts permit unchecked indexing in the validated device hot path.

On a Modal Tesla T4, the verified 1,024-candidate, 32-history, 8,192-element
benchmark scores in 2.567 ms, compared with the original 6.687 ms baseline.
An eight-region batch with 1,024 candidates per region completes in a median
11.210 ms. Both measurements use FP32 distance and acquisition accumulation.
The focused gate also checks CPU parity for eight suites and runs CUDA Compute
Sanitizer over the trial path.

T4 does not provide TMA, thread-block clusters, WGMMA, or the newer reduction
instructions. Those belong in future `sm_80+` or `sm_90+` kernel families, not
in the `sm_75` baseline.

For Colab, run the wrappers from the repository root:

```bash
python ops/colab_cuda.py setup
python ops/colab_cuda.py doctor
python ops/colab_cuda.py ennx
python ops/colab_cuda.py resident
python ops/colab_cuda.py sanitize
python ops/colab_cuda.py bench
python ops/colab_cuda.py python
```

The Buck2 wheel target is the development and release build entry point on a
Linux x86-64 T4 runtime:

```bash
./buck2w --isolation-dir cuda build //:cuda-wheel \
  --target-platforms //:linux-x86_64-platform \
  --local-only --num-threads 4 --show-output
```

This action enters the CUDA workspace, which pins `nightly-2026-04-03`, then
compiles the CUDA-Oxide device crate for `sm_75`, embeds that artifact in the
Python extension, and packages the CPython 3.12 wheel. It requires
`cargo-oxide` and the pinned nightly to be installed in the execution
environment; the Colab setup cell provides both.

After JAX is available on the T4 runtime, build the wheel and verify direct
BF16 DLPack import, resident perturbation, and export with one target:

```bash
./buck2w --isolation-dir cuda build //:cuda-parity \
  --target-platforms //:linux-x86_64-platform \
  --local-only --num-threads 4 --show-output
```

For the hosted T4 release gate on Modal, build the pinned toolchain image and
then run the Buck2 CUDA wheel target, parity target, and batched MJX integration:

```bash
cargo run --manifest-path Cargo.toml -p ennx-modal -- image
cargo run --manifest-path Cargo.toml -p ennx-modal -- \
  wheel /tmp/ennx-cuda-wheel.whl --mjx
```

CUDA events are the authoritative kernel timer. Tracy zones cover the parity
and benchmark host paths. `TrialEngine::set_profiling(true)` records score,
argmax, materialization, and total CUDA event times as Tracy plots; setting
`ENNX_CUDA_PROFILE` enables the same plots outside the benchmark.

## Integration order

1. Materialize packed trial rows with exact CPU parity. Done.
2. Port trial distance, scoring, and selection while rows remain on the GPU. Done.
3. Add a resident CUDA compute backend and expose it through Python. Done.
4. Batch regions and compact center trees to amortize launch overhead. Done.

CUDA support is opt-in through the ENNx `cuda` Cargo feature so CUDA-Oxide's
pinned nightly compiler remains outside normal CPU, Metal, and OpenCL builds.
Do not run CUDA commands through Pixi: Pixi owns the Python environment, while
this workspace owns the Rust and CUDA-Oxide toolchain.
