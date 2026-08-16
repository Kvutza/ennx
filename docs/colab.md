# Colab development

Hosted Google Colab with an NVIDIA T4 is the compatibility target for ENNx's
CUDA work. The development loop must exercise the environment that notebook
users receive, including its Python ABI, CUDA toolkit, NVIDIA driver, and
ephemeral filesystem.

Open `examples/colab_cuda_oxide_dev.ipynb` in Colab, select a T4 GPU runtime,
and run the cells in order. The notebook fingerprints the assigned runtime,
clones ENNx, installs the pinned CUDA-Oxide toolchain, runs `cargo oxide doctor`,
executes CUDA-Oxide's `vecadd`, and verifies ENNx's trial materialization kernel.
It then imports the Cargo-built extension in Python 3.12 and checks the public
resident-session API against the CPU backend.

For an applied walkthrough, open `examples/colab_jax_cuda_ennx.ipynb`. It uses
Colab's preinstalled CUDA-enabled JAX to evaluate a quantized CNN on the T4 and the public
`ennx.experimental.WeightSearch` API with `backend="cuda"` to optimize the CNN
from scalar task rewards. The tutorial installs the prebuilt CPython 3.12,
`sm_75` wheel, times proposal and evaluation separately, and identifies the
selected packed-row transfer as the remaining host boundary. It does not clone
the repository, replace Colab's JAX stack, or install Rust, LLVM, and CUDA
compiler tooling.

For the high-dimensional control experiment, open
[`examples/colab_mjx_humanoid_ennx.ipynb`](https://colab.research.google.com/github/Kvutza/ennx/blob/cuda/examples/colab_mjx_humanoid_ennx.ipynb).
It runs a roughly 972,000-parameter JAX policy in a pure MJX Humanoid simulation,
optimizes packed policy mutations with the ENNx CUDA backend, and renders the
incumbent policy to an MP4. The notebook installs the released ENNx wheel and
`mujoco-mjx`; it does not require a source checkout or Rust toolchain.
`ennx.experimental.TurboSearch` owns acceptance and TuRBO trust-region updates
in Rust while packed history, candidate scoring, and selected rows remain on
the CUDA backend. The Python API exports a synchronized batch of pending-row
device addresses; CuPy retains the Rust search as their allocation owner and
passes all rows to one vectorized JAX/MJX evaluation through DLPack without
staging policies in NumPy. Rewards return together through `tell_batch`.
The wheel is installed without dependency resolution so Colab's compatible
NumPy, SciPy, and CUDA-enabled JAX stack remains unchanged. The MJX dependency
install is also constrained to the numerical package versions supplied by the
fresh runtime.

The development notebook only orchestrates the environment. Toolchain setup lives in
`ops/colab_cuda_oxide_smoke.py`, and CUDA or ENNx implementation work belongs in
normal repository source files. This keeps experiments reviewable and prevents
the notebook from becoming a second implementation.

## Compatibility boundary

The hosted Colab runtime line uses CPython 3.12. ENNx supports CPython 3.12 and
3.13 with separate native wheels. Release artifacts keep their version-specific
ABI tags; a `cp313` extension must never be relabeled as `cp312`.

Install the CUDA-enabled Linux wheel directly from GitHub:

```python
!pip install "https://github.com/Kvutza/ennx/releases/download/cuda-v0.1.5/ennx-0.1.5%2Bcuda75-cp312-cp312-manylinux_2_28_x86_64.whl"
```

The Colab gate is complete when a clean hosted runtime can:

1. Install the ENNx wheel without changing the runtime Python installation.
2. Import ENNx and select the CUDA backend.
3. Run deterministic CPU/CUDA parity tests for an ENNx kernel.
4. Execute a representative ENN fit and prediction workload.

The first kernel development stage is available now. Run its parity, memory
check, and benchmark from the repository checkout:

```bash
python ops/colab_cuda_oxide_smoke.py ennx
python ops/colab_cuda_oxide_smoke.py resident
python ops/colab_cuda_oxide_smoke.py sanitize
python ops/colab_cuda_oxide_smoke.py bench
python ops/colab_cuda_oxide_smoke.py python
```

The `ennx` command compares 36 deterministic CPU and CUDA materialization cases.
`resident` additionally checks exact distance, ENN scoring, selection, and the
selected row against CPU. `sanitize` runs the resident path through Compute
Sanitizer memcheck. `bench` uses CUDA events for timing and reports effective
read-plus-write bandwidth. `python` compiles `ennx-py` for the T4 and runs two
ask/tell rounds through `ennx.experimental.ResidentBoSession` on both CPU and
CUDA. This source smoke intentionally uses the native-free Faiss bridge; release
wheels continue to bundle the real Faiss runtime.

The CUDA crate remains a separate nightly workspace and is enabled in ENNx only
for Linux x86_64 builds with the `cuda` feature. The existing experimental
Python resident session selects it with `backend="cuda"`.

The compiler revision, Rust nightly, and LLVM major version are pinned in
`ops/cuda_oxide_toolchain.py`. Update those pins deliberately and rerun both the
Modal release gate and Colab checks before accepting a toolchain change.

Build the pinned CUDA-Oxide Modal image from Rust using:

```bash
cargo run --manifest-path rust/Cargo.toml -p ennx-modal -- image
```

Run the release wheel gate with the real batched MJX integration check using:

```bash
cargo run --manifest-path rust/Cargo.toml -p ennx-modal -- \
  wheel /tmp/ennx-cuda-wheel.whl --mjx
```

Colab runtimes and GPU assignments change over time. Record the notebook's
runtime fingerprint with every benchmark result instead of treating "Colab T4"
as a complete software specification.
