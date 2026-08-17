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
optimizes dense BF16 whole-policy perturbations with the ENNx CUDA backend, and renders the
incumbent policy to an MP4. The notebook installs the released ENNx wheel and
`mujoco-mjx`; it does not require a source checkout or Rust toolchain.
Its TOML parameter cell defaults to ten sequential repetitions and 32 BO rounds.
MJX reset keys vary by repetition and round but remain reproducible. The notebook
records Yubo-style proposal, evaluation, tell, total-round, environment-step, and
incumbent fields to `trace.jsonl`, then writes `y_best` mean plus or minus SEM,
timing curves, a CUDA proposal breakdown, and the runtime fingerprint under
`/content/ennx_mjx_runs`.
For a first T4 validation pass, set `ENNX_REPETITIONS=1` and `ENNX_ROUNDS=2`
before running the experiment cell; the TOML defaults remain the benchmark
settings.
`ennx.experimental.Bf16Search` keeps acceptance, bounded history, TuRBO
trust-region updates, hierarchical distance scoring, acquisition, selection,
and selected rows CUDA-resident. The Python API leases a
synchronized batch of pending rows directly to JAX through DLPack without
CuPy or NumPy policy staging. No candidate-by-history distance matrix is
materialized; tile blocks emit small FP32 partials that a second kernel reduces
before acquisition. Contiguous JAX FP32 rewards and estimated variances return
through `tell_batch` via DLPack; only the accepted flags and scalar public state
cross back for experiment control and logging.
The wheel is installed without dependency resolution so Colab's compatible
NumPy, SciPy, and CUDA-enabled JAX stack remains unchanged. The MJX dependency
install is also constrained to the numerical package versions supplied by the
fresh runtime.

The development notebook only orchestrates the environment. Toolchain setup lives in
`ops/colab_cuda.py`, and CUDA or ENNx implementation work belongs in
normal repository source files. This keeps experiments reviewable and prevents
the notebook from becoming a second implementation.

## Compatibility boundary

The hosted Colab runtime line uses CPython 3.12. ENNx publishes separate
CPython 3.12 through 3.14 native wheels as GitHub Release assets, rather than
through PyPI. Release artifacts keep their version-specific ABI tags; a
`cp314` extension must never be relabeled as `cp312`.

Install the CUDA-enabled Linux wheel directly from GitHub:

```python
!pip install "https://github.com/Kvutza/ennx/releases/download/vX.Y.Z/ennx-X.Y.Z%2Bcuda75-cp312-cp312-manylinux_2_28_x86_64.whl"
```

The Colab gate is complete when a clean hosted runtime can:

1. Install the ENNx wheel without changing the runtime Python installation.
2. Import ENNx and select the CUDA backend.
3. Run deterministic CPU/CUDA parity tests for an ENNx kernel.
4. Execute a representative ENN fit and prediction workload.

The first kernel development stage is available now. Run its parity, memory
check, and benchmark from the repository checkout:

```bash
python ops/colab_cuda.py ennx
python ops/colab_cuda.py resident
python ops/colab_cuda.py sanitize
python ops/colab_cuda.py bench
python ops/colab_cuda.py python
```

The `ennx` command compares 36 deterministic CPU and CUDA materialization cases.
`resident` additionally checks exact distance, ENN scoring, selection, and the
selected row against CPU. `sanitize` runs the resident path through Compute
Sanitizer memcheck. `bench` uses CUDA events for timing and reports effective
read-plus-write bandwidth. `python` compiles `ennx-py` for the T4 and runs two
ask/tell rounds through `ennx.experimental.ResidentBoSession` on both CPU and
CUDA. This source check intentionally uses the native-free Faiss bridge; release
wheels continue to bundle the real Faiss runtime.

The CUDA crate remains a separate nightly workspace and is enabled in ENNx only
for Linux x86_64 builds with the `cuda` feature. The existing experimental
Python resident session selects it with `backend="cuda"`.

The BF16 T4 release gate also measures the full resident proposal path. With
one million weights, eight candidates, and eight history rows, the current
hierarchical CUDA-Oxide implementation measures 2.074 ms median: 1.834 ms for
distance scoring and acquisition, 0.007 ms for selection, and 0.184 ms to write
the selected full row. Treat these values as the checked T4 baseline, not as a
hardware-independent performance claim.

The CUDA-Oxide revision is pinned in `cuda/Cargo.toml`, the Rust nightly in
`cuda/rust-toolchain.toml`, and the Colab LLVM major version in
`ops/cuda_oxide_toolchain.py`. Update those pins deliberately and rerun both
the Modal release gate and Colab checks before accepting a toolchain change.

Build the pinned CUDA-Oxide Modal image from Rust using:

```bash
cargo run --manifest-path rust/Cargo.toml -p ennx-modal -- image
```

Run the Buck2 release wheel and parity targets with the real batched MJX
integration check using:

```bash
cargo run --manifest-path rust/Cargo.toml -p ennx-modal -- \
  wheel /tmp/ennx-cuda-wheel.whl --mjx
```

After the T4 gate succeeds, attach its wheel to the matching GitHub Release:

```bash
./ennx release upload vX.Y.Z /tmp/ennx-cuda-wheel.whl
```

Colab runtimes and GPU assignments change over time. Record the notebook's
runtime fingerprint with every benchmark result instead of treating "Colab T4"
as a complete software specification.
