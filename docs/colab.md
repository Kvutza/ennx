# Colab development

Hosted Google Colab with an NVIDIA T4 is the compatibility target for ENNx's
CUDA work. The development loop must exercise the environment that notebook
users receive, including its Python ABI, CUDA toolkit, NVIDIA driver, and
ephemeral filesystem.

Open `examples/colab_cuda_oxide_dev.ipynb` in Colab, select a T4 GPU runtime,
and run the cells in order. The notebook fingerprints the assigned runtime,
clones ENNx, installs the pinned CUDA-Oxide toolchain, runs `cargo oxide doctor`,
and executes `vecadd`.

The notebook only orchestrates the environment. Toolchain setup lives in
`ops/colab_cuda_oxide_smoke.py`, and CUDA or ENNx implementation work belongs in
normal repository source files. This keeps experiments reviewable and prevents
the notebook from becoming a second implementation.

## Compatibility boundary

The hosted Colab runtime line uses CPython 3.12. ENNx supports CPython 3.12 and
3.13 with separate native wheels. Release artifacts keep their version-specific
ABI tags; a `cp313` extension must never be relabeled as `cp312`.

Install a released Linux wheel directly from GitHub:

```python
!pip install "https://github.com/Kvutza/ennx/releases/download/<tag>/ennx-<version>-cp312-cp312-manylinux_2_28_x86_64.whl"
```

The Colab gate is complete when a clean hosted runtime can:

1. Install the ENNx wheel without changing the runtime Python installation.
2. Import ENNx and select the CUDA backend.
3. Run deterministic CPU/CUDA parity tests for an ENNx kernel.
4. Execute a representative ENN fit and prediction workload.

The compiler revision, Rust nightly, and LLVM major version are pinned in
`ops/cuda_oxide_toolchain.py`. Update those pins deliberately and rerun both the
Modal and Colab smoke tests before accepting a toolchain change.

Colab runtimes and GPU assignments change over time. Record the notebook's
runtime fingerprint with every benchmark result instead of treating "Colab T4"
as a complete software specification.
