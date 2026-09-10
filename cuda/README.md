# CUDA-Oxide

ENNX CUDA kernels written in Rust with CUDA-Oxide. The current target is
NVIDIA T4 (`sm_75`). CUDA is built separately from the CPU, Metal, and OpenCL code.

On a machine with the CUDA-Oxide toolchain installed:

```sh
cd cuda
cargo oxide run --arch sm_75 -- parity
cargo oxide run --arch sm_75 -- resident
cargo oxide run --arch sm_75 -- bench
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ../target/release/ennx-cuda resident
```

To install the toolchain and run the checks on [Colab](../docs/colab.md), run
these commands from the repository root on the VM:

```sh
python ops/colab_cuda.py setup
python ops/colab_cuda.py doctor
python ops/colab_cuda.py vecadd
python ops/colab_cuda.py ennx
python ops/colab_cuda.py resident
python ops/colab_cuda.py sanitize
python ops/colab_cuda.py bench
python ops/colab_cuda.py python
```

The CUDA-Oxide revision is pinned in `Cargo.toml`; Rust and LLVM versions are in
`ops/cuoxtool.py`.
