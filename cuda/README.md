# CUDA-Oxide

Rust CUDA kernels for ENNX. This workspace targets T4 `sm_75` through
CUDA-Oxide and stays outside the normal local Metal/OpenCL build.

From a prepared CUDA host:

```sh
cd cuda
cargo oxide run --arch sm_75 -- parity
cargo oxide run --arch sm_75 -- resident
cargo oxide run --arch sm_75 -- bench
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ../target/release/ennx-cuda resident
```

From the repository root on Colab:

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

Checked on Colab T4 with CUDA 12.8 and Python 3.13.15:

- `doctor`, `vecadd`, `ennx`, `resident`, `sanitize`, `bench`, `python` passed.
- `sanitize` reported zero Compute Sanitizer errors.
- `bench` reported 16,777,216 elements, 4-bit rows, 0.286112 ms, 54.611 GiB/s.
