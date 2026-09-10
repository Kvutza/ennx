# Colab CUDA

Use Google Colab as the NVIDIA test machine for CUDA-Oxide work.

```sh
uv tool install google-colab-cli --with jupyter-kernel-client==0.14.0 --force
colab new -s ennx-cuda --gpu T4
```

Prepare `/content/ennx` on the VM:

```sh
colab console -s ennx-cuda
cd /content
jj git clone --depth 1 --branch main https://github.com/Kvutza/ennx.git ennx
cd ennx
```

Run the CUDA checks from `/content/ennx`:

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

Stop the machine:

```sh
colab stop -s ennx-cuda
```

The pinned CUDA-Oxide revision is in `Cargo.toml`. The pinned nightly and LLVM
major are in `ops/cuoxtool.py`. The CUDA crate is opt-in and outside normal
local CPU, Metal, and OpenCL builds.
