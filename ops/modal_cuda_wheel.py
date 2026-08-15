"""Build and test the ENNx CPython 3.12 CUDA wheel on a Modal T4."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal
from cuda_oxide_toolchain import CUDA_OXIDE_REV, LLVM_MAJOR, RUST_TOOLCHAIN

CUDA_DIR = Path("/opt/cuda-oxide")
ENNX_DIR = Path("/opt/ennx")
LLVM_DIR = Path("/opt/llvm/.pixi/envs/default")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-rockylinux8",
        add_python="3.12",
    )
    .entrypoint([])
    .run_commands(
        "dnf install -y ca-certificates curl gcc gcc-c++ git libffi-devel make patch "
        "pkgconf-pkg-config xz && dnf clean all",
        "curl -fsSL https://pixi.sh/install.sh | bash && "
        "/root/.pixi/bin/pixi init /opt/llvm --channel conda-forge && "
        "/root/.pixi/bin/pixi add --manifest-path /opt/llvm/pixi.toml "
        f"'llvmdev={LLVM_MAJOR}.*' 'clang={LLVM_MAJOR}.*' "
        f"'libclang={LLVM_MAJOR}.*' 'lld={LLVM_MAJOR}.*'",
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --profile minimal --default-toolchain {RUST_TOOLCHAIN} && "
        f"/root/.cargo/bin/rustup component add --toolchain {RUST_TOOLCHAIN} "
        "rust-src rustc-dev llvm-tools rust-analyzer clippy rustfmt",
        f"git init {CUDA_DIR} && "
        f"git -C {CUDA_DIR} remote add origin https://github.com/NVlabs/cuda-oxide.git && "
        f"git -C {CUDA_DIR} fetch --depth 1 origin {CUDA_OXIDE_REV} && "
        f"git -C {CUDA_DIR} checkout --detach FETCH_HEAD && "
        f"PATH={LLVM_DIR}/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH "
        f"LLVM_CONFIG_PATH={LLVM_DIR}/bin/llvm-config "
        f"LIBCLANG_PATH={LLVM_DIR}/lib "
        f"/root/.cargo/bin/cargo +{RUST_TOOLCHAIN} install "
        f"--path {CUDA_DIR}/crates/cargo-oxide --locked && "
        f"cd {CUDA_DIR} && "
        f"PATH={LLVM_DIR}/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH "
        f"LLVM_CONFIG_PATH={LLVM_DIR}/bin/llvm-config "
        f"LIBCLANG_PATH={LLVM_DIR}/lib "
        f"CUDA_OXIDE_LLC={LLVM_DIR}/bin/llc "
        f"RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN} /root/.cargo/bin/cargo oxide setup",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": str(LLVM_DIR / "bin/llc"),
            "CUDA_OXIDE_BACKEND": str(
                CUDA_DIR
                / "crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so"
            ),
            "ENNX_FAISS_UNAVAILABLE": "1",
            "LIBCLANG_PATH": str(LLVM_DIR / "lib"),
            "LLVM_CONFIG_PATH": str(LLVM_DIR / "bin/llvm-config"),
            "PATH": f"{LLVM_DIR}/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": str(ENNX_DIR / "ops"),
            "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
        }
    )
    .add_local_dir(
        ".",
        remote_path=str(ENNX_DIR),
        copy=True,
        ignore=[
            ".git/**",
            ".jj/**",
            ".pixi/**",
            "buck-out/**",
            "cuda/target/**",
            "results/**",
            "rust/target/**",
            "target/**",
        ],
    )
)

app = modal.App(
    "ennx-cuda-wheel",
    image=image,
    tags={"project": "ennx", "artifact": "cuda-wheel"},
)


def _run(command: list[str], *, cwd: Path = ENNX_DIR, timeout: int = 3_600) -> float:
    print("+", " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=cwd, env=os.environ.copy(), timeout=timeout, check=True)
    return time.monotonic() - started


@app.function(gpu="T4", cpu=8.0, memory=16_384, timeout=3_600)
def build() -> tuple[str, bytes]:
    _run(
        [
            "cargo",
            "oxide",
            "build",
            "--arch",
            "sm_75",
            "--device-codegen-crate",
            "ennx_cuda_kernels",
            "--",
            "-p",
            "ennx-py",
            "--features",
            "cuda",
            "--release",
        ],
        cwd=ENNX_DIR / "rust",
    )
    extension = ENNX_DIR / "rust/target/release/libennx_rust.so"
    _run(
        ["python", str(ENNX_DIR / "ops/cuda_python_smoke.py"), str(extension)]
    )
    wheel_dir = Path("/tmp/ennx-wheel")
    _run(
        [
            "python",
            str(ENNX_DIR / "ops/cuda_wheel.py"),
            str(extension),
            str(wheel_dir),
        ]
    )
    wheel = next(wheel_dir.glob("*.whl"))
    env_dir = Path("/tmp/ennx-wheel-env")
    _run(["python", "-m", "venv", str(env_dir)])
    python = env_dir / "bin/python"
    _run([str(python), "-m", "pip", "install", "--quiet", str(wheel)])
    _run(
        [
            str(python),
            "-c",
            (
                "import numpy as np; "
                "from ennx.experimental import WeightSearch; "
                "base=np.zeros(8,dtype=np.uint8); "
                "search=WeightSearch(base,0.0,[(0,16,4,0.25,1.0,0.25)],2,backend='cuda'); "
                "search.ask(np.array([11,13],dtype=np.uint64),0.5,1); "
                "assert search.row().shape==(8,); "
                "search.tell(1.0,True); "
                "print('CUDA_WHEEL ok=true')"
            ),
        ]
    )
    return wheel.name, wheel.read_bytes()


@app.local_entrypoint()
def main(output: str = "/tmp/ennx-cuda-wheel") -> None:
    name, data = build.remote()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_bytes(data)
    print(target)
