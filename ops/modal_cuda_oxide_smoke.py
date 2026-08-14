"""Run CUDA-Oxide's smallest Rust kernel on a bounded Modal T4 job."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal
from cuda_oxide_toolchain import CUDA_OXIDE_REV, LLVM_MAJOR, RUST_TOOLCHAIN

CUDA_OXIDE_DIR = Path("/opt/cuda-oxide")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .run_commands(
        "export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update && "
        "apt-get install -y --no-install-recommends "
        "ca-certificates curl g++ gcc git gnupg libc6-dev make pkg-config "
        "python3 xz-utils && "
        "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key "
        "| gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] "
        "https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' "
        "> /etc/apt/sources.list.d/llvm-toolchain-noble-21.list && "
        "apt-get update && "
        "apt-get install -y --no-install-recommends "
        f"clang-{LLVM_MAJOR} libclang-common-{LLVM_MAJOR}-dev "
        f"lld-{LLVM_MAJOR} llvm-{LLVM_MAJOR} llvm-{LLVM_MAJOR}-dev && "
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
        f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --profile minimal --default-toolchain {RUST_TOOLCHAIN} && "
        f"/root/.cargo/bin/rustup component add --toolchain {RUST_TOOLCHAIN} "
        "rust-src rustc-dev llvm-tools",
        f"git init {CUDA_OXIDE_DIR} && "
        f"git -C {CUDA_OXIDE_DIR} remote add origin "
        "https://github.com/NVlabs/cuda-oxide.git && "
        f"git -C {CUDA_OXIDE_DIR} fetch --depth 1 origin {CUDA_OXIDE_REV} && "
        f"git -C {CUDA_OXIDE_DIR} checkout --detach FETCH_HEAD && "
        f"/root/.cargo/bin/cargo +{RUST_TOOLCHAIN} install "
        f"--path {CUDA_OXIDE_DIR}/crates/cargo-oxide --locked",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": f"/usr/bin/llc-{LLVM_MAJOR}",
            "LIBCLANG_PATH": f"/usr/lib/llvm-{LLVM_MAJOR}/lib",
            "LLVM_CONFIG_PATH": f"/usr/bin/llvm-config-{LLVM_MAJOR}",
            "PATH": f"/root/.cargo/bin:/usr/lib/llvm-{LLVM_MAJOR}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
)

app = modal.App(
    "ennx-cuda-oxide-smoke",
    image=image,
    tags={"project": "ennx", "experiment": "cuda-oxide-smoke"},
)


def _run(command: list[str], timeout: int) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=CUDA_OXIDE_DIR,
        env=os.environ.copy(),
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
    }


@app.function(gpu="T4", cpu=8.0, memory=16_384, timeout=3_600)
def smoke() -> dict[str, object]:
    commands = [
        _run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=30,
        ),
        _run(["cargo", "oxide", "doctor"], timeout=300),
        _run(["cargo", "oxide", "run", "vecadd"], timeout=3_000),
    ]
    return {
        "cuda_oxide_rev": CUDA_OXIDE_REV,
        "rust_toolchain": RUST_TOOLCHAIN,
        "ok": all(result["returncode"] == 0 for result in commands),
        "commands": commands,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke.remote(), indent=2))
