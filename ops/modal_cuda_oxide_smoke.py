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
ENNX_DIR = Path("/opt/ennx")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.12",
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
        "rust-src rustc-dev llvm-tools rust-analyzer clippy rustfmt",
        f"git init {CUDA_OXIDE_DIR} && "
        f"git -C {CUDA_OXIDE_DIR} remote add origin "
        "https://github.com/NVlabs/cuda-oxide.git && "
        f"git -C {CUDA_OXIDE_DIR} fetch --depth 1 origin {CUDA_OXIDE_REV} && "
        f"git -C {CUDA_OXIDE_DIR} checkout --detach FETCH_HEAD && "
        f"/root/.cargo/bin/cargo +{RUST_TOOLCHAIN} install "
        f"--path {CUDA_OXIDE_DIR}/crates/cargo-oxide --locked && "
        f"cd {CUDA_OXIDE_DIR} && "
        f"RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN} /root/.cargo/bin/cargo oxide setup",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": f"/usr/bin/llc-{LLVM_MAJOR}",
            "CUDA_OXIDE_BACKEND": str(
                CUDA_OXIDE_DIR
                / "crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so"
            ),
            "ENNX_FAISS_UNAVAILABLE": "1",
            "LIBCLANG_PATH": f"/usr/lib/llvm-{LLVM_MAJOR}/lib",
            "LLVM_CONFIG_PATH": f"/usr/bin/llvm-config-{LLVM_MAJOR}",
            "PATH": f"/root/.cargo/bin:/usr/lib/llvm-{LLVM_MAJOR}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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
    "ennx-cuda-oxide-smoke",
    image=image,
    tags={"project": "ennx", "experiment": "cuda-oxide-smoke"},
)

GPU_OPTIONS = {"gpu": "T4", "cpu": 8.0, "memory": 16_384, "timeout": 3_600}


def _run(
    command: list[str], timeout: int, *, cwd: Path = CUDA_OXIDE_DIR
) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
    }


def _result(commands: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cuda_oxide_rev": CUDA_OXIDE_REV,
        "rust_toolchain": RUST_TOOLCHAIN,
        "ok": all(result["returncode"] == 0 for result in commands),
        "commands": commands,
    }


def _gpu_identity() -> dict[str, object]:
    return _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        timeout=30,
    )


def _python_build_commands() -> list[dict[str, object]]:
    return [
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
            timeout=3_000,
            cwd=ENNX_DIR / "rust",
        ),
        _run(
            [
                "python",
                str(ENNX_DIR / "ops/cuda_python_smoke.py"),
                str(ENNX_DIR / "rust/target/release/libennx_rust.so"),
            ],
            timeout=300,
            cwd=ENNX_DIR,
        ),
    ]


def _cuda_dense_commands() -> list[dict[str, object]]:
    return [
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
                "ennx",
                "--example",
                "cuda_dense",
                "--features",
                "cuda",
                "--release",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "rust",
        ),
        _run(
            [str(ENNX_DIR / "rust/target/release/examples/cuda_dense")],
            timeout=300,
            cwd=ENNX_DIR / "rust",
        ),
    ]


def _cuda_trial_commands(*, sanitize: bool) -> list[dict[str, object]]:
    commands = [
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
                "ennx",
                "--example",
                "cuda_trial",
                "--example",
                "trial_bench",
                "--features",
                "cuda",
                "--release",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "rust",
        ),
        _run(
            [str(ENNX_DIR / "rust/target/release/examples/cuda_trial")],
            timeout=300,
            cwd=ENNX_DIR / "rust",
        ),
        _run(
            [
                str(ENNX_DIR / "rust/target/release/examples/trial_bench"),
                "8192",
                "32",
                "1024",
                "20",
                "cuda",
                "8",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "rust",
        ),
    ]
    if sanitize:
        commands.append(
            _run(
                [
                    "compute-sanitizer",
                    "--tool",
                    "memcheck",
                    "--error-exitcode",
                    "99",
                    str(ENNX_DIR / "rust/target/release/examples/cuda_trial"),
                ],
                timeout=3_000,
                cwd=ENNX_DIR / "rust",
            )
        )
    commands.append(
        _run(
            [
                "cargo",
                "oxide",
                "run",
                "--arch",
                "sm_75",
                "--",
                "trial-bench",
                "1024",
                "32",
                "8192",
                "20",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "cuda",
        )
    )
    return commands


@app.function(**GPU_OPTIONS)
def python_smoke() -> dict[str, object]:
    commands = [_gpu_identity(), _run(["cargo", "oxide", "doctor"], timeout=300)]
    commands.extend(_python_build_commands())
    return _result(commands)


@app.function(**GPU_OPTIONS)
def dense_smoke() -> dict[str, object]:
    commands = [_gpu_identity(), _run(["cargo", "oxide", "doctor"], timeout=300)]
    commands.extend(_cuda_dense_commands())
    return _result(commands)


@app.function(**GPU_OPTIONS)
def trial_smoke() -> dict[str, object]:
    commands = [_gpu_identity(), _run(["cargo", "oxide", "doctor"], timeout=300)]
    commands.extend(_cuda_trial_commands(sanitize=True))
    return _result(commands)


@app.function(**GPU_OPTIONS)
def smoke() -> dict[str, object]:
    commands = [
        _gpu_identity(),
        _run(["cargo", "oxide", "doctor"], timeout=300),
        _run(["cargo", "oxide", "run", "vecadd"], timeout=3_000),
        _run(
            ["cargo", "oxide", "run", "--arch", "sm_75", "--", "parity"],
            timeout=3_000,
            cwd=ENNX_DIR / "cuda",
        ),
        _run(
            ["cargo", "oxide", "run", "--arch", "sm_75", "--", "resident"],
            timeout=3_000,
            cwd=ENNX_DIR / "cuda",
        ),
        _run(
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--error-exitcode",
                "99",
                "target/release/ennx-cuda",
                "resident",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "cuda",
        ),
        _run(
            [
                "cargo",
                "oxide",
                "run",
                "--arch",
                "sm_75",
                "--",
                "bench",
                str(16 * 1024 * 1024),
                "100",
                "4",
            ],
            timeout=3_000,
            cwd=ENNX_DIR / "cuda",
        ),
    ]
    commands.extend(_cuda_trial_commands(sanitize=False))
    commands.extend(_cuda_dense_commands())
    commands.extend(_python_build_commands())
    return _result(commands)


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke.remote(), indent=2))
