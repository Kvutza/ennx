"""Set up and exercise the pinned CUDA-Oxide toolchain on hosted Colab."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cuda_oxide_toolchain import CUDA_OXIDE_REV, LLVM_MAJOR, RUST_TOOLCHAIN

DEFAULT_WORKSPACE = Path("/content/cuda-oxide")
DEFAULT_PROJECT = Path(__file__).resolve().parents[1] / "cuda"
LLVM_KEY_URL = "https://apt.llvm.org/llvm-snapshot.gpg.key"
TOOLCHAIN_STATE = Path.home() / ".cache/ennx/cuda-oxide-toolchain.json"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> float:
    print(f"+ {shlex.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return time.monotonic() - started


def _capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _fingerprint() -> dict[str, object]:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("CUDA-Oxide Colab development requires Linux x86_64")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "No NVIDIA runtime found; select a GPU under Runtime > Change runtime type"
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "cuda": _capture(["nvcc", "--version"]),
    }


def _install_system_packages() -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Colab setup expects its default root notebook runtime")

    llvm_tools = [
        Path(f"/usr/bin/clang-{LLVM_MAJOR}"),
        Path(f"/usr/bin/llc-{LLVM_MAJOR}"),
        Path(f"/usr/bin/llvm-config-{LLVM_MAJOR}"),
    ]
    if all(tool.exists() for tool in llvm_tools):
        print(f"LLVM {LLVM_MAJOR} is already installed", flush=True)
        return

    base_packages = [
        "ca-certificates",
        "curl",
        "g++",
        "gcc",
        "git",
        "gnupg",
        "libc6-dev",
        "make",
        "pkg-config",
    ]
    _run(["apt-get", "update"])
    _run(["apt-get", "install", "-y", "--no-install-recommends", *base_packages])

    os_release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    codename = os_release.get("VERSION_CODENAME")
    if not codename:
        raise RuntimeError("Could not determine the Ubuntu release codename")

    key_download = Path("/tmp/llvm-snapshot.gpg.key")
    keyring = Path("/usr/share/keyrings/apt.llvm.org.gpg")
    source = Path(f"/etc/apt/sources.list.d/llvm-toolchain-{LLVM_MAJOR}.list")
    _run(["curl", "-fsSL", LLVM_KEY_URL, "-o", str(key_download)])
    _run(
        [
            "gpg",
            "--dearmor",
            "--yes",
            "--output",
            str(keyring),
            str(key_download),
        ]
    )
    source.write_text(
        f"deb [signed-by={keyring}] https://apt.llvm.org/{codename}/ "
        f"llvm-toolchain-{codename}-{LLVM_MAJOR} main\n"
    )
    _run(["apt-get", "update"])
    _run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            f"clang-{LLVM_MAJOR}",
            f"libclang-common-{LLVM_MAJOR}-dev",
            f"lld-{LLVM_MAJOR}",
            f"llvm-{LLVM_MAJOR}",
            f"llvm-{LLVM_MAJOR}-dev",
        ]
    )


def _install_rust() -> tuple[Path, Path]:
    cargo = Path.home() / ".cargo/bin/cargo"
    rustup = Path.home() / ".cargo/bin/rustup"
    if not rustup.exists():
        _run(
            [
                "bash",
                "-c",
                (
                    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                    "| sh -s -- -y --profile minimal --default-toolchain none"
                ),
            ]
        )
    _run(
        [
            str(rustup),
            "toolchain",
            "install",
            RUST_TOOLCHAIN,
            "--profile",
            "minimal",
            "--component",
            "rust-src",
            "--component",
            "rustc-dev",
            "--component",
            "llvm-tools",
            "--component",
            "rust-analyzer",
            "--component",
            "clippy",
            "--component",
            "rustfmt",
        ]
    )
    return cargo, rustup


def _checkout(workspace: Path) -> None:
    if workspace.exists() and not (workspace / ".git").is_dir():
        raise RuntimeError(f"Refusing to replace non-git directory: {workspace}")
    if not workspace.exists():
        workspace.mkdir(parents=True)
        _run(["git", "init", str(workspace)])
        _run(
            [
                "git",
                "-C",
                str(workspace),
                "remote",
                "add",
                "origin",
                "https://github.com/NVlabs/cuda-oxide.git",
            ]
        )
    _run(
        [
            "git",
            "-C",
            str(workspace),
            "fetch",
            "--depth",
            "1",
            "origin",
            CUDA_OXIDE_REV,
        ]
    )
    _run(["git", "-C", str(workspace), "checkout", "--detach", "FETCH_HEAD"])


def _toolchain_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        str(Path.home() / ".cargo/bin"),
        f"/usr/lib/llvm-{LLVM_MAJOR}/bin",
        "/usr/local/cuda/bin",
        env.get("PATH", ""),
    ]
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": f"/usr/bin/llc-{LLVM_MAJOR}",
            "LIBCLANG_PATH": f"/usr/lib/llvm-{LLVM_MAJOR}/lib",
            "LLVM_CONFIG_PATH": f"/usr/bin/llvm-config-{LLVM_MAJOR}",
            "PATH": os.pathsep.join(paths),
        }
    )
    return env


def setup(workspace: Path) -> dict[str, object]:
    timings = {}
    started = time.monotonic()
    _install_system_packages()
    timings["system_packages"] = time.monotonic() - started
    cargo, _ = _install_rust()
    timings["rust"] = time.monotonic() - started - timings["system_packages"]
    _checkout(workspace)
    install_started = time.monotonic()
    expected_state = {
        "cuda_oxide_rev": CUDA_OXIDE_REV,
        "llvm_major": LLVM_MAJOR,
        "rust_toolchain": RUST_TOOLCHAIN,
    }
    cargo_oxide = Path.home() / ".cargo/bin/cargo-oxide"
    installed_state = None
    if TOOLCHAIN_STATE.exists():
        installed_state = json.loads(TOOLCHAIN_STATE.read_text())
    if cargo_oxide.exists() and installed_state == expected_state:
        print("Pinned cargo-oxide is already installed", flush=True)
    else:
        _run(
            [
                str(cargo),
                f"+{RUST_TOOLCHAIN}",
                "install",
                "--path",
                str(workspace / "crates/cargo-oxide"),
                "--locked",
                "--force",
            ],
            env=_toolchain_env(),
        )
        TOOLCHAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
        TOOLCHAIN_STATE.write_text(json.dumps(expected_state, sort_keys=True) + "\n")
    timings["cargo_oxide"] = time.monotonic() - install_started
    backend_started = time.monotonic()
    _run(
        [str(cargo), f"+{RUST_TOOLCHAIN}", "oxide", "setup"],
        cwd=workspace,
        env=_toolchain_env(),
    )
    timings["backend"] = time.monotonic() - backend_started
    return timings


def exercise(workspace: Path, command: list[str]) -> dict[str, object]:
    cargo = Path.home() / ".cargo/bin/cargo"
    if not cargo.exists() or not workspace.exists():
        raise RuntimeError("Run the setup command first")
    seconds = _run(
        [str(cargo), f"+{RUST_TOOLCHAIN}", "oxide", *command],
        cwd=workspace,
        env=_toolchain_env(),
    )
    return {"command": command, "seconds": seconds}


def exercise_project(project: Path, command: list[str]) -> dict[str, object]:
    cargo = Path.home() / ".cargo/bin/cargo"
    if not cargo.exists() or not project.is_dir():
        raise RuntimeError("Run setup first and ensure the ENNx repository is present")
    seconds = _run(
        [str(cargo), f"+{RUST_TOOLCHAIN}", "oxide", *command],
        cwd=project,
        env=_toolchain_env(),
    )
    return {"command": command, "project": str(project), "seconds": seconds}


def sanitize_project(project: Path) -> dict[str, object]:
    executable = project / "target/release/ennx-cuda"
    if not executable.is_file():
        raise RuntimeError("Run the ennx command before the sanitizer")
    command = [
        "compute-sanitizer",
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        str(executable),
        "resident",
    ]
    seconds = _run(command, cwd=project, env=_toolchain_env())
    return {"command": command, "project": str(project), "seconds": seconds}


def exercise_python(project: Path) -> dict[str, object]:
    repo = project.resolve().parent
    rust = repo / "rust"
    smoke = repo / "ops/cuda_python_smoke.py"
    if not rust.is_dir() or not smoke.is_file():
        raise RuntimeError("The CUDA project must be inside an ENNx checkout")

    env = _toolchain_env()
    env.update(
        {
            "ENNX_FAISS_UNAVAILABLE": "1",
            "PYO3_PYTHON": sys.executable,
            "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
        }
    )
    cargo = Path.home() / ".cargo/bin/cargo"
    build = [
        str(cargo),
        f"+{RUST_TOOLCHAIN}",
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
    ]
    build_seconds = _run(build, cwd=rust, env=env)
    extension = rust / "target/release/libennx_rust.so"
    smoke_command = [sys.executable, str(smoke), str(extension)]
    smoke_seconds = _run(smoke_command, cwd=repo, env=env)
    return {
        "build": build,
        "build_seconds": build_seconds,
        "smoke": smoke_command,
        "smoke_seconds": smoke_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "fingerprint",
            "setup",
            "doctor",
            "vecadd",
            "ennx",
            "resident",
            "sanitize",
            "bench",
            "python",
            "all",
        ],
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()

    result: dict[str, object] = {
        "cuda_oxide_rev": CUDA_OXIDE_REV,
        "rust_toolchain": RUST_TOOLCHAIN,
        "runtime": _fingerprint(),
    }
    if args.action in {"setup", "all"}:
        result["setup"] = setup(args.workspace)
    if args.action in {"doctor", "all"}:
        result["doctor"] = exercise(args.workspace, ["doctor"])
    if args.action in {"vecadd", "all"}:
        result["vecadd"] = exercise(args.workspace, ["run", "vecadd"])
    if args.action in {"ennx", "all"}:
        result["ennx"] = exercise_project(
            args.project, ["run", "--arch", "sm_75", "--", "parity"]
        )
    if args.action in {"resident", "all"}:
        result["resident"] = exercise_project(
            args.project, ["run", "--arch", "sm_75", "--", "resident"]
        )
    if args.action in {"sanitize", "all"}:
        if args.action == "sanitize":
            result["resident"] = exercise_project(
                args.project, ["run", "--arch", "sm_75", "--", "resident"]
            )
        result["sanitize"] = sanitize_project(args.project)
    if args.action in {"bench", "all"}:
        result["bench"] = exercise_project(
            args.project, ["run", "--arch", "sm_75", "--", "bench"]
        )
    if args.action in {"python", "all"}:
        result["python"] = exercise_python(args.project)
    result["ok"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
