"""Check the Cargo-built CUDA extension through ENNx's public Python API."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_round(session, seeds: list[int], neighbors: int, reward: float, accept: bool):
    trial = session.ask(
        seeds,
        0.8,
        neighbors,
        beta=1.3,
        seed=23,
    )
    session.tell(reward, accept)
    return trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "extension",
        nargs="?",
        type=Path,
        default=ROOT / "rust/target/release/libennx_rust.so",
    )
    args = parser.parse_args()
    extension = args.extension.resolve()
    if not extension.is_file():
        raise SystemExit(f"CUDA extension not found: {extension}")

    with tempfile.TemporaryDirectory(prefix="ennx-cuda-python-") as directory:
        package = Path(directory) / "ennx"
        shutil.copytree(ROOT / "src/ennx", package)
        shutil.copy2(extension, package / "ennx_rust.so")
        sys.path.insert(0, directory)

        from ennx.experimental import ResidentBoSession

        leaves = [
            (0, 257, 4, 0.125, 1.0, 0.75),
            (257, 259, 8, 0.03125, 1.0, 0.5),
        ]
        row_bytes = (257 + 1) // 2 + 259
        base = [(index * 37 + 11) & 0xFF for index in range(row_bytes)]
        cpu = ResidentBoSession(base, -0.75, leaves, 4, device="cpu")
        cuda = ResidentBoSession(base, -0.75, leaves, 4, device="cuda")

        rounds = [
            ([3, 17, 0xDEADBEEFCAFEBABE, (1 << 64) - 10], 1, 1.25, True),
            ([5, 29, 0x0123456789ABCDEF, (1 << 64) - 4], 2, 0.5, False),
        ]
        for seeds, neighbors, reward, accept in rounds:
            cpu_trial = run_round(cpu, seeds, neighbors, reward, accept)
            cuda_trial = run_round(cuda, seeds, neighbors, reward, accept)
            if cpu_trial[:2] != cuda_trial[:2]:
                raise SystemExit(
                    f"CUDA Python choice {cuda_trial[:2]} differs from CPU {cpu_trial[:2]}"
                )
            tolerance = 2.0e-5 * max(abs(cpu_trial[2]), 1.0)
            if not math.isclose(cpu_trial[2], cuda_trial[2], abs_tol=tolerance):
                raise SystemExit(
                    f"CUDA Python score {cuda_trial[2]} differs from CPU {cpu_trial[2]}"
                )
            if cpu_trial[3] != cuda_trial[3]:
                raise SystemExit("CUDA Python program version differs from CPU")

        if cpu.rewards != cuda.rewards:
            raise SystemExit("CUDA Python rewards differ from CPU")
        print(f"CUDA_PYTHON ok=true rounds={len(rounds)} python={sys.version_info[:3]}")


if __name__ == "__main__":
    main()
