"""Package a CUDA-enabled ENNx extension as a CPython 3.12 Linux wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import shutil
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from pathlib import Path

TAG = "cp312-cp312-manylinux_2_28_x86_64"


def _digest(path: Path) -> str:
    value = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
    return "sha256=" + value.rstrip(b"=").decode("ascii")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def package_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def build_wheel(root: Path, extension: Path, output: Path) -> Path:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("CUDA wheel packaging requires CPython 3.12")
    if sys.platform != "linux":
        raise RuntimeError("CUDA wheels are built on Linux")
    if not extension.is_file():
        raise FileNotFoundError(extension)

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix or "cpython-312" not in suffix:
        raise RuntimeError(f"unexpected extension suffix: {suffix!r}")

    version = package_version(root) + "+cuda75"
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"ennx-{version}-{TAG}.whl"
    with tempfile.TemporaryDirectory(prefix="ennx-cuda-wheel-") as directory:
        stage = Path(directory)
        package = stage / "ennx"
        shutil.copytree(
            root / "src/ennx",
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
        shutil.copy2(extension, package / f"ennx_rust{suffix}")

        dist = stage / f"ennx-{version}.dist-info"
        _write(
            dist / "METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.3",
                    "Name: ennx",
                    f"Version: {version}",
                    "Summary: Epistemic Nearest Neighbors with CUDA-Oxide sm_75 support",
                    "Requires-Python: >=3.12,<3.13",
                    "Requires-Dist: numpy>=2.1",
                    "",
                ]
            ),
        )
        _write(
            dist / "WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: ennx.cuda_wheel",
                    "Root-Is-Purelib: false",
                    f"Tag: {TAG}",
                    "",
                ]
            ),
        )
        _write(dist / "top_level.txt", "ennx\n")
        license_dir = dist / "licenses"
        license_dir.mkdir(parents=True)
        shutil.copy2(root / "LICENSE", license_dir / "LICENSE")
        shutil.copy2(root / "NOTICE", license_dir / "NOTICE")

        record = dist / "RECORD"
        rows = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path != record:
                rows.append(
                    (
                        path.relative_to(stage).as_posix(),
                        _digest(path),
                        str(path.stat().st_size),
                    )
                )
        rows.append((record.relative_to(stage).as_posix(), "", ""))
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        _write(record, stream.getvalue())

        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(build_wheel(args.root, args.extension, args.output))


if __name__ == "__main__":
    main()
