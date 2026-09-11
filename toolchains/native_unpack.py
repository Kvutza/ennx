"""Unpack pinned LLVM archives or merge the data files of Linux SDK packages."""

import argparse
from pathlib import Path
import tarfile
import zipfile


def extract(stream, output, package=None):
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if package:
                name = member.name.removeprefix("./")
                if name.startswith("info/licenses/"):
                    member.name = "share/licenses/" + package + "/" + name[14:]
                elif name.split("/", 1)[0] in {"info", "bin"}:
                    # The sysroot's optional clang config embeds a conda prefix.
                    # Buck supplies the sysroot explicitly; no relocation is needed.
                    continue
            archive.extract(member, output, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", action="store_true")
    parser.add_argument("output", type=Path)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.archives:
        package = path.name if args.sdk else None
        if path.suffix == ".conda":
            with zipfile.ZipFile(path) as archive:
                payloads = [
                    name
                    for name in archive.namelist()
                    if name.startswith("pkg-") and name.endswith(".tar.zst")
                ]
                if len(payloads) != 1:
                    raise ValueError(f"Expected one package payload in {path}")
                with archive.open(payloads[0]) as stream:
                    extract(stream, args.output, package)
                # License files live in the separate metadata archive.
                for name in archive.namelist():
                    if name.startswith("info-") and name.endswith(".tar.zst"):
                        with archive.open(name) as stream:
                            extract(stream, args.output, package)
        else:
            with path.open("rb") as stream:
                extract(stream, args.output, package)


if __name__ == "__main__":
    main()
