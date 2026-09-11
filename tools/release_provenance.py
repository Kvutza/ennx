"""Resolve release tags and verify source and wheel versions before publishing."""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
from pathlib import Path
import subprocess
import tomllib
import zipfile


PYTHON_ABIS = ("cp312", "cp313", "cp314")
PLATFORM_TAGS = (
    "macosx_11_0_arm64",
    "manylinux_2_28_aarch64",
    "manylinux_2_28_x86_64",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def resolve_release(root: Path, tag: str) -> str:
    if not tag.startswith("v") or len(tag) == 1:
        raise ValueError("release tag must be v followed by the package version")
    ref = f"refs/tags/{tag}"
    git(root, "check-ref-format", ref)
    commit = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    workspace = tomllib.loads(git(root, "show", f"{commit}:Cargo.toml"))
    versions = {workspace["workspace"]["package"]["version"]}
    for path in ("cuda/Cargo.toml", "cuda/kernels/Cargo.toml"):
        package = tomllib.loads(git(root, "show", f"{commit}:{path}"))
        versions.add(package["package"]["version"])
    buck = configparser.ConfigParser()
    buck.read_string(git(root, "show", f"{commit}:.buckconfig"))
    versions.add(buck["ennx"]["release_version"])
    if versions != {tag[1:]}:
        raise ValueError(
            f"tag {tag} disagrees with package versions: {sorted(versions)}"
        )
    return commit


def verify_checkout(root: Path, tag: str, commit: str) -> None:
    if resolve_release(root, tag) != commit:
        raise ValueError("release tag moved after resolution")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("checkout is not the resolved release commit")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("release checkout contains modified tracked files")


def expected_wheel_names(version: str, platform_tags: tuple[str, ...]) -> set[str]:
    return {
        f"ennx-{version}-{abi}-{abi}-{platform_tag}.whl"
        for abi in PYTHON_ABIS
        for platform_tag in platform_tags
    }


def verify_wheels(
    directory: Path, version: str, platform_tags: tuple[str, ...] = PLATFORM_TAGS
) -> None:
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise ValueError("no release wheels found")
    actual = {wheel.name for wheel in wheels}
    expected = expected_wheel_names(version, platform_tags)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "release wheels do not match expected set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_paths = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"expected one wheel metadata document: {wheel.name}")
            metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if metadata.get_all("Name") != ["ennx"] or metadata.get_all("Version") != [
            version
        ]:
            raise ValueError(f"wheel metadata disagrees with release: {wheel.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--commit")
    parser.add_argument("--platform-tag", action="append", choices=PLATFORM_TAGS)
    parser.add_argument("--wheels", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    commit = resolve_release(root, args.tag)
    if args.commit:
        verify_checkout(root, args.tag, args.commit)
    if args.wheels:
        verify_wheels(
            args.wheels, args.tag[1:], tuple(args.platform_tag or PLATFORM_TAGS)
        )
    print(f"tag={args.tag}\ncommit={commit}")


if __name__ == "__main__":
    main()
