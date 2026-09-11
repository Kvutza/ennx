from __future__ import annotations

import subprocess
import zipfile

import pytest

from tools.release_provenance import (
    PLATFORM_TAGS,
    PYTHON_ABIS,
    git,
    resolve_release,
    verify_checkout,
    verify_wheels,
)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Release test")
    git(tmp_path, "config", "user.email", "release@example.invalid")
    (tmp_path / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.3"\n')
    for directory in ("cuda", "cuda/kernels"):
        package = tmp_path / directory
        package.mkdir(parents=True, exist_ok=True)
        (package / "Cargo.toml").write_text('[package]\nversion = "1.2.3"\n')
    (tmp_path / ".buckconfig").write_text("[ennx]\nrelease_version = 1.2.3\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "release source")
    git(tmp_path, "tag", "v1.2.3")
    return tmp_path


def test_001(repository):
    commit = resolve_release(repository, "v1.2.3")
    git(repository, "commit", "--allow-empty", "-m", "later development")
    assert resolve_release(repository, "v1.2.3") == commit
    with pytest.raises(ValueError, match="checkout"):
        verify_checkout(repository, "v1.2.3", commit)
    git(repository, "switch", "--detach", commit)
    verify_checkout(repository, "v1.2.3", commit)


def test_002(repository):
    commit = resolve_release(repository, "v1.2.3")
    (repository / "Cargo.toml").write_text("changed")
    with pytest.raises(ValueError, match="modified tracked"):
        verify_checkout(repository, "v1.2.3", commit)
    git(repository, "commit", "--allow-empty", "-m", "later commit")
    git(repository, "tag", "-f", "v1.2.3")
    with pytest.raises(ValueError, match="tag moved"):
        verify_checkout(repository, "v1.2.3", commit)


def test_003(repository):
    git(repository, "tag", "v9.9.9")
    with pytest.raises(ValueError, match="package versions"):
        resolve_release(repository, "v9.9.9")
    with pytest.raises(subprocess.CalledProcessError):
        resolve_release(repository, "v0.0.0")
    with pytest.raises(ValueError, match="release tag"):
        resolve_release(repository, "main")


def test_004(repository):
    git(repository, "tag", "-f", "-a", "v1.2.3", "-m", "release")
    assert resolve_release(repository, "v1.2.3") == git(repository, "rev-parse", "HEAD")


def write_wheel(directory, version, abi, platform_tag, metadata_version=None):
    wheel = directory / f"ennx-{version}-{abi}-{abi}-{platform_tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"ennx-{version}.dist-info/METADATA",
            f"Name: ennx\nVersion: {metadata_version or version}\n",
        )
    return wheel


@pytest.mark.parametrize("version", ["1.2.3", "9.9.9"])
def test_005(tmp_path, version):
    for abi in PYTHON_ABIS:
        write_wheel(
            tmp_path,
            "1.2.3",
            abi,
            "macosx_11_0_arm64",
            metadata_version=version,
        )
    if version == "1.2.3":
        verify_wheels(tmp_path, "1.2.3", ("macosx_11_0_arm64",))
    else:
        with pytest.raises(ValueError, match="metadata"):
            verify_wheels(tmp_path, "1.2.3", ("macosx_11_0_arm64",))
    with pytest.raises(ValueError, match="expected set"):
        verify_wheels(tmp_path, "2.0.0", ("macosx_11_0_arm64",))


def test_006(tmp_path):
    with pytest.raises(ValueError, match="no release wheels"):
        verify_wheels(tmp_path, "1.2.3")


def test_007(tmp_path):
    for platform_tag in PLATFORM_TAGS:
        for abi in PYTHON_ABIS:
            write_wheel(tmp_path, "1.2.3", abi, platform_tag)
    verify_wheels(tmp_path, "1.2.3")
    (tmp_path / "ennx-1.2.3-cp312-cp312-manylinux_2_28_aarch64.whl").unlink()
    with pytest.raises(ValueError, match="manylinux_2_28_aarch64"):
        verify_wheels(tmp_path, "1.2.3")
