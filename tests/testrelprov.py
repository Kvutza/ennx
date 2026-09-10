from __future__ import annotations

import subprocess
import zipfile

import pytest

from tools.release_provenance import (
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


@pytest.mark.parametrize("version", ["1.2.3", "9.9.9"])
def test_005(tmp_path, version):
    wheel = tmp_path / "ennx-1.2.3-cp313-cp313-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "ennx-1.2.3.dist-info/METADATA", f"Name: ennx\nVersion: {version}\n"
        )
    if version == "1.2.3":
        verify_wheels(tmp_path, "1.2.3")
    else:
        with pytest.raises(ValueError, match="metadata"):
            verify_wheels(tmp_path, "1.2.3")
    with pytest.raises(ValueError, match="filename"):
        verify_wheels(tmp_path, "2.0.0")


def test_006(tmp_path):
    with pytest.raises(ValueError, match="no release wheels"):
        verify_wheels(tmp_path, "1.2.3")
