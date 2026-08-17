import configparser
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_version(path: Path, key: tuple[str, ...]) -> str:
    with path.open("rb") as file:
        data = tomllib.load(file)
    for part in key:
        data = data[part]
    return data


def test_version():
    config = configparser.ConfigParser()
    config.read(ROOT / ".buckconfig")
    versions = {
        read_version(ROOT / "pyproject.toml", ("project", "version")),
        read_version(ROOT / "rust/Cargo.toml", ("workspace", "package", "version")),
        read_version(ROOT / "cuda/Cargo.toml", ("package", "version")),
        read_version(ROOT / "cuda/kernels/Cargo.toml", ("package", "version")),
        config["ennx"]["release_version"],
    }
    assert len(versions) == 1, versions
