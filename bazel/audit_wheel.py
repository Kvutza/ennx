import os
import subprocess
import sys
import tempfile
import zipfile


def wheel_paths(root: str) -> tuple[str, str]:
    dist_infos = [
        name
        for name in os.listdir(root)
        if name.startswith("ennx-") and name.endswith(".dist-info")
    ]
    extensions = [
        name
        for name in os.listdir(os.path.join(root, "ennx"))
        if name.startswith("ennx_rust") and name.endswith(".so")
    ]
    assert len(dist_infos) == 1, f"Expected one dist-info directory, got {dist_infos}"
    assert len(extensions) == 1, f"Expected one native extension, got {extensions}"
    return os.path.join(root, dist_infos[0]), os.path.join(root, "ennx", extensions[0])


def audit_macos(extension: str) -> None:
    output = subprocess.check_output(["otool", "-l", extension], text=True)
    assert "cmd LC_BUILD_VERSION" in output, "LC_BUILD_VERSION missing"
    blocks = output.split("cmd LC_BUILD_VERSION")
    assert any("minos 11.0" in block for block in blocks[1:]), "Expected minos 11.0"

    linked = subprocess.check_output(["otool", "-L", extension], text=True)
    print("Dynamic libraries linked:\n", linked)
    forbidden = ["homebrew", "anaconda", "miniconda", "pixi", ".cache"]
    for line in linked.lower().splitlines():
        assert not any(word in line for word in forbidden), line


def audit_wheel(wheel_path: str):
    print(f"Auditing wheel: {wheel_path}")
    assert os.path.exists(wheel_path), f"Wheel not found at {wheel_path}"
    assert "cp313-cp313-macosx_11_0_arm64" in os.path.basename(wheel_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(wheel_path, "r") as archive:
            archive.extractall(tmpdir)
        dist_info, extension = wheel_paths(tmpdir)
        assert os.path.isfile(os.path.join(dist_info, "licenses", "LICENSE"))
        assert os.path.isfile(os.path.join(dist_info, "licenses", "NOTICE"))
        audit_macos(extension)

        sys.path.insert(0, tmpdir)
        from ennx.ennx_rust import optimizer

        assert hasattr(optimizer, "PackedSearch"), (
            "Missing PackedSearch class on PyO3 optimizer module"
        )
        print(
            "Successfully imported ennx and verified native PackedSearch API from audited release wheel!"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: audit_wheel.py <path_to_wheel>")
        sys.exit(1)
    audit_wheel(sys.argv[1])
