from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_oxide_build_command_is_atomic():
    source = (ROOT / "buck2/cuda/defs.bzl").read_text(encoding="utf-8")
    lockfile = "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;"
    build = (
        "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target "
        "--device-codegen-crate ennx_cuda_kernels -- -p ennx-py --features cuda "
        "--release --locked --manifest-path $ROOT/Cargo.toml;"
    )

    assert lockfile in source
    assert build in source
    assert source.index(lockfile) < source.index(build)
    assert '"cargo oxide build",' not in source
    assert '"--arch sm_75",' not in source
    assert "; --arch sm_75" not in " ".join(source.split())
