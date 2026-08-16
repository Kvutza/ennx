def cuda_oxide(name, package, workspace, bpann, ennx, python, modal, cuda, parity):
    """Build a CUDA-Oxide CPython wheel and its GPU parity action."""
    wheel = name + "-wheel"
    native.genrule(
        name = wheel,
        srcs = [package, workspace, bpann, ennx, python, modal, cuda],
        out = "ennx-0.1.5+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl",
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 --version | grep -q '^Python 3\\.12\\.';",
            "OUT_FILE=$PWD/$OUT;",
            "ROOT=$TMP/repo;",
            "mkdir -p $ROOT/rust/crates $ROOT/cuda;",
            "cp -R $(location {})/. $ROOT/;".format(package),
            "cp -R $(location {})/. $ROOT/rust/;".format(workspace),
            "cp -R $(location {}) $ROOT/rust/crates/bpann;".format(bpann),
            "cp -R $(location {}) $ROOT/rust/crates/ennx;".format(ennx),
            "cp -R $(location {}) $ROOT/rust/crates/ennx-py;".format(python),
            "cp -R $(location {}) $ROOT/rust/crates/modal-runner;".format(modal),
            "cp -R $(location {})/. $ROOT/cuda/;".format(cuda),
            "cd $ROOT/rust;",
            "cargo +nightly-2026-04-03 oxide build",
            "--arch sm_75",
            "--cargo-target-dir $TMP/target",
            "--device-codegen-crate ennx_cuda_kernels --",
            "-p ennx-py --features cuda --release;",
            "cd $ROOT;",
            "python3 ops/cuda_wheel.py",
            "$TMP/target/release/libennx_rust.so",
            "$TMP/wheel --root $ROOT;",
            "mkdir -p ${OUT_FILE%/*};",
            "cp $TMP/wheel/ennx-0.1.5+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl $OUT_FILE",
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
    native.genrule(
        name = name + "-parity",
        srcs = [parity],
        out = "bf16-parity.txt",
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 -m pip install --quiet --no-deps",
            "--target $TMP/site $(location :{});".format(wheel),
            "PYTHONPATH=$TMP/site python3 $SRCDIR/ops/bf16_parity.py > $OUT",
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
