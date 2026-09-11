def cuda_oxide(name, package, workspace, bpann, ennx, python, modal, cuda, parity):
    """Build a CUDA-Oxide CPython wheel and its GPU parity action."""
    wheel = name + "-wheel"
    filename = "ennx-{}+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl".format(read_config("ennx", "release_version", "0.1.1"))
    native.genrule(
        name = wheel,
        srcs = [package, workspace, bpann, ennx, python, modal, cuda],
        outs = {"wheel": [filename], "cargo-target": ["target"]},
        default_outs = [filename],
        # Cargo owns this incremental cache; Buck keeps it between local runs.
        no_outputs_cleanup = True,
        cacheable = False,
        labels = ["uses_undeclared_inputs"],
        env = {"RUSTUP_TOOLCHAIN": "nightly-2026-04-03"},
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 --version | grep -q '^Python 3\\.12\\.';",
            "OUT_DIR=$PWD/$OUT;",
            "OUT_FILE=$OUT_DIR/{};".format(filename),
            "ROOT=$TMP/repo;",
            "mkdir -p $ROOT/rust/crates $ROOT/cuda;",
            "cp -R $(location {})/. $ROOT/;".format(package),
            "cp -R $(location {})/. $ROOT/;".format(workspace),
            "cp -R $(location {}) $ROOT/rust/crates/bpann;".format(bpann),
            "cp -R $(location {}) $ROOT/rust/crates/ennx;".format(ennx),
            "cp -R $(location {}) $ROOT/rust/crates/ennx-py;".format(python),
            "cp -R $(location {}) $ROOT/rust/crates/modal-runner;".format(modal),
            "cp -R $(location {})/. $ROOT/cuda/;".format(cuda),
            "cd $ROOT/cuda;",
            "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;",
            "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target --device-codegen-crate ennx_cuda_kernels -- -p ennx-py --features cuda --release --locked --manifest-path $ROOT/Cargo.toml;",
            "cd $ROOT;",
            "python3 ops/cuda_wheel.py",
            "$OUT_DIR/target/release/libennx_rust.so",
            "$TMP/wheel --root $ROOT --version {};".format(read_config("ennx", "release_version", "0.1.1")),
            "mkdir -p ${OUT_FILE%/*};",
            "cp $TMP/wheel/ennx-{}+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl $OUT_FILE".format(read_config("ennx", "release_version", "0.1.1")),
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
