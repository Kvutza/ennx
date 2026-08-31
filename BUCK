load(
    "//buck2:python.bzl",
    "PYTHON_ABI",
    "PYTHON_REQUIRES",
    "python_extension_suffix",
)
load("//buck2/cuda:defs.bzl", "cuda_oxide")
load("//buck2/wheel:wheel.bzl", "python_wheel")

config_setting(
    name = "linux-arm64",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
)

config_setting(
    name = "linux-x86_64",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
)

platform(
    name = "linux-arm64-platform",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
    visibility = ["PUBLIC"],
)

platform(
    name = "linux-x86_64-platform",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
    visibility = ["PUBLIC"],
)

config_setting(
    name = "macos-arm64",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
)

filegroup(
    name = "pixi-lock",
    srcs = ["pixi.lock"],
    visibility = ["PUBLIC"],
)

filegroup(
    name = "cargo-workspace",
    srcs = [
        "Cargo.lock",
        "Cargo.toml",
    ],
    visibility = ["PUBLIC"],
)

filegroup(
    name = "cuda-package",
    srcs = glob(["src/ennx/**/*.py"]) + [
        "LICENSE",
        "NOTICE",
        "README.md",
        "ops/cuda_wheel.py",
        "rust-toolchain.toml",
    ],
    visibility = ["PUBLIC"],
)

cuda_oxide(
    name = "cuda",
    package = ":cuda-package",
    workspace = ":cargo-workspace",
    bpann = "//rust/crates/bpann:bpann-source",
    ennx = "//rust/crates/ennx:ennx-source",
    python = "//rust/crates/ennx-py:python-source",
    modal = "//rust/crates/modal-runner:modal-source",
    cuda = "//cuda:source",
    parity = "ops/bf16_parity.py",
)

python_wheel(
    name = "wheel-linux-arm64",
    extension = "//rust/crates/ennx-py:ennx-py[shared]",
    extension_suffix = python_extension_suffix("linux-aarch64"),
    license = "LICENSE",
    notice = "NOTICE",
    package = "ennx",
    platform_tag = "manylinux_2_28_aarch64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["src/ennx/**/*.py"]),
    readme = "README.md",
    runtime_libraries = [],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
    version = read_config("ennx", "release_version", "0.1.1"),
)

python_wheel(
    name = "wheel-linux-x86_64",
    extension = "//rust/crates/ennx-py:ennx-py[shared]",
    extension_suffix = python_extension_suffix("linux-x86_64"),
    license = "LICENSE",
    notice = "NOTICE",
    package = "ennx",
    platform_tag = "manylinux_2_28_x86_64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["src/ennx/**/*.py"]),
    readme = "README.md",
    runtime_libraries = [],
    target_compatible_with = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
    version = read_config("ennx", "release_version", "0.1.1"),
)

python_wheel(
    name = "wheel-macos-arm64",
    extension = "//rust/crates/ennx-py:ennx-py[shared]",
    extension_suffix = python_extension_suffix("macos-arm64"),
    license = "LICENSE",
    notice = "NOTICE",
    package = "ennx",
    platform_tag = "macosx_11_0_arm64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["src/ennx/**/*.py"]),
    readme = "README.md",
    runtime_libraries = ["//buck2/native:openmp-build[lib]"],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
    version = read_config("ennx", "release_version", "0.1.1"),
)

alias(
    name = "wheel",
    actual = select({
        ":linux-arm64": ":wheel-linux-arm64",
        ":linux-x86_64": ":wheel-linux-x86_64",
        ":macos-arm64": ":wheel-macos-arm64",
    }),
    visibility = ["PUBLIC"],
)
