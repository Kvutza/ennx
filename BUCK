load("//buck2/wheel:wheel.bzl", "python_wheel")
load(
    "//buck2:python.bzl",
    "PYTHON_ABI",
    "PYTHON_REQUIRES",
    "python_extension_suffix",
)

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

config_setting(
    name = "windows-x86_64",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:windows",
    ],
)

platform(
    name = "windows-x86_64-platform",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:windows",
    ],
    visibility = ["PUBLIC"],
)

filegroup(
    name = "pixi-lock",
    srcs = ["pixi.lock"],
    visibility = ["PUBLIC"],
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
    runtime_libraries = ["//buck2/native:linux-native"],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
    version = "0.0.0",
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
    runtime_libraries = ["//buck2/native:linux-native"],
    target_compatible_with = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
    version = "0.0.0",
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
    runtime_libraries = ["//buck2/native:faiss-build[openmp]"],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
    version = "0.0.0",
)

python_wheel(
    name = "wheel-windows-x86_64",
    extension = "//rust/crates/ennx-py:ennx-py[shared]",
    extension_suffix = python_extension_suffix("windows-x86_64"),
    license = "LICENSE",
    notice = "NOTICE",
    package = "ennx",
    platform_tag = "win_amd64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["src/ennx/**/*.py"]),
    runtime_libraries = ["//buck2/native:windows-native"],
    target_compatible_with = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:windows",
    ],
    version = "0.0.0",
)

alias(
    name = "wheel",
    actual = select({
        ":linux-arm64": ":wheel-linux-arm64",
        ":linux-x86_64": ":wheel-linux-x86_64",
        ":macos-arm64": ":wheel-macos-arm64",
        ":windows-x86_64": ":wheel-windows-x86_64",
    }),
    visibility = ["PUBLIC"],
)
