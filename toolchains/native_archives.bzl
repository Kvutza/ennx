# LLVM binaries: hermeticbuild/hermetic-llvm, release llvm-22.1.8-1.
# Linux executables are statically linked against musl, independent of host glibc.
_LLVM = {
    "macos-arm64": ("darwin-arm64", "53b7a8a8d4ff638c967e8cc4d19a796a82e1e082289afb2f1e30ca7432435b11"),
    "linux-x86_64": ("linux-amd64-musl", "89f29294a584267251edac00ab723a0a74514b8e1acb7f036840d13635c66bfa"),
    "linux-aarch64": ("linux-arm64-musl", "2914968f1fff964c654627599d15ade423e17ed310b09e2d6b61e82c16a1993d"),
}

# Standalone Python 3.14 provides zstd extraction without host Python or unzstd.
_UNPACK_PYTHON = {
    "aarch64-apple-darwin": "4632cb1a6edad9e73d3c81b6d2e69131637d995173e3e85005df14102b0592ba",
    "x86_64-unknown-linux-gnu": "3959f92825141e04adf44982d3a83ee57af0877e893b0796e04c1468749d9b04",
    "aarch64-unknown-linux-gnu": "8a0798baa8a2c27b5751d590aced543eaa85e20b3f73d93c1af049688acdc9c5",
}

# Conda-forge data packages, pinned independently of Pixi. GCC 8.5 headers and
# startup objects use the existing tested GCC runtime and glibc 2.28 sysroot.
_SDK = {
    "x86_64": [
        ("linux-64/libgcc-16.2.0-ha9f2e26_4.conda", "24090e675d34403b4ee1cd4372d8f6c0937da7ecfd66a19a57cac2ed0f4ea793"),
        ("linux-64/libgcc-devel_linux-64-8.5.0-ha5d7ed8_19.tar.bz2", "b5ac7b49304140e55cb4cdf6936a93ac800c59609f9c51965c761cf4dcf85830"),
        ("linux-64/libstdcxx-16.2.0-h934c35e_4.conda", "40b792b0186c1e8859280a1f6f19a54fc50a11b32724fc7b637009c1a9bd302b"),
        ("linux-64/libstdcxx-devel_linux-64-8.5.0-ha5d7ed8_19.tar.bz2", "4a97d23fd7fafc3e0c8b28e9957efdb0b01984b8c19924816b664a05e9c4684d"),
        ("noarch/kernel-headers_linux-64-4.18.0-he073ed8_9.conda", "41557eeadf641de6aeae49486cef30d02a6912d8da98585d687894afd65b356a"),
        ("noarch/sysroot_linux-64-2.28-h4ee821c_9.conda", "c47299fe37aebb0fcf674b3be588e67e4afb86225be4b0d452c7eb75c086b851"),
    ],
    "aarch64": [
        ("linux-aarch64/libgcc-16.2.0-h205dda4_4.conda", "a132d78d49d0fa0a08e9e2a77528a12d974e8e123bc32895c0bd0a737b8abd89"),
        ("linux-aarch64/libgcc-devel_linux-aarch64-8.5.0-ha065938_19.tar.bz2", "25733f02509e3931389296dc565e8f2aaa30996272a9f06460384c5225c4c795"),
        ("linux-aarch64/libstdcxx-16.2.0-hef695bb_4.conda", "84d2b667bce6549325243235952110b1849ff2dfd7091a1479108930b88057d5"),
        ("linux-aarch64/libstdcxx-devel_linux-aarch64-8.5.0-ha065938_19.tar.bz2", "5565b07af9453e9a9cb8a0c8bcc734ac5ee53eab7f7a2d09b9c15dbe8918f9ad"),
        ("noarch/kernel-headers_linux-aarch64-4.18.0-h05a177a_9.conda", "5d224bf4df9bac24e69de41897c53756108c5271a0e5d2d2f66fd4e2fbc1d84b"),
        ("noarch/sysroot_linux-aarch64-2.28-h585391f_9.conda", "1bd2db6b2e451247bab103e4a0128cf6c7595dd72cb26d70f7fadd9edd1d1bc3"),
    ],
}

def _unpack_impl(ctx):
    python = ctx.attrs.python[DefaultInfo].default_outputs[0]
    output = ctx.actions.declare_output("prefix", dir = True)
    command = cmd_args(python.project("bin/python3"), ctx.attrs.script)
    if ctx.attrs.sdk:
        command.add("--sdk")
    command.add(output.as_output())
    for archive in ctx.attrs.archives:
        command.add(archive[DefaultInfo].default_outputs[0])
    # Keep the interpreter's shared libraries and standard library materialized.
    command.add(cmd_args(hidden = [python]))
    ctx.actions.run(command, category = "native_unpack")
    return [DefaultInfo(default_output = output)]

_unpack = rule(
    impl = _unpack_impl,
    attrs = {
        "archives": attrs.list(attrs.dep()),
        "python": attrs.exec_dep(),
        "script": attrs.source(),
        "sdk": attrs.bool(default = False),
    },
)

def native_archives():
    for triple, checksum in _UNPACK_PYTHON.items():
        native.http_archive(
            name = "native-unpack-python-" + triple,
            urls = ["https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.14.7%2B20260901-" + triple + "-install_only_stripped.tar.gz"],
            sha256 = checksum,
            strip_prefix = "python",
        )
    python = select({
        "prelude//os:macos": ":native-unpack-python-aarch64-apple-darwin",
        "prelude//os:linux": select({
            "prelude//cpu:x86_64": ":native-unpack-python-x86_64-unknown-linux-gnu",
            "prelude//cpu:arm64": ":native-unpack-python-aarch64-unknown-linux-gnu",
        }),
    })
    for platform, (suffix, checksum) in _LLVM.items():
        name = "llvm-" + platform
        native.http_file(
            name = name + "-download",
            urls = ["https://github.com/hermeticbuild/hermetic-llvm/releases/download/llvm-22.1.8-1/llvm-toolchain-minimal-22.1.8-" + suffix + ".tar.zst"],
            sha256 = checksum,
            out = "llvm.tar.zst",
        )
        _unpack(
            name = name,
            archives = [":" + name + "-download"],
            python = python,
            script = "native_unpack.py",
            visibility = ["PUBLIC"],
        )
    for arch, packages in _SDK.items():
        downloads = []
        for path, checksum in packages:
            filename = path.split("/")[-1]
            name = "sdk-" + filename
            native.http_file(
                name = name,
                urls = ["https://conda.anaconda.org/conda-forge/" + path],
                sha256 = checksum,
                out = filename,
            )
            downloads.append(":" + name)
        _unpack(
            name = "sdk-linux-" + arch,
            archives = downloads,
            python = python,
            script = "native_unpack.py",
            sdk = True,
            visibility = ["PUBLIC"],
        )
