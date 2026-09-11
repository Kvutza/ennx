load("@prelude//toolchains:cxx.bzl", "system_cxx_toolchain")

def ennx_cxx_toolchain(name):
    prefix = read_config("ennx", "cxx_root", "")
    if not prefix:
        system_cxx_toolchain(name = name, visibility = ["PUBLIC"])
        return

    sdk = read_config("ennx", "cxx_sdk", "")
    linux_flags = select({
        "prelude//cpu:arm64": [
            "--sysroot=" + prefix + "/aarch64-conda-linux-gnu/sysroot",
            "--gcc-install-dir=" + prefix + "/lib/gcc/aarch64-conda-linux-gnu/8.5.0",
        ],
        "prelude//cpu:x86_64": [
            "--sysroot=" + prefix + "/x86_64-conda-linux-gnu/sysroot",
            "--gcc-install-dir=" + prefix + "/lib/gcc/x86_64-conda-linux-gnu/8.5.0",
        ],
    })
    flags = ["--no-default-config"] + select({
        "prelude//os:linux": linux_flags,
        "prelude//os:macos": ["-isysroot", sdk],
    })
    cxx_flags = select({
        "prelude//os:linux": [],
        # Use Apple's headers and runtime, not the compiler environment's libc++.
        "prelude//os:macos": ["-nostdinc++", "-isystem", sdk + "/usr/include/c++/v1"],
    })
    system_cxx_toolchain(
        name = name,
        compiler = prefix + "/bin/clang",
        cxx_compiler = prefix + "/bin/clang++",
        linker = prefix + "/bin/clang++",
        archiver = prefix + "/bin/llvm-ar",
        c_flags = flags,
        cxx_flags = flags + cxx_flags,
        link_flags = flags + select({
            "prelude//os:linux": ["--ld-path=" + prefix + "/bin/ld.lld", "-L" + prefix + "/lib"],
            "prelude//os:macos": ["--ld-path=" + prefix + "/bin/ld64.lld"],
        }),
        visibility = ["PUBLIC"],
    )
