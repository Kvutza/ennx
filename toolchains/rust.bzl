load("@prelude//rust:rust_toolchain.bzl", "PanicRuntime", "RustToolchainInfo")
load(":rust_pins.bzl", "RUST_ARCHIVES", "RUST_DATE", "RUST_EDITION")

def _sysroot_impl(ctx):
    compiler = ctx.attrs.compiler[DefaultInfo].default_outputs[0]
    std = ctx.attrs.std[DefaultInfo].default_outputs[0]
    clippy = ctx.attrs.clippy[DefaultInfo].default_outputs[0]
    sysroot = ctx.actions.declare_output("sysroot", dir = True)
    merge = ctx.actions.write(
        "merge.sh",
        '#!/bin/sh\nset -eu\nmkdir -p "$4"\ncp -R "$1"/. "$4"/\ncp -R "$2"/. "$4"/\ncp -R "$3"/. "$4"/\n',
        is_executable = True,
    )
    ctx.actions.run([merge, compiler, std, clippy, sysroot.as_output()], category = "rust_sysroot")
    return [DefaultInfo(default_output = sysroot)]

_rust_sysroot = rule(
    impl = _sysroot_impl,
    attrs = {
        "compiler": attrs.dep(),
        "std": attrs.dep(),
        "clippy": attrs.dep(),
    },
)

def _rust_toolchain_impl(ctx):
    sysroot = ctx.attrs.sysroot[DefaultInfo].default_outputs[0]
    profile = ctx.attrs.profile
    flags = ["-Copt-level=1", "-Cdebug-assertions=on", "-Coverflow-checks=on"]
    if profile == "release":
        flags = ["-Copt-level=3", "-Cdebug-assertions=off", "-Coverflow-checks=off", "-Ccodegen-units=16", "-Cembed-bitcode=yes"]
    elif profile == "tool":
        flags = ["-Copt-level=0", "-Cdebug-assertions=off", "-Coverflow-checks=off", "-Ccodegen-units=16"]
    elif profile != "dev":
        fail("unknown Rust profile: " + profile)
    return [
        DefaultInfo(default_output = sysroot),
        RustToolchainInfo(
            compiler = RunInfo(args = [sysroot.project("bin/rustc")]),
            rustdoc = RunInfo(args = [sysroot.project("bin/rustdoc")]),
            clippy_driver = RunInfo(args = [sysroot.project("bin/clippy-driver")]),
            sysroot_path = sysroot,
            rustc_target_triple = ctx.attrs.triple,
            default_edition = RUST_EDITION,
            panic_runtime = PanicRuntime("unwind"),
            rustc_flags = flags,
            rustc_binary_flags = ["-Clto=thin"] if profile == "release" else [],
            rustc_test_flags = ["-Clto=thin"] if profile == "release" else [],
        ),
    ]

_rust_toolchain = rule(
    impl = _rust_toolchain_impl,
    attrs = {
        "sysroot": attrs.exec_dep(),
        "triple": attrs.string(),
        "profile": attrs.string(),
    },
    is_toolchain_rule = True,
)

def pinned_rust_toolchain(name):
    for triple, hashes in RUST_ARCHIVES.items():
        for component, checksum in zip(["rustc", "rust-std", "clippy"], hashes):
            archive = "{}-nightly-{}".format(component, triple)
            native.http_archive(
                name = archive,
                urls = ["https://static.rust-lang.org/dist/{}/{}.tar.xz".format(RUST_DATE, archive)],
                sha256 = checksum,
                strip_prefix = archive + "/" + ({"rustc": "rustc", "rust-std": "rust-std-" + triple, "clippy": "clippy-preview"}[component]),
            )
    triple = select({
        "prelude//os:macos": "aarch64-apple-darwin",
        "prelude//os:linux": select({
            "prelude//cpu:arm64": "aarch64-unknown-linux-gnu",
            "prelude//cpu:x86_64": "x86_64-unknown-linux-gnu",
        }),
    })
    _rust_sysroot(
        name = name + "-sysroot",
        compiler = select_map(triple, lambda value: ":rustc-nightly-" + value, recurse = True),
        std = select_map(triple, lambda value: ":rust-std-nightly-" + value, recurse = True),
        clippy = select_map(triple, lambda value: ":clippy-nightly-" + value, recurse = True),
    )
    _rust_toolchain(
        name = name,
        triple = triple,
        sysroot = ":" + name + "-sysroot",
        # Execution dependencies (build scripts, proc macros, packagers) do not
        # ship in the library. Keep them shared across dev/release builds.
        profile = select({
            ":build-tool": "tool",
            "root//:release-mode": "release",
            "DEFAULT": "dev",
        }),
        visibility = ["PUBLIC"],
    )
