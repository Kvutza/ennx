load("@prelude//rust:rust_toolchain.bzl", "PanicRuntime", "RustToolchainInfo")

# Official channel-rust-1.96.0.toml, released 2026-05-28.
_ARCHIVES = {
    "aarch64-apple-darwin": (
        "1bb7b0bad1d2a42fc4173ede6dd460de2774fc1858a8369329d3e081e4e3426c",
        "439c4f71060b913e00db3a2e01340b2da0aa49978b843e36871f3250267c63f8",
        "5ebcd03b05f70b0ffc330fd884b24e34189281ab6cc8b4c9ac5f1e32b0fdd85c",
    ),
    "aarch64-unknown-linux-gnu": (
        "76b1a6e8dd1636e364d4bbba685485ff44eee5ff6434add089bab4c703c7e19d",
        "538e85452709687797d990579a491ff9b02f8bffba4a5d54cfa945e28868053e",
        "f0304a59688cccc77da29bbfbb96cb29407b4fdc152337f8de198d81a01eac05",
    ),
    "x86_64-unknown-linux-gnu": (
        "7d7fa1d0cfb0fab71a956bb78f41107202c17f30ab56c45288e869a37fd9633d",
        "c09c7c646248f14f473f5f7a029af15ee57c3a9f9bc93dfa72d9621938586b82",
        "9b0b89c67d5ce8195e1c8733587a63e326353fdda65f62455a64fa03e25659f2",
    ),
}

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
            default_edition = "2021",
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
    for triple, hashes in _ARCHIVES.items():
        for component, checksum in zip(["rustc", "rust-std", "clippy"], hashes):
            archive = "{}-1.96.0-{}".format(component, triple)
            native.http_archive(
                name = archive,
                urls = ["https://static.rust-lang.org/dist/2026-05-28/{}.tar.xz".format(archive)],
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
        compiler = select_map(triple, lambda value: ":rustc-1.96.0-" + value, recurse = True),
        std = select_map(triple, lambda value: ":rust-std-1.96.0-" + value, recurse = True),
        clippy = select_map(triple, lambda value: ":clippy-1.96.0-" + value, recurse = True),
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
