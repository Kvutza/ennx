load("@prelude//cxx:cxx_toolchain_types.bzl", "LinkerType")
load("@prelude//toolchains:cxx.bzl", "CxxToolsInfo", "cxx_tools_info_toolchain")
load(":native_archives.bzl", "native_archives")

def _tools_impl(ctx):
    llvm = ctx.attrs.llvm[DefaultInfo].default_outputs[0]
    clang = llvm.project("bin/clang")
    clangxx = llvm.project("bin/clang++")
    if ctx.attrs.os == "macos":
        compiler = cmd_args(ctx.attrs.macos_wrapper, clang, "c", hidden = [llvm])
        cxx_compiler = cmd_args(ctx.attrs.macos_wrapper, clangxx, "cxx", hidden = [llvm])
        linker = cmd_args(cxx_compiler, "-fuse-ld=lld", cmd_args(llvm.project("bin/ld64.lld"), format = "--ld-path={}"))
    else:
        sdk = ctx.attrs.sdk[DefaultInfo].default_outputs[0]
        triple = ctx.attrs.arch + "-conda-linux-gnu"
        flags = cmd_args(
            "--no-default-config",
            "--target=" + ctx.attrs.arch + "-unknown-linux-gnu",
            cmd_args(sdk.project(triple + "/sysroot"), format = "--sysroot={}"),
            cmd_args(sdk.project("lib/gcc/" + triple + "/8.5.0"), format = "--gcc-install-dir={}"),
            hidden = [llvm, sdk],
        )
        compiler = cmd_args(clang, flags)
        cxx_compiler = cmd_args(clangxx, flags)
        linker = cmd_args(
            cxx_compiler,
            cmd_args(llvm.project("bin/ld.lld"), format = "--ld-path={}"),
            cmd_args(sdk.project("lib"), format = "-L{}"),
        )
    return [
        DefaultInfo(),
        CxxToolsInfo(
            compiler = compiler,
            compiler_type = "clang",
            cxx_compiler = cxx_compiler,
            asm_compiler = compiler,
            asm_compiler_type = "clang",
            linker = linker,
            linker_type = LinkerType("darwin" if ctx.attrs.os == "macos" else "gnu"),
            archiver = cmd_args(llvm.project("bin/llvm-ar"), hidden = [llvm]),
            archiver_type = "gnu",
        ),
    ]

_tools = rule(
    impl = _tools_impl,
    attrs = {
        "llvm": attrs.dep(),
        "sdk": attrs.option(attrs.dep(), default = None),
        "os": attrs.string(),
        "arch": attrs.string(),
        "macos_wrapper": attrs.source(),
    },
)

def ennx_cxx_toolchain(name):
    native_archives()
    arch = select({"prelude//cpu:arm64": "aarch64", "prelude//cpu:x86_64": "x86_64"})
    _tools(
        name = name + "-tools",
        macos_wrapper = "clang-macos",
        llvm = select({
            "prelude//os:macos": ":llvm-macos-arm64",
            "prelude//os:linux": select_map(arch, lambda value: ":llvm-linux-" + value),
        }),
        sdk = select({
            "prelude//os:macos": None,
            "prelude//os:linux": select_map(arch, lambda value: ":sdk-linux-" + value),
        }),
        os = select({"prelude//os:macos": "macos", "prelude//os:linux": "linux"}),
        arch = arch,
    )
    cxx_tools_info_toolchain(name = name, cxx_tools_info = ":" + name + "-tools", visibility = ["PUBLIC"])
