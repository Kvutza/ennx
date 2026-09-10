def _zig_static_library_impl(ctx):
    out = ctx.actions.declare_output(ctx.attrs.out)
    cmd = cmd_args(
        "./tools/zigw",
        "build-lib",
        ctx.attrs.root,
        "-OReleaseFast",
        "-fPIC",
        "-fno-emit-h",
        cmd_args(out.as_output(), format = "-femit-bin={}"),
    )
    ctx.actions.run(cmd, category = "zig_build_lib", identifier = ctx.label.name)
    return [DefaultInfo(default_output = out)]

zig_static_library = rule(
    impl = _zig_static_library_impl,
    attrs = {
        "out": attrs.string(),
        "root": attrs.source(),
        "srcs": attrs.list(attrs.source(), default = []),
    },
)
