def _copy_extension_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    ctx.actions.symlink(
        output = output,
        target_file = ctx.file.src,
    )
    return DefaultInfo(files = depset([output]))

copy_extension = rule(
    implementation = _copy_extension_impl,
    attrs = {
        "output": attr.string(mandatory = True),
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)
