def _unpack_openmp_impl(ctx):
    archive = ctx.attrs.archive[DefaultInfo].default_outputs[0]
    package = archive.project("pkg-llvm-openmp-22.1.8-hc7d1edf_0.tar.zst")
    out = ctx.actions.declare_output("openmp", dir = True)
    script = ctx.actions.write(
        "unpack-openmp.sh",
        "#!/bin/sh\nset -eu\nmkdir -p \"$2\"\n/usr/bin/bsdtar -xf \"$1\" -C \"$2\"\n",
        is_executable = True,
    )
    ctx.actions.run([script, package, out.as_output()], category = "openmp_unpack")
    outputs = {
        "header": out.project("include/omp.h"),
        "lib": out.project("lib/libomp.dylib"),
    }
    return [
        DefaultInfo(
            default_output = outputs["lib"],
            sub_targets = {
                name: [DefaultInfo(default_output = output)]
                for name, output in outputs.items()
            },
        ),
    ]

unpack_openmp = rule(
    impl = _unpack_openmp_impl,
    attrs = {
        "archive": attrs.dep(),
    },
)
