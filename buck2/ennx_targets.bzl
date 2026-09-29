"""Buck2 renderer for ENNX's shared logical program catalogue."""

load("//buck2:reindeer_rules.bzl", "app_rust_binary")

def _local_program_dep(name):
    return "//rust/crates/" + name + ":" + name

def _program_deps(program):
    deps = [_local_program_dep(name) for name in program["local_deps"]]
    deps += ["//rust:" + crate for crate in program["crate_deps"]]
    deps += select({
        "//:macos-arm64": ["//buck2/native:openmp-macos"],
        "DEFAULT": [],
    })
    return deps

def ennx_program_targets(programs, edition, sources):
    """Declare the Buck2 half of the shared logical program catalogue."""
    for program in programs:
        if "cuda" in program["required_features"]:
            continue
        app_rust_binary(
            name = program["name"],
            srcs = sources,
            crate = program["crate"],
            crate_root = program["root"],
            edition = edition,
            env = {"CARGO_PKG_VERSION": program["version"]},
            deps = _program_deps(program),
            visibility = ["PUBLIC"],
            rpath = True,
            runtime_dependency_handling = "symlink",
            rustc_flags = select({
                "//:macos-arm64": [
                    "-Clink-arg=-Wl,-rpath,@loader_path/../../../../buck2/native/__openmp-build__/openmp/lib",
                ],
                "DEFAULT": [],
            }),
            target_compatible_with = ["prelude//os/constraints:macos"] if "metal" in program["required_features"] else [],
        )
