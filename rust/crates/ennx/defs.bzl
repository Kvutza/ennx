load("@crates//:defs.bzl", "aliases", "crate_deps", "crate_edition")
load("@rules_rust//rust:defs.bzl", "rust_binary")

def ennx_crate_deps(deps):
    return crate_deps(deps)

def _local_program_dep(name):
    if name == "ennx":
        return "//rust/crates/ennx:ennx_accelerators"
    return "//rust/crates/" + name + ":" + name

def ennx_program_targets(programs, sources):
    """Declare the Bazel half of the shared logical program catalogue."""
    for program in programs:
        if "cuda" in program["required_features"]:
            continue
        rust_binary(
            name = program["name"],
            srcs = sources,
            aliases = aliases(),
            crate_name = program["crate"],
            crate_root = program["root"],
            edition = crate_edition(),
            rustc_env = {"CARGO_PKG_VERSION": program["version"]},
            target_compatible_with = ["@platforms//os:macos"] if "metal" in program["required_features"] else [],
            deps = [_local_program_dep(name) for name in program["local_deps"]] + crate_deps(program["crate_deps"]),
        )
