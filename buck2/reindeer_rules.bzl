load("@prelude//rust:cargo_package.bzl", "cargo")
load("@prelude//rust:cargo_buildscript.bzl", _buildscript_run = "buildscript_run")
load("//buck2:python.bzl", "PYTHON_VERSION")

_PROFILE = read_config("ennx", "profile", "dev")

_PROFILE_FLAGS = {
    "dev": [
        "-Copt-level=1",
        "-Cdebug-assertions=on",
        "-Coverflow-checks=on",
    ],
    "release": [
        "-Copt-level=3",
        "-Cdebug-assertions=off",
        "-Coverflow-checks=off",
    ],
}

_PYO3_CONFIG = {
    "3.12": "696d706c656d656e746174696f6e3d43507974686f6e0a76657273696f6e3d332e31320a7368617265643d747275650a706f696e7465725f77696474683d36340a6275696c645f666c6167733d0a73757070726573735f6275696c645f7363726970745f6c696e6b5f6c696e65733d66616c73650a",
    "3.13": "696d706c656d656e746174696f6e3d43507974686f6e0a76657273696f6e3d332e31330a7368617265643d747275650a706f696e7465725f77696474683d36340a6275696c645f666c6167733d0a73757070726573735f6275696c645f7363726970745f6c696e6b5f6c696e65733d66616c73650a",
    "3.14": "696d706c656d656e746174696f6e3d43507974686f6e0a76657273696f6e3d332e31340a7368617265643d747275650a706f696e7465725f77696474683d36340a6275696c645f666c6167733d0a73757070726573735f6275696c645f7363726970745f6c696e6b5f6c696e65733d66616c73650a",
}

if _PROFILE not in _PROFILE_FLAGS:
    fail("ennx.profile must be dev or release, got {}".format(_PROFILE))

def profile_rustc_flags():
    return _PROFILE_FLAGS[_PROFILE]

def app_rust_library(name, rustc_flags = None, **kwargs):
    cargo.rust_library(
        name = name,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        **kwargs
    )

def app_rust_binary(name, rustc_flags = None, **kwargs):
    cargo.rust_binary(
        name = name,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        **kwargs
    )

def reindeer_rust_library(
        name,
        crate,
        rustc_flags = None,
        visibility = None,
        **kwargs):
    cargo.rust_library(
        name = name,
        crate = crate,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        visibility = visibility,
        **kwargs
    )

def reindeer_rust_binary(name, rustc_flags = None, **kwargs):
    cargo.rust_binary(
        name = name,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        **kwargs
    )

def buildscript_run(package_name, env = None, **kwargs):
    resolved_env = dict(env or {})
    if package_name == "pyo3-ffi":
        resolved_env["PYO3_CROSS"] = "1"
        resolved_env["PYO3_CROSS_PYTHON_VERSION"] = PYTHON_VERSION
    if package_name in ["numpy", "pyo3"]:
        resolved_env["DEP_PYTHON_PYO3_CONFIG"] = _PYO3_CONFIG[PYTHON_VERSION]
    _buildscript_run(
        package_name = package_name,
        env = resolved_env,
        **kwargs
    )
