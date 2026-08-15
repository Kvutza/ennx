PYTHON_VERSION = read_config("ennx", "python_version", "3.13")

_PYTHON_CONFIGS = {
    "3.12": struct(
        abi = "cp312",
        environment = "ennx-py312",
        extension_infix = "312",
    ),
    "3.13": struct(
        abi = "cp313",
        environment = "ennx",
        extension_infix = "313",
    ),
}

if PYTHON_VERSION not in _PYTHON_CONFIGS:
    fail("ennx.python_version must be 3.12 or 3.13, got {}".format(PYTHON_VERSION))

_PYTHON = _PYTHON_CONFIGS[PYTHON_VERSION]

PYTHON_ABI = _PYTHON.abi
PYTHON_ENVIRONMENT = _PYTHON.environment
PYTHON_REQUIRES = ">=3.12,<3.14"

def python_extension_suffix(platform):
    if platform == "linux-aarch64":
        return ".cpython-{}-aarch64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "linux-x86_64":
        return ".cpython-{}-x86_64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "macos-arm64":
        return ".cpython-{}-darwin.so".format(_PYTHON.extension_infix)
    fail("unsupported Python extension platform: {}".format(platform))
