"""Exercise build setup in an isolated checkout with no developer tools on PATH."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ennx-bootstrap-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (
            "ennx",
            "tools/build-wheels",
            "tools/pixiw",
            "tools/pixi",
        ):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in (
            "bash",
            "sh",
            "cat",
            "dirname",
            "mkdir",
            "mktemp",
            "rm",
            "cp",
            "basename",
        ):
            (self.bin / name).symlink_to(shutil.which(name))
        self.script("bin/uname", 'case "$1" in -s) echo Darwin;; -m) echo arm64;; esac')
        self.script(
            "tools/buck2-wheel-verify", 'echo "verify:$ENNX_PYTHON_VERSION" >> events'
        )
        self.script(
            "buck2w",
            """
echo build >> events
while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-report) report=$2; shift;;
    esac
    shift
done
: > wheel.whl
echo wheel.whl > "$report"
""",
        )
        # The interpreter stub handles the readiness probe and report extraction.
        self.script(
            "python-stub",
            """
case "$2" in
    3.*) [ ! -f "$0.broken" ]; exit $?;;
    *) echo wheel.whl;;
esac
""",
        )
        self.script(
            "fake-pixi",
            """
echo "pixi:$*" >> events
[ "${FAIL_INSTALL:-0}" = 0 ] || exit 42
environment=$3
bin=".pixi/envs/$environment/bin"
mkdir -p "$bin"
cp python-stub "$bin/python"
rm -f "$bin/python.broken"
""",
        )
        self.script(
            "tools/bootstrap-dotslash",
            """
echo dotslash >> events
mkdir -p .buck2-tools/bin
cp dotslash-stub .buck2-tools/bin/dotslash
""",
        )
        self.script("dotslash-stub", 'shift\nexec ./fake-pixi "$@"')
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("ENNX_")}
        self.env["PATH"] = str(self.bin)

    def script(self, name, body):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def prepare_python(self):
        for name in ("ennx-py312", "ennx-py314", "ennx"):
            directory = self.root / ".pixi/envs" / name / "bin"
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / "python-stub", directory / "python")

    def run_build(self, *args):
        return subprocess.run(
            [str(self.root / "ennx"), "build", *args],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def events(self):
        path = self.root / "events"
        return path.read_text().splitlines() if path.exists() else []

    def test_fresh_build_bootstraps_before_compiling_and_reuses_environments(self):
        first = self.run_build("--out", "output with spaces")
        self.assertEqual(first.returncode, 0, first.stderr)
        events = self.events()
        installs = [line for line in events if line.startswith("pixi:")]
        self.assertEqual(
            installs,
            [
                f"pixi:install --environment {name}"
                for name in ("ennx-py312", "ennx-py314", "ennx")
            ],
        )
        self.assertLess(events.index(installs[-1]), events.index("build"))
        self.assertEqual(events.count("build"), 3)
        self.assertEqual(
            [line for line in events if line.startswith("verify:")],
            ["verify:3.12", "verify:3.14", "verify:3.13"],
        )
        second = self.run_build()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.events()[len(events) :].count("build"), 3)
        self.assertFalse(
            any(
                "pixi" in line or "dotslash" in line
                for line in self.events()[len(events) :]
            )
        )

    def test_existing_pixi_is_used(self):
        (self.bin / "pixi").symlink_to(self.root / "fake-pixi")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("dotslash", self.events())

    def test_install_failure_stops_before_compilation(self):
        self.env["FAIL_INSTALL"] = "1"
        result = self.run_build()
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertNotIn("build", self.events())

    def test_incomplete_environment_is_repaired(self):
        self.prepare_python()
        (self.root / ".pixi/envs/ennx-py314/bin/python.broken").touch()
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [x for x in self.events() if x.startswith("pixi:")],
            ["pixi:install --environment ennx-py314"],
        )

    def test_invalid_override_is_not_replaced(self):
        self.prepare_python()
        self.env["ENNX_PYTHON_312"] = "missing-python"
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ENNX_PYTHON_312", result.stderr)
        self.assertEqual(self.events(), [])

    def test_wrong_python_version_fails_before_compilation(self):
        self.prepare_python()
        version = "312" if sys.version_info[:2] != (3, 12) else "314"
        self.env[f"ENNX_PYTHON_{version}"] = sys.executable
        self.env["PYTHONOPTIMIZE"] = "1"
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"ENNX_PYTHON_{version}", result.stderr)
        self.assertEqual(self.events(), [])

    def test_override_pointing_at_default_environment_is_not_modified(self):
        self.prepare_python()
        self.env["ENNX_PYTHON_312"] = ".pixi/envs/ennx-py312/bin/python"
        marker = self.root / ".pixi/envs/ennx-py312/bin/python.broken"
        marker.touch()
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(marker.exists())
        self.assertEqual(self.events(), [])

    def test_valid_override_with_spaces_is_reused(self):
        self.prepare_python()
        override = self.root / "custom python"
        shutil.copy2(self.root / "python-stub", override)
        self.env["ENNX_PYTHON_312"] = str(override)
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(line.startswith("pixi:") for line in self.events()))

    def test_generic_override_is_reused(self):
        self.env["ENNX_PYTHON"] = str(self.root / "python-stub")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events().count("build"), 3)
        self.assertFalse(any(line.startswith("pixi:") for line in self.events()))

    def test_linux_wheel_tools_use_each_verifier_environment(self):
        self.prepare_python()
        self.script("bin/uname", 'case "$1" in -s) echo Linux;; -m) echo x86_64;; esac')
        for name in ("ennx-py312", "ennx-py314", "ennx"):
            self.script(
                f".pixi/envs/{name}/bin/auditwheel",
                'echo "auditwheel:$0:$1" >> events',
            )
        self.script(
            "tools/buck2-wheel-verify",
            'echo "verifier-tool:$(command -v auditwheel)" >> events',
        )
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [
                line.replace(str(self.root) + "/", "").replace("./.pixi", ".pixi")
                for line in self.events()
                if line != "build"
            ],
            [
                event
                for name in ("ennx-py312", "ennx-py314", "ennx")
                for event in (
                    f"auditwheel:.pixi/envs/{name}/bin/auditwheel:repair",
                    f"auditwheel:.pixi/envs/{name}/bin/auditwheel:show",
                    f"verifier-tool:.pixi/envs/{name}/bin/auditwheel",
                )
            ],
        )

    def test_version_override_takes_precedence(self):
        self.prepare_python()
        self.env["ENNX_PYTHON"] = str(self.root / "python-stub")
        self.env["ENNX_PYTHON_312"] = "missing-python"
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ENNX_PYTHON_312", result.stderr)
        self.assertEqual(self.events(), [])

    def test_help_and_invalid_options_do_not_bootstrap(self):
        self.assertEqual(self.run_build("--help").returncode, 0)
        for args in (
            ("--bad",),
            ("--out",),
            ("--out", ""),
            ("--out", "--tests"),
            ("--tests", "--tests"),
            ("--out", "dist", "--out", "other"),
        ):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_build(*args).returncode, 0)
                self.assertEqual(self.events(), [])


if __name__ == "__main__":
    unittest.main()
