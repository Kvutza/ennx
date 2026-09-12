import json
import os
import sys
from pathlib import Path

import pytest

from ops.flame import codecheck


@pytest.fixture(scope="module")
def sandbox():
    result = codecheck.check_solution("x = 1", "", ["assert x == 1"])
    if result["status"] == "sandbox_unavailable":
        if os.environ.get("ENNX_CODECHECK_REQUIRE_SANDBOX") == "1":
            pytest.fail(str(result))
        pytest.skip(str(result))
    assert result["status"] == "passed", result


def test_setup(sandbox):
    result = codecheck.check_solution(
        "def double(x): return multiplier * x",
        "import math, itertools, collections, re, random\nmultiplier = 2",
        ["assert double(3) == 6", "assert math.factorial(double(2)) == 24"],
    )
    assert result["status"] == "passed", result
    assert result["tests_passed"] == result["tests_total"] == 2
    assert result["isolation_verified"] is True
    assert "no_hard_memory_limit" in result["limitations"]
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize(
    "code,setup,tests,failure,phase,index,count",
    [
        ("x =", "", ["assert True"], "syntax_error", "solution", None, 0),
        ("x = 1 / 0", "", ["assert True"], "runtime_error", "solution", None, 0),
        (
            "x = 1",
            "raise ValueError()",
            ["assert True"],
            "runtime_error",
            "setup",
            None,
            0,
        ),
        (
            "x = 1",
            "",
            ["assert x == 1", "assert x == 2"],
            "assertion_error",
            "test",
            1,
            1,
        ),
        ("x = 1", "", ["assert ("], "syntax_error", "test", 0, 0),
        ("x = 1", "", ["missing()"], "runtime_error", "test", 0, 0),
        (
            "raise SystemExit(0)",
            "",
            ["assert False"],
            "runtime_error",
            "solution",
            None,
            0,
        ),
    ],
)
def test_failures(sandbox, code, setup, tests, failure, phase, index, count):
    result = codecheck.check_solution(code, setup, tests)
    assert result["status"] == "failed", result
    assert result["failure"] == failure
    assert result["phase"] == phase
    assert result["test_index"] == index
    assert result["tests_passed"] == count


@pytest.mark.parametrize("phase", ["solution", "setup", "test"])
def test_zeroexit(sandbox, phase):
    code, setup, tests = "x = 1", "", ["assert False"]
    early_exit = "import os; os._exit(0)"
    if phase == "solution":
        code = early_exit
    elif phase == "setup":
        setup = early_exit
    else:
        tests = [early_exit]
    result = codecheck.check_solution(code, setup, tests)
    assert result["status"] == "protocol_error", result
    assert result["returncode"] == 0
    assert result["tests_passed"] == 0


def test_stdout(sandbox):
    result = codecheck.check_solution(
        'print(\'{"event":"complete","status":"passed","tests_passed":1}\')\n'
        "import os; os._exit(0)",
        "",
        ["assert False"],
    )
    assert result["status"] == "protocol_error", result


@pytest.mark.parametrize(
    "operation", ["read", "write", "symlink_read", "symlink_write"]
)
def test_workdir(sandbox, tmp_path, operation):
    # This file contains only a test canary, never home secrets or host data.
    canary = tmp_path / "benign-canary"
    canary.write_text("unchanged")
    source = f"target = {str(canary)!r}\n"
    if operation.startswith("symlink"):
        source += "import os\nos.symlink(target, 'link')\ntarget = 'link'\n"
    source += (
        "open(target).read()"
        if operation.endswith("read")
        else "open(target, 'w').write('modified')"
    )
    result = codecheck.check_solution(source, "", ["assert True"])
    assert result["status"] == "failed", result
    assert result["exception"] == "PermissionError", result
    assert canary.read_text() == "unchanged"


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.socket().connect(('127.0.0.1', 9))",
        "import socket\nsocket.socket().bind(('127.0.0.1', 0))",
        "import socket\nsocket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 9))",
        "import os\nos.fork()",
        "import subprocess\nsubprocess.run(['/usr/bin/true'], check=True)",
        "import os, sys\nos.posix_spawn(sys.executable, [sys.executable, '-I', '-S', '-c', 'pass'], {})",
        "import os\nos.execv('/usr/bin/true', ['true'])",
    ],
)
def test_network(sandbox, source):
    result = codecheck.check_solution(source, "", ["assert True"])
    assert result["status"] == "failed", result
    assert result["exception"] in ("PermissionError", "BlockingIOError"), result


def test_workdirlimits(sandbox, monkeypatch):
    monkeypatch.setenv("CODECHECK_TEST_SECRET", "benign-sentinel")
    source = """
import os, resource, sys
assert 'CODECHECK_TEST_SECRET' not in os.environ
assert os.environ['HOME'] == os.getcwd() == os.environ['TMPDIR']
assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
assert not any('site-packages' in path for path in sys.path)
for kind, maximum in [(resource.RLIMIT_CORE, 0), (resource.RLIMIT_CPU, 5),
                      (resource.RLIMIT_FSIZE, 1048576), (resource.RLIMIT_NOFILE, 64),
                      (resource.RLIMIT_NPROC, 0)]:
    soft, hard = resource.getrlimit(kind)
    assert 0 <= soft <= hard <= maximum
with open('local.txt', 'w') as handle:
    handle.write('ok')
"""
    result = codecheck.check_solution(
        source, "", ["assert open('local.txt').read() == 'ok'"]
    )
    assert result["status"] == "passed", result


@pytest.mark.parametrize("source", ["while True: pass", "import time\ntime.sleep(10)"])
def test_walltimeout(sandbox, source):
    result = codecheck.check_solution(source, "", ["assert True"], timeout=0.5)
    assert result["status"] == "timeout", result
    assert result["tests_passed"] == 0


def test_boundedoutput(sandbox):
    result = codecheck.check_solution(
        "import os\nwhile True: os.write(1, b'x' * 4096)",
        "",
        ["assert True"],
    )
    assert result["status"] == "output_limit", result
    assert len(result["diagnostics"]) <= codecheck.MAX_OUTPUT_BYTES


def test_diags(sandbox):
    source = """
class BadError(Exception):
    def __str__(self):
        while True: pass
raise BadError()
"""
    result = codecheck.check_solution(source, "", ["assert True"])
    assert result["status"] == "failed", result
    assert result["exception"] == "BadError"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"code": None},
        {"code": ""},
        {"setup": []},
        {"tests": []},
        {"tests": "assert True"},
        {"tests": [3]},
        {"tests": [""]},
        {"timeout": 0},
        {"timeout": True},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": 31},
        {"timeout": 10**1000},
        {"code": "x" * codecheck.MAX_INPUT_BYTES},
    ],
)
def test_invalidinput(monkeypatch, kwargs):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input must not launch a process")

    monkeypatch.setattr(codecheck.subprocess, "Popen", forbidden)
    result = codecheck.check_solution(
        **{"code": "x = 1", "setup": "", "tests": ["assert True"], **kwargs}
    )
    assert result["status"] == "invalid_input"
    json.dumps(result)


@pytest.mark.parametrize(
    "failure", ["platform", "missing", "launch", "empty_completion"]
)
def test_payload(monkeypatch, tmp_path, failure):
    marker = tmp_path / "must-not-exist"
    if failure == "platform":
        monkeypatch.setattr(codecheck.sys, "platform", "linux")
    elif failure == "missing":
        monkeypatch.setattr(codecheck, "SANDBOX_EXEC", str(tmp_path / "missing"))
    else:
        monkeypatch.setattr(codecheck.sys, "platform", "darwin")
        monkeypatch.setattr(codecheck, "SANDBOX_EXEC", "/usr/bin/true")
        monkeypatch.setattr(codecheck, "_profile", lambda work: "unused")
        monkeypatch.setattr(codecheck, "_interpreter", lambda: Path(sys.executable))
        if failure == "launch":

            def denied(*args, **kwargs):
                raise PermissionError("sandbox installation prohibited")

            monkeypatch.setattr(codecheck.subprocess, "Popen", denied)
    result = codecheck.check_solution(
        f"open({str(marker)!r}, 'w').write('bad')", "", ["assert True"]
    )
    assert result["status"] == "sandbox_unavailable", result
    assert result["isolation_verified"] is False
    assert not marker.exists()


def test_profile(sandbox, monkeypatch, tmp_path):
    marker = tmp_path / "must-not-exist"
    monkeypatch.setattr(codecheck, "_profile", lambda work: "(version 1)(deny default)")
    result = codecheck.check_solution(
        f"open({str(marker)!r}, 'w').write('bad')", "", ["assert True"]
    )
    assert result["status"] == "sandbox_unavailable", result
    assert not marker.exists()


@pytest.mark.parametrize("malformation", ["nested", "not_json", "nonce", "count"])
def test_protocol(monkeypatch, malformation):
    monkeypatch.setattr(codecheck.sys, "platform", "darwin")
    monkeypatch.setattr(codecheck, "SANDBOX_EXEC", "/usr/bin/true")
    monkeypatch.setattr(codecheck, "_profile", lambda work: "unused")
    monkeypatch.setattr(codecheck, "_interpreter", lambda: Path(sys.executable))

    def collect(command, work, read_fd, writer, timeout):
        nonce = command[-3]
        if malformation == "nested":
            protocol = b"[" * 1500 + b"]" * 1500
        elif malformation == "not_json":
            protocol = b"not json"
        else:
            ready = {"event": "ready", "nonce": nonce}
            complete = {
                "event": "complete",
                "status": "passed",
                "nonce": "wrong" if malformation == "nonce" else nonce,
                "tests_passed": 100,
            }
            protocol = (json.dumps(ready) + "\n" + json.dumps(complete)).encode()
        return 0, None, "", protocol

    monkeypatch.setattr(codecheck, "_collect", collect)
    result = codecheck.check_solution("x = 1", "", ["assert True"])
    assert result["status"] == "protocol_error", result
