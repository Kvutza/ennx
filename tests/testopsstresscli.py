from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.slow

_STRESS_ROW_RE = re.compile(r" *\d+ \d+\.\d{4} \d+\.\d{4}")


def test_001(monkeypatch):
    from click.testing import CliRunner

    from ennx.ennx.enn_fitter import ENNStatefulFitter
    from ops.stress import cli

    tell_calls: list[int] = []
    ask_calls: list[int] = []

    def _counttell2(self, *args, **kwargs):
        tell_calls.append(1)

    def _countask2(self, *args, **kwargs):
        ask_calls.append(1)

    monkeypatch.setattr(ENNStatefulFitter, "tell", _counttell2)
    monkeypatch.setattr(ENNStatefulFitter, "ask", _countask2)

    num_obs = 10
    result = CliRunner().invoke(cli, ["enn", "flat", str(num_obs)])
    assert result.exit_code == 0
    assert tell_calls == []
    assert ask_calls == []


def test_ennstresscli():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10"],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=10 num_obs=10"
    assert len(lines) == 4
    data_lines = lines[1:]
    for line in data_lines:
        assert _STRESS_ROW_RE.fullmatch(line)
    assert data_lines[0].startswith(" ")
    assert not data_lines[-1].startswith(" ")


def test_002():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10", "--work-dir", "/tmp/enn_work"],
    )
    assert result.exit_code != 0
    assert "work_dir requires index_type in" in result.output


def test_003():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "bpann_disk", "10"])
    assert result.exit_code != 0
    assert "bpann_disk requires --work-dir" in result.output


@pytest.mark.parametrize(
    "index_type,subdir",
    [("bpann_disk", "enn_cli_bpann")],
)
def test_ennstressclidisk(tmp_path, index_type, subdir):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / subdir
    result = CliRunner().invoke(
        cli,
        ["enn", index_type, "10", "--work-dir", str(work_dir)],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == f"num_dim=10 num_obs=10 work_dir={work_dir}"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)


def test_004(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_cli_batch"
    result = CliRunner().invoke(
        cli,
        ["enn", "--batch", "bpann_disk", "10", "--work-dir", str(work_dir)],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == f"num_dim=10 num_obs=10 work_dir={work_dir}"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)


def test_005():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "--index-type", "flat", "--num-obs", "10"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_006():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "10", "flat"])
    assert result.exit_code != 0


def test_007():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10", "--num-dim", "4"],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=4 num_obs=10"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)


def test_008():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "flat", "10", "--num-dim", "0"])
    assert result.exit_code != 0
    assert "num_dim must be >= 1" in result.output


def test_stressmain(monkeypatch):
    from ops.stress import main

    monkeypatch.setattr(
        "sys.argv",
        ["stress", "enn", "flat", "3"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
