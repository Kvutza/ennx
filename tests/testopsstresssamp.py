from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config.eidxdrv import ENNIndexDriver

pytestmark = pytest.mark.slow


def _bpannstore(work_dir, *, num_obs: int = 10, num_dim: int = 4):
    from ennx.ennx.enn_class import ENN
    from ops.stress import make_observations

    train_x, train_y = make_observations(num_obs, num_dim=num_dim, seed=0)
    ENN(
        train_x,
        train_y,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )


def test_001():
    from ops.stress import make_points2

    x = make_points2(50, num_dim=3, seed=7)
    assert x.shape == (50, 3)
    assert np.all(x >= -1.0)
    assert np.all(x <= 1.0)
    x2 = make_points2(50, num_dim=3, seed=7)
    np.testing.assert_allclose(x, x2)


def test_002(tmp_path):
    from ops.stress import SampleStressConfig, run_stress

    work_dir = tmp_path / "enn_sample"
    _bpannstore(work_dir, num_obs=10, num_dim=4)
    result = run_stress(
        work_dir=str(work_dir),
        config=SampleStressConfig(num_samples=5, seed=1),
    )
    assert result.num_dim == 4
    assert result.num_obs == 10
    assert result.num_samples == 5
    assert result.draws_shape == (5, 1, 1)
    assert result.all_finite
    assert result.init_s >= 0.0
    assert result.sample_s >= 0.0


def test_003(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_sample_cli"
    _bpannstore(work_dir, num_obs=10, num_dim=4)
    result = CliRunner().invoke(
        cli,
        ["sample", str(work_dir), "5", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        f"num_dim=4 num_obs=10 work_dir={work_dir} num_samples=5 seed=1"
    )
    assert lines[1].startswith(
        "draws_shape=(5, 1, 1) function_seeds=1 all_finite=true init_s="
    )
    assert " sample_s=" in lines[1]


def test_004():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["sample", "/nonexistent/enn_store", "5"],
    )
    assert result.exit_code != 0
    assert "work_dir does not exist" in result.output


def test_005():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["sample"])
    assert result.exit_code != 0


@pytest.mark.slow
def test_006(tmp_path):
    """After persist_index, 10k-row reopen init should be mmap-fast."""
    import time

    from ennx.ennx.enn_class import ENN
    from ops.stress import make_observations

    work_dir = tmp_path / "enn_persist_reopen_10k"
    num_obs = 10_000
    num_dim = 32
    model = ENN(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    batch = 500
    for start in range(0, num_obs, batch):
        end = min(start + batch, num_obs)
        xs, ys = make_observations(end - start, num_dim=num_dim, seed=0 + start)
        model.add(xs, ys)
        model.schedule_flush()
    model.persist_index()

    t0 = time.perf_counter()
    ENN(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    init_s = time.perf_counter() - t0
    assert init_s < 1.0, f"reopen init_s={init_s:.3f}s expected < 1.0s after persist"


def test_007(tmp_path):
    """After persist_index, reopen init should be mmap-fast (not O(n·dim) rebuild).

    Rebuild local ``_enn/`` fixtures with ``enn_stress`` after this lands so
    ``./ops/stress.py sample _enn/`` benchmarks reflect persisted indices.
    """
    import time

    from ennx.ennx.enn_class import ENN
    from ops.stress import make_observations

    work_dir = tmp_path / "enn_persist_reopen"
    num_obs = 2000
    num_dim = 32
    model = ENN(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    batch = 500
    for start in range(0, num_obs, batch):
        end = min(start + batch, num_obs)
        xs, ys = make_observations(end - start, num_dim=num_dim, seed=0 + start)
        model.add(xs, ys)
        model.schedule_flush()
    model.persist_index()

    t0 = time.perf_counter()
    ENN(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    init_s = time.perf_counter() - t0
    assert init_s < 1.0, f"reopen init_s={init_s:.3f}s expected < 1.0s after persist"


def test_008(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_sample_invalid"
    _bpannstore(work_dir)
    result = CliRunner().invoke(
        cli,
        ["sample", str(work_dir), "0"],
    )
    assert result.exit_code != 0
    assert "num_samples must be >= 1" in result.output
