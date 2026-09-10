from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def _rssstress(tmp_path, num_obs: int, *, num_dim: int = 10, query_n: int = 50):
    from ops.stress import DiskRssStressConfig, rss_stress

    work_dir = tmp_path / f"enn_disk_rss_{num_obs}"
    return rss_stress(
        num_obs=num_obs,
        work_dir=str(work_dir),
        config=DiskRssStressConfig(num_dim=num_dim, query_n=query_n, batch_size=500),
    )


def _rssceiling(num_dim: int):
    from ops.stress import DEFAULT_SHARD_MAX_ROWS, rss_bytes2

    return rss_bytes2(num_dim=num_dim, shard_max_rows=DEFAULT_SHARD_MAX_ROWS)


def _rssceiling2(result, *, num_dim: int, num_obs: int) -> None:
    ceiling = _rssceiling(num_dim)
    assert result.rss_delta_bytes < ceiling, (
        f"RSS delta {result.rss_delta_bytes} >= ceiling {ceiling} "
        f"(baseline={result.baseline_rss_bytes} final={result.final_rss_bytes})"
    )
    assert result.index_bytes >= 0
    assert result.num_obs == num_obs
    print(
        "disk_rss_stress "
        f"N={num_obs} delta={result.rss_delta_bytes} "
        f"ceiling={ceiling} index_mem={result.index_bytes}"
    )


@pytest.mark.parametrize("num_obs", [1_000, 2_000])
def test_001(tmp_path, num_obs: int):
    num_dim = 10
    result = _rssstress(tmp_path, num_obs, num_dim=num_dim)
    _rssceiling2(result, num_dim=num_dim, num_obs=num_obs)


def test_002(tmp_path):
    num_dim = 10
    num_obs = 1_000
    result = _rssstress(tmp_path, num_obs, num_dim=num_dim)
    expected_train_x = num_obs * num_dim * 8
    assert abs(result.train_x_bytes - expected_train_x) <= num_dim * 8
    _rssceiling2(result, num_dim=num_dim, num_obs=num_obs)


def test_003(tmp_path):
    """N just above default 250-row pending threshold still stays under RSS ceiling."""
    num_dim = 10
    num_obs = 251
    result = _rssstress(tmp_path, num_obs, num_dim=num_dim)
    _rssceiling2(result, num_dim=num_dim, num_obs=num_obs)


def test_004(tmp_path):
    """RSS delta at N=2000 should stay below the same ceiling as N=1000."""
    num_dim = 10
    ceiling = _rssceiling(num_dim)
    deltas = [
        _rssstress(tmp_path, num_obs, num_dim=num_dim).rss_delta_bytes
        for num_obs in (1_000, 2_000)
    ]
    assert all(delta < ceiling for delta in deltas)
    print(f"disk_rss_metamorphic deltas={deltas} ceiling={ceiling}")
