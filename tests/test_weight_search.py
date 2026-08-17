import sys

import numpy as np
import pytest
from ennx.ennx_rust import optimizer


def _leaves():
    return [
        (0, 257, 4, 0.25, 1.0, 0.75),
        (257, 263, 8, 0.5, 0.5, 1.0),
    ]


def _base():
    row_bytes = (257 + 1) // 2 + 263
    return np.asarray(
        [(index * 37 + 11) & 0xFF for index in range(row_bytes)],
        dtype=np.uint8,
    )


def _ask(backend):
    search = optimizer.PackedSearch(_base(), 0.25, _leaves(), 4, backend)
    _, _, _ = search.ask(np.asarray([17], dtype=np.uint64), 1.0, 1)
    search.tell(0.75, True)
    index, seed, score = search.ask(
        np.asarray([19, 23, 29, 31], dtype=np.uint64),
        0.65,
        2,
        beta=1.3,
    )
    return index, seed, score, np.asarray(search.row())


def test_weight_search_keeps_state_across_ask_and_tell():
    cpu = _ask("cpu")
    assert cpu[0] in range(4)
    assert cpu[1] in {19, 23, 29, 31}
    assert np.isfinite(cpu[2])
    assert cpu[3].shape == _base().shape
    assert not np.array_equal(cpu[3], _base())


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend requires macOS")
def test_weight_search_metal_matches_cpu():
    cpu = _ask("cpu")
    metal = _ask("metal")
    assert metal[:2] == cpu[:2]
    assert np.isclose(metal[2], cpu[2], atol=1.0e-5)
    assert np.array_equal(metal[3], cpu[3])


def test_bpann_history_shortlists_stable_observation_ids(tmp_path):
    history = optimizer.BpannHistory(str(tmp_path / "history"), 2)
    assert history.append(np.asarray([0.0, 0.0]), 10.0) == 0
    assert history.append(np.asarray([1.0, 0.0]), 20.0) == 1
    assert history.append(np.asarray([4.0, 0.0]), 40.0) == 2
    history.sync()

    queries = np.asarray([[0.1, 0.0], [3.9, 0.0]])
    assert history.search(queries, 1) == [[0], [2]]
    assert history.shortlist(queries, 1, 2) == [(0, 10.0), (2, 40.0)]


def test_weight_search_accepts_bpann_resolved_history():
    base = _base()
    rows = np.stack(
        [
            np.bitwise_xor(base, np.uint8(0x11)),
            np.bitwise_xor(base, np.uint8(0x22)),
        ]
    )
    search = optimizer.PackedSearch(base, 0.25, _leaves(), 4, "cpu")
    search.replace_history(rows, np.asarray([3.0, 7.0], dtype=np.float32))
    assert search.history_len == 2
    assert search.history_capacity == 4
    index, seed, score = search.ask(
        np.asarray([19, 23], dtype=np.uint64),
        0.65,
        2,
    )
    assert index in range(2)
    assert seed in {19, 23}
    assert np.isfinite(score)
