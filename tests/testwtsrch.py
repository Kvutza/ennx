import sys

import numpy as np
import pytest
from ennx.ennx_rust import optimizer
from ennx.search import Parameter, Search


def test_parameter():
    parameter = Parameter(
        offset=0, length=4, encoding="int4", scale=0.25, weight=0.5, radius=0.75
    )
    assert (parameter.offset, parameter.length, parameter.bits) == (0, 4, 4)
    assert (
        parameter.encoding,
        parameter.scale,
        parameter.weight,
        parameter.radius,
    ) == ("int4", 0.25, 0.5, 0.75)
    assert 'encoding="int4"' in repr(parameter)
    with pytest.raises(AttributeError):
        parameter.radius = 2.0
    for kwargs in [
        {"length": 0},
        {"encoding": "unknown"},
        {"scale": 0.0},
        {"weight": float("nan")},
        {"radius": -1.0},
    ]:
        options = {"offset": 0, "length": 4}
        options.update(kwargs)
        with pytest.raises(ValueError):
            Parameter(**options)


def test_tuple():
    with pytest.raises(TypeError):
        Search(
            np.asarray([0], dtype=np.uint8), 0.0, [(0, 1, 8, 1.0, 1.0, 1.0)], 2, "cpu"
        )


def _leaves():
    return [
        Parameter(
            offset=0, length=257, encoding="int4", scale=0.25, weight=1.0, radius=0.75
        ),
        Parameter(
            offset=257, length=263, encoding="int8", scale=0.5, weight=0.5, radius=1.0
        ),
    ]


def _base():
    row_bytes = (257 + 1) // 2 + 263
    return np.asarray(
        [(index * 37 + 11) & 0xFF for index in range(row_bytes)],
        dtype=np.uint8,
    )


def _ask(backend):
    search = Search(_base(), 0.25, _leaves(), 4, backend)
    trial = search.ask(np.asarray([17], dtype=np.uint64), 1.0, 1)
    search.tell(trial, 0.75, True)
    trial = search.ask(
        np.asarray([19, 23, 29, 31], dtype=np.uint64),
        0.65,
        2,
        beta=1.3,
    )
    return trial.index, trial.seed, trial.score, np.asarray(search.row(trial))


def test_001():
    cpu = _ask("cpu")
    assert cpu[0] in range(4)
    assert cpu[1] in {19, 23, 29, 31}
    assert np.isfinite(cpu[2])
    assert cpu[3].shape == _base().shape
    assert not np.array_equal(cpu[3], _base())


def test_002():
    with pytest.raises(ValueError, match="unknown compute device"):
        Search(_base(), 0.25, _leaves(), 4, "agx")


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend requires macOS")
def test_003():
    cpu = _ask("cpu")
    try:
        metal = _ask("metal")
    except ValueError as error:
        message = str(error)
        if (
            "Metal trial search is not available in this build" in message
            or "no default Metal device found" in message
        ):
            pytest.skip(message)
        raise
    assert metal[:2] == cpu[:2]
    assert np.isclose(metal[2], cpu[2], atol=1.0e-5)
    assert np.array_equal(metal[3], cpu[3])


def test_004(tmp_path):
    history = optimizer.BpannHistory(str(tmp_path / "history"), 2)
    assert history.append(np.asarray([0.0, 0.0]), 10.0) == 0
    assert history.append(np.asarray([1.0, 0.0]), 20.0) == 1
    assert history.append(np.asarray([4.0, 0.0]), 40.0) == 2
    history.sync()

    queries = np.asarray([[0.1, 0.0], [3.9, 0.0]])
    assert history.search(queries, 1) == [[0], [2]]
    assert history.shortlist(queries, 1, 2) == [(0, 10.0), (2, 40.0)]


def test_005():
    base = _base()
    rows = np.stack(
        [
            np.bitwise_xor(base, np.uint8(0x11)),
            np.bitwise_xor(base, np.uint8(0x22)),
        ]
    )
    search = Search(base, 0.25, _leaves(), 4, "cpu")
    search.replace_history(rows, np.asarray([3.0, 7.0], dtype=np.float32))
    assert search.history_len == 2
    assert search.history_capacity == 4
    trial = search.ask(
        np.asarray([19, 23], dtype=np.uint64),
        0.65,
        2,
    )
    assert trial.index in range(2)
    assert trial.seed in {19, 23}
    assert np.isfinite(trial.score)
    search.tell(trial, 0.5, False)
    trial = search.ask_stream(19, 2, 0.65, 2)
    assert trial.index in range(2)
    assert np.isfinite(trial.score)
    assert search.row(trial).shape == base.shape
    search.tell(trial, 0.75, False)


@pytest.mark.parametrize("acquisition", ["ucb", "thompson"])
def test_stream(acquisition):
    from ennx.search import Optimizer

    def create():
        return Optimizer(_base(), 0.25, _leaves(), 8, "cpu", num_pert=2)

    first, second = create(), create()
    with pytest.raises(ValueError):
        first.ask_stream(17, 0, 1)
    for value in [0.5, 0.75]:
        left = first.ask_stream(17, 8, 1, acquisition=acquisition, seed=23)
        right = second.ask_stream(17, 8, 1, acquisition=acquisition, seed=23)
        assert (left.index, left.seed, left.score) == (
            right.index,
            right.seed,
            right.score,
        )
        np.testing.assert_array_equal(first.row(left), second.row(right))
        assert first.tell(left, value) == second.tell(right, value)
    assert first.history_len == second.history_len == 3


def test_batch():
    from ennx.search import Optimizer

    search = Optimizer(_base(), 0.25, _leaves(), 8, "cpu", num_pert=2, max_pending=2)
    with pytest.raises(ValueError):
        search.batch_stream(17, 3, 8, 1)
    trials = search.batch_stream(17, 2, 8, 1)
    assert len(trials) == 2
    with pytest.raises(ValueError):
        search.ask_stream(19, 8, 1)
    for trial in reversed(trials):
        assert search.row(trial).shape == _base().shape
        search.tell(trial, 0.5)
    trial = search.ask_stream(23, 8, 1)
    search.tell(trial, 0.75)
    assert search.history_len == 4


def test_identity():
    from ennx.search import Optimizer, Trial

    def create():
        return Optimizer(_base(), 0.25, _leaves(), 8, "cpu", num_pert=2, max_pending=2)

    first, second = create(), create()
    left = first.ask_stream(17, 8, 1)
    right = second.ask_stream(17, 8, 1)
    assert isinstance(left, Trial)
    assert not hasattr(left, "length")
    assert not hasattr(left, "probability")
    for operation in [lambda: first.row(right), lambda: first.tell(right, 1.0)]:
        with pytest.raises(ValueError, match="outstanding ask"):
            operation()
    assert first.history_len == 1
    first.tell(left, 0.5)
    with pytest.raises(ValueError, match="outstanding ask"):
        first.tell(left, 1.0)
    second.tell(right, 0.5)

    trials = first.batch_stream(23, 2, 8, 1)
    for invalid, values in [
        ([trials[0], trials[0]], [0.75, 1.0]),
        (trials, [0.75, float("nan")]),
        ([trials[0], right], [0.75, 1.0]),
    ]:
        with pytest.raises(ValueError):
            first.tell_batch(invalid, np.asarray(values, dtype=np.float32))
        assert first.history_len == 2
        for trial in trials:
            assert first.row(trial).shape == _base().shape
    first.tell_batch(trials, np.asarray([0.75, 1.0], dtype=np.float32))
    assert first.history_len == 4


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal backend requires macOS")
def test_metal():
    from ennx.search import Optimizer

    cpu = Optimizer(_base(), 0.25, _leaves(), 8, "cpu", num_pert=2)
    try:
        metal = Optimizer(_base(), 0.25, _leaves(), 8, "metal", num_pert=2)
    except ValueError as error:
        if "Metal trial search is not available in this build" in str(
            error
        ) or "no default Metal device found" in str(error):
            pytest.skip(str(error))
        raise
    for value in [0.5, 0.75]:
        left = cpu.ask_stream(17, 8, 1)
        right = metal.ask_stream(17, 8, 1)
        assert (left.index, left.seed) == (right.index, right.seed)
        assert np.isclose(left.score, right.score, atol=1.0e-5)
        np.testing.assert_array_equal(cpu.row(left), metal.row(right))
        assert cpu.tell(left, value) == metal.tell(right, value)
