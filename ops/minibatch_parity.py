"""GPU parity for native paired BF16 acceptance and incumbent DLPack leases.

Run with the CUDA wheel: python ops/minibatch_parity.py
"""

import gc
from functools import partial

import numpy as np
from correlated_parity import candidate, check_rounding, decode, read_proposals


def fixture(sampler="correlated", capacity=2, value=10.0, variance=0.125):
    import jax.numpy as jnp

    from ennx.experimental import ParamBlock, SearchState

    bits = np.resize(np.array([0x3FC0, 0xBFC0], dtype=np.uint16), 513)
    leaves = [(17, 0, bits.size, 1.0)]
    search = SearchState(
        jnp.asarray(bits).view(jnp.bfloat16),
        value,
        [ParamBlock(*leaf) for leaf in leaves],
        capacity,
        base_variance=variance,
        sampler=sampler,
        reference_seed=123,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
    )
    return search, bits, leaves


def ask(search, seed=42):
    return search.ask(
        1, 4, min(2, search.history_len), seed, acquisition="ucb", beta=0.5
    )


def fails(action, text=None, error_types=(ValueError, BufferError)):
    try:
        action()
    except error_types as error:
        if text is not None:
            assert text.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError("invalid operation unexpectedly succeeded")


def score(bits, history, epistemic_scale=0.7, aleatoric_scale=0.05):
    row = decode(bits).astype(np.float64)
    distances = [
        np.square(row - decode(item[0]).astype(np.float64)).sum() for item in history
    ]
    weights = 1.0 / (
        1e-9
        + epistemic_scale * np.asarray(distances)
        + aleatoric_scale
        + [item[2] for item in history]
    )
    return np.dot(
        weights, [item[1] for item in history]
    ) / weights.sum() + 0.5 / np.sqrt(weights.sum())


def ask_relative(search, seed=42, acquisition="ucb"):
    return search.ask(
        1,
        4,
        search.history_len,
        seed,
        epistemic_scale=1.0,
        aleatoric_scale=0.0,
        y_scale=1.0,
        acquisition=acquisition,
        beta=0.5,
        draw_seed=seed + 1000,
    )


def check_history():
    # All neighbors are used so the score observes every FIFO reward and variance.
    for capacity in (2, 3):
        search, base, _ = fixture(capacity=capacity)
        search.enable_relative(failure_tolerance=4)
        assert search.best == 10.0 and search.best_variance == 0.125
        history = [(base.copy(), 0.0, 0.0)]
        deltas = [-0.25, -0.5, -0.125, -0.75, 0.125, -0.25, -0.5, 0.25, -0.125]
        for step, delta in enumerate(deltas):
            proposals = ask_relative(search, 100 + step)
            actual = read_proposals(proposals)[0]
            np.testing.assert_allclose(
                proposals.describe()[0][1],
                score(actual, history, 1.0, 0.0),
                rtol=4e-5,
                atol=4e-5,
            )
            incumbent = 20.0 if step % 2 else -20.0
            paired_variance = (step + 1) / 32
            accept = delta > 0
            search.tell_relative(
                proposals,
                incumbent + delta,
                0.75,
                incumbent,
                0.5,
                delta,
                paired_variance,
                accept,
            )
            assert search.sync() == [accept]
            if accept:
                base = actual.copy()
                history = [(base.copy(), 0.0, 0.0)]
            else:
                rejected = [*history[1:], (actual.copy(), delta, paired_variance)]
                history = [history[0], *rejected[-(capacity - 1) :]]
            assert search.history_len == len(history)
            assert search.best == (incumbent + delta if accept else incumbent)
            assert search.best_variance == (0.75 if accept else 0.5)
            assert search.restarts == 0
            np.testing.assert_array_equal(search.read_best(), base)
        # Read the last insertion as well, including the new-center anchor.
        proposals = ask_relative(search, 200)
        actual = read_proposals(proposals)[0]
        np.testing.assert_allclose(
            proposals.describe()[0][1],
            score(actual, history, 1.0, 0.0),
            rtol=4e-5,
            atol=4e-5,
        )
        search.tell_relative(proposals, -21.0, 0.75, -20.0, 0.5, -1.0, 0.25, False)
        assert search.sync() == [False]


def check_offsets():
    for acquisition in ("ucb", "thompson"):
        left, _, _ = fixture(capacity=3)
        right, _, _ = fixture(capacity=3, value=110.0, variance=0.875)
        for search in (left, right):
            search.enable_relative(failure_tolerance=4)
        for step, delta in enumerate([-0.5, -0.25, -0.75, -0.125, 0.5, -0.25, -0.5]):
            a = ask_relative(left, 300 + step, acquisition)
            b = ask_relative(right, 300 + step, acquisition)
            assert a.geometry() == b.geometry()
            assert a.describe() == b.describe()
            np.testing.assert_array_equal(read_proposals(a), read_proposals(b))
            incumbent = -20.0 + step
            # Offsets change every batch; marginal variances deliberately differ too.
            offset = 100.0 if step % 2 else -40.0
            accept = delta > 0
            left.tell_relative(
                a, incumbent + delta, 0.5, incumbent, 0.75, delta, 0.0625, accept
            )
            right.tell_relative(
                b,
                incumbent + offset + delta,
                1.5,
                incumbent + offset,
                1.75,
                delta,
                0.0625,
                accept,
            )
            assert left.sync() == right.sync() == [accept]
            assert right.best - left.best == offset
            assert right.best_variance - left.best_variance == 1.0
            assert left.history_len == right.history_len
            assert left.length == right.length
            np.testing.assert_array_equal(left.read_best(), right.read_best())
            np.testing.assert_array_equal(left.read_reference(), right.read_reference())


def check_contraction():
    search, base, _ = fixture()
    search.enable_relative(failure_tolerance=4)
    reference = search.read_reference()
    for step in range(12):
        proposals = ask_relative(search, 400 + step)
        search.tell_relative(proposals, 9.0, 0.5, 10.0, 0.75, -1.0, 0.25, False)
        assert search.sync() == [False]
        assert search.length == max(0.03125, 0.125 / 2 ** ((step + 1) // 4))
        assert search.restarts == 0
        np.testing.assert_array_equal(search.read_best(), base)
        np.testing.assert_array_equal(search.read_reference(), reference)

    # An acceptance interrupts the failure streak; the next four failures start anew.
    search, _, _ = fixture()
    search.enable_relative(failure_tolerance=4)
    failures = 0
    for step in range(8):
        old_radius = search.length
        proposals = ask_relative(search, 500 + step)
        radius = proposals.describe()[0][2]
        accept = step == 3
        delta = 1.0 if accept else -1.0
        search.tell_relative(
            proposals, 10.0 + delta, 0.5, 10.0, 0.75, delta, 0.25, accept
        )
        assert search.sync() == [accept]
        failures = 0 if accept else failures + 1
        expected = radius if accept else old_radius
        if failures == 4:
            expected = max(0.03125, old_radius / 2)
            failures = 0
        assert search.length == expected and search.restarts == 0


def check_validation():
    valid = [11.0, 0.5, 10.0, 0.75, 1.0, 0.25, True]
    undersized, _, _ = fixture(capacity=1)
    fails(undersized.enable_relative)
    search, _, _ = fixture()
    for tolerance in (0, -1, 2**32):
        fails(
            partial(search.enable_relative, failure_tolerance=tolerance),
            error_types=(ValueError, OverflowError),
        )
    search.enable_relative(failure_tolerance=4)
    fails(search.enable_relative)
    proposals = ask_relative(search)
    for index in range(6):
        invalids = [float("nan"), float("inf"), -float("inf"), 1e100]
        if index in (1, 3, 5):
            invalids += [-0.1, -1e-100]
        for value in invalids:
            args = valid.copy()
            args[index] = value
            fails(
                partial(search.tell_relative, proposals, *args),
                error_types=(ValueError, OverflowError),
            )
    fails(partial(search.tell, proposals, [100.0]))
    fails(partial(search.tell_paired, proposals, *valid[:4], True))
    foreign_search, _, _ = fixture()
    foreign_search.enable_relative()
    foreign = ask_relative(foreign_search)
    fails(partial(search.tell_relative, foreign, *valid))
    search.tell_relative(proposals, *valid)
    fails(partial(search.tell_relative, proposals, *valid))
    assert search.sync() == [True] and search.sync() == []
    fails(search.enable_relative)
    foreign_search.tell_relative(foreign, *valid)
    assert foreign_search.sync() == [True]
    for sampler in ("independent", "gaussian"):
        state, _, _ = fixture(sampler)
        fails(state.enable_relative)
        pending = ask(state)
        fails(partial(state.tell_relative, pending, *valid))
        state.tell(pending, [9.0])
        assert state.sync() == [False]
    for legacy in ("tell", "tell_paired"):
        state, _, _ = fixture()
        pending = ask(state)
        fails(state.enable_relative)
        fails(partial(state.tell_relative, pending, *valid))
        if legacy == "tell":
            state.tell(pending, [9.0])
        else:
            state.tell_paired(pending, 9.0, 0.5, 10.0, 0.75, False)
        assert state.sync() == [False]
        fails(state.enable_relative)


def check_pairedleases():
    import torch

    search, base, _ = fixture()
    snapshot = search.incumbent()
    tensor = torch.from_dlpack(LegacyExport(snapshot))
    fails(search.enable_relative, "release live")
    del tensor
    gc.collect()
    search.enable_relative()
    proposals = ask_relative(search)
    tensor = torch.from_dlpack(LegacyExport(proposals))
    tell = partial(
        search.tell_relative, proposals, 9.0, 0.5, 10.0, 0.75, -1.0, 0.25, False
    )
    fails(tell, "release live")
    fails(search.sync, "release live")
    del tensor
    gc.collect()
    tell()
    assert search.sync() == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    for accept in (False, True):
        state, _, _ = fixture()
        state.enable_relative()
        pending = ask_relative(state)
        tensor = torch.from_dlpack(LegacyExport(pending))
        tensor.fill_(0.5)
        del tensor
        gc.collect()
        state.tell_relative(pending, 11.0, 0.5, 10.0, 0.75, 1.0, 0.25, accept)
        fails(state.sync, "invalid")


def check_updates():
    for capacity in (1, 2):
        search, base, leaves = fixture(capacity=capacity)
        reference = search.read_reference()
        history = [(base.copy(), 10.0, 0.125)]
        # Easier-batch false winner, then a real improvement below the stale cache.
        # Last two decisions deliberately disagree with scalar comparisons.
        rounds = [
            (20.0, 0.25, 21.0, 0.5, False),
            (-2.0, 0.75, -3.0, 0.125, True),
            (5.0, 0.5, -4.0, 0.25, False),
            (-8.0, 0.125, -7.0, 0.75, True),
        ]
        for step, (value, variance, incumbent, incumbent_variance, accept) in enumerate(
            rounds
        ):
            snapshot = search.incumbent()
            np.testing.assert_array_equal(read_proposals(snapshot), base[None, :])
            fails(snapshot.__dlpack__, "already exported")
            old_radius = search.length
            proposals = ask(search, 42 + step)
            fails(search.incumbent, "outstanding")
            index, _ = proposals.geometry()[0]
            _, actual_score, radius, _ = proposals.describe()[0]
            actual = read_proposals(proposals)[0]
            # This checks candidate absolute rewards AND variances in FIFO, including
            # rejected observations and eviction, through unchanged surrogate math.
            np.testing.assert_allclose(
                actual_score, score(actual, history), rtol=4e-5, atol=4e-5
            )
            expected = candidate(base, leaves, reference, 42 + step, index, radius)
            check_rounding(actual, expected[3], expected[4])
            search.tell_paired(
                proposals, value, variance, incumbent, incumbent_variance, accept
            )
            fails(search.incumbent, "sync")
            fails(
                partial(
                    search.tell_paired,
                    proposals,
                    value,
                    variance,
                    incumbent,
                    incumbent_variance,
                    accept,
                )
            )
            assert search.sync() == [accept]
            assert search.sync() == []
            assert search.best == (value if accept else incumbent)
            assert search.best_variance == (variance if accept else incumbent_variance)
            assert search.restarts == 0
            assert search.length == (radius if accept else old_radius)
            stored = search.read_reference()
            if accept:
                check_rounding(stored, expected[1], expected[2])
                assert np.any(stored != reference)
                base = actual
            else:
                np.testing.assert_array_equal(stored, reference)
            reference = stored
            np.testing.assert_array_equal(search.read_best(), base)
            history.append((actual.copy(), value, variance))
            history = history[-capacity:]
            assert search.history_len == len(history)
            fails(proposals.__dlpack__, "outstanding")
        # Observe the final insertion too, without leaving an outstanding round.
        proposals = ask(search, 99)
        actual = read_proposals(proposals)[0]
        np.testing.assert_allclose(
            proposals.describe()[0][1], score(actual, history), rtol=4e-5, atol=4e-5
        )
        search.tell_paired(proposals, 0.0, 0.0, -8.0, 0.125, False)
        assert search.sync() == [False]


def check_mixed():
    search, base, _ = fixture()
    proposals = ask(search)
    valid = [1.0, 0.0, 0.0, 0.0, True]
    for index in range(4):
        invalids = [float("nan"), float("inf"), -float("inf"), 1e100]
        if index in (1, 3):
            invalids.extend([-0.1, -1e-100])
        for value in invalids:
            args = valid.copy()
            args[index] = value
            fails(
                partial(search.tell_paired, proposals, *args),
                error_types=(ValueError, OverflowError),
            )
    other, _, _ = fixture()
    foreign = ask(other)
    fails(lambda: search.tell_paired(foreign, *valid), "outstanding")
    fails(lambda: search.tell(foreign, [1.0]), "outstanding")
    # Invalid paired calls must not select paired mode or consume the pending ask.
    search.tell(proposals, [9.0], [0.5])
    assert search.sync() == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    assert search.best == 10.0 and search.best_variance == 0.125
    proposals = ask(search, 43)
    fails(lambda: search.tell_paired(proposals, *valid), "mix")
    search.tell(proposals, [11.0])
    assert search.sync() == [True]

    other.tell_paired(foreign, *valid)
    assert other.sync() == [True]
    proposals = ask(other, 43)
    fails(lambda: other.tell(proposals, [100.0]), "mix")
    import jax.numpy as jnp

    fails(lambda: other.tell(proposals, jnp.asarray([100.0], dtype=jnp.float32)), "mix")
    fails(lambda: other.tell_paired(foreign, *valid), "outstanding")
    other.tell_paired(proposals, *valid)
    assert other.sync() == [True]
    for sampler in ("independent", "gaussian"):
        state, _, _ = fixture(sampler)
        pending = ask(state)
        fails(partial(state.tell_paired, pending, *valid), "correlated")
        state.tell(pending, [9.0])
        assert state.sync() == [False]


class LegacyExport:
    def __init__(self, source):
        self.source = source

    def __dlpack_device__(self):
        return self.source.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        return self.source.__dlpack__(stream=stream)


def check_contract():
    import jax
    import jax.numpy as jnp

    search, base, _ = fixture()
    old = search.incumbent()
    current = search.incumbent()
    fails(old.__dlpack__, "stale")
    for kwargs in (
        {"copy": True},
        {"dl_device": (2, 1)},
        {"dl_device": (1, 0)},
        {"stream": 0},
        {"stream": -2},
    ):
        fails(partial(current.__dlpack__, **kwargs))
    assert current.__dlpack_device__() == (2, 0)
    array = jax.dlpack.from_dlpack(current)
    assert array.shape == (1, base.size) and array.dtype == jnp.bfloat16
    assert next(iter(array.devices())).id == 0
    for action in (
        partial(ask, search),
        search.incumbent,
        search.sync,
        search.read_best,
        search.read_reference,
        partial(search.profile, True),
        partial(getattr, search, "length"),
        current.__dlpack__,
    ):
        fails(action, "release live")
    np.testing.assert_array_equal(np.asarray(array.view(jnp.uint16)), base[None, :])
    array.delete()
    del array
    gc.collect()
    fails(current.__dlpack__, "already exported")
    stale = search.incumbent()
    proposals = ask(search)
    fails(stale.__dlpack__, "stale")
    array = jax.dlpack.from_dlpack(proposals)
    for action in (
        search.incumbent,
        partial(ask, search),
        search.sync,
        partial(search.tell, proposals, [0.0]),
        partial(search.tell_paired, proposals, 0.0, 0.0, 0.0, 0.0, False),
    ):
        fails(action, "release live")
    array.block_until_ready()
    array.delete()
    del array
    gc.collect()
    search.tell_paired(proposals, 0.0, 0.0, 10.0, 0.125, False)
    assert search.sync() == [False]
    # Both capsule versions must release leases even without a consumer.
    for version in ((0, 8), (1, 0), (1, 3), (2, 0)):
        snapshot = search.incumbent()
        capsule = snapshot.__dlpack__(
            max_version=version, dl_device=(2, 0), copy=False, stream=-1
        )
        fails(search.incumbent, "release live")
        del capsule
        gc.collect()
        np.testing.assert_array_equal(search.read_best(), base)
    # The managed capsule keeps both exporter and search allocation alive.
    snapshot = search.incumbent()
    array = jax.dlpack.from_dlpack(snapshot)
    del snapshot, search, proposals, old, current, stale, action
    gc.collect()
    np.testing.assert_array_equal(np.asarray(array.view(jnp.uint16)), base[None, :])
    array.delete()


def check_scratch():
    import torch

    for delayed in (False, True):
        search, base, _ = fixture()
        replay, _, _ = fixture()
        snapshot = search.incumbent()
        stream = torch.cuda.Stream()
        complete = torch.cuda.Event()
        with torch.cuda.stream(stream):
            tensor = torch.from_dlpack(LegacyExport(snapshot))
            scratch_pointer = tensor.data_ptr()
            if delayed:
                torch.cuda._sleep(200_000_000)
            tensor.fill_(float("nan"))
            complete.record()
        fails(search.incumbent, "release live")
        del tensor
        gc.collect()
        pending = ask(search)
        assert complete.query(), "scratch reuse did not wait for the consumer stream"
        clean = ask(replay)
        # The incumbent scratch and single-arm candidate reuse the same allocation.
        tensor = torch.from_dlpack(LegacyExport(pending))
        assert tensor.data_ptr() == scratch_pointer
        del tensor
        gc.collect()
        np.testing.assert_array_equal(read_proposals(pending), read_proposals(clean))
        search.tell_paired(pending, 20.0, 0.5, 21.0, 0.25, False)
        replay.tell_paired(clean, 20.0, 0.5, 21.0, 0.25, False)
        assert search.sync() == replay.sync() == [False]
        np.testing.assert_array_equal(search.read_best(), base)
        np.testing.assert_array_equal(read_proposals(search.incumbent()), base[None, :])
    for accept in (False, True):
        search, _, _ = fixture()
        pending = ask(search)
        tensor = torch.from_dlpack(LegacyExport(pending))
        tensor.fill_(0.5)
        del tensor
        gc.collect()
        search.tell_paired(pending, 20.0, 0.5, 21.0, 0.25, accept)
        fails(search.sync, "invalid")


def main():
    import jax

    assert jax.devices()[0].platform == "gpu", "requires CUDA device 0"
    check_updates()
    check_mixed()
    check_contract()
    check_scratch()
    check_history()
    check_offsets()
    check_contraction()
    check_validation()
    check_pairedleases()
    print(
        "MINIBATCH_PARITY ok=true paired=true fifo=true variance=true reference=true "
        "radius=true mixed_api=true leases=true scratch_reused=true mutation=true "
        "paired_relative=true pinned_anchor=true recenter=true batch_offsets=true "
        "failure_contraction=true relative_validation=true relative_leases=true"
    )


if __name__ == "__main__":
    main()
