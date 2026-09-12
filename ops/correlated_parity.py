"""Validate dense Gaussian CUDA proposals; CPU libm agreement is approximate."""

import gc
import math
import struct

import click
import numpy as np

MASK64 = (1 << 64) - 1
EPS = np.finfo(np.float32).eps
NOISE_DOMAIN = 0x8EBC6AF09C88C6E3
REFERENCE_DOMAIN = 0xE7037ED1A0B428DB


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def decode(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def encode(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def trial_hash(low, high, index):
    value = low ^ ((index * 0x9E3779B9) & 0xFFFFFFFF)
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= high
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 15)


def candidate_seed(root, stream):
    low, high = root & 0xFFFFFFFF, root >> 32
    return trial_hash(low, high, stream) | (
        trial_hash(high, low, stream ^ 0x9E3779B9) << 32
    )


def dense_normal(seed, key, element):
    """Mirror FP32 Box-Muller operations with host libm, not CUDA libdevice."""
    first = mix64(
        seed
        ^ mix64(key ^ 0xD6E8FEB86659FD93)
        ^ mix64((element // 2) ^ 0xA0761D6478BD642F)
    )
    second = mix64(first ^ 0xD2B74407B1CE6E93)
    u1 = f32(((first >> 11) + 1) / (1 << 53))
    u2 = f32((second >> 11) / (1 << 53))
    log_u1 = f32(math.log(min(max(u1, f32(1e-12)), f32(0.99999994))))
    radius = f32(math.sqrt(f32(-2.0 * log_u1)))
    angle = f32(f32(math.tau) * u2)
    trig = math.cos if element % 2 == 0 else math.sin
    return f32(radius * f32(trig(angle)))


def draw_normal(seed, slot):
    # Acquisition draws use a distinct FP64 Box-Muller stream keyed by FIFO slot.
    combined = ((seed * 1_000_003 + slot) * 1_000_003) & MASK64
    first, second = mix64(combined), mix64(combined ^ 0xD2B74407B1CE6E93)
    u1 = min(max((first >> 11) / (1 << 53), 1e-12), 1 - 1e-12)
    u2 = (second >> 11) / (1 << 53)
    return f32(math.sqrt(-2 * math.log(u1)) * math.cos(math.tau * u2))


def initial_reference(leaves, seed):
    return np.asarray(
        [
            dense_normal(seed ^ REFERENCE_DOMAIN, key, i)
            for key, _, length, _ in leaves
            for i in range(length)
        ],
        dtype=np.float32,
    )


def direction(leaves, reference, seed, persistence):
    values, errors = [], []
    for key, offset, length, _ in leaves:
        if persistence:
            stored = decode(reference[offset : offset + length]).astype(np.float64)
            inverse_rms = f32(math.sqrt(length / np.square(stored).sum()))
        for element in range(length):
            noise = dense_normal(seed ^ NOISE_DOMAIN, key, element)
            if persistence:
                normalized = f32(float(stored[element]) * inverse_rms)
                left = f32(persistence * normalized)
                right = f32(f32(math.sqrt(0.4375)) * noise)
                value = f32(left + right)
                magnitude = abs(left) + abs(right)
            else:
                value, magnitude = noise, abs(noise)
            values.append(value)
            # FP32 libm, possible FMA contraction, and reference RMS reduction.
            # This is a numerical test allowance, not a proved libdevice bound.
            errors.append(16 * EPS * (1 + magnitude))
    return np.asarray(values), np.asarray(errors)


def candidate(base, leaves, reference, root, index, radius, correlated=True):
    seed = candidate_seed(root, index // 2 if correlated else index)
    persistence = 0.75 if correlated and index < 2 else 0.0
    values, errors = direction(leaves, reference, seed, persistence)
    old = decode(base).astype(np.float64)
    raw, allowance = np.empty(len(base)), np.empty(len(base))
    for _, offset, length, scale in leaves:
        sl = slice(offset, offset + length)
        step = f32(f32(scale) * f32(radius))
        delta = (step * values[sl]).astype(np.float32).astype(np.float64)
        raw[sl] = (old[sl] + delta).astype(np.float32)
        allowance[sl] = abs(step) * errors[sl] + 4 * EPS * (abs(old[sl]) + abs(delta))
    return seed, values, errors, raw, allowance


def rounded_bounds(raw, allowance):
    raw, allowance = np.asarray(raw), np.asarray(allowance)
    return decode(encode(raw - allowance)).astype(np.float64), decode(
        encode(raw + allowance)
    ).astype(np.float64)


def check_rounding(actual, raw, allowance):
    """Permit only adjacent BF16 rounding cells crossed by a tiny FP32 interval."""
    actual = np.asarray(actual, dtype=np.uint16)
    expected = encode(raw)
    assert actual.shape == expected.shape
    low, high = rounded_bounds(raw, allowance)
    decoded = decode(actual).astype(np.float64)
    assert np.isfinite(decoded).all()
    assert np.all((decoded >= low) & (decoded <= high)), (
        "Outside FP32 rounding-boundary allowance"
    )
    mismatch = actual != expected
    for observed, wanted in zip(actual[mismatch], expected[mismatch]):
        assert (int(observed) ^ int(wanted)) & 0x8000 == 0
        assert abs(int(observed) - int(wanted)) == 1
    return int(np.count_nonzero(mismatch))


def score(bits, history, draw_seed):
    values = decode(bits).astype(np.float64)
    distances = [np.square(values - row).sum() for _, row, _ in history]
    weights = 1 / (1e-9 + 0.7 * np.asarray(distances) + 0.05)
    rewards = np.asarray([reward for _, _, reward in history])
    noises = np.asarray([draw_normal(draw_seed, slot) for slot, _, _ in history])
    return np.dot(weights, rewards) / weights.sum() + (
        np.dot(weights, noises) / np.linalg.norm(weights) / np.sqrt(weights.sum())
    )


def score_bounds(raw, allowance, history, draw_seed):
    """Conservative TS interval when a CPU value straddles a BF16 midpoint."""
    low, high = rounded_bounds(raw, allowance)
    minimum, maximum = [], []
    for _, row, _ in history:
        closest = np.maximum(np.maximum(low - row, row - high), 0)
        farthest = np.maximum(abs(low - row), abs(high - row))
        minimum.append(np.square(closest).sum())
        maximum.append(np.square(farthest).sum())
    wlo = 1 / (1e-9 + 0.7 * np.asarray(maximum) + 0.05)
    whi = 1 / (1e-9 + 0.7 * np.asarray(minimum) + 0.05)

    def ratio(values, denom_lo, denom_hi):
        a, b = wlo * values, whi * values
        numerator = (np.minimum(a, b).sum(), np.maximum(a, b).sum())
        corners = [n / d for n in numerator for d in (denom_lo, denom_hi)]
        return min(corners), max(corners)

    mean = ratio(np.asarray([r for _, _, r in history]), wlo.sum(), whi.sum())
    noise = ratio(
        np.asarray([draw_normal(draw_seed, slot) for slot, _, _ in history]),
        np.linalg.norm(wlo) * np.sqrt(wlo.sum()),
        np.linalg.norm(whi) * np.sqrt(whi.sum()),
    )
    return mean[0] + noise[0], mean[1] + noise[1]


def read_proposals(proposals):
    import jax
    import jax.numpy as jnp

    batch = jax.dlpack.from_dlpack(proposals)
    try:
        return np.array(
            jax.device_get(jax.lax.bitcast_convert_type(batch, jnp.uint16)), copy=True
        )
    finally:
        batch.delete()
        del batch
        gc.collect()


def check_changes(changes, leaves, old, actual):
    assert len(changes) == len(leaves)
    for (_, offset, length, _), (changed, squared) in zip(leaves, changes):
        before, after = old[offset : offset + length], actual[offset : offset + length]
        assert changed == np.count_nonzero(before != after)
        delta = decode(after).astype(np.float64) - decode(before).astype(np.float64)
        np.testing.assert_allclose(
            squared, np.square(delta).sum(), rtol=3e-6, atol=1e-10
        )


def append_history(history, bits, reward):
    slot = len(history) + 1 if len(history) < 2 else history.pop(0)[0]
    history.append((slot, decode(bits).astype(np.float64), reward))


def fixture(sampler):
    import jax
    import jax.numpy as jnp

    from ennx.experimental import ParamBlock, turbo_enn

    # Odd leaf lengths test Box-Muller pairing reset; 513 crosses thread chunks.
    leaves = [
        (17, 0, 1, 2**-20),
        (19, 1, 30, 0.03125),
        (23, 31, 3, 0.5),
        (29, 34, 513, 1.0),
        (31, 547, 485, 2.0),
    ]
    size = sum(leaf[2] for leaf in leaves)
    bits = np.resize(np.asarray([0x3FC0, 0xBFC0], dtype=np.uint16), size)
    base = jax.device_put(jnp.asarray(bits)).view(jnp.bfloat16)
    search = turbo_enn(
        base,
        0.0,
        [ParamBlock(*leaf) for leaf in leaves],
        2,
        sampler=sampler,
        reference_seed=0x123456789ABCDEF0,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
        failure_tolerance=4 if sampler == "gaussian" else None,
    )
    return leaves, bits, search


def ask(search, root, draw):
    return search.ask(
        1,
        4,
        2,
        root,
        epistemic_scale=0.7,
        aleatoric_scale=0.05,
        y_scale=1.0,
        acquisition="thompson",
        draw_seed=draw,
    )


def check_correlated():
    leaves, base, search = fixture("correlated")
    _, _, replay = fixture("correlated")
    reference = search.read_reference()
    initial = initial_reference(leaves, 0x123456789ABCDEF0)
    mismatches = check_rounding(reference, initial, 16 * EPS * (1 + abs(initial)))
    np.testing.assert_array_equal(reference, replay.read_reference())
    history = [(1, decode(base).astype(np.float64), 0.0)]
    radius, best, seen = 0.125, 0.0, set()
    for step in range(48):
        root, draw = mix64(step + 100), mix64(step + 900)
        radii = [f32(max(0.03125, radius / 2)), f32(min(0.5, radius * 2))] * 2
        expected = [
            candidate(base, leaves, reference, root, i, radii[i]) for i in range(4)
        ]
        for a, b in ((0, 1), (2, 3)):
            assert expected[a][0] == expected[b][0]
            np.testing.assert_array_equal(expected[a][1], expected[b][1])
        proposals, repeated = ask(search, root, draw), ask(replay, root, draw)
        geometry, description = proposals.geometry(), proposals.describe()
        assert geometry == repeated.geometry() and description == repeated.describe()
        index, persistence = geometry[0]
        seed, actual_score, actual_radius, changes = description[0]
        assert 0 <= index < 4
        seen.add(index)
        assert persistence == (0.75 if index < 2 else 0.0)
        assert seed == expected[index][0] and actual_radius == radii[index]
        actual = read_proposals(proposals)[0]
        np.testing.assert_array_equal(actual, read_proposals(repeated)[0])
        _, values, errors, raw, allowance = expected[index]
        mismatches += check_rounding(actual, raw, allowance)
        np.testing.assert_allclose(
            actual_score, score(actual, history, draw), rtol=3e-5, atol=3e-5
        )
        bounds = [score_bounds(e[3], e[4], history, draw) for e in expected]
        slack = 6e-5 * max(1, abs(actual_score))
        assert actual_score >= max(lo for lo, _ in bounds) - slack
        assert bounds[index][0] - slack <= actual_score <= bounds[index][1] + slack
        check_changes(changes, leaves, base, actual)
        improve = step % 3 == 0
        reward = best + 1.0 if improve else best - 0.5
        for state, pending in ((search, proposals), (replay, repeated)):
            state.tell(pending, [reward], [0.0])
            assert state.sync() == [improve]
        append_history(history, actual, reward)
        stored = search.read_reference()
        if improve:
            mismatches += check_rounding(stored, values, errors)
            best, radius, base = reward, radii[index], actual
        else:
            np.testing.assert_array_equal(stored, reference)
        # Anchor each CPU round to exact GPU BF16 state, never CPU libm recursion.
        reference = stored
        np.testing.assert_array_equal(reference, replay.read_reference())
        for state in (search, replay):
            assert state.length == radius and state.best == best
            assert state.restarts == 0 and state.history_len == 2
            np.testing.assert_array_equal(state.read_best(), base)
    assert seen == {0, 1, 2, 3}, seen
    return mismatches


def check_gaussian():
    leaves, base, search = fixture("gaussian")
    try:
        search.read_reference()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Independent Gaussian sampling must not expose a reference"
        )
    history = [(1, decode(base).astype(np.float64), 0.0)]
    mismatches = 0
    for step in range(4):
        root, draw = mix64(step + 70), mix64(step + 90)
        radius = search.length
        proposals = ask(search, root, draw)
        index, persistence = proposals.geometry()[0]
        seed, actual_score, actual_radius, changes = proposals.describe()[0]
        assert 0 <= index < 4 and persistence == 0
        assert seed == candidate_seed(root, index) and actual_radius == f32(radius)
        expected = [
            candidate(base, leaves, None, root, i, radius, False) for i in range(4)
        ]
        assert len({e[0] for e in expected}) == 4
        actual = read_proposals(proposals)[0]
        mismatches += check_rounding(actual, expected[index][3], expected[index][4])
        np.testing.assert_allclose(
            actual_score, score(actual, history, draw), rtol=3e-5, atol=3e-5
        )
        lower = [score_bounds(e[3], e[4], history, draw)[0] for e in expected]
        assert actual_score >= max(lower) - 6e-5 * max(1, abs(actual_score))
        check_changes(changes, leaves, base, actual)
        search.tell(proposals, [-1.0])
        assert search.sync() == [False]
        append_history(history, actual, -1.0)
        np.testing.assert_array_equal(search.read_best(), base)
    assert search.length == 0.0625 and search.restarts == 0
    return mismatches


def check_copyorder():
    import jax.numpy as jnp

    from ennx.experimental import ParamBlock, turbo_enn

    size = 262_147
    base = jnp.ones(size, dtype=jnp.bfloat16)
    search = turbo_enn(
        base,
        0.0,
        [ParamBlock(17, 0, size, 1.0)],
        2,
        max_pending=5,
        sampler="gaussian",
        failure_tolerance=1,
        length_init=0.125,
        length_min=0.125,
        length_max=0.5,
    )
    proposals = search.ask(5, 2, 1, 123)
    expected = read_proposals(proposals)
    rewards = [1.0, 3.0, 2.0, 4.0, 0.0]
    search.tell(proposals, rewards)
    assert search.sync() == [True, True, False, True, False]
    assert search.history_len == 2 and search.best == 4.0
    np.testing.assert_array_equal(search.read_best(), expected[3])
    history = [
        (1, decode(np.full(size, 0x3F80, dtype=np.uint16)).astype(np.float64), 0.0)
    ]
    for bits, reward in zip(expected, rewards):
        append_history(history, bits, reward)
    # Scoring the next round checks both surviving FIFO copies, not only the best.
    proposals = ask(search, 456, 789)
    actual = read_proposals(proposals)[0]
    actual_score = proposals.describe()[0][1]
    np.testing.assert_allclose(
        actual_score, score(actual, history, 789), rtol=3e-5, atol=3e-5
    )
    search.tell(proposals, [-1.0])
    assert search.sync() == [False]
    assert search.restarts == 1 and search.history_len == 1
    assert search.length == 0.125 and search.best == 4.0
    np.testing.assert_array_equal(search.read_best(), expected[3])
    proposals = search.ask(1, 4, 1, 987, acquisition="thompson", draw_seed=654)
    actual = read_proposals(proposals)[0]
    history = [(1, decode(expected[3]).astype(np.float64), 4.0)]
    np.testing.assert_allclose(
        proposals.describe()[0][1], score(actual, history, 654), rtol=3e-5, atol=3e-5
    )
    search.tell(proposals, [-1.0])
    assert search.sync() == [False]


def null_case():
    leaves = [(17, 0, 33, 1.0)]
    base = encode(np.full(33, 1.5))
    reference = encode(initial_reference(leaves, 123))
    radius = 0.0015
    for root in range(128):
        pool = [
            candidate(
                base, leaves, reference, root, i, radius * (0.5 if i % 2 == 0 else 2)
            )
            for i in range(4)
        ]
        baseline = [
            candidate(base, leaves, None, root, i, radius, False) for i in range(4)
        ]

        def null(item):
            low, high = rounded_bounds(item[3], item[4])
            return np.array_equal(low, decode(base)) and np.array_equal(
                high, decode(base)
            )

        def changed(item):
            low, high = rounded_bounds(item[3], item[4])
            return np.any((low > decode(base)) | (high < decode(base)))

        if (
            null(pool[0])
            and null(pool[2])
            and changed(pool[1])
            and null(baseline[0])
            and any(changed(item) for item in baseline[1:])
        ):
            return leaves, base, radius, root
    raise AssertionError("No rounding-safe null/non-null fixture found")


def check_nulls():
    import jax.numpy as jnp

    from ennx.experimental import ParamBlock, turbo_enn

    leaves, bits, radius, root = null_case()
    base = jnp.asarray(bits).view(jnp.bfloat16)
    for sampler, step in (
        ("correlated", radius),
        ("gaussian", radius),
        ("correlated", 1e-7),
    ):
        all_null = step == 1e-7
        reward = 2.0 if all_null else 0.0
        search = turbo_enn(
            base,
            reward,
            [ParamBlock(*leaf) for leaf in leaves],
            2,
            sampler=sampler,
            reference_seed=123,
            length_init=step,
            length_min=step / 4,
            length_max=step * 4,
        )
        # UCB with beta=0 is predictive mean: every candidate has an exact tie.
        proposals = search.ask(1, 4, 1, root, acquisition="ucb", beta=0)
        index, _ = proposals.geometry()[0]
        _, actual_score, _, changes = proposals.describe()[0]
        actual = read_proposals(proposals)[0]
        assert actual_score == reward
        check_changes(changes, leaves, bits, actual)
        if sampler == "correlated" and not all_null:
            assert index == 1 and sum(count for count, _ in changes) > 0
        else:
            assert index == 0 and changes == [(0, 0.0)]
            np.testing.assert_array_equal(actual, bits)
        search.tell(proposals, [reward])
        assert search.sync() == [False]


def check_tiles():
    import jax.numpy as jnp

    from ennx.experimental import ParamBlock, turbo_enn

    leaves = [(17, 0, 65_539, 1.0)]
    base = encode(np.full(65_539, 1.5))
    search = turbo_enn(
        jnp.asarray(base).view(jnp.bfloat16),
        0.0,
        [ParamBlock(*leaves[0])],
        2,
        sampler="correlated",
        reference_seed=123,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
    )
    reference = search.read_reference()
    initial = initial_reference(leaves, 123)
    mismatches = check_rounding(reference, initial, 16 * EPS * (1 + abs(initial)))
    for step in range(2):
        # Mean ties force persistent candidate 0, exercising reference RMS on both rounds.
        proposals = search.ask(
            1, 4, 2, step, acquisition="ucb", beta=0, epistemic_scale=0
        )
        index, persistence = proposals.geometry()[0]
        seed, _, radius, changes = proposals.describe()[0]
        assert index == 0 and persistence == 0.75
        expected = candidate(base, leaves, reference, step, index, radius)
        assert seed == expected[0]
        actual = read_proposals(proposals)[0]
        mismatches += check_rounding(actual, expected[3], expected[4])
        check_changes(changes, leaves, base, actual)
        # Zero epistemic scale keeps weights equal across candidates after acceptance.
        search.tell(proposals, [1.0 if step == 0 else -1.0])
        assert search.sync() == [step == 0]
        stored = search.read_reference()
        if step == 0:
            mismatches += check_rounding(stored, expected[1], expected[2])
            base = actual
        else:
            np.testing.assert_array_equal(stored, reference)
        reference = stored
        np.testing.assert_array_equal(search.read_best(), base)
    return mismatches


def check_cpuhelpers():
    null_case()
    values = np.asarray([dense_normal(123, 17, i) for i in range(16_384)])
    assert abs(values.mean()) < 0.03 and abs(values.var() - 1) < 0.04
    assert np.unique(encode(values)).size > 1000
    for raw, neighbor in ((1.0, 0x3F81), (-1.0, 0xBF81)):
        assert check_rounding(encode([raw]), [raw], [1e-7]) == 0
        try:
            check_rounding([neighbor], [raw], [1e-7])
        except AssertionError:
            pass
        else:
            raise AssertionError("Tolerance accepted a BF16 ULP away from a midpoint")
    midpoint = 1 + 1 / 256
    assert check_rounding([0x3F81], [midpoint], [1e-7]) == 1
    assert check_rounding([0xBF81], [-midpoint], [1e-7]) == 1
    leaves = [(17, 0, 33, 1.0), (19, 33, 17, 0.5)]
    reference = encode(initial_reference(leaves, 123))
    base = encode(np.ones(50))
    near = [candidate(base, leaves, reference, 456, i, 0.1) for i in range(4)]
    for a, b in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(near[a][1], near[b][1])
    independent = candidate(base, leaves, None, 456, 1, 0.1, False)
    assert independent[0] == candidate_seed(456, 1)
    history = [(1, decode(base).astype(np.float64), -0.5), (2, np.zeros(50), 0.75)]
    for _, _, _, raw, allowance in near:
        lo, hi = score_bounds(raw, allowance, history, 789)
        assert lo - 1e-12 <= score(encode(raw), history, 789) <= hi + 1e-12


@click.command(help=__doc__)
@click.option(
    "--cpu-only", is_flag=True, help="Run numerical helper checks without CUDA imports."
)
def main(cpu_only):
    check_cpuhelpers()
    if cpu_only:
        click.echo(
            "CORRELATED_CPU_HELPERS ok=true gaussian=true rounding_boundaries=true"
        )
        return
    import jax

    from ops.bf16_parity import check_export

    device = jax.devices()[0]
    if device.platform != "gpu" or "T4" not in device.device_kind:
        raise click.ClickException(f"Expected a CUDA T4, got {device}")
    mismatches = check_correlated() + check_gaussian() + check_tiles()
    check_copyorder()
    check_nulls()
    check_export("correlated")
    click.echo(
        "CORRELATED_PARITY ok=true cpu=rounding_boundary_tolerance gpu_replay=exact "
        "acquisition=true reference=true radius=true gaussian=true fifo_copy=true "
        "reference_tiles=true restart_copy=true null_filter=true "
        f"cpu_boundary_mismatches={mismatches}"
    )


if __name__ == "__main__":
    main()
