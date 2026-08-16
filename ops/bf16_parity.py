"""Verify CUDA BF16 perturbations bit-for-bit against an independent reference."""

import gc
import math
import struct

import jax
import jax.numpy as jnp

from ennx.experimental import Bf16Search, Bf16Tree

MASK64 = (1 << 64) - 1
MAX_F32 = 3.4028234663852886e38


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def sign(seed: int, key: int, element: int) -> float:
    leaf = mix64(key ^ 0xD6E8FEB86659FD93)
    coordinate = mix64(element ^ 0xA0761D6478BD642F)
    return -1.0 if mix64(seed ^ leaf ^ coordinate) & 1 == 0 else 1.0


def decode(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def encode(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16


def next_value(value: int, positive: bool) -> int:
    if value & 0x7FFF == 0:
        return 1 if positive else 0x8001
    grows = (value & 0x8000 == 0) == positive
    candidate = (value + 1 if grows else value - 1) & 0xFFFF
    if math.isfinite(decode(candidate)):
        return candidate
    return (value - 1 if grows else value + 1) & 0xFFFF


def cpu_values(base, leaves, terms):
    output = list(base)
    for key, offset, length, scale in leaves:
        for local in range(length):
            total = 0.0
            strongest = 0.0
            positive = True
            for seed, coefficient in terms:
                if coefficient == 0.0:
                    continue
                direction = sign(seed, key, local)
                total = f32(total + f32(coefficient * direction))
                if abs(coefficient) > strongest:
                    strongest = abs(coefficient)
                    positive = (coefficient > 0.0) == (direction > 0.0)
            value = f32(decode(base[offset + local]) + f32(scale * total))
            candidate = encode(value)
            output[offset + local] = (
                next_value(base[offset + local], positive)
                if total == 0.0 or candidate == base[offset + local]
                else candidate
            )
    return output


def search_score(base, candidate, leaves, base_value, base_variance):
    distance = 0.0
    for (_, offset, length, _), weight in zip(leaves, (1.0, 0.5, 2.0)):
        for index in range(offset, offset + length):
            delta = decode(candidate[index]) - decode(base[index])
            distance += delta * delta * weight
    variance = 1.0e-9 + 0.7 * distance + 0.05 + base_variance
    return base_value + 1.3 * math.sqrt(variance)


def check_search(base, base_bits, leaves):
    base_value = -0.75
    base_variance = 0.04
    blocks = [(*leaf, weight) for leaf, weight in zip(leaves, (1.0, 0.5, 2.0))]
    search = Bf16Search(
        base,
        base_value,
        blocks,
        8,
        max_pending=2,
        base_variance=base_variance,
    )
    search.profile(True)
    seeds = [
        [3, 17, 0xDEADBEEFCAFEBABE, (1 << 64) - 10],
        [5, 29, 0x0123456789ABCDEF, (1 << 64) - 4],
    ]
    trials = search.ask_batch(seeds, 1, beta=1.3, seed=23)
    assert len(trials) == 2
    assert search.last_profile is not None
    assert search.last_profile[3] > 0.0

    views = search.rows(trials)
    candidates = [jax.dlpack.from_dlpack(view) for view in views]
    batch = jnp.stack(candidates)
    batch.block_until_ready()
    assert batch.shape == (len(trials), len(base_bits))
    for trial, candidate in zip(trials, candidates):
        expected = cpu_values(base_bits, leaves, [(trial.seed, trial.length)])
        actual = jax.device_get(
            jax.lax.bitcast_convert_type(candidate, jnp.uint16)
        ).tolist()
        assert actual == expected
        expected_score = search_score(
            base_bits,
            expected,
            leaves,
            base_value,
            base_variance,
        )
        assert math.isclose(trial.score, expected_score, rel_tol=2.0e-5)

    try:
        search.tell_batch(trials, [1.0, 0.5], [0.01, 0.09])
    except ValueError as error:
        assert "live JAX BF16 rows" in str(error)
    else:
        raise AssertionError("live JAX BF16 rows did not hold their leases")

    del candidate
    del candidates
    del views
    gc.collect()
    assert search.tell_batch(trials, [1.0, 0.5], [0.01, 0.09]) == [True, False]
    assert search.best == 1.0
    assert math.isclose(search.best_variance, 0.01, rel_tol=1.0e-6)
    assert search.history_len == 3
    assert search.len == len(base_bits)
    assert batch.shape == (2, len(base_bits))
    return len(trials)


def main() -> None:
    size = 1_030
    base = jnp.linspace(-4.0, 4.0, size, dtype=jnp.bfloat16)
    base = base.at[0].set(jnp.bfloat16(0.0))
    base = base.at[1].set(jnp.bfloat16(-0.0))
    base = jax.device_put(base)
    leaves = [
        (17, 0, 257, 1.0e-6),
        (19, 257, 516, 1.0e-3),
        (23, 773, 257, 0.25),
    ]
    terms = [(41, 0.25), (73, -0.125), (89, 0.03125)]
    base_bits = jax.device_get(jax.lax.bitcast_convert_type(base, jnp.uint16)).tolist()
    expected = cpu_values(base_bits, leaves, terms)

    tree = Bf16Tree(base, leaves)
    tree.materialize(terms)
    candidate = jax.dlpack.from_dlpack(tree)
    actual = jax.device_get(jax.lax.bitcast_convert_type(candidate, jnp.uint16)).tolist()
    assert actual == expected

    try:
        tree.materialize([(97, 1.0)])
    except ValueError as error:
        assert "candidate is alive" in str(error)
    else:
        raise AssertionError("live JAX candidate did not hold the BF16 lease")

    del candidate
    gc.collect()
    tree.materialize([(97, 1.0)])

    invalid = jax.device_put(jnp.array([0x7F80], dtype=jnp.uint16)).view(jnp.bfloat16)
    try:
        Bf16Tree(invalid, [(29, 0, 1, 1.0)])
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite BF16 base was accepted")

    maximum = jax.device_put(jnp.array([0x7F7F], dtype=jnp.uint16)).view(jnp.bfloat16)
    overflow = Bf16Tree(maximum, [(31, 0, 1, MAX_F32)])
    try:
        overflow.materialize([(101, MAX_F32)])
    except ValueError as error:
        assert "overflowed" in str(error)
    else:
        candidate = jax.dlpack.from_dlpack(overflow)
        bits = jax.device_get(
            jax.lax.bitcast_convert_type(candidate, jnp.uint16)
        ).tolist()
        raise AssertionError(f"BF16 perturbation overflow was accepted: {bits}")

    trials = check_search(base, base_bits, leaves)

    print(
        f"BF16_PARITY ok=true exact={size} trials={trials} "
        "leases=true noise=true profile=true validation=true"
    )


if __name__ == "__main__":
    main()
