"""Verify CUDA BF16 perturbations bit-for-bit against an independent reference."""

import gc
import math
import struct

import jax
import jax.numpy as jnp
import numpy as np

from ennx._rust import EpistemicNearestNeighbors
from ennx.experimental import ParamBlock, ParamBuffer, turbo_enn

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


def draw_normal(seed: int, index: int, metric: int) -> float:
    prime = 1_000_003
    base = (((seed & MASK64) * prime + index) * prime) & MASK64
    combined = (base + metric) & MASK64
    first = mix64(combined)
    second = mix64(combined ^ 0xD2B74407B1CE6E93)
    u1 = min(max((first >> 11) / float(1 << 53), 1.0e-12), 1.0 - 1.0e-12)
    u2 = (second >> 11) / float(1 << 53)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def draw_expected(indices, weights, l2, means, errors, seeds):
    draws = np.empty((len(seeds), means.shape[0], means.shape[1]), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for query in range(means.shape[0]):
            for metric in range(means.shape[1]):
                weighted = 0.0
                for neighbor, index in enumerate(indices[query]):
                    weighted += float(weights[query, neighbor, metric]) * draw_normal(
                        seed, int(index), metric
                    )
                scale = float(errors[query, metric]) / max(
                    float(l2[query, metric]), 1.0e-12
                )
                draws[seed_index, query, metric] = (
                    float(means[query, metric]) + scale * weighted
                )
    return draws


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


def check_search(base, dimensions, leaves):
    base_value = -0.75
    base_variance = 0.04
    blocks = [
        ParamBlock(*leaf, weight) for leaf, weight in zip(leaves, (1.0, 0.5, 2.0))
    ]
    search = turbo_enn(
        base,
        base_value,
        blocks,
        8,
        max_pending=2,
        base_variance=base_variance,
    )
    search.profile(True)
    proposals = search.ask(2, 4, 1, 3, beta=1.3, draw_seed=23)
    assert proposals.arms == 2
    assert search.last_profile is not None
    assert search.last_profile[3] > 0.0

    batch = jax.dlpack.from_dlpack(proposals)
    batch.block_until_ready()
    assert batch.shape == (proposals.arms, dimensions)
    assert bool(jnp.all(jnp.isfinite(batch)).block_until_ready())

    try:
        search.tell(proposals, [1.0, 0.5], [0.01, 0.09])
    except ValueError as error:
        assert "live JAX proposals" in str(error)
    else:
        raise AssertionError("live JAX proposals did not hold their leases")

    del batch
    gc.collect()
    search.tell(proposals, [1.0, 0.5], [0.01, 0.09])
    assert search.sync() == [True, False]
    assert search.best == 1.0
    assert math.isclose(search.best_variance, 0.01, rel_tol=1.0e-6)
    assert search.history_len == 3
    assert search.len == dimensions
    proposals = search.ask(2, 4, 1, 5, beta=1.3, draw_seed=29)
    device_values = jax.device_put(jnp.array([1.5, 0.25], dtype=jnp.float32))
    device_variances = jax.device_put(jnp.array([0.02, 0.03], dtype=jnp.float32))
    search.tell(proposals, device_values, device_variances)
    assert search.sync() == [True, False]
    assert search.best == 1.5
    assert math.isclose(search.best_variance, 0.02, rel_tol=1.0e-6)
    assert search.history_len == 5

    proposals = search.ask(2, 4, 1, 7, beta=1.3, draw_seed=31)
    device_values = jax.device_put(jnp.array([2.0, 0.75], dtype=jnp.float32))
    search.tell(proposals, device_values)
    assert search.sync() == [True, False]
    assert search.best == 2.0
    assert math.isclose(search.length, 1.6, rel_tol=1.0e-12)
    assert search.history_len == 7
    return proposals.arms


def knn_expected(data, queries, values, skip=0, aleatoric=0.1):
    data = data.astype(np.float32)
    query = queries.astype(np.float32)
    value = values.astype(np.float32)
    distances = np.sum((query[:, None, :] - data[None, :, :]) ** 2, axis=2)
    indices = np.argsort(distances, axis=1, kind="stable")[:, skip : skip + 6]
    nearest = np.take_along_axis(distances, indices, axis=1)
    weights = np.float32(1.0) / (np.float32(1.0e-9) + nearest + np.float32(aleatoric))
    norm = np.sum(weights, axis=1, dtype=np.float32)
    means = np.sum(weights[:, :, None] * value[indices], axis=1, dtype=np.float32)
    means /= norm[:, None]
    scale = np.std(values, axis=0, ddof=0).astype(np.float32)
    errors = (
        np.sqrt(np.maximum(np.float32(1.0) / norm, np.float32(1.0e-9)))[:, None]
        * scale[None, :]
    )
    return indices, means, errors


def knn_weighted(
    data,
    queries,
    values,
    variances,
    neighbors=6,
    epistemic=0.7,
    aleatoric=0.13,
):
    data = data.astype(np.float32)
    query = queries.astype(np.float32)
    value = values.astype(np.float32)
    yvar = variances.astype(np.float32)
    distances = np.sum((query[:, None, :] - data[None, :, :]) ** 2, axis=2)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    nearest = np.take_along_axis(distances, indices, axis=1)
    scale = np.std(values, axis=0, ddof=0).astype(np.float32)
    scaled_yvar = yvar[indices] / (scale[None, None, :] ** 2)
    weights = np.float32(1.0) / (
        np.float32(1.0e-9)
        + np.float32(epistemic) * nearest[:, :, None]
        + np.float32(aleatoric)
        + scaled_yvar
    )
    norm = np.sum(weights, axis=1, dtype=np.float32)
    normalized = weights / norm[:, None, :]
    l2 = np.sqrt(np.sum(normalized * normalized, axis=1, dtype=np.float32))
    means = np.sum(normalized * value[indices], axis=1, dtype=np.float32)
    epistemic_var = np.float32(1.0) / norm
    aleatoric_var = np.sum(
        normalized * (np.float32(aleatoric) + scaled_yvar),
        axis=1,
        dtype=np.float32,
    )
    errors = np.sqrt(epistemic_var + aleatoric_var) * scale[None, :]
    epistemic_error = np.sqrt(epistemic_var) * scale[None, :]
    aleatoric_error = np.sqrt(aleatoric_var) * scale[None, :]
    return (
        indices,
        means,
        errors,
        epistemic_error,
        aleatoric_error,
        normalized,
        l2,
    )


def weighted_case(train, queries, values) -> None:
    row_ids = np.arange(train.shape[0], dtype=np.float64)[:, None]
    metric_ids = np.arange(values.shape[1], dtype=np.float64)[None, :]
    variances = 0.02 + 0.01 * np.abs(np.sin(row_ids * 0.071 + metric_ids * 0.29))
    cuda = EpistemicNearestNeighbors(train, values, variances, False, "cuda")
    cuda.set_tie_break_neighbors(False)
    actual = cuda.posterior(queries, 6, 0.7, 0.13, False, True)
    expected = knn_weighted(train, queries, values, variances)
    np.testing.assert_array_equal(actual[4], expected[0])
    for observed, reference in zip(actual[:4], expected[1:]):
        np.testing.assert_allclose(observed, reference, rtol=5.0e-5, atol=5.0e-6)

    k_values = [3, 6]
    epi_values = [0.4, 0.7]
    ale_values = [0.05, 0.13]
    batch = cuda.batch_posterior(
        queries,
        k_values,
        epi_values,
        ale_values,
        False,
        True,
    )
    references = [
        knn_weighted(
            train,
            queries,
            values,
            variances,
            neighbors=k,
            epistemic=epi,
            aleatoric=ale,
        )
        for k, epi, ale in zip(k_values, epi_values, ale_values)
    ]
    for output_index, observed in enumerate(batch):
        reference = np.stack([result[output_index + 1] for result in references])
        np.testing.assert_allclose(observed, reference, rtol=5.0e-5, atol=5.0e-6)

    seeds = [7, -3, 101]
    draws, draw_indices = cuda.posterior_function_draw(
        queries,
        6,
        0.7,
        0.13,
        seeds,
        False,
        True,
    )
    expected_draws = draw_expected(
        expected[0],
        expected[5],
        expected[6],
        expected[1],
        expected[2],
        seeds,
    )
    assert draw_indices == expected[0].tolist()
    np.testing.assert_allclose(draws, expected_draws, rtol=2.0e-4, atol=2.0e-5)

    whatif_x = queries[:2] + 0.003
    whatif_y = np.array([[1.25, -0.4], [0.8, 0.3]], dtype=np.float64)
    combined_x = np.concatenate((train, whatif_x), axis=0)
    combined_y = np.concatenate((values, whatif_y), axis=0)
    combined_var = np.concatenate((variances, np.zeros_like(whatif_y)), axis=0)
    expected_cond = knn_weighted(
        combined_x,
        queries,
        combined_y,
        combined_var,
    )
    conditional = cuda.conditional_posterior(
        whatif_x,
        whatif_y,
        queries,
        6,
        0.7,
        0.13,
        False,
        True,
    )
    np.testing.assert_array_equal(conditional[4], expected_cond[0])
    for observed, reference in zip(conditional[:4], expected_cond[1:]):
        np.testing.assert_allclose(observed, reference, rtol=5.0e-5, atol=5.0e-6)

    cond_draws, cond_indices = cuda.conditional_posterior_function_draw(
        whatif_x,
        whatif_y,
        queries,
        6,
        0.7,
        0.13,
        seeds,
        False,
        True,
    )
    expected_cond_draws = draw_expected(
        expected_cond[0],
        expected_cond[5],
        expected_cond[6],
        expected_cond[1],
        expected_cond[2],
        seeds,
    )
    assert cond_indices == expected_cond[0].tolist()
    np.testing.assert_allclose(
        cond_draws,
        expected_cond_draws,
        rtol=2.0e-4,
        atol=2.0e-5,
    )

    restored = cuda.posterior(queries, 6, 0.7, 0.13, False, True)
    np.testing.assert_array_equal(restored[4], expected[0])
    np.testing.assert_allclose(restored[0], expected[1], rtol=5.0e-5, atol=5.0e-6)


def knn_case(rows: int, dims: int) -> None:
    row_ids = np.arange(rows, dtype=np.float64)[:, None]
    dim_ids = np.arange(dims, dtype=np.float64)[None, :]
    train = np.sin(row_ids * 0.173 + dim_ids * 0.131)
    queries = np.cos(np.arange(5, dtype=np.float64)[:, None] * 0.271 + dim_ids * 0.097)
    values = np.concatenate((row_ids / rows, np.sin(row_ids * 0.113)), axis=1)
    cuda = EpistemicNearestNeighbors(train, values, None, False, "cuda")
    cuda.set_tie_break_neighbors(False)

    cuda_out = cuda.posterior(queries, 6, 1.0, 0.1, False, False)
    indices, means, errors = knn_expected(train, queries, values)
    np.testing.assert_array_equal(cuda_out[4], indices)
    np.testing.assert_allclose(cuda_out[0], means, rtol=3.0e-5, atol=3.0e-6)
    np.testing.assert_allclose(cuda_out[1], errors, rtol=3.0e-5, atol=3.0e-6)

    weighted_case(train, queries, values)

    extra = np.sin((row_ids[:1] + rows) * 0.173 + dim_ids * 0.131)
    extra_value = np.array([[1.5, -0.25]], dtype=np.float64)
    cuda.add(extra, extra_value)
    train = np.concatenate((train, extra), axis=0)
    values = np.concatenate((values, extra_value), axis=0)
    cuda_out = cuda.posterior(queries, 6, 1.0, 0.1, False, False)
    indices, means, errors = knn_expected(train, queries, values)
    np.testing.assert_array_equal(cuda_out[4], indices)
    np.testing.assert_allclose(cuda_out[0], means, rtol=3.0e-5, atol=3.0e-6)
    np.testing.assert_allclose(cuda_out[1], errors, rtol=3.0e-5, atol=3.0e-6)

    self_queries = train[:3]
    cuda_out = cuda.posterior(self_queries, 6, 1.0, 0.1, True, False)
    indices, means, errors = knn_expected(train, self_queries, values, skip=1)
    np.testing.assert_array_equal(cuda_out[4], indices)
    np.testing.assert_allclose(cuda_out[0], means, rtol=3.0e-5, atol=3.0e-6)
    np.testing.assert_allclose(cuda_out[1], errors, rtol=3.0e-5, atol=3.0e-6)


def check_knn() -> None:
    knn_case(517, 7)
    knn_case(79, 73)


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

    blocks = [ParamBlock(*leaf) for leaf in leaves]
    buffer = ParamBuffer(base, blocks)
    buffer.materialize(terms)
    candidate = jax.dlpack.from_dlpack(buffer)
    actual = jax.device_get(
        jax.lax.bitcast_convert_type(candidate, jnp.uint16)
    ).tolist()
    assert actual == expected

    try:
        buffer.materialize([(97, 1.0)])
    except ValueError as error:
        assert "candidate is alive" in str(error)
    else:
        raise AssertionError("live JAX candidate did not hold the BF16 lease")

    del candidate
    gc.collect()
    buffer.materialize([(97, 1.0)])

    invalid = jax.device_put(jnp.array([0x7F80], dtype=jnp.uint16)).view(jnp.bfloat16)
    try:
        ParamBuffer(invalid, [ParamBlock(29, 0, 1, 1.0)])
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite BF16 base was accepted")

    maximum = jax.device_put(jnp.array([0x7F7F], dtype=jnp.uint16)).view(jnp.bfloat16)
    overflow = ParamBuffer(maximum, [ParamBlock(31, 0, 1, MAX_F32)])
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

    proposals = check_search(base, len(base_bits), leaves)
    check_knn()

    print(
        f"BF16_PARITY ok=true exact={size} proposals={proposals} "
        "leases=true noise=true profile=true validation=true knn=true weighted=true "
        "batch=true draws=true conditional=true device_tell=true"
    )


if __name__ == "__main__":
    main()
