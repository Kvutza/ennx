"""NumPy-only coverage of paired finite-population screening."""

from itertools import product
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ops.flame.minibatch import PairedLosses, paired_losses, sample_indices


def test_samplerrepro():
    rng = np.random.default_rng(123)
    reference = np.random.default_rng(123)
    batches = [sample_indices(rng, 20, 7) for _ in range(8)]
    for batch in batches:
        np.testing.assert_array_equal(
            batch, reference.choice(20, size=7, replace=False)
        )
        assert batch.shape == (7,) and batch.dtype.kind in "iu"
        assert len(set(batch)) == 7
        assert np.all((batch >= 0) & (batch < 20))
    assert not np.array_equal(batches[0], batches[1])
    assert set(batches[0]) & set(batches[1])  # No cross-call exclusion.


def test_fullbatch():
    result = sample_indices(np.random.default_rng(1), np.int64(5), np.int32(5))
    np.testing.assert_array_equal(np.sort(result), np.arange(5))


@pytest.mark.parametrize(
    "population,batch_size",
    [
        (1, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (3, 4),
        (3.0, 2),
        (3, 2.0),
        (True, 2),
        (3, True),
        (np.bool_(True), 2),
        (3, "2"),
    ],
)
def test_samplersizes(population, batch_size):
    with pytest.raises(ValueError):
        sample_indices(np.random.default_rng(0), population, batch_size)


def test_generator():
    with pytest.raises(TypeError, match="Generator"):
        sample_indices(None, 5, 2)


def test_corrnoise():
    result = paired_losses([1, 101, 201], [2, 102, 202], population=30)
    assert isinstance(result, PairedLosses)
    assert result.candidate_loss == 101
    assert result.incumbent_loss == 102
    assert result.candidate_variance == pytest.approx(3000)
    assert result.incumbent_variance == pytest.approx(3000)
    assert result.improvement == 1
    assert result.improvement_variance == 0
    assert result.improvement_se == result.threshold == 0
    assert result.accepted is True
    assert result.deteriorated is False
    with pytest.raises(FrozenInstanceError):
        result.accepted = False


def test_variance():
    result = paired_losses([1, 3], [2, 6], population=4)
    assert result.candidate_loss == 2
    assert result.incumbent_loss == 4
    assert result.candidate_variance == pytest.approx(0.5)
    assert result.incumbent_variance == pytest.approx(2)
    assert result.improvement == 2
    assert result.improvement_variance == pytest.approx(0.5)
    assert result.improvement_se == pytest.approx(np.sqrt(0.5))
    assert result.threshold == pytest.approx(2 * np.sqrt(0.5))
    assert result.accepted
    assert not paired_losses([1, 3], [2, 6], 4, acceptance_se=3).accepted


@pytest.mark.parametrize(
    "acceptance_se,conclusive,reverse",
    [
        (*case_0, case_1)
        for case_0, case_1 in product(
            [
                (0, True),
                (np.nextafter(2.0, 0.0), True),
                (2.0, False),
                (np.nextafter(2.0, np.inf), False),
                (3.0, False),
            ],
            [False, True],
        )
    ],
)
def test_threshold(reverse, acceptance_se, conclusive):
    candidate, incumbent = [1, 1, 1], [1, 2, 3]
    if reverse:
        candidate, incumbent = incumbent, candidate
    result = paired_losses(
        candidate, incumbent, population=12, acceptance_se=acceptance_se
    )
    assert result.improvement_se == 0.5
    assert result.improvement == (-1 if reverse else 1)
    assert result.threshold == acceptance_se * 0.5
    assert result.accepted is (conclusive and not reverse)
    assert result.deteriorated is (conclusive and reverse)


@pytest.mark.parametrize("reverse", [False, True])
def test_population(reverse):
    candidate, incumbent = [1, 100], [2, 100]
    if reverse:
        candidate, incumbent = incumbent, candidate
    result = paired_losses(candidate, incumbent, population=2, acceptance_se=1e300)
    assert result.candidate_variance == result.incumbent_variance == 0
    assert result.improvement_se == result.threshold == 0
    assert result.improvement_variance == 0
    assert result.accepted is (not reverse)
    assert result.deteriorated is reverse


@pytest.mark.parametrize(
    "candidate,incumbent,accepted,deteriorated",
    [
        ([0, 0], [1, 1], True, False),
        ([0, 0], [0, 0], False, False),
        ([1, 2], [1, 2], False, False),
        ([1, 2], [2, 1], False, False),
        ([2, 2], [1, 1], False, True),
        ([1, 1], [0, 0], False, True),
    ],
)
def test_zeroties(candidate, incumbent, accepted, deteriorated):
    result = paired_losses(candidate, incumbent, population=10)
    assert result.accepted is accepted
    assert result.deteriorated is deteriorated


def test_bias():
    stale_absolute_best = 10.0
    result = paired_losses([2, 3], [1, 2], population=20)
    assert result.candidate_loss < stale_absolute_best
    assert result.improvement == -1 and not result.accepted


@pytest.mark.parametrize(
    "low,high,population,reverse",
    [
        (*case_0, case_1, case_2)
        for case_0, case_1, case_2 in product(
            [(1, 1 + 1e-8), (0, 1e-50)], [2, 20], [False, True]
        )
    ],
)
def test_fp32ties(reverse, population, low, high):
    candidate, incumbent = [low, low], [high, high]
    if reverse:
        candidate, incumbent = incumbent, candidate
    result = paired_losses(candidate, incumbent, population=population)
    assert abs(result.improvement) > result.threshold
    assert result.accepted is False and result.deteriorated is False


def test_ordered():
    next_value = float(np.nextafter(np.float32(1), np.float32(2)))
    assert paired_losses([1, 1], [next_value, next_value], 2).accepted
    assert paired_losses([next_value, next_value], [1, 1], 2).deteriorated


@pytest.mark.parametrize("reverse", [False, True])
def test_uncertaingain(reverse):
    candidate, incumbent = [2, 2], [1, 4]
    if reverse:
        candidate, incumbent = incumbent, candidate
    result = paired_losses(candidate, incumbent, population=20)
    assert 0 < abs(result.improvement) < result.threshold
    assert result.accepted is False and result.deteriorated is False


@pytest.mark.parametrize(
    "side,values",
    [
        (case_0, case_1)
        for case_0, case_1 in product(
            ["candidate", "incumbent"],
            [
                [],
                [1],
                [[1, 2]],
                [1, -1],
                [np.nan, 1],
                [np.inf, 1],
                [-np.inf, 1],
                [1j, 2],
                ["1", "2"],
            ],
        )
    ],
)
def test_invalidlosses(values, side):
    inputs = {"candidate": [1, 2], "incumbent": [1, 2]}
    inputs[side] = values
    with pytest.raises(ValueError):
        paired_losses(**inputs, population=4)


def test_losses():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_losses([1, 2], [1, 2, 3], population=4)


@pytest.mark.parametrize("population", [0, 1, -2, 4.0, True, "4"])
def test_pop(population):
    with pytest.raises(ValueError):
        paired_losses([1, 2], [2, 3], population)


@pytest.mark.parametrize("acceptance_se", [-1, np.inf, -np.inf, np.nan, True, "2", 1j])
def test_acceptance(acceptance_se):
    with pytest.raises(ValueError, match="acceptance_se"):
        paired_losses([1, 2], [2, 3], 4, acceptance_se)


@pytest.mark.parametrize(
    "values,population,statistic,side",
    [
        (*case_0, case_1)
        for case_0, case_1 in product(
            [
                ([1e39, 1e39], 2, "mean"),
                ([1e308, 1e308], 2, "mean"),
                ([0, 1e20], 4, "variance"),
            ],
            ["candidate", "incumbent"],
        )
    ],
)
def test_fp32overflow(side, values, population, statistic):
    inputs = {"candidate": [1, 2], "incumbent": [1, 2]}
    inputs[side] = values
    with pytest.raises(ValueError, match=f"{side} {statistic}.*float32"):
        paired_losses(**inputs, population=population)


def test_varcensus():
    result = paired_losses([0, 1e30], [1e30, 1e30], population=2)
    assert result.candidate_variance == result.incumbent_variance == 0
    assert result.accepted
