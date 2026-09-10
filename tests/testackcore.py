import numpy as np

from ennx.benchmarks.ackley_core import ackley_core


def test_ackleycore1d():
    x = np.array([1.0, 2.0])
    result = ackley_core(x)
    assert result.shape == (1,)


def test_ackleycore2d():
    x = np.array([[1.0, 2.0], [0.0, 0.0]])
    result = ackley_core(x)
    assert result.shape == (2,)
    assert result[1] > result[0]
