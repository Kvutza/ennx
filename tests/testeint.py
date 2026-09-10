import numpy as np

from ennx.ennx.draw_internals import DrawInternals
from ennx.ennx.neighbor_data import NeighborData
from ennx.ennx.weighted_stats import WeightedStats


def test_drawinternals2():
    di = DrawInternals(
        idx=np.array([[0, 1]]),
        w_normalized=np.array([[[0.5], [0.5]]]),
        l2=np.array([[1.0]]),
        mu=np.array([[0.5]]),
        se=np.array([[0.1]]),
        se_epi=np.array([[0.1]]),
        se_ale=np.array([[0.0]]),
    )
    assert di.idx.shape == (1, 2)
    assert di.mu.shape == (1, 1)


def test_neighbordata2():
    nd = NeighborData(
        dist2s=np.array([[0.1, 0.2]]),
        idx=np.array([[0, 1]]),
        y_neighbors=np.array([[[1.0], [2.0]]]),
        k=2,
    )
    assert nd.k == 2
    assert nd.idx.shape == (1, 2)


def test_weightedstats():
    ws = WeightedStats(
        w_normalized=np.array([[[0.5], [0.5]]]),
        l2=np.array([[1.0]]),
        mu=np.array([[0.5]]),
        se=np.array([[0.1]]),
        se_epi=np.array([[0.1]]),
        se_ale=np.array([[0.0]]),
    )
    assert ws.mu.shape == (1, 1)
