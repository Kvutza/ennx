from __future__ import annotations

import pytest
from enn_helpers import enn_rows


from testefit import _fitmodel, _makedata, _ybatch


def test_ennfitwithyvarnone():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _makedata(rng=rng, n=n, d=d, noise_std=0.1, yvar=None)
    model = ENN(x, y, train_yvar=yvar)
    result = _fitmodel(
        model,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
    )
    assert isinstance(result, ENNParams)
    assert result.k_neighbors == 5
    assert result.epistemic_scale > 0.0
    assert result.aleatoric_scale >= 0.0


def test_ennfitwithwarmstart():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _makedata(rng=rng, n=n, d=d, noise_std=0.1, yvar=0.01)
    model = ENN(x, y, yvar)
    result1 = _fitmodel(
        model,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
    )
    result2 = _fitmodel(
        model,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
        params_warm_start=result1,
    )
    assert isinstance(result2, ENNParams)
    assert result2.k_neighbors == 5
    assert result2.epistemic_scale > 0.0
    assert result2.aleatoric_scale >= 0.0


def test_003():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fit import subsample_loglik
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(123)
    x = rng.standard_normal((60, 3))
    y1 = x @ [1.0, -2.0, 0.5] + 0.1 * rng.standard_normal(60)
    y2 = np.sin(x @ [-0.5, 0.25, 1.25]) + 0.3 * rng.standard_normal(60)
    y = np.column_stack([y1, y2]).astype(float)
    model = ENN(x, y, np.ones_like(y) * [[0.01, 0.09]])
    params = _fitmodel(
        model,
        k=12,
        num_candidates=40,
        num_samples=25,
        rng=np.random.default_rng(456),
    )
    assert isinstance(params, ENNParams) and params.k_neighbors == 12
    lls = subsample_loglik(
        model, x, y, paramss=[params], P=25, rng=np.random.default_rng(789)
    )
    assert len(lls) == 1 and np.isfinite(lls[0])


def test_004():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(42)
    x, y, yvar = _makedata(rng=rng, n=40, d=3, noise_std=0.2, yvar=0.01)
    model = ENN(x, y, yvar)
    warm = ENNParams(
        k_neighbors=5,
        epistemic_scale=1.0,
        aleatoric_scale=123.0,
    )
    result = _fitmodel(
        model,
        k=5,
        num_candidates=25,
        num_samples=15,
        rng=rng,
        params_warm_start=warm,
        infer_aleatoric_variance_scale=False,
    )
    assert isinstance(result, ENNParams)
    assert result.k_neighbors == 5
    assert result.aleatoric_scale == 0.0


def test_005():
    import numpy as np
    import pytest

    from ennx.ennx.enn_fitter import ENNStatefulFitter

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="finite"):
        fitter.tell([[float("nan"), 0.0]], [[0.0]])


def test_006():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(2020)
    x = rng.standard_normal((12, 2))
    y = (10.0 * rng.standard_normal((12, 1))).astype(float)
    model = ENN(x, y)

    fitter_batch = ENNStatefulFitter(k=3, rng=np.random.default_rng(100))
    fitter_batch.tell(x, y)
    p_batch = fitter_batch.ask(model, num_candidates=8, num_samples=6)

    fitter_inc = ENNStatefulFitter(k=3, rng=np.random.default_rng(100))
    for row_x, row_y in zip(x, y):
        fitter_inc.tell(row_x.reshape(1, -1), row_y.reshape(1, -1))
    p_inc = fitter_inc.ask(model, num_candidates=8, num_samples=6)

    assert p_batch.k_neighbors == p_inc.k_neighbors
    assert abs(p_batch.epistemic_scale - p_inc.epistemic_scale) < 1e-9
    assert abs(p_batch.aleatoric_scale - p_inc.aleatoric_scale) < 1e-9


def test_007():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter
    from ennx.ennx.enn_params import ENNParams

    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0], [1.5]], dtype=float)
    model = ENN(x, y)
    warm = ENNParams(k_neighbors=2, epistemic_scale=0.01, aleatoric_scale=0.01)

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(55))
    fitter.tell(x, y)
    p_cold = fitter.ask(model, num_candidates=6, num_samples=4)
    p_warm = fitter.ask(
        model,
        num_candidates=6,
        num_samples=4,
        params_warm_start=warm,
    )
    assert np.isfinite(p_cold.epistemic_scale)
    assert np.isfinite(p_warm.epistemic_scale)
    assert (
        abs(p_cold.epistemic_scale - p_warm.epistemic_scale) > 1e-12
        or abs(p_cold.aleatoric_scale - p_warm.aleatoric_scale) > 1e-12
    )


def test_008():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(88)
    x_all = rng.standard_normal((8, 2))
    y_all = rng.standard_normal((8, 1))
    model = ENN(np.empty((0, 2)), np.empty((0, 1)))
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(88))
    for row_x, row_y in zip(x_all, y_all):
        model.add(row_x.reshape(1, -1), row_y.reshape(1, -1))
        fitter.tell(row_x.reshape(1, -1), row_y.reshape(1, -1))
    params = fitter.ask(model, num_candidates=5, num_samples=4)
    assert np.isfinite(params.epistemic_scale)
    assert params.epistemic_scale > 0.0


def test_009():
    import numpy as np
    import pytest

    from ennx.ennx.enn_fitter import ENNStatefulFitter

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="finite"):
        fitter.tell([[0.0]], [[float("nan")]])


def test_010():
    import numpy as np
    import pytest

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    y = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=float)
    model = ENN(x, y)
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(77))

    with pytest.raises(ValueError, match="tell"):
        fitter.ask(model, num_candidates=5, num_samples=5)


def test_011():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0], [1.5]], dtype=float)
    ENN(x, y)
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    for i in range(y.shape[0]):
        fitter.tell(x[i : i + 1], y[i : i + 1])
        _ybatch(fitter.y_std(), y[: i + 1].std(axis=0))


def test_012():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array(
        [[0.0, 1.0], [1.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 1.5]],
        dtype=float,
    )
    ENN(x, y)
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    for i in range(y.shape[0]):
        fitter.tell(x[i : i + 1], y[i : i + 1])
        _ybatch(fitter.y_std(), y[: i + 1].std(axis=0))


def test_013():
    import numpy as np
    import pytest

    from ennx.ennx.enn_fitter import ENNStatefulFitter

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="finite"):
        fitter.tell([[0.0, 0.0]], [[0.0]], [[float("nan")]])
    fitter.tell([[0.0, 0.0]], [[0.0]], [[0.1]])


def test_014():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter
    from ennx.ennx.enn_params import ENNParams

    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    model = ENN(x, y)
    warm = ENNParams(k_neighbors=2, epistemic_scale=2.5, aleatoric_scale=0.3)

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(7))
    fitter.tell(x, y)
    params = fitter.ask(
        model,
        num_candidates=0,
        num_samples=2,
        params_warm_start=warm,
    )
    assert params.k_neighbors == 2
    assert abs(params.epistemic_scale - 2.5) < 1e-12


def test_015():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    y = np.array([[0.0], [0.0], [0.0], [100.0], [100.0], [100.0]], dtype=float)
    model = ENN(x, y)

    fitter_sync = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    fitter_sync.tell(x, y)
    _, y_all, _ = enn_rows(model)
    model_std = y_all.std(axis=0)

    fitter_desync = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    fitter_desync.tell(x[:3], y[:3])
    assert abs(fitter_desync.y_std()[0] - model_std[0]) > 1.0
    assert abs(fitter_sync.y_std()[0] - model_std[0]) < 1e-6

    p_desync = fitter_desync.ask(model, num_candidates=6, num_samples=5)
    assert np.isfinite(p_desync.epistemic_scale)


def test_016():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    model = ENN(
        np.array([[0.0, 0.0]]),
        np.array([[0.0]]),
    )
    fitter = ENNStatefulFitter(k=7, rng=np.random.default_rng(1))
    x_all, y_all, yvar_all = enn_rows(model)
    fitter.tell(x_all, y_all, yvar_all)
    params = fitter.ask(model, num_candidates=5, num_samples=3)
    assert params.k_neighbors == 7
    assert params.epistemic_scale == 1.0
    assert params.aleatoric_scale == 0.0
