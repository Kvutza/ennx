from __future__ import annotations

import pytest
from enn_helpers import enn_rows


def _fitmodel(
    model,
    *,
    k: int,
    num_candidates: int,
    num_samples: int,
    rng,
    params_warm_start=None,
    infer_aleatoric_variance_scale: bool = True,
):
    from ennx.ennx.enn_fit import ENNIncrementalDelta, enn_fit
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    fitter = ENNStatefulFitter(
        k=k,
        rng=rng,
        infer_aleatoric_variance_scale=infer_aleatoric_variance_scale,
    )
    return enn_fit(
        model,
        k=k,
        num_candidates=num_candidates,
        num_samples=num_samples,
        rng=rng,
        params_warm_start=params_warm_start,
        incremental=ENNIncrementalDelta(
            fitter,
            *enn_rows(model),
        ),
    )


def _ybatch(inc_std, batch_std) -> None:
    for a, b in zip(inc_std, batch_std):
        expected = b if b > 1e-10 else 1.0
        assert abs(a - expected) < 1e-10


def test_001():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fit import subsample_loglik
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 2))
    y = (x @ np.array([1.5, -0.5]) + 0.1 * rng.standard_normal(40)).reshape(-1, 1)
    model = ENN(x, y, 0.01 * np.ones_like(y))
    result = _fitmodel(
        model,
        k=10,
        num_candidates=30,
        num_samples=20,
        rng=np.random.default_rng(1),
    )
    assert (
        isinstance(result, ENNParams)
        and result.k_neighbors == 10
        and result.epistemic_scale > 0.0
    )
    tuned_ll = subsample_loglik(
        model, x, y[:, 0], paramss=[result], P=20, rng=np.random.default_rng(2)
    )[0]
    assert np.isfinite(tuned_ll), "tuned log-likelihood must be finite"


def _captureresult(model, params, num_samples, eval_x, eval_y):
    import numpy as np

    from ennx.ennx.enn_fit import subsample_loglik

    loglik = subsample_loglik(
        model,
        eval_x,
        eval_y,
        paramss=[params],
        P=len(eval_x),
        rng=np.random.default_rng(2000 + num_samples),
    )[0]
    return {
        "num_samples": num_samples,
        "params": {
            "k_neighbors": params.k_neighbors,
            "epistemic_scale": params.epistemic_scale,
            "aleatoric_scale": params.aleatoric_scale,
        },
        "subsample_loglik": loglik,
    }


def _runsweep(x_train, y_train, y_var_train, sample_sizes):
    import numpy as np

    from ennx.ennx.enn_class import ENN

    captured = []
    for num_samples in sample_sizes:
        model = ENN(x_train, y_train, y_var_train)
        params = _fitmodel(
            model,
            k=10,
            num_candidates=100,
            num_samples=num_samples,
            rng=np.random.default_rng(1000 + num_samples),
        )
        captured.append(_captureresult(model, params, num_samples, x_train, y_train))
    return captured


def _runsweep2(x_train, y_train, y_var_train, sample_sizes):
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fit import ENNIncrementalDelta, enn_fit
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    incremental_model = ENN(
        np.empty((0, x_train.shape[1])),
        np.empty((0, y_train.shape[1])),
        np.empty((0, y_var_train.shape[1])),
    )
    fitter = ENNStatefulFitter(k=10, rng=np.random.default_rng(4242))
    params_warm_start = None
    captured = []
    for num_samples, (x, y, y_var) in enumerate(
        zip(x_train, y_train, y_var_train), start=1
    ):
        row_x = x.reshape(1, -1)
        row_y = y.reshape(1, -1)
        row_yvar = y_var.reshape(1, -1)
        incremental_model.add(row_x, row_y, row_yvar)
        params_warm_start = enn_fit(
            incremental_model,
            k=10,
            num_candidates=1,
            num_samples=100,
            rng=np.random.default_rng(4242),
            params_warm_start=params_warm_start,
            incremental=ENNIncrementalDelta(fitter, row_x, row_y, row_yvar),
        )
        if num_samples in sample_sizes:
            captured.append(
                _captureresult(
                    incremental_model,
                    params_warm_start,
                    num_samples,
                    x_train[:num_samples],
                    y_train[:num_samples],
                )
            )
    return captured


@pytest.mark.slow
def test_002():
    import json

    import numpy as np

    rng = np.random.default_rng(20260522)
    x_train = rng.standard_normal((1000, 3))
    y_train = (
        x_train @ np.array([1.5, -0.5, 0.25]) + 0.1 * rng.standard_normal(1000)
    ).reshape(-1, 1)
    y_var_train = 0.01 * np.ones_like(y_train)
    sample_sizes = [10, 30, 100, 300, 1000]

    batch_captured = _runsweep(x_train, y_train, y_var_train, sample_sizes)
    incremental_captured = _runsweep2(x_train, y_train, y_var_train, sample_sizes)

    print(
        json.dumps(
            {"batch": batch_captured, "incremental": incremental_captured}, indent=2
        )
    )


def _makedata(
    *,
    rng,
    n: int,
    d: int,
    noise_std: float,
    yvar: float | None,
):
    import numpy as np

    x = rng.standard_normal((n, d))
    y = x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * float(noise_std)
    if yvar is None:
        return x, y, None
    return x, y, float(yvar) * np.ones_like(y)


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
