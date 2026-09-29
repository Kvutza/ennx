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
