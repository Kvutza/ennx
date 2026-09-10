from __future__ import annotations


def test_001():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fit import ENNIncrementalDelta, enn_fit
    from ennx.ennx.enn_fitter import ENNStatefulFitter
    from ennx.ennx.enn_params import ENNParams

    rng = np.random.default_rng(7)
    x_all = rng.standard_normal((12, 2))
    y_all = rng.standard_normal((12, 1))
    yvar_all = 0.01 * np.ones_like(y_all)

    model = ENN(np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)))
    fitter = ENNStatefulFitter(k=3, rng=np.random.default_rng(7))
    params: ENNParams | None = None
    for row_x, row_y, row_yvar in zip(x_all, y_all, yvar_all):
        row_x = row_x.reshape(1, -1)
        row_y = row_y.reshape(1, -1)
        row_yvar = row_yvar.reshape(1, -1)
        model.add(row_x, row_y, row_yvar)
        params = enn_fit(
            model,
            k=3,
            num_candidates=1,
            num_samples=8,
            rng=np.random.default_rng(7),
            params_warm_start=params,
            incremental=ENNIncrementalDelta(fitter, row_x, row_y, row_yvar),
        )
    assert isinstance(params, ENNParams)
    assert params.k_neighbors == 3
    assert np.isfinite(params.epistemic_scale)


def test_002():
    import numpy as np

    from ennx.ennx.enn_class import ENN
    from ennx.ennx.enn_fit import ENNIncrementalDelta, enn_fit
    from ennx.ennx.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(99)
    x_all = rng.standard_normal((8, 2))
    y_all = rng.standard_normal((8, 1))

    via_enn_fit = ENN(np.empty((0, 2)), np.empty((0, 1)))
    fitter_a = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    params_a = None

    via_manual = ENN(np.empty((0, 2)), np.empty((0, 1)))
    fitter_b = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    params_b = None

    for row_x, row_y in zip(x_all, y_all):
        row_x = row_x.reshape(1, -1)
        row_y = row_y.reshape(1, -1)
        via_enn_fit.add(row_x, row_y)
        params_a = enn_fit(
            via_enn_fit,
            k=2,
            num_candidates=1,
            num_samples=6,
            rng=np.random.default_rng(99),
            params_warm_start=params_a,
            incremental=ENNIncrementalDelta(fitter_a, row_x, row_y),
        )

        via_manual.add(row_x, row_y)
        fitter_b.tell(row_x, row_y)
        params_b = fitter_b.ask(
            via_manual,
            num_candidates=1,
            num_samples=6,
            params_warm_start=params_b,
        )

    assert params_a is not None and params_b is not None
    assert params_a.k_neighbors == params_b.k_neighbors
    assert abs(params_a.epistemic_scale - params_b.epistemic_scale) < 1e-12
    assert abs(params_a.aleatoric_scale - params_b.aleatoric_scale) < 1e-12
