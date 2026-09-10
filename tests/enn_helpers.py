from __future__ import annotations


def sphere_objective(x):
    import numpy as np

    return -np.sum(x**2, axis=1)


def enn_model(n=20, d=3, seed=0, yvar_scale=0.1):
    import numpy as np

    from ennx.ennx.enn_class import ENN

    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = yvar_scale * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    return model, train_x, train_y, train_yvar, rng


def enn_rows(model):
    """Return (x, y, yvar?) for all rows via index-based gather."""
    n = len(model)
    return model.train_rows(list(range(n)))
