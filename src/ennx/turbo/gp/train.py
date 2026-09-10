from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ennx.ennx.enn_util import standardize_y

from .model import Exact, Noisy


@dataclass(slots=True)
class _Fit:
    model: Any
    likelihood: Any
    mean: Any
    std: Any


@dataclass(slots=True)
class _Data:
    x: Any
    y: Any
    multi: bool
    mean: Any
    std: Any
    raw: np.ndarray


def _prep(x: Any, y: Any, var: Any | None) -> _Data:
    import torch

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim not in (1, 2):
        raise ValueError(y.shape)
    multi = y.ndim == 2 and y.shape[1] > 1
    if var is not None and np.asarray(var).shape != y.shape:
        raise ValueError(f"y_var has shape {np.asarray(var).shape}, expected {y.shape}")
    if var is not None and multi:
        raise ValueError("y_var is not supported for multi-output GP")
    if multi:
        mean, std = y.mean(0), y.std(0)
        std = np.where(std < 1e-6, 1.0, std)
        train_y = torch.as_tensor(((y - mean) / std).T, dtype=torch.float64)
    else:
        mean, std = standardize_y(y)
        train_y = torch.as_tensor((y - mean) / std, dtype=torch.float64)
    return _Data(torch.as_tensor(x, dtype=torch.float64), train_y, multi, mean, std, y)


def _build(data: _Data, d: int, var: Any | None) -> tuple[Any, Any]:
    import torch
    from gpytorch.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood

    ls = Interval(0.005, 2.0)
    scale = Interval(0.05, 20.0)
    if var is not None:
        noise = torch.as_tensor(np.asarray(var) / data.std**2, dtype=torch.float64)
        gp = Noisy(data.x, data.y, noise, ls, scale, d).to(dtype=data.x.dtype)
        return gp, gp.likelihood

    noise = Interval(5e-4, 0.2)
    m = int(data.raw.shape[1]) if data.multi else None
    batch = torch.Size([m]) if data.multi else torch.Size()
    like = GaussianLikelihood(noise_constraint=noise, batch_shape=batch).to(
        dtype=data.y.dtype
    )
    gp = Exact(data.x, data.y, like, ls, scale, d).to(dtype=data.x.dtype)
    like.noise = (
        torch.full((m,), 0.005, dtype=data.y.dtype)
        if data.multi
        else torch.tensor(0.005, dtype=data.y.dtype)
    )
    return gp, like


def _init(gp: Any, data: _Data, d: int) -> None:
    import torch

    if data.multi:
        m = int(data.raw.shape[1])
        gp.covar_module.outputscale = torch.ones(m, dtype=data.x.dtype)
        gp.covar_module.base_kernel.lengthscale = torch.full(
            (m, 1, d), 0.5, dtype=data.x.dtype
        )
    else:
        gp.covar_module.outputscale = torch.tensor(1.0, dtype=data.x.dtype)
        gp.covar_module.base_kernel.lengthscale = torch.full(
            (d,), 0.5, dtype=data.x.dtype
        )


def _train(gp: Any, like: Any, data: _Data, steps: int) -> None:
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    gp.train()
    like.train()
    mll = ExactMarginalLogLikelihood(like, gp)
    opt = torch.optim.Adam(gp.parameters(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(gp(data.x), data.y)
        (loss.sum() if loss.ndim else loss).backward()
        opt.step()
    gp.eval()
    like.eval()


def fit(
    x: Any,
    y: Any,
    d: int,
    *,
    var: Any | None = None,
    steps: int = 50,
) -> _Fit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    multi = y.ndim == 2 and y.shape[1] > 1
    if len(x) == 0:
        if multi:
            return _Fit(None, None, np.zeros(y.shape[1]), np.ones(y.shape[1]))
        return _Fit(None, None, 0.0, 1.0)
    if len(x) == 1 and multi:
        return _Fit(None, None, y[0].copy(), np.ones(y.shape[1]))

    data = _prep(x, y, var)
    gp, like = _build(data, d, var)
    _init(gp, data, d)
    _train(gp, like, data, steps)
    return _Fit(gp, like, data.mean, data.std)
