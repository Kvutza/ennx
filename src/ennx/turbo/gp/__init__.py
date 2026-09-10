from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

if TYPE_CHECKING:
    import torch
    from numpy.random import Generator


@dataclass(slots=True)
class Posterior:
    mu: np.ndarray
    sigma: np.ndarray


@contextlib.contextmanager
def _seed(seed: int, dev: torch.device | Any | None = None) -> Iterator[None]:
    import torch

    ids: list[int] | None = None
    if dev is not None and getattr(dev, "type", None) == "cuda":
        ids = [0 if getattr(dev, "index", None) is None else int(dev.index)]
    with torch.random.fork_rng(devices=ids, enabled=True):
        torch.manual_seed(int(seed))
        if dev is not None and getattr(dev, "type", None) == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        if (
            dev is not None
            and getattr(dev, "type", None) == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "manual_seed")
        ):
            torch.mps.manual_seed(int(seed))
        yield


def _post(model: Any, x: Any) -> Any:
    try:
        from gpytorch.utils.warnings import GPInputWarning
    except ImportError:
        return model.posterior(x)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The input matches the stored training data\..*",
            category=GPInputWarning,
        )
        return model.posterior(x)


class Surrogate:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._mean: float | Any = 0.0
        self._std: float | Any = 1.0
        self._ls: np.ndarray | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return self._ls

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
        *,
        steps: int = 0,
        rng: Generator | None = None,
    ) -> Self:
        from .train import fit

        del rng
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
            if yvar is not None:
                yvar = np.asarray(yvar, dtype=float).ravel()
        out = fit(x, y, x.shape[1], var=yvar, steps=steps)
        self._model = out.model
        self._mean = out.mean
        self._std = out.std
        self._ls = self._lengths()
        return self

    def _lengths(self) -> np.ndarray | None:
        if self._model is None:
            return None
        ls = self._model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()
        if ls.ndim == 3:
            ls = ls.mean(axis=0)
        ls = ls.ravel()
        ls /= ls.mean()
        return ls / np.prod(ls ** (1.0 / len(ls)))

    def _scale(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError(y.shape)
        mean = np.asarray(self._mean, dtype=float).reshape(-1)
        std = np.asarray(self._std, dtype=float).reshape(-1)
        m = y.shape[1]
        if mean.size == 1 and m != 1:
            mean = np.full(m, mean[0])
        if std.size == 1 and m != 1:
            std = np.full(m, std[0])
        return mean[None, :] + std[None, :] * y

    def predict(self, x: np.ndarray) -> Posterior:
        import torch

        if self._model is None:
            raise RuntimeError("GP is not fitted")
        xt = torch.as_tensor(x, dtype=torch.float64)
        with torch.no_grad():
            post = _post(self._model, xt)
            mu = post.mean.cpu().numpy()
            var = post.variance.cpu().numpy()
        if mu.ndim == 1:
            mu = mu[:, None]
            var = var[:, None]
        elif mu.ndim == 2:
            mu = mu.T
            var = var.T
        else:
            raise ValueError(mu.shape)

        std = np.asarray(self._std, dtype=float).reshape(-1)
        if std.size == 1 and mu.shape[1] != 1:
            std = np.full(mu.shape[1], std[0])
        return Posterior(self._scale(mu), std[None, :] * np.sqrt(var))

    def sample(self, x: np.ndarray, n: int, rng: Generator) -> np.ndarray:
        return self.draw(x, n, int(rng.integers(2**31 - 1)))

    def draw(self, x: np.ndarray, n: int, seed: int) -> np.ndarray:
        import gpytorch
        import torch

        if self._model is None:
            raise RuntimeError("GP is not fitted")
        xt = torch.as_tensor(x, dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), _seed(seed, xt.device):
            vals = self._model.posterior(xt).sample(torch.Size([n]))
        vals = vals.detach().cpu().numpy()
        m = np.asarray(self._mean).size
        if vals.ndim == 2:
            vals = vals[:, :, None]
        else:
            vals = np.transpose(vals, (0, 2, 1))
        shape = (n, len(x), m)
        if vals.shape != shape:
            raise ValueError(f"GP samples have shape {vals.shape}, expected {shape}")
        mean = np.asarray(self._mean, dtype=float).reshape(1, 1, -1)
        std = np.asarray(self._std, dtype=float).reshape(1, 1, -1)
        return mean + std * vals


__all__ = ["Posterior", "Surrogate"]
