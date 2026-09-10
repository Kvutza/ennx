from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray
    se_epi: np.ndarray
    se_ale: np.ndarray
    idx: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    def sample(
        self,
        num_samples: int,
        rng: Generator,
        clip: float | None = None,
    ) -> np.ndarray:
        import numpy as np

        size = (*self.se.shape, num_samples)
        eps = rng.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)
        draws = np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps
        if self.y_bounds is None:
            return draws
        bounds = np.asarray(self.y_bounds, dtype=float)
        if np.all(np.isneginf(bounds[:, 0]) & np.isposinf(bounds[:, 1])):
            return draws
        for j, (lower, upper) in enumerate(bounds):
            mu = np.asarray(self.mu[..., j], dtype=float)
            se = np.asarray(self.se[..., j], dtype=float)
            if np.isfinite(lower) and np.isfinite(upper):
                u = (mu - lower) / (upper - lower)
                z = np.log(u / (1.0 - u))
                sigmoid = 1.0 / (1.0 + np.exp(-z))
                jac = (upper - lower) * sigmoid * (1.0 - sigmoid)
                z_draws = z[..., None] + (se / jac)[..., None] * eps[..., j, :]
                s = 1.0 / (1.0 + np.exp(-z_draws))
                draws[..., j, :] = lower + (upper - lower) * s
            elif np.isfinite(lower):
                z = np.log(mu - lower)
                z_draws = z[..., None] + (se / np.exp(z))[..., None] * eps[..., j, :]
                draws[..., j, :] = lower + np.exp(z_draws)
            elif np.isfinite(upper):
                z = -np.log(upper - mu)
                z_draws = z[..., None] + (se / np.exp(-z))[..., None] * eps[..., j, :]
                draws[..., j, :] = upper - np.exp(-z_draws)
        return draws
