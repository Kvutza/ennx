from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .. import _rust
from .config.encode import encode, enn_k, is_lhd, supports
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, GPSurrogateConfig, NoSurrogateConfig


class _View:
    def __init__(self, x: np.ndarray) -> None:
        self._x = np.asarray(x, dtype=float)

    def view(self) -> np.ndarray:
        return self._x


@dataclass(frozen=True, slots=True)
class _Obs:
    x: np.ndarray
    y: np.ndarray
    yvar: np.ndarray | None


@dataclass(frozen=True, slots=True)
class Telemetry:
    dt_fit: float
    dt_sel: float
    dt_gen: float = 0.0
    dt_tell: float = 0.0
    num_candidates: int = 0


@dataclass(frozen=True, slots=True)
class _Plan:
    kind: str
    k: int = 10
    gp: Any | None = None


def _obs(x: np.ndarray, y: np.ndarray, yvar: np.ndarray | None, d: int) -> _Obs:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(f"x must have shape (n, {d}), got {x.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must have shape ({x.shape[0]}, m), got {y.shape}")
    if yvar is not None:
        yvar = np.asarray(yvar, dtype=float)
        if yvar.ndim == 1:
            yvar = yvar.reshape(-1, 1)
        if yvar.shape != y.shape:
            raise ValueError(f"y_var must have shape {y.shape}, got {yvar.shape}")
    return _Obs(x, y, yvar)


def _plan(cfg: OptimizerConfig) -> _Plan:
    if is_lhd(cfg):
        return _Plan("lhd")
    if isinstance(cfg.surrogate, ENNSurrogateConfig):
        return _Plan("enn", enn_k(cfg))
    if isinstance(cfg.surrogate, NoSurrogateConfig):
        return _Plan("zero")
    if isinstance(cfg.surrogate, GPSurrogateConfig):
        from .gp import Surrogate

        return _Plan("gp", gp=Surrogate())
    raise ValueError(f"unsupported surrogate: {type(cfg.surrogate)!r}")


class Optimizer:
    """NumPy facade over the optimizer core."""

    def __init__(self, d: int, rng: Generator, inner: Any) -> None:
        self._d = d
        self._rng = rng
        self._inner = inner

    @property
    def _x_obs(self) -> _View:
        x = self._inner.x_obs()
        return _View(np.empty((0, self._d)) if x is None else x)

    @property
    def _y_obs(self) -> _View:
        y = self._inner.y_obs()
        return _View(np.empty((0, 1)) if y is None else y)

    @property
    def tr_obs_count(self) -> int:
        return int(self._inner.tr_obs_count())

    @property
    def tr_length(self) -> float:
        return float(self._inner.tr_length())

    @property
    def init_progress(self) -> tuple[int, int] | None:
        return self._inner.init_progress()

    def telemetry(self) -> Telemetry:
        t = self._inner.telemetry()
        return Telemetry(
            dt_fit=t.dt_fit,
            dt_gen=t.dt_gen,
            dt_sel=t.dt_sel,
            dt_tell=t.dt_tell,
            num_candidates=int(t.num_candidates),
        )

    def ask(self, num_arms: int) -> np.ndarray:
        """Return `(num_arms, d)` candidates inside the configured bounds."""
        n = int(num_arms)
        if n <= 0:
            raise ValueError(f"num_arms must be > 0, got {n}")
        seed = int(self._rng.integers(2**63 - 1))
        return np.asarray(self._inner.ask(n, seed), dtype=float)

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        """Add an observation batch and return its 2D objective array."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr.flags.writeable = False
        y_arr.flags.writeable = False

        y_var_arr = None
        if y_var is not None:
            y_var_arr = np.asarray(y_var, dtype=float)
            y_var_arr.flags.writeable = False

        obs = _obs(x_arr, y_arr, y_var_arr, self._d)
        seed = int(self._rng.integers(2**63 - 1))
        self._inner.tell(obs.x, obs.y, seed, obs.yvar)
        return obs.y


def create_optimizer(
    *, bounds: np.ndarray, config: OptimizerConfig, rng: Generator
) -> Optimizer:
    """Build an optimizer from the Python config."""
    if not supports(config):
        raise ValueError(f"unsupported optimizer config: {config!r}")

    bounds = np.asarray(bounds, dtype=float)
    plan = _plan(config)
    n = config.init.num_init or 10
    seed = int(rng.integers(2**63 - 1))
    inner = _rust.create_optimizer(
        bounds,
        plan.kind,
        plan.k,
        n,
        4,
        seed,
        cfg=encode(config),
        gp=plan.gp,
    )
    return Optimizer(len(bounds), rng, inner)
