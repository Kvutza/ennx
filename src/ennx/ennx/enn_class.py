from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ennx._rust import ENN as _RustENN
from ennx.turbo.config.eidxdrv import ENNIndexDriver

from .eclssup import _rustname, _rustseeds

if TYPE_CHECKING:
    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


def _posteriorcoerced(flags):
    from .enn_params import PosteriorFlags

    return flags if flags is not None else PosteriorFlags()


def _rustkwargs(params, flags, seeds: list[int]) -> dict[str, Any]:
    return {
        "k_neighbors": params.k_neighbors,
        "epistemic_scale": params.epistemic_scale,
        "aleatoric_scale": params.aleatoric_scale,
        "function_seeds": seeds,
        "exclude_nearest": flags.exclude_nearest,
        "observation_noise": flags.observation_noise,
    }


def _finalizedraw(
    x: np.ndarray, draws: np.ndarray, idx: list | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return draws as (batch, metrics, num_samples), matching ENNNormal.sample()."""
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(
            f"function draws must be 3D (num_samples, batch, metrics), "
            f"got shape {draws_arr.shape}"
        )
    # Rust returns (num_samples, batch, metrics); match posterior().sample().
    draws_arr = np.transpose(draws_arr, (1, 2, 0))
    idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0), dtype=int)
    return draws_arr, idx_arr


class ENN:
    _EPS_VAR = 1e-9

    @staticmethod
    def _inputs(train_x, train_y, train_yvar):
        train_x, train_y = (
            np.asarray(train_x, dtype=float),
            np.asarray(train_y, dtype=float),
        )
        if (
            train_x.ndim != 2
            or train_y.ndim != 2
            or train_x.shape[0] != train_y.shape[0]
        ):
            raise ValueError((train_x.shape, train_y.shape))
        if train_yvar is not None:
            train_yvar = np.asarray(train_yvar, dtype=float)
            if train_yvar.ndim != 2 or train_y.shape != train_yvar.shape:
                raise ValueError((train_y.shape, train_yvar.shape))
        return train_x, train_y, train_yvar

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        *,
        scale_x: bool = False,
        index_driver: ENNIndexDriver = ENNIndexDriver.FLAT,
        work_dir: str | os.PathLike[str] | None = None,
        enn_storage: str | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> None:
        train_x, train_y, train_yvar = self._inputs(train_x, train_y, train_yvar)
        if scale_x and index_driver == ENNIndexDriver.BPANN_DISK:
            raise ValueError("scale_x=True is not compatible with BPANN_DISK")
        self._index_driver = index_driver
        idx_driver = _rustname(index_driver)
        rust_kwargs: dict[str, Any] = {
            "train_x": train_x,
            "train_y": train_y,
            "train_yvar": train_yvar,
            "scale_x": scale_x,
            "index_driver": idx_driver,
        }
        if work_dir is not None:
            rust_kwargs["work_dir"] = os.fspath(work_dir)
        if enn_storage is not None:
            rust_kwargs["enn_storage"] = enn_storage
        if y_bounds is not None:
            rust_kwargs["y_bounds"] = np.asarray(y_bounds, dtype=float)
        self._rust_model = _RustENN(**rust_kwargs)
        self._tie_break_neighbors: bool = True

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
    ) -> None:
        x, y, yvar = self._inputs(x, y, yvar)
        self._rust_model.add(x, y, yvar)

    def ensure_sync(self) -> None:
        self._rust_model.ensure_sync()

    def schedule_flush(self) -> None:
        self._rust_model.schedule_flush()

    def persist_index(self) -> None:
        self._rust_model.persist_index()

    def train_rows(
        self, indices: list[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        idx = [int(i) for i in np.asarray(indices, dtype=int).ravel()]
        x, y, yvar = self._rust_model.train_rows(idx)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        yvar_arr = None if yvar is None else np.asarray(yvar, dtype=float)
        return x_arr, y_arr, yvar_arr

    def index_bytes(self) -> int:
        return int(self._rust_model.index_bytes())

    @property
    def num_outputs(self) -> int:
        return int(self._rust_model.num_outputs)

    @property
    def rust_backend(self):
        """Rust surrogate implementation (same lifetime as this wrapper)."""
        return self._rust_model

    @property
    def _numdim(self) -> int:
        return int(self._rust_model.num_dim)

    @property
    def _nummetrics(self) -> int:
        return int(self._rust_model.num_outputs)

    @property
    def _xscale(self) -> np.ndarray:
        return np.asarray(self._rust_model.x_scale, dtype=float)

    @property
    def _yscale(self) -> np.ndarray:
        return np.asarray(self._rust_model.y_scale, dtype=float)

    @property
    def y_bounds(self) -> np.ndarray:
        return np.asarray(self._rust_model.y_bounds, dtype=float)

    @property
    def _scalesx(self) -> bool:
        return bool(self._rust_model.scale_x)

    @property
    def _ytrain(self) -> np.ndarray:
        n = len(self)
        _, y, _ = self.train_rows(list(range(n)))
        return y

    @property
    def _trainyvar(self) -> np.ndarray | None:
        n = len(self)
        _, _, yvar = self.train_rows(list(range(n)))
        return yvar

    def __len__(self) -> int:
        return len(self._rust_model)

    def posterior(
        self,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posteriorcoerced(flags)
        if self._tie_break_neighbors != flags.tie_neighbors:
            self._rust_model.set_neighbors(flags.tie_neighbors)
            self._tie_break_neighbors = flags.tie_neighbors

        mu, se, se_epi, se_ale, idx = self._rust_model.posterior(
            x,
            k_neighbors=params.k_neighbors,
            epistemic_scale=params.epistemic_scale,
            aleatoric_scale=params.aleatoric_scale,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=self.y_bounds)

    def conditional_posterior(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posteriorcoerced(flags)
        self._rust_model.set_neighbors(flags.tie_neighbors)

        mu, se, se_epi, se_ale, _ = self._rust_model.conditional_posterior(
            x_whatif,
            y_whatif,
            x,
            k_neighbors=params.k_neighbors,
            epistemic_scale=params.epistemic_scale,
            aleatoric_scale=params.aleatoric_scale,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        return ENNNormal(mu, se, se_epi, se_ale, y_bounds=self.y_bounds)

    def batch_posterior(
        self,
        x: np.ndarray,
        paramss: list[ENNParams],
        *,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posteriorcoerced(flags)
        self._rust_model.set_neighbors(flags.tie_neighbors)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._numdim:
            raise ValueError(x.shape)
        if not paramss:
            raise ValueError("paramss must be non-empty")

        k_values = [p.k_neighbors for p in paramss]
        epistemic_scales = [p.epistemic_scale for p in paramss]
        aleatoric_scales = [p.aleatoric_scale for p in paramss]
        mu_all, se_all, se_epi_all, se_ale_all = self._rust_model.batch_posterior(
            x,
            k_values=k_values,
            epistemic_scales=epistemic_scales,
            aleatoric_scales=aleatoric_scales,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        return ENNNormal(mu_all, se_all, se_epi_all, se_ale_all, y_bounds=self.y_bounds)

    def neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] != self._numdim:
            raise ValueError(
                f"x must be single point with {self._numdim} dims, got {x.shape}"
            )
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if len(self) == 0:
            return np.zeros((0,), dtype=np.int64)
        if exclude_nearest and len(self) <= 1:
            raise ValueError(
                f"exclude_nearest=True requires at least 2 observations, got {len(self)}"
            )

        idx_2d = self._rust_model.neighbors(x, k, exclude_nearest=exclude_nearest)
        idx = idx_2d[0, :] if idx_2d.size > 0 else np.array([], dtype=np.int64)
        return idx.astype(np.int64, copy=False)

    def posterior_draw(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        flags = _posteriorcoerced(flags)
        self._rust_model.set_neighbors(flags.tie_neighbors)
        seeds = _rustseeds(function_seeds)
        kw = _rustkwargs(params, flags, seeds)
        draws, idx = self._rust_model.posterior_draw(x, **kw)
        return _finalizedraw(x, draws, idx)

    def conditional_draw(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        flags = _posteriorcoerced(flags)
        self._rust_model.set_neighbors(flags.tie_neighbors)
        x_whatif = np.asarray(x_whatif, dtype=float)
        if x_whatif.ndim != 2 or x_whatif.shape[1] != self._numdim:
            raise ValueError(x_whatif.shape)
        if x_whatif.shape[0] == 0:
            return self.posterior_draw(
                x,
                params,
                function_seeds=function_seeds,
                flags=flags,
            )

        seeds = _rustseeds(function_seeds)
        kw = _rustkwargs(params, flags, seeds)
        draws, idx = self._rust_model.conditional_draw(
            x_whatif,
            y_whatif,
            x,
            **kw,
        )
        return _finalizedraw(x, draws, idx)
