"""Shared numerical boundary for Python trial managers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .turbo.optimizer import create_optimizer
from .turbo.optimizer_config import OptimizerConfig, turbo_enn


class Space:
    def __init__(self, bounds: Mapping[str, tuple[float, float]], logs=()) -> None:
        if not bounds or any(not isinstance(name, str) for name in bounds):
            raise ValueError("the search space needs named continuous parameters")
        self.names = tuple(sorted(bounds))
        self.limits = np.asarray([bounds[name] for name in self.names], dtype=float)
        if self.limits.shape != (len(bounds), 2):
            raise ValueError("each parameter needs a lower and upper bound")
        if not np.isfinite(self.limits).all() or np.any(
            self.limits[:, 0] >= self.limits[:, 1]
        ):
            raise ValueError("bounds must be finite and strictly increasing")
        if not set(logs).issubset(self.names):
            raise ValueError("log parameters must belong to the search space")
        self.logs = np.asarray([name in logs for name in self.names])
        if np.any(self.limits[self.logs] <= 0):
            raise ValueError("log bounds must be positive")
        self.bounds = self.limits.copy()
        self.bounds[self.logs] = np.log(self.bounds[self.logs])

    def encode(self, parameters: Mapping[str, float]) -> np.ndarray:
        if set(parameters) != set(self.names):
            raise ValueError("parameters must match the fixed search space exactly")
        values = np.asarray([parameters[name] for name in self.names], dtype=float)
        if not np.isfinite(values).all() or np.any(
            (values < self.limits[:, 0]) | (values > self.limits[:, 1])
        ):
            raise ValueError("parameters must be finite and inside their bounds")
        values[self.logs] = np.log(values[self.logs])
        return values

    def decode(self, values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=float).copy()
        values[self.logs] = np.exp(values[self.logs])
        values = np.clip(values, self.limits[:, 0], self.limits[:, 1])
        return dict(zip(self.names, values.tolist(), strict=True))


class Loop:
    """Retain a live optimizer; ingest each completed observation exactly once.

    Reconstructing this object from observations is a warm restart, not a native
    optimizer checkpoint. Trial managers must have a single generating process.
    """

    def __init__(self, space: Space, config: OptimizerConfig | None, seed: int) -> None:
        if config is not None and config.num_metrics is not None:
            raise ValueError(
                "trial adapters require a single-objective optimizer config"
            )
        self.space = space
        self.optimizer = create_optimizer(
            bounds=space.bounds,
            config=turbo_enn() if config is None else config,
            rng=np.random.default_rng(seed),
        )
        self._seen: dict[str, tuple[np.ndarray, float, float | None]] = {}

    def update(self, records: Mapping[str, tuple[Mapping, float, float | None]]):
        if not self._seen.keys() <= records.keys():
            raise ValueError("completed observations were removed from the history")
        checked = {key: self._record(*record) for key, record in records.items()}
        if len({record[2] is None for record in checked.values()}) > 1:
            raise ValueError(
                "mixed known and unknown observation variances are unsupported"
            )
        for key, record in checked.items():
            if key in self._seen and not self._equal(self._seen[key], record):
                raise ValueError("completed observations changed; create a new adapter")
        for key, (point, score, variance) in checked.items():
            if key in self._seen:
                continue
            noise = None if variance is None else np.array([[variance]])
            self.optimizer.tell(point[None, :], np.array([[score]]), noise)
            self._seen[key] = (point, score, variance)

    def _record(self, parameters, score, variance):
        point = self.space.encode(parameters)
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("completed objective values must be finite")
        if variance is not None:
            variance = float(variance)
            if not np.isfinite(variance) or variance < 0:
                raise ValueError("observation variance must be finite and nonnegative")
        return point, score, variance

    @staticmethod
    def _equal(left, right):
        return np.array_equal(left[0], right[0]) and left[1:] == right[1:]

    def suggest(self, pending: Sequence[Mapping[str, float]]) -> dict[str, float]:
        occupied = [self.space.encode(point) for point in pending]
        for _ in range(32):
            proposal = self.space.decode(self.optimizer.ask(1)[0])
            point = self.space.encode(proposal)
            if not any(np.array_equal(point, other) for other in occupied):
                return proposal
        raise RuntimeError(
            "ENNX could not propose a point distinct from pending trials"
        )
