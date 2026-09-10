"""ENNX candidate generation for Optuna studies."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import override

from optuna.distributions import FloatDistribution
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState

from ._trials import Loop, Space
from .turbo.optimizer_config import OptimizerConfig


class Sampler(BaseSampler):
    """Use ENNX for a fixed continuous, single-objective Optuna study.

    Declare the complete search space up front. Log-uniform floats are supported;
    integers, steps, conditional spaces, and multi-objective studies are not.
    Use one sampler and one generating process per study. Pending points are
    excluded, but are not assigned fabricated observations. New sampler instances
    warm-start from completed trials; they do not restore the original trajectory.
    """

    def __init__(
        self,
        space: Mapping[str, FloatDistribution],
        *,
        config: OptimizerConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._space = deepcopy(dict(space))
        for distribution in self._space.values():
            if not isinstance(distribution, FloatDistribution):
                raise ValueError("ENNX supports only continuous FloatDistribution")
            if distribution.step is not None:
                raise ValueError("stepped parameters are not supported")
        domain = Space(
            {name: (dist.low, dist.high) for name, dist in self._space.items()},
            [name for name, dist in self._space.items() if dist.log],
        )
        self._loop = Loop(domain, config, seed)
        self._study = None
        self._direction = None

    def _bind(self, study):
        if len(study.directions) != 1:
            raise ValueError("ENNX's Optuna adapter supports one objective")
        if self._study is not None and self._study is not study:
            raise ValueError("use a separate ENNX sampler for each Study instance")
        if self._direction is not None and self._direction != study.direction:
            raise ValueError("the objective direction changed")
        self._study = study
        self._direction = study.direction

    @override
    def infer_relative_search_space(self, study, trial):
        self._bind(study)
        return deepcopy(self._space)

    @override
    def sample_relative(self, study, trial, search_space):
        self._bind(study)
        if search_space != self._space:
            raise ValueError("the search space changed")
        records, pending = self._history(study, trial.number)
        self._loop.update(records)
        return self._loop.suggest(pending)

    def _history(self, study, number):
        records, pending = {}, []
        sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        for trial in study.get_trials(deepcopy=False):
            if trial.number == number:
                continue
            if trial.state == TrialState.COMPLETE:
                self._validate(trial.distributions)
                records[str(trial.number)] = (trial.params, sign * trial.value, None)
                continue
            if trial.state != TrialState.RUNNING:
                continue
            if set(trial.params) != set(self._space):
                raise RuntimeError("finish suggesting the pending trial first")
            self._validate(trial.distributions)
            pending.append(trial.params)
        return records, pending

    def _validate(self, distributions):
        if distributions != self._space:
            raise ValueError("every trial must use the declared fixed search space")

    @override
    def sample_independent(self, study, trial, param_name, param_distribution):
        raise ValueError(
            f"{param_name!r} is outside the declared continuous search space; "
            "ENNX does not silently fall back to another sampler"
        )

    @override
    def after_trial(self, study, trial, state, values):
        if state == TrialState.COMPLETE:
            self._validate(trial.distributions)

    @override
    def reseed_rng(self):
        raise RuntimeError("use n_jobs=1 and one generating process with ENNX")


__all__ = ["Sampler"]
