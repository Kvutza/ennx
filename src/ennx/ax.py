"""ENNX candidate generation for Ax experiments."""

from __future__ import annotations

from typing import override

import numpy as np
from ax.core.parameter import ParameterType, RangeParameter
from ax.generation_strategy.external_generation_node import ExternalGenerationNode

from ._trials import Loop, Space
from .turbo.optimizer_config import OptimizerConfig


def _space(experiment):
    search = experiment.search_space
    if search.parameter_constraints:
        raise ValueError("parameter constraints are not supported by the ENNX adapter")
    bounds, logs = {}, []
    for name, parameter in search.parameters.items():
        if (
            not isinstance(parameter, RangeParameter)
            or parameter.parameter_type != ParameterType.FLOAT
        ):
            raise ValueError("ENNX requires continuous floating-point range parameters")
        if (
            parameter.is_fidelity
            or parameter.logit_scale
            or parameter.digits is not None
        ):
            raise ValueError(
                "fidelity, logit, and rounded parameters are not supported"
            )
        bounds[name] = (parameter.lower, parameter.upper)
        if parameter.log_scale:
            logs.append(name)
    return Space(bounds, logs)


def _objective(experiment):
    config = experiment.optimization_config
    if config is None or not config.objective.is_single_objective:
        raise ValueError("ENNX's Ax adapter requires one unscalarized objective")
    if config.outcome_constraints:
        raise ValueError("outcome constraints are not supported by the ENNX adapter")
    if config.pruning_target_parameterization is not None:
        raise ValueError("parameter pruning is not supported by the ENNX adapter")
    metric, weight = config.objective.metric_weights[0]
    if not np.isfinite(weight) or weight == 0:
        raise ValueError("the objective coefficient must be finite and nonzero")
    return metric, weight


class Node(ExternalGenerationNode):
    """Keep Ax orchestration and use ENNX to generate continuous candidates.

    Known SEMs are squared before passing observation variance to ENNX. Failed,
    abandoned, and running trials are not treated as completed observations.
    Reattaching a fresh node warm-starts from data, not from an optimizer checkpoint.
    Use one generating process; pending points are excluded without fantasizing.
    """

    def __init__(
        self,
        *,
        config: OptimizerConfig | None = None,
        seed: int = 0,
        name: str = "ENNX",
    ) -> None:
        super().__init__(name=name)
        self._config = config
        self._seed = seed
        self._loop = None
        self._bound = None

    @override
    def update_generator_state(self, experiment, data):
        domain = _space(experiment)
        metric, weight = _objective(experiment)
        signature = (
            id(experiment),
            domain.names,
            domain.limits.tolist(),
            domain.logs.tolist(),
            metric,
            weight,
        )
        if self._bound is not None and self._bound != signature:
            raise ValueError(
                "use a new ENNX node when the experiment or its contract changes"
            )
        if self._loop is None:
            self._loop = Loop(domain, self._config, self._seed)
            self._bound = signature
        self._loop.update(self._records(experiment, data, metric, weight))

    def _records(self, experiment, data, metric, weight):
        frame = data.df
        records = {}
        for index, trial in sorted(experiment.trials.items()):
            if not trial.status.is_completed:
                continue
            for arm in trial.arms:
                rows = frame[
                    (frame.trial_index == index)
                    & (frame.arm_name == arm.name)
                    & (frame.metric_signature == metric)
                ]
                if len(rows) != 1:
                    raise ValueError(
                        "each completed arm needs exactly one objective observation"
                    )
                row = rows.iloc[0]
                sem = float(row["sem"])
                if not np.isnan(sem) and (not np.isfinite(sem) or sem < 0):
                    raise ValueError(
                        "SEM must be nonnegative and finite, or NaN if unknown"
                    )
                variance = None if np.isnan(sem) else (sem * weight) ** 2
                score = float(row["mean"]) * weight
                records[f"{index}:{arm.name}"] = (arm.parameters, score, variance)
        return records

    @override
    def get_next_candidate(self, pending_parameters):
        if self._loop is None:
            raise RuntimeError(
                "update the ENNX node with the experiment before generating"
            )
        return self._loop.suggest(pending_parameters)


__all__ = ["Node"]
