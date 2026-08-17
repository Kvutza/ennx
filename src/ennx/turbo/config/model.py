# ruff: noqa: TRY004
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from .candidate_gen_config import CandidateGenConfig, RAASPDriver
from .enn_surrogate_config import ENNSurrogateConfig
from .init_config import InitConfig

if TYPE_CHECKING:
    from .candidate_rv import CandidateRV


class Rescalarize(Enum):
    ON_RESTART = "on_restart"
    ON_PROPOSE = "on_propose"


@dataclass(frozen=True)
class DrawAcquisitionConfig:
    pass


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    pass


@dataclass(frozen=True)
class RandomAcquisitionConfig:
    pass


@dataclass(frozen=True)
class UCBAcquisitionConfig:
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")


@dataclass(frozen=True)
class RAASPOptimizerConfig:
    pass


@dataclass(frozen=True)
class NDSOptimizerConfig:
    pass


@dataclass(frozen=True)
class GPSurrogateConfig:
    pass


@dataclass(frozen=True)
class NoSurrogateConfig:
    pass


@dataclass(frozen=True)
class NoTRConfig:
    noise_aware: bool = False


@dataclass(frozen=True)
class ObservationHistoryConfig:
    pass


@dataclass(frozen=True)
class TRLengthConfig:
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6

    def __post_init__(self) -> None:
        if self.length_init <= 0:
            raise ValueError(f"length_init must be > 0, got {self.length_init}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be > 0, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be > 0, got {self.length_max}")
        if self.length_min >= self.length_max:
            raise ValueError(
                f"length_min must be < length_max, got {self.length_min} >= {self.length_max}"
            )
        if self.length_init > self.length_max:
            raise ValueError(
                f"length_init must be <= length_max, got {self.length_init} > {self.length_max}"
            )
        if self.length_min > self.length_init:
            raise ValueError(
                f"length_min must be <= length_init, got {self.length_min} > {self.length_init}"
            )


@dataclass(frozen=True)
class TurboTRConfig:
    length: TRLengthConfig = TRLengthConfig()
    noise_aware: bool = False

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MultiObjectiveConfig:
    num_metrics: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.num_metrics < 2:
            raise ValueError(
                f"num_metrics must be >= 2 for MORBO, got {self.num_metrics}"
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


@dataclass(frozen=True)
class RescalePolicyConfig:
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE


@dataclass(frozen=True)
class MorboTRConfig:
    multi_objective: MultiObjectiveConfig
    length: TRLengthConfig = TRLengthConfig()
    rescale_policy: RescalePolicyConfig = RescalePolicyConfig()
    noise_aware: bool = False

    @property
    def rescalarize(self) -> Rescalarize:
        return self.rescale_policy.rescalarize

    @property
    def num_metrics(self) -> int:
        return self.multi_objective.num_metrics

    @property
    def alpha(self) -> float:
        return self.multi_objective.alpha

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MultiTRConfig:
    num_regions: int = 4
    length: TRLengthConfig = TRLengthConfig()
    succ_tolerance: int = 3
    fail_tolerance: int = 5
    sharing_policy: Literal["shared", "nearest_center", "independent"] = "shared"

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


AcquisitionConfig = (
    UCBAcquisitionConfig
    | DrawAcquisitionConfig
    | ParetoAcquisitionConfig
    | RandomAcquisitionConfig
)
AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig


def _random_acquisition() -> RandomAcquisitionConfig:
    return RandomAcquisitionConfig()


def _raasp_optimizer() -> RAASPOptimizerConfig:
    return RAASPOptimizerConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    trust_region: TrustRegionConfig = TurboTRConfig()
    candidates: CandidateGenConfig = field(default_factory=CandidateGenConfig)
    init: InitConfig = field(default_factory=InitConfig)
    surrogate: SurrogateConfig = NoSurrogateConfig()
    acquisition: AcquisitionConfig = field(default_factory=_random_acquisition)
    acq_optimizer: AcqOptimizerConfig = field(default_factory=_raasp_optimizer)
    observation_history: ObservationHistoryConfig = ObservationHistoryConfig()

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def num_metrics(self) -> int | None:
        if isinstance(self.trust_region, MorboTRConfig):
            return self.trust_region.num_metrics
        return None

    @property
    def candidate_rv(self) -> CandidateRV:
        return self.candidates.candidate_rv

    @property
    def num_candidates(self) -> int | None:
        return self.candidates.num_candidates

    @property
    def raasp_driver(self) -> RAASPDriver:
        return self.candidates.raasp_driver


SurrogateConfig = NoSurrogateConfig | GPSurrogateConfig | ENNSurrogateConfig
TrustRegionConfig = NoTRConfig | TurboTRConfig | MorboTRConfig | MultiTRConfig


def _validate(cfg: OptimizerConfig) -> None:
    from .init_config import LHDOnlyInit

    if isinstance(cfg.init.init_strategy, LHDOnlyInit) and not isinstance(
        cfg.surrogate, NoSurrogateConfig
    ):
        raise ValueError(
            "init_strategy='lhd_only' requires NoSurrogateConfig surrogate"
        )
    if isinstance(cfg.surrogate, NoSurrogateConfig):
        if isinstance(cfg.acquisition, DrawAcquisitionConfig):
            raise ValueError(
                "DrawAcquisitionConfig (Thompson sampling) requires a surrogate. "
                "NoSurrogateConfig is not compatible with DrawAcquisitionConfig."
            )
        if isinstance(cfg.acquisition, UCBAcquisitionConfig):
            raise ValueError(
                "UCBAcquisitionConfig requires a surrogate. "
                "NoSurrogateConfig is not compatible with UCBAcquisitionConfig."
            )
    if isinstance(cfg.acquisition, ParetoAcquisitionConfig) and not isinstance(
        cfg.acq_optimizer, NDSOptimizerConfig
    ):
        raise ValueError("ParetoAcquisitionConfig requires NDSOptimizerConfig")
    if not isinstance(
        cfg.surrogate, (NoSurrogateConfig, GPSurrogateConfig, ENNSurrogateConfig)
    ):
        raise ValueError(f"unsupported surrogate: {type(cfg.surrogate)!r}")
