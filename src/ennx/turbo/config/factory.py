from __future__ import annotations

from . import model as cfg
from .acq_type import AcqType
from .candgencfg import CandidateGenConfig
from .candidate_rv import CandidateRV
from .esurrcfg import ENNSurrogateConfig
from .init_config import InitConfig, LHDOnlyInit


def _candidates(
    candidate_rv: CandidateRV,
    num_candidates: int | None,
    *,
    num_candidates_per_arm: int | None = None,
) -> CandidateGenConfig:
    return CandidateGenConfig(
        candidate_rv=candidate_rv,
        num_candidates=num_candidates,
        num_candidates_per_arm=num_candidates_per_arm,
    )


def _acqconfigs(
    acq_type: AcqType,
) -> tuple[cfg.AcquisitionConfig, cfg.AcqOptimizerConfig]:
    if acq_type == AcqType.PARETO:
        return cfg.ParetoAcquisitionConfig(), cfg.NDSOptimizerConfig()
    if acq_type == AcqType.UCB:
        return cfg.UCBAcquisitionConfig(), cfg.RAASPOptimizerConfig()
    if acq_type == AcqType.THOMPSON:
        return cfg.DrawAcquisitionConfig(), cfg.RAASPOptimizerConfig()
    raise ValueError(
        f"acq_type must be AcqType.THOMPSON, AcqType.PARETO, or AcqType.UCB, got {acq_type!r}"
    )


def turbo_one(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trust_region: cfg.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
    acq_type: AcqType = AcqType.THOMPSON,
) -> cfg.OptimizerConfig:
    acquisition, acq_optimizer = _acqconfigs(acq_type)
    return cfg.OptimizerConfig(
        trust_region=trust_region or cfg.TurboTRConfig(),
        candidates=_candidates(candidate_rv, num_candidates),
        init=InitConfig(num_init=num_init),
        surrogate=cfg.GPSurrogateConfig(),
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
    )


def turbo_zero(
    *,
    num_candidates: int | None = None,
    num_candidates_per_arm: int | None = None,
    num_init: int | None = None,
    trust_region: cfg.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> cfg.OptimizerConfig:
    return cfg.OptimizerConfig(
        trust_region=trust_region or cfg.TurboTRConfig(),
        candidates=_candidates(
            candidate_rv,
            num_candidates,
            num_candidates_per_arm=num_candidates_per_arm,
        ),
        init=InitConfig(num_init=num_init),
        surrogate=cfg.NoSurrogateConfig(),
        acquisition=cfg.RandomAcquisitionConfig(),
        acq_optimizer=cfg.RAASPOptimizerConfig(),
    )


def turbo_enn(
    *,
    enn: ENNSurrogateConfig | None = None,
    trust_region: cfg.TrustRegionConfig | None = None,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    acq_type: AcqType = AcqType.PARETO,
) -> cfg.OptimizerConfig:
    acquisition, acq_optimizer = _acqconfigs(acq_type)
    surrogate = enn if enn is not None else ENNSurrogateConfig()
    if surrogate.num_samples is None and acq_type != AcqType.PARETO:
        raise ValueError(f"enn.num_samples required for acq_type={acq_type!r}")
    return cfg.OptimizerConfig(
        trust_region=trust_region or cfg.TurboTRConfig(),
        candidates=candidates or CandidateGenConfig(),
        init=InitConfig(num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
    )


def lhd_only(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trust_region: cfg.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> cfg.OptimizerConfig:
    return cfg.OptimizerConfig(
        trust_region=trust_region or cfg.NoTRConfig(),
        candidates=_candidates(candidate_rv, num_candidates),
        init=InitConfig(init_strategy=LHDOnlyInit(), num_init=num_init),
        surrogate=cfg.NoSurrogateConfig(),
        acquisition=cfg.RandomAcquisitionConfig(),
        acq_optimizer=cfg.RAASPOptimizerConfig(),
    )
