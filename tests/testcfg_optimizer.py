from __future__ import annotations

import pytest

from ennx.turbo import config as cfg
from ennx.turbo.config import (
    AcqType,
    CandidateGenConfig,
    CandidateRV,
    DrawAcquisitionConfig,
    ENNDistanceMetric,
    ENNFitConfig,
    ENNIndexDriver,
    ENNSurrogateConfig,
    GPSurrogateConfig,
    InitConfig,
    LHDOnlyInit,
    MorboTRConfig,
    MultiObjectiveConfig,
    MultiTRConfig,
    NDSOptimizerConfig,
    NoSurrogateConfig,
    NoTRConfig,
    ObservationHistoryConfig,
    OptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    TurboTRConfig,
    UCBAcquisitionConfig,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)


def test_021():
    with pytest.raises(ValueError, match="init_strategy must be"):
        InitConfig(init_strategy="invalid")


def test_022():
    with pytest.raises(ValueError, match="num_init must be > 0"):
        InitConfig(num_init=0)


def test_023():
    cfg = ENNSurrogateConfig()
    assert cfg.k is None
    assert cfg.num_samples is None
    assert cfg.num_candidates is None
    assert cfg.scale_x is False


def test_024():
    cfg = ENNSurrogateConfig(k=10, fit=ENNFitConfig(num_samples=50), scale_x=True)
    assert cfg.k == 10
    assert cfg.num_samples == 50
    assert cfg.scale_x is True


def test_025():
    with pytest.raises(
        ValueError, match=r"scale_x=True is not compatible with BPANN_DISK"
    ):
        ENNSurrogateConfig(scale_x=True, index_driver=ENNIndexDriver.BPANN_DISK)


def test_026():
    cfg = ENNFitConfig()
    assert cfg.num_samples is None
    assert cfg.num_candidates is None


def test_ennfitconfigcustom():
    cfg = ENNFitConfig(num_samples=50, num_candidates=100)
    assert cfg.num_samples == 50
    assert cfg.num_candidates == 100


def test_027():
    with pytest.raises(ValueError, match="num_samples must be > 0"):
        ENNFitConfig(num_samples=0)


def test_028():
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        ENNFitConfig(num_candidates=0)


def test_029():
    cfg = ENNSurrogateConfig(fit=ENNFitConfig(num_candidates=100))
    assert cfg.num_candidates == 100


def test_030():
    acq = UCBAcquisitionConfig()
    assert acq.beta == 2.0


def test_031():
    cfg = OptimizerConfig()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert cfg.candidate_rv == CandidateRV.SOBOL
    assert cfg.init.num_init is None
    assert isinstance(cfg.surrogate, NoSurrogateConfig)
    assert isinstance(cfg.observation_history, ObservationHistoryConfig)


def test_032():
    with pytest.raises(
        ValueError, match="init_strategy='lhd_only' requires NoSurrogateConfig"
    ):
        OptimizerConfig(
            init=InitConfig(init_strategy=LHDOnlyInit()),
            surrogate=GPSurrogateConfig(),
        )
    config = OptimizerConfig(
        init=InitConfig(init_strategy=LHDOnlyInit()),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
        surrogate=NoSurrogateConfig(),
    )
    assert config.num_metrics == 2


def test_033():
    with pytest.raises(
        ValueError,
        match="NoSurrogateConfig is not compatible with DrawAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogateConfig(),
            acquisition=DrawAcquisitionConfig(),
        )


def test_034():
    with pytest.raises(
        ValueError,
        match="NoSurrogateConfig is not compatible with UCBAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogateConfig(),
            acquisition=UCBAcquisitionConfig(),
        )


def test_035():
    with pytest.raises(
        ValueError, match="ParetoAcquisitionConfig requires NDSOptimizerConfig"
    ):
        OptimizerConfig(
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )


def test_036():
    cfg = turbo_one()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_037():
    cfg = turbo_one(acq_type=AcqType.PARETO)
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_038():
    cfg = turbo_one(acq_type=AcqType.UCB)
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_039():
    cfg = turbo_one(acq_type=AcqType.THOMPSON)
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_040():
    cfg = turbo_zero()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert isinstance(cfg.surrogate, NoSurrogateConfig)
    assert isinstance(cfg.acquisition, RandomAcquisitionConfig)


def test_041():
    cfg = turbo_enn(acq_type=AcqType.PARETO)
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_042():
    cfg = turbo_enn(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(fit=ENNFitConfig(num_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_043():
    cfg = turbo_enn(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(fit=ENNFitConfig(num_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_044():
    with pytest.raises(ValueError, match="num_samples required"):
        turbo_enn(acq_type=AcqType.UCB)


def test_045():
    cfg = lhd_only()
    assert isinstance(cfg.trust_region, NoTRConfig)
    assert isinstance(cfg.init.init_strategy, LHDOnlyInit)
    assert isinstance(cfg.surrogate, NoSurrogateConfig)


def test_046():
    cfg = turbo_enn(
        enn=ENNSurrogateConfig(k=15),
        candidates=CandidateGenConfig(num_candidates=200),
        num_init=10,
    )
    assert cfg.surrogate.k == 15
    assert cfg.candidates.resolve_candidates(num_dim=3, num_arms=7) == 200
    assert cfg.init.num_init == 10


def test_047():
    cfg_turbo = OptimizerConfig(trust_region=TurboTRConfig())
    assert cfg_turbo.num_metrics is None
    cfg_morbo = OptimizerConfig(
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    )
    assert cfg_morbo.num_metrics == 2
    cfg_none = OptimizerConfig(trust_region=NoTRConfig())
    assert cfg_none.num_metrics is None


def test_048():
    config = cfg.turbo_enn(
        enn=cfg.ENNSurrogateConfig(fit=cfg.ENNFitConfig(num_samples=4))
    )

    assert isinstance(config, cfg.OptimizerConfig)
    assert config.candidates == cfg.CandidateGenConfig()
    assert config.raasp_driver is cfg.RAASPDriver.ORIG


def test_enumvalues():
    assert ENNDistanceMetric.SQUARED_L2.value == "squared_l2"
    assert ENNDistanceMetric.COSINE.value == "cosine"


def test_multitrlengths():
    tr = MultiTRConfig()
    assert (tr.length_init, tr.length_min, tr.length_max) == (
        tr.length.length_init,
        tr.length.length_min,
        tr.length.length_max,
    )
