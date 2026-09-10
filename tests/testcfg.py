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
    HybridInit,
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
    Rescalarize,
    RescalePolicyConfig,
    TurboTRConfig,
    UCBAcquisitionConfig,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)


def test_001():
    cfg = TurboTRConfig()
    assert cfg.length_init == 0.8
    assert cfg.length_min == 0.5**7
    assert cfg.length_max == 1.6


def test_turbotrconfigcustom():
    tr = TurboTRConfig(
        length=cfg.TRLengthConfig(length_init=0.5, length_min=0.01, length_max=2.0)
    )
    assert tr.length_init == 0.5
    assert tr.length_min == 0.01
    assert tr.length_max == 2.0


def test_002():
    with pytest.raises(ValueError, match="length_init must be > 0"):
        cfg.TRLengthConfig(length_init=0)
    with pytest.raises(ValueError, match="length_min must be < length_max"):
        cfg.TRLengthConfig(length_min=1.0, length_max=0.5)


def test_003():
    with pytest.raises(ValueError, match="length_init must be <= length_max"):
        cfg.TRLengthConfig(length_init=2.0, length_max=1.0)


def test_004():
    with pytest.raises(ValueError, match="length_min must be <= length_init"):
        cfg.TRLengthConfig(length_init=0.05, length_min=0.1)


def test_morbotrconfig():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=3))
    assert cfg.num_metrics == 3
    assert cfg.alpha == 0.05


def test_005():
    with pytest.raises(ValueError, match="num_metrics must be >= 2"):
        MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=1))


def test_006():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2, alpha=0.1))
    assert cfg.alpha == 0.1


def test_007():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2, alpha=0))


def test_008():
    cfg = MultiObjectiveConfig(num_metrics=2)
    assert cfg.num_metrics == 2
    assert cfg.alpha == 0.05


def test_009():
    cfg = MultiObjectiveConfig(num_metrics=3, alpha=0.1)
    assert cfg.num_metrics == 3
    assert cfg.alpha == 0.1


def test_010():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    assert cfg.length_init == 0.8
    assert cfg.length_min == 0.5**7
    assert cfg.length_max == 1.6


def test_011():
    tr = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        length=cfg.TRLengthConfig(
            length_init=0.5,
            length_min=0.01,
            length_max=2.0,
        ),
    )
    assert tr.length_init == 0.5
    assert tr.length_min == 0.01
    assert tr.length_max == 2.0


def test_012():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    assert cfg.rescalarize == Rescalarize.ON_PROPOSE


def test_013():
    cfg = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        rescale_policy=RescalePolicyConfig(rescalarize=Rescalarize.ON_RESTART),
    )
    assert cfg.rescalarize == Rescalarize.ON_RESTART


def test_014():
    cfg = RescalePolicyConfig()
    assert cfg.rescalarize == Rescalarize.ON_PROPOSE


def test_015():
    cfg = RescalePolicyConfig(rescalarize=Rescalarize.ON_RESTART)
    assert cfg.rescalarize == Rescalarize.ON_RESTART


def test_notrconfig():
    assert NoTRConfig().noise_aware is False


def test_016():
    cfg = CandidateGenConfig()
    assert cfg.candidate_rv == CandidateRV.SOBOL
    assert cfg.num_candidates is None
    assert cfg.resolve_candidates(num_dim=1, num_arms=1) == 100
    assert cfg.resolve_candidates(num_dim=100, num_arms=1) == 5000


def test_017():
    cfg = CandidateGenConfig(
        candidate_rv=CandidateRV.UNIFORM,
        num_candidates=100,
    )
    assert cfg.candidate_rv == CandidateRV.UNIFORM
    assert cfg.resolve_candidates(num_dim=3, num_arms=7) == 100


def test_018():
    with pytest.raises(ValueError, match="candidate_rv must be"):
        CandidateGenConfig(candidate_rv="invalid")


def test_019():
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        CandidateGenConfig(num_candidates=0)


def test_020():
    cfg = CandidateGenConfig(num_candidates_per_arm=100)
    assert cfg.resolve_candidates(num_dim=3, num_arms=7) == 700


def test_initconfigdefaults():
    cfg = InitConfig()
    assert cfg.init_strategy is None
    assert isinstance(cfg.get_strategy(), HybridInit)
    assert cfg.num_init is None


def test_initconfiglhdonly():
    cfg = InitConfig(init_strategy=LHDOnlyInit(), num_init=20)
    assert isinstance(cfg.init_strategy, LHDOnlyInit)
    assert cfg.num_init == 20


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
