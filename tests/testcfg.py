from __future__ import annotations

import pytest

from ennx.turbo import config as cfg
from ennx.turbo.config import (
    CandidateGenConfig,
    CandidateRV,
    HybridInit,
    InitConfig,
    LHDOnlyInit,
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    Rescalarize,
    RescalePolicyConfig,
    TurboTRConfig,
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
