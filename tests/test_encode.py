from __future__ import annotations

import pytest

from ennx.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    UCBAcquisitionConfig,
    turbo_enn,
)
from ennx.turbo.config.encode import encode
from ennx.turbo.config.eidxdrv import ENNIndexDriver


@pytest.mark.parametrize(
    ("samples", "candidates"),
    [(100, None), (None, 500), (50, 200)],
)
def test_fitparams(samples, candidates):
    cfg = turbo_enn(
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(
                num_samples=samples,
                num_candidates=candidates,
            ),
        )
    )
    out = encode(cfg)
    assert out is not None
    if samples is not None:
        assert out["num_samples"] == samples
    if candidates is not None:
        assert out["num_candidates"] == candidates


def test_indexdriver():
    cfg = turbo_enn(enn=ENNSurrogateConfig(index_driver=ENNIndexDriver.BPANN_DISK))
    assert encode(cfg)["index_driver"] == "bpann_disk"


def test_ucbbeta():
    cfg = turbo_enn(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(fit=ENNFitConfig(num_samples=8)),
    )
    out = encode(cfg)
    assert out == {
        "acquisition": "ucb",
        "acquisition_beta": UCBAcquisitionConfig().beta,
        "candidate_rv": "sobol",
        "index_driver": "exact",
        "max_candidates": 5000,
        "num_candidates_factor": 100.0,
        "num_samples": 8,
    }
