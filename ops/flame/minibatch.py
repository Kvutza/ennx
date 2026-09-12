"""Uniform minibatches and equal-problem-weight paired loss screening.

The default two-SE rule is a screening heuristic, not guaranteed statistical
significance under adaptive repeated selection. Candidate and incumbent must be
evaluated on the same sampled problems; no confirmation batch is used. Sampling
is without replacement within each batch and independent across calls. A full
population has zero sampling variance and uses an exact, strict comparison,
subject to preserving score order at the native float32 precision.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class PairedLosses:
    candidate_loss: float
    incumbent_loss: float
    candidate_variance: float
    incumbent_variance: float
    improvement: float
    improvement_variance: float
    improvement_se: float
    threshold: float
    accepted: bool
    deteriorated: bool


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    return int(value)


def sample_indices(
    rng: np.random.Generator, population: int, batch_size: int
) -> np.ndarray:
    """Draw a fresh uniform batch, reproducibly using the supplied Generator."""
    population = _integer(population, "population")
    batch_size = _integer(batch_size, "batch_size")
    if not 2 <= batch_size <= population:
        raise ValueError("require 2 <= batch_size <= population")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng.choice(population, size=batch_size, replace=False)


def _losses(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array of at least 2 losses")
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric losses")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} losses must be finite and nonnegative")
    return values


def _nativescalar(value: float, name: str) -> np.float32:
    with np.errstate(over="ignore", invalid="ignore"):
        native = np.float32(value)
    if not np.isfinite(native):
        raise ValueError(f"{name} must be finite at native float32 precision")
    return native


def paired_losses(
    candidate, incumbent, population: int, acceptance_se: float = 2.0
) -> PairedLosses:
    """Screen aligned per-problem losses with a paired finite-population SE.

    Each reported variance is the variance of a sample mean: the unbiased sample
    variance times ``(1 - batch_size / population) / batch_size``. Improvement
    uncertainty uses incumbent-minus-candidate differences, retaining covariance.
    Acceptance requires improvement strictly above the threshold and strictly
    ordered negative mean scores after float32 conversion. Deterioration mirrors
    this below the negative threshold; borderline values and FP32 score ties are
    inconclusive. Invalid inputs or
    native float32 mean/variance overflow raise ValueError.
    """
    population = _integer(population, "population")
    candidate = _losses(candidate, "candidate")
    incumbent = _losses(incumbent, "incumbent")
    if candidate.size != incumbent.size:
        raise ValueError("candidate and incumbent must have equal lengths")
    batch_size = candidate.size
    if population < batch_size:
        raise ValueError("population must be at least the batch size")
    if (
        isinstance(acceptance_se, (bool, np.bool_))
        or not isinstance(acceptance_se, Real)
        or not np.isfinite(acceptance_se)
        or acceptance_se < 0
    ):
        raise ValueError("acceptance_se must be finite and nonnegative")

    with np.errstate(over="ignore", invalid="ignore"):
        candidate_loss = float(candidate.mean())
        incumbent_loss = float(incumbent.mean())
    candidate_score = _nativescalar(-candidate_loss, "candidate mean")
    incumbent_score = _nativescalar(-incumbent_loss, "incumbent mean")
    differences = incumbent - candidate
    improvement = float(differences.mean())
    # Skip the variance computation for a census, including large finite losses.
    correction = (1.0 - batch_size / population) / batch_size
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_variance = (
            float(correction * candidate.var(ddof=1)) if correction else 0.0
        )
        incumbent_variance = (
            float(correction * incumbent.var(ddof=1)) if correction else 0.0
        )
        difference_variance = (
            float(correction * differences.var(ddof=1)) if correction else 0.0
        )
        improvement_se = float(np.sqrt(difference_variance))
        threshold = float(acceptance_se) * improvement_se
    _nativescalar(candidate_variance, "candidate variance")
    _nativescalar(incumbent_variance, "incumbent variance")
    _nativescalar(difference_variance, "improvement variance")
    if not all(np.isfinite(v) for v in (improvement, improvement_se, threshold)):
        raise ValueError("paired statistics must be finite")
    return PairedLosses(
        candidate_loss=candidate_loss,
        incumbent_loss=incumbent_loss,
        candidate_variance=candidate_variance,
        incumbent_variance=incumbent_variance,
        improvement=improvement,
        improvement_variance=difference_variance,
        improvement_se=improvement_se,
        threshold=threshold,
        accepted=bool(improvement > threshold and candidate_score > incumbent_score),
        deteriorated=bool(
            improvement < -threshold and candidate_score < incumbent_score
        ),
    )
