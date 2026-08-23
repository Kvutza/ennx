from typing import Any, NamedTuple

from .multi_trust_region_policy import SharingPolicy


class RegionBatch(NamedTuple):
    region: int
    start: int
    length: int


class RegionCandidate(NamedTuple):
    index: int
    region: int
    seed: int
    score: float


class CandidateProposal(NamedTuple):
    region: int
    seed: int
    payload: Any


class RegionRound(NamedTuple):
    batches: list[RegionBatch]
    proposals: list[CandidateProposal]
    candidates: list[RegionCandidate]
    selected: list[RegionCandidate]


__all__ = [
    "CandidateProposal",
    "RegionBatch",
    "RegionCandidate",
    "RegionRound",
    "SharingPolicy",
]
