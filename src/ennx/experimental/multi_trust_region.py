from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, NamedTuple

import numpy as np


class SharingPolicy(str, Enum):
    SHARED = "shared"
    NEAREST_CENTER = "nearest_center"
    INDEPENDENT = "independent"


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


def _policy_value(sharing_policy: SharingPolicy | str) -> str:
    if isinstance(sharing_policy, SharingPolicy):
        return sharing_policy.value
    return str(sharing_policy)


def make_multi_trust_region(
    num_dim: int,
    num_regions: int = 4,
    sharing_policy: SharingPolicy | str = SharingPolicy.SHARED,
    seed: int = 42,
):
    from . import MultiTrustRegion

    return MultiTrustRegion(
        num_dim=num_dim,
        num_regions=num_regions,
        sharing_policy=_policy_value(sharing_policy),
        seed=seed,
    )


def allocate_region_batches(
    state,
    budget: int,
    utility: Sequence[float] | np.ndarray | None = None,
) -> list[RegionBatch]:
    raw_batches = (
        state.allocate(budget)
        if utility is None
        else state.allocate_with(budget, np.asarray(utility, dtype=float))
    )
    return [RegionBatch(*batch) for batch in raw_batches]


def select_region_candidates(
    state,
    candidates: Sequence[RegionCandidate | Sequence[object]],
    num_arms: int,
) -> list[RegionCandidate]:
    raw_candidates = [
        candidate
        if isinstance(candidate, RegionCandidate)
        else RegionCandidate(*candidate)
        for candidate in candidates
    ]
    selected = state.select(
        [
            (candidate.index, candidate.region, candidate.seed, candidate.score)
            for candidate in raw_candidates
        ],
        num_arms,
    )
    return [RegionCandidate(*candidate) for candidate in selected]


def _coerce_proposal(
    candidate: CandidateProposal | Sequence[object],
) -> CandidateProposal:
    if isinstance(candidate, CandidateProposal):
        return candidate
    region, seed, payload = candidate
    return CandidateProposal(int(region), int(seed), payload)


class MultiTrustRegionLoop:
    """End-to-end experimental loop around a multi-trust-region state."""

    def __init__(self, state) -> None:
        self.state = state

    def allocate(
        self, budget: int, utility: Sequence[float] | np.ndarray | None = None
    ) -> list[RegionBatch]:
        return allocate_region_batches(self.state, budget, utility)

    def propose(
        self,
        budget: int,
        proposal_fn: Callable[
            [RegionBatch],
            Sequence[CandidateProposal | Sequence[object]] | CandidateProposal | None,
        ],
        utility: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[list[RegionBatch], list[CandidateProposal]]:
        batches = self.allocate(budget, utility)
        proposals: list[CandidateProposal] = []
        for batch in batches:
            produced = proposal_fn(batch)
            if produced is None:
                continue
            if isinstance(produced, CandidateProposal):
                proposals.append(produced)
                continue
            for candidate in produced:
                proposals.append(_coerce_proposal(candidate))
        return batches, proposals

    def score(
        self,
        proposals: Sequence[CandidateProposal | Sequence[object]],
        scorer: Callable[[CandidateProposal], float],
    ) -> list[RegionCandidate]:
        scored: list[RegionCandidate] = []
        for index, proposal in enumerate(map(_coerce_proposal, proposals)):
            scored.append(
                RegionCandidate(
                    index=index,
                    region=proposal.region,
                    seed=proposal.seed,
                    score=float(scorer(proposal)),
                )
            )
        return scored

    def select(
        self, candidates: Sequence[RegionCandidate | Sequence[object]], num_arms: int
    ) -> list[RegionCandidate]:
        return select_region_candidates(self.state, candidates, num_arms)

    def run_round(
        self,
        *,
        budget: int,
        proposal_fn: Callable[
            [RegionBatch],
            Sequence[CandidateProposal | Sequence[object]] | CandidateProposal | None,
        ],
        scorer: Callable[[CandidateProposal], float],
        num_arms: int | None = None,
        utility: Sequence[float] | np.ndarray | None = None,
    ) -> RegionRound:
        batches, proposals = self.propose(budget, proposal_fn, utility)
        candidates = self.score(proposals, scorer)
        selected = self.select(candidates, num_arms or len(candidates))
        return RegionRound(
            batches=batches,
            proposals=proposals,
            candidates=candidates,
            selected=selected,
        )

    def tell(self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x.flags.writeable = False
        y.flags.writeable = False
        if y_var is not None:
            y_var = np.asarray(y_var, dtype=float)
            y_var.flags.writeable = False
        if y_var is None:
            return self.state.tell(x, y)
        try:
            return self.state.tell(x, y, y_var)
        except TypeError:
            return self.state.tell(x, y)

    def restart_region(self, region: int, new_center: np.ndarray):
        return self.state.restart_region(region, new_center)

    def variance(self, region: int):
        return self.state.variance(region)


multi_trust_region = sys.modules[__name__]


__all__ = [
    "CandidateProposal",
    "MultiTrustRegionLoop",
    "RegionBatch",
    "RegionCandidate",
    "RegionRound",
    "SharingPolicy",
    "allocate_region_batches",
    "make_multi_trust_region",
    "multi_trust_region",
    "select_region_candidates",
]
