from __future__ import annotations


def test_experimental_multi_trust_region_namespace():
    import ennx
    import ennx.experimental as experimental_mod
    import ennx.experimental.multi_trust_region as mtr
    from ennx import experimental

    assert experimental is experimental_mod
    assert experimental.multi_trust_region is mtr
    assert experimental.SharingPolicy is mtr.SharingPolicy
    assert experimental.RegionBatch is mtr.RegionBatch
    assert experimental.RegionCandidate is mtr.RegionCandidate
    assert experimental.CandidateProposal is mtr.CandidateProposal
    assert experimental.RegionRound is mtr.RegionRound
    assert experimental.MultiTrustRegionLoop is mtr.MultiTrustRegionLoop
    assert experimental.make_multi_trust_region is mtr.make_multi_trust_region
    assert mtr.SharingPolicy.SHARED.value == "shared"
    assert mtr.SharingPolicy.NEAREST_CENTER.value == "nearest_center"
    assert mtr.SharingPolicy.INDEPENDENT.value == "independent"
    assert callable(mtr.allocate_region_batches)
    assert callable(mtr.select_region_candidates)
    assert mtr.multi_trust_region is mtr
    assert ennx.__file__


def test_experimental_quantization():
    import numpy as np

    import ennx.experimental as experimental

    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(experimental.quantize_int4(x), np.array([0x10, 0x02]))
    np.testing.assert_array_equal(
        experimental.quantize_fp4_e2m1(x), np.array([0x20, 0x04])
    )


def test_experimental_multi_trust_region_loop_round():
    import numpy as np

    import ennx.experimental.multi_trust_region as mtr

    class _FakeState:
        def allocate(self, budget, utility=None):
            assert budget == 5
            assert utility is None or len(utility) == 2
            return [
                mtr.RegionBatch(region=0, start=0, length=2),
                mtr.RegionBatch(1, 2, 3),
            ]

        def select(self, candidates, num_arms):
            assert num_arms == 2
            return sorted(candidates, key=lambda candidate: candidate[3], reverse=True)[
                :num_arms
            ]

        def tell(self, x, y):
            return (x, y)

        def restart_region(self, region, new_center):
            return (region, new_center)

        def variance(self, region):
            return float(region)

    loop = mtr.MultiTrustRegionLoop(_FakeState())

    def proposal_fn(batch):
        return [
            mtr.CandidateProposal(
                batch.region, batch.start + 1, np.array([batch.length])
            ),
            mtr.CandidateProposal(
                batch.region, batch.start + 2, np.array([batch.length + 1])
            ),
        ]

    def scorer(proposal):
        return float(proposal.seed)

    round_result = loop.run_round(
        budget=5,
        proposal_fn=proposal_fn,
        scorer=scorer,
        num_arms=2,
    )
    assert [batch.region for batch in round_result.batches] == [0, 1]
    assert len(round_result.proposals) == 4
    assert len(round_result.candidates) == 4
    assert [candidate.seed for candidate in round_result.selected] == [4, 3]
