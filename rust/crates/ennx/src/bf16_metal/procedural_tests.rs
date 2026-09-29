use super::*;

fn state(pool: ProceduralPool, perturbation: Perturbation) -> SearchState {
    let base = vec![0x3400; TILE_ELEMENTS + 17];
    let block = ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
    let mut state = SearchState::new_implicit(
        &base,
        vec![block],
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        perturbation,
    )
    .unwrap();
    state
        .configure_enn(crate::config::ResidentEnnConfig {
            pool,
            proposal_method: crate::procedural_pool::ProposalMethod::Independent,
            ask: Ask {
                neighbors: 1,
                ..Ask::default()
            },
            num_candidates: 4,
            num_samples: 2,
            fit_neighbors: false,
            distance_scaling: crate::config::DistanceScaling::Global,
            history_geometry: crate::config::HistoryGeometry::Realized,
            local_scale_neighbors: 1,
        })
        .unwrap();
    state.observe_initial(0.0, 0.0).unwrap();
    state
}

fn check_layout(pool: ProceduralPool, perturbation: Perturbation) {
    let mut search = state(pool, perturbation);
    let mut elapsed = Vec::new();
    for round in 0..8 {
        let candidate = round % 4;
        let root = crate::hash::splitmix64(round as u64);
        let expected = search.test_candidate(root, candidate).unwrap();
        let bits = read::<u16>(&expected, search.dimensions).to_vec();
        let radius = search.radius(candidate);
        let start = std::time::Instant::now();
        search
            .begin_forced(root, Ask::default(), candidate)
            .unwrap();
        let proposal = search.finish_ask().unwrap();
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(proposal.seed, pool.seed(root, proposal.identity()).unwrap());
        assert_eq!(proposal.length.to_bits(), radius.to_bits());
        assert_eq!(proposal.arms(), pool.arms() as usize);
        assert_eq!(read::<u16>(&search.proposal, search.dimensions), bits);
        assert_eq!(proposal.procedural_candidates().count(), 4);
        search.tell_noisy(&proposal, -1.0, 0.0).unwrap();
        search.sync().unwrap();
    }
    elapsed.sort_by(f64::total_cmp);
    eprintln!(
        "procedural {}x{} {:?}: {} weights, tiny ask p50={:.3}ms max={:.3}ms (not full BO)",
        pool.arms(),
        pool.slots(),
        perturbation,
        search.dimensions,
        elapsed[4],
        elapsed[7]
    );
}

#[test]
fn resident_layouts() {
    autoreleasepool(|| {
        for (arms, slots) in [(1, 4), (2, 2), (4, 1)] {
            let pool = ProceduralPoolConfig {
                arms,
                candidates_per_arm: slots,
            }
            .resolve_resident()
            .unwrap();
            for perturbation in [Perturbation::Rademacher, Perturbation::Gaussian] {
                check_layout(pool, perturbation);
            }
        }
    });
}
