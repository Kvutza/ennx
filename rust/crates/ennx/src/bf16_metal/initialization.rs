use super::*;
use std::sync::atomic::Ordering;

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                Some((base_value, base_variance)),
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                Perturbation::Gaussian,
            )
        })
    }

    pub(crate) fn new_unscored(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                Perturbation::Gaussian,
            )
        })
    }

    /// Full FP16 weight rows with independent per-coordinate innovations.
    pub(crate) fn new_fp16(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        length: TRLengthConfig,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                capacity,
                1,
                length,
                true,
                false,
                perturbation,
            )
        })
    }

    pub(crate) fn new_implicit(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                resident_capacity,
                1,
                length,
                true,
                true,
                perturbation,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_inner(
        base: &[u16],
        initial: Option<(f32, f32)>,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
        independent_fp16: bool,
        implicit_history: bool,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        let length = checked_length(length)?;
        if capacity == 0 || capacity > MAX_HISTORY || pending_capacity != 1 {
            return Err(
                "Metal correlated search requires capacity in 1..=128 and max_pending=1".into(),
            );
        }
        if let Some((value, variance)) = initial {
            check_scores(&[value], &[variance])?;
        }
        let (tiles, offsets, leaves) = make_layout(&blocks, base.len())?;
        let exponent_mask = if independent_fp16 { 0x7c00 } else { 0x7f80 };
        if base
            .iter()
            .any(|bits| bits & exponent_mask == exponent_mask)
        {
            return Err("Base weights must be finite".into());
        }
        let exact_history = independent_fp16 && implicit_history;
        let (partial_count, aggregate_count, geometry_count, replay_count, sizes, resident_bytes) =
            allocation_sizes(
                base.len(),
                blocks.len(),
                tiles.len(),
                offsets.len(),
                capacity,
                exact_history,
            )?;
        let runtime = Runtime::shared()?;
        preflight(&runtime, &sizes, resident_bytes)?;
        let source = search_source(independent_fp16, perturbation);
        let propose_pipeline = runtime.pipeline(&source, "Dense search", "bf16_propose")?;
        let initial_pipeline = runtime.pipeline(&source, "Dense search", "bf16_propose_initial")?;
        let pool_pipeline = runtime.pipeline(&source, "Dense search", "bf16_propose_pool")?;
        let reduction_pipeline = runtime.precise(&source, "Dense search", "bf16_reduce_pool")?;
        let replay_short_pipeline =
            runtime.precise(&source, "Dense search", "bf16_replay_short")?;
        let replay_pipeline = runtime.precise(&source, "Dense search", "bf16_replay_history")?;
        let replay_reduction_pipeline =
            runtime.precise(&source, "Dense search", "bf16_reduce_replay")?;
        let selection_pipeline = runtime.precise(&source, "Dense search", "bf16_select")?;
        let materialize_pipeline = runtime.pipeline(&source, "Dense search", "bf16_materialize")?;
        let reference_pipeline = runtime.precise(&source, "Dense search", "bf16_reference")?;
        let rms_pipeline = runtime.precise(&source, "Dense search", "bf16_reference_rms")?;
        for pipeline in [
            &propose_pipeline,
            &initial_pipeline,
            &pool_pipeline,
            &reduction_pipeline,
            &replay_short_pipeline,
            &replay_pipeline,
            &replay_reduction_pipeline,
            &selection_pipeline,
            &materialize_pipeline,
            &reference_pipeline,
            &rms_pipeline,
        ] {
            if pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("BF16 Metal kernels require 256 threads per group".into());
            }
        }
        let history_rows: Vec<_> = (0..capacity)
            .map(|_| runtime.buffer::<u16>(base.len()))
            .collect();
        let replay_scale_count = if exact_history {
            MAX_HISTORY * blocks.len()
        } else {
            1
        };
        let replay_family_count = if exact_history {
            4 * MAX_HISTORY * FAMILIES
        } else {
            1
        };
        let mut trust = TurboTrustRegion::new(base.len(), length);
        trust.set_arms(1);
        let mut state = Self {
            independent_fp16,
            perturbation,
            leaves_gpu: runtime.buffer_with(&leaves),
            tiles_gpu: runtime.buffer_with(&tiles),
            offsets_gpu: runtime.buffer_with(&offsets),
            base: runtime.buffer_with(base),
            replay_origin: exact_history.then(|| runtime.buffer_with(base)),
            anchor: history_rows[0].clone(),
            rejected: history_rows[capacity.min(2) - 1].clone(),
            history_rows,
            proposal: runtime.buffer::<u16>(base.len()),
            reference: None,
            reference_scales: runtime.buffer_with(&vec![1.0f32; blocks.len()]),
            reference_partials: runtime.buffer::<f32>(tiles.len()),
            partials: runtime.buffer::<Partial>(partial_count),
            pool_aggregates: runtime.buffer::<Partial>(aggregate_count),
            decision: runtime.buffer::<Decision>(1),
            pool_distances: runtime.buffer::<f32>(5 * MAX_HISTORY),
            replay_steps: runtime.buffer::<ReplayStep>(MAX_HISTORY),
            replay_scales: runtime.buffer::<f32>(replay_scale_count),
            replay_partials: runtime.buffer::<f32>(replay_count),
            replay_components: runtime.buffer::<f32>(replay_count),
            pool_family_distances: runtime.buffer::<f32>(replay_family_count),
            family_groups: runtime.buffer_with(&vec![0u32; blocks.len()]),
            family_base_weights: runtime
                .buffer_with(&blocks.iter().map(|block| block.weight).collect::<Vec<_>>()),
            pool_geometry: runtime.buffer::<f32>(geometry_count),
            threshold_tables: runtime.buffer::<u64>(4 * crate::threshold::TABLE_WORDS),
            runtime,
            blocks,
            tiles,
            propose_pipeline,
            initial_pipeline,
            pool_pipeline,
            reduction_pipeline,
            replay_short_pipeline,
            replay_pipeline,
            replay_reduction_pipeline,
            selection_pipeline,
            materialize_pipeline,
            reference_pipeline,
            rms_pipeline,
            dimensions: base.len(),
            resident_bytes,
            length_config: length,
            length: length.length_init,
            best: f32::NEG_INFINITY,
            best_variance: 0.0,
            objective_selector: None,
            pool_layout: crate::procedural_pool::ProceduralPool::legacy(),
            proposal_method: crate::procedural_pool::ProposalMethod::Independent,
            threshold: None,
            objective_history: crate::objective_observation::ObjectiveWindow::new(
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                },
            ),
            outcomes: vec![
                0.0;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            variances: vec![
                0.0;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            identities: vec![
                1;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            observation: 1,
            base_id: 1,
            trust,
            reliability: None,
            observed: Vec::new(),
            restart_count: 0,
            history: 0,
            resident_history: 0,
            resident_identities: vec![1; capacity],
            pairwise_distances: vec![0.0; MAX_HISTORY * MAX_HISTORY],
            distance_scaling: crate::config::DistanceScaling::Global,
            local_scale_neighbors: 8,
            family: None,
            metric_gpu: None,
            implicit_history,
            exact_history,
            latent_history: false,
            initial_observations: 0,
            fit_candidates: 0,
            fit_samples: 0,
            fit_neighbors: false,
            fit_seed: 0,
            fitter: None,
            fitted_enn: None,
            failures: 0,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            next_id: 0,
            pending: None,
            queued: None,
            reference_seed: None,
            relative: false,
            started: false,
            poisoned: false,
            profiling: false,
            last_profile: None,
            async_command: None,
            profile_commands: Vec::new(),
            tell_profile: None,
            axis: None,
        };
        state.copy(&state.base, &state.anchor)?;
        if let Some((value, variance)) = initial {
            state.observe_initial(value, variance)?;
        }
        Ok(state)
    }
}

fn allocation_sizes(
    dimensions: usize,
    blocks: usize,
    tiles: usize,
    offsets: usize,
    capacity: usize,
    exact_history: bool,
) -> Result<(usize, usize, usize, usize, [u64; 19], u64), String> {
    let row_bytes = bytes::<u16>(dimensions)?;
    let partial_count = tiles
        .checked_mul(4)
        .and_then(|n| n.checked_mul(capacity.div_ceil(2)))
        .ok_or("BF16 partial count overflow")?;
    let aggregate_count = 4usize
        .checked_mul(capacity.div_ceil(2))
        .ok_or("BF16 aggregate count overflow")?;
    let geometry_count = tiles
        .checked_mul(POOL_METRICS)
        .ok_or("BF16 geometry count overflow")?;
    let replay_count = if exact_history {
        tiles
            .checked_mul(4 * MAX_HISTORY)
            .ok_or("BF16 replay partial count overflow")?
    } else {
        1
    };
    let replay_scales = if exact_history {
        MAX_HISTORY
            .checked_mul(blocks)
            .ok_or("BF16 replay scale count overflow")?
    } else {
        1
    };
    let replay_families = if exact_history {
        4 * MAX_HISTORY * FAMILIES
    } else {
        1
    };
    let sizes = [
        row_bytes,
        bytes::<Leaf>(blocks)?,
        bytes::<Tile>(tiles)?,
        bytes::<u32>(offsets)?,
        bytes::<f32>(blocks)?,
        bytes::<f32>(tiles)?,
        bytes::<Partial>(partial_count)?,
        bytes::<Partial>(aggregate_count)?,
        bytes::<Decision>(1)?,
        bytes::<f32>(5 * MAX_HISTORY)?,
        bytes::<f32>(geometry_count)?,
        bytes::<ReplayStep>(MAX_HISTORY)?,
        bytes::<f32>(replay_scales)?,
        bytes::<f32>(replay_count)?,
        bytes::<f32>(replay_count)?,
        bytes::<f32>(replay_families)?,
        bytes::<u32>(blocks)?,
        bytes::<f32>(blocks)?,
        bytes::<u64>(4 * crate::threshold::TABLE_WORDS)?,
    ];
    let resident_bytes = sizes.iter().try_fold(
        row_bytes
            .checked_mul(capacity as u64 + 2 + u64::from(exact_history))
            .ok_or("BF16 memory overflow")?,
        |sum, size| sum.checked_add(*size).ok_or("BF16 memory overflow"),
    )?;
    Ok((
        partial_count,
        aggregate_count,
        geometry_count,
        replay_count,
        sizes,
        resident_bytes,
    ))
}

pub(super) fn search_source(independent_fp16: bool, perturbation: Perturbation) -> String {
    let mut defines = String::new();
    if independent_fp16 {
        defines.push_str("#define FP16_INDEPENDENT\n");
    }
    if perturbation == Perturbation::Rademacher {
        defines.push_str("#define RADEMACHER_ONLY\n");
    }
    #[cfg(test)]
    defines.push_str("#define POOL_GEOMETRY_DIAGNOSTICS\n");
    let tables = if perturbation == Perturbation::Gaussian {
        metal_tables()
    } else {
        String::new()
    };
    format!(
        "{defines}{}",
        SOURCE.replacen("// ENNX_ZIGGURAT_TABLES", &tables, 1)
    )
}
