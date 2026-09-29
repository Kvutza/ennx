use super::*;

const OBJECTIVE_BLOCK: usize = 128;
const OBJECTIVE_BLOCKS: usize = ROWS as usize / OBJECTIVE_BLOCK;

#[derive(Clone, Copy)]
pub(super) struct ObjectiveStats {
    pub(super) reward: f32,
    pub(super) variance: f32,
    pub(super) sequence_nlls: [f32; 2],
    pub(super) block_nlls: [f32; OBJECTIVE_BLOCKS],
}

pub(super) fn mean_variance(samples: &[f32]) -> (f32, f32) {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let centered = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f32>();
    let variance = centered / (samples.len() * (samples.len() - 1)) as f32;
    (mean, variance)
}

pub(super) fn sequence_objective(buffers: &Buffers) -> Result<ObjectiveStats, String> {
    let values = unsafe {
        std::slice::from_raw_parts(
            buffers.sequence_scores.contents().cast::<f32>(),
            BATCH as usize,
        )
    };
    let scores = [values[0], values[1]];
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate objective produced invalid sequence scores: {scores:?}"
        ));
    }
    let losses = unsafe {
        std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize)
    };
    let mask = unsafe {
        std::slice::from_raw_parts(buffers.score_mask.contents().cast::<u8>(), ROWS as usize)
    };
    let mut block_nlls = [0.0f32; OBJECTIVE_BLOCKS];
    for (block, output) in block_nlls.iter_mut().enumerate() {
        let start = block * OBJECTIVE_BLOCK;
        let end = start + OBJECTIVE_BLOCK;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for index in start..end {
            if mask[index] != 0 {
                let loss = losses[index];
                if !loss.is_finite() {
                    return Err(format!(
                        "candidate objective produced invalid token loss at {index}"
                    ));
                }
                sum += loss;
                count += 1;
            }
        }
        if count == 0 {
            return Err(format!(
                "candidate objective block {block} has no scored tokens"
            ));
        }
        *output = sum / count as f32;
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let (_, variance) = mean_variance(&block_nlls);
    Ok(ObjectiveStats {
        reward: -mean,
        variance,
        sequence_nlls: scores,
        block_nlls,
    })
}

pub(super) fn paired_improvement(
    candidate: &ObjectiveStats,
    incumbent: &ObjectiveStats,
) -> (f32, f32) {
    let samples: [f32; OBJECTIVE_BLOCKS] =
        std::array::from_fn(|index| incumbent.block_nlls[index] - candidate.block_nlls[index]);
    let mean = std::array::from_fn::<_, 2, _>(|index| {
        incumbent.sequence_nlls[index] - candidate.sequence_nlls[index]
    })
    .iter()
    .sum::<f32>()
        / 2.0;
    let (_, variance) = mean_variance(&samples);
    (mean, variance)
}

#[cfg(test)]
mod objective_tests {
    use super::*;

    #[test]
    pub(super) fn variance_contract() {
        let (mean, variance) = mean_variance(&[1.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert_eq!(variance, 1.0);
    }

    #[test]
    pub(super) fn paired_noise() {
        let candidate = ObjectiveStats {
            reward: -1.0,
            variance: 0.0,
            sequence_nlls: [1.0, 1.0],
            block_nlls: [1.0; OBJECTIVE_BLOCKS],
        };
        let incumbent = ObjectiveStats {
            reward: -3.0,
            variance: 0.0,
            sequence_nlls: [2.0, 4.0],
            block_nlls: [2.0; OBJECTIVE_BLOCKS],
        };
        let (improvement, variance) = paired_improvement(&candidate, &incumbent);
        assert_eq!(improvement, 2.0);
        assert_eq!(variance, 0.0);
    }
}

pub(super) fn reliability_json(state: crate::ReliabilityTelemetry) -> ennx_wire::json::Value {
    ennx_wire::json::json!({
        "action": format!("{:?}", state.action),
        "concordance": state.concordance,
        "coverage": state.coverage,
        "posterior_mean": state.reliability_mean,
        "posterior_lower": state.reliability_lower,
        "evidence": state.evidence,
        "progress": state.progress,
        "center_radius": state.center_radius,
        "normalized_step": state.normalized_step,
        "conversion": state.conversion,
        "escape_remaining": state.escape_remaining,
        "escapes": state.escapes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn controller_record(
    round: u32,
    initializing: bool,
    proposal: &Proposals,
    decision: NoisyDecision,
    controller: ControllerInfo,
    reliability: Option<crate::ReliabilityTelemetry>,
    wall_seconds: f64,
    controller_seconds: f64,
) -> ennx_wire::json::Value {
    ennx_wire::json::json!({
        "schema": "ennx.reliability_controller.v1",
        "round": round,
        "phase": if initializing { "initialization" } else { "guided" },
        "candidate_index": proposal.index,
        "candidate_seed": proposal.seed,
        "nominal_radius": proposal.length,
        "predicted_mean": proposal.predicted_mean,
        "predicted_standard_error": proposal.predicted_standard_error,
        "incumbent_mean": proposal.incumbent_mean,
        "observed_improvement": decision.improvement,
        "acceptance_threshold": decision.threshold,
        "agreement_ratio": decision.agreement_ratio,
        "trust_outcome": format!("{:?}", decision.trust_outcome),
        "accepted": decision.accepted,
        "next_length": controller.length,
        "reliability": reliability.map(reliability_json),
        "wall_seconds": wall_seconds,
        "controller_seconds": controller_seconds,
    })
}

pub(super) fn log_reliability(round: u32, state: crate::ReliabilityTelemetry) {
    eprintln!(
        "TURBO_ENN_RELIABILITY round={round} action={:?} concordance={} coverage={:.6} posterior_mean={:.6} posterior_lower={:.6} evidence={:.6} progress={:.6} center_radius={} normalized_step={} conversion={} escape_remaining={} escapes={} next_length={:.9}",
        state.action,
        state
            .concordance
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.coverage,
        state.reliability_mean,
        state.reliability_lower,
        state.evidence,
        state.progress,
        state
            .center_radius
            .map_or_else(|| "none".into(), |value| format!("{value:.9}")),
        state
            .normalized_step
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state
            .conversion
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.escape_remaining,
        state.escapes,
        state.length,
    );
}
