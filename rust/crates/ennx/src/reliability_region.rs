//! Reliability-aware, density-normalized trust-region control.

use serde::{Deserialize, Serialize};

use crate::trust_region::TRLengthConfig;

const PRIOR: f64 = 1.0;
const MIN_POSITIVE: f64 = 1.0e-12;

/// User-tunable policy for the full-weight ENN controller.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReliabilityControllerConfig {
    /// Neighbor rank used for the incumbent's raw metric radius.
    pub local_scale_neighbors: usize,
    /// Discount applied to past correct/incorrect ordering evidence.
    pub evidence_decay: f64,
    /// Discount applied to normalized realized improvement.
    pub progress_decay: f64,
    /// Normal quantile used for the conservative reliability bound.
    pub confidence_z: f64,
    /// Minimum conservative concordance considered reliable.
    pub reliability_threshold: f64,
    /// Minimum EWMA progress considered sustained progress.
    pub progress_threshold: f64,
    /// Decisive round-equivalents required before reliability controls length.
    pub minimum_evidence: f64,
    /// Desired realized step divided by incumbent neighbor radius.
    pub target_step_ratio: f64,
    /// Fraction of the density correction applied per guided round.
    pub density_gain: f64,
    /// Multiplicative expansion when ranking and progress agree.
    pub expand_factor: f64,
    /// Multiplicative contraction when ranking and progress both fail.
    pub shrink_factor: f64,
    /// Reliable non-progress rounds before an escape burst.
    pub stagnation_patience: u32,
    /// Maximum-radius fresh-direction rounds in an escape burst.
    pub escape_rounds: u32,
}

impl Default for ReliabilityControllerConfig {
    fn default() -> Self {
        Self {
            local_scale_neighbors: 8,
            evidence_decay: 0.9,
            progress_decay: 0.8,
            confidence_z: 1.0,
            reliability_threshold: 0.6,
            progress_threshold: 0.1,
            minimum_evidence: 4.0,
            target_step_ratio: 0.5,
            density_gain: 0.25,
            expand_factor: 1.25,
            shrink_factor: 0.8,
            stagnation_patience: 8,
            escape_rounds: 4,
        }
    }
}

impl ReliabilityControllerConfig {
    pub fn validate(self) -> Result<Self, String> {
        let probabilities = [
            ("evidence_decay", self.evidence_decay),
            ("progress_decay", self.progress_decay),
            ("reliability_threshold", self.reliability_threshold),
            ("density_gain", self.density_gain),
        ];
        for (name, value) in probabilities {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("reliability controller {name} must be in 0..=1"));
            }
        }
        let positive = [
            ("confidence_z", self.confidence_z),
            ("progress_threshold", self.progress_threshold),
            ("minimum_evidence", self.minimum_evidence),
            ("target_step_ratio", self.target_step_ratio),
            ("expand_factor", self.expand_factor),
            ("shrink_factor", self.shrink_factor),
        ];
        for (name, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "reliability controller {name} must be positive and finite"
                ));
            }
        }
        if self.local_scale_neighbors == 0 || self.local_scale_neighbors > 128 {
            return Err("reliability controller local_scale_neighbors must be in 1..=128".into());
        }
        if self.expand_factor <= 1.0 || self.shrink_factor >= 1.0 {
            return Err(
                "reliability controller requires expand_factor > 1 and shrink_factor < 1".into(),
            );
        }
        if self.stagnation_patience == 0 || self.escape_rounds == 0 {
            return Err(
                "reliability controller patience and escape rounds must be positive".into(),
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityAction {
    Hold,
    Expand,
    Shrink,
    Escape,
}

#[derive(Debug, Clone, Copy)]
enum FeedbackRegime {
    Learning,
    ReliableProgress,
    ReliableStagnation,
    UnreliableProgress,
    UnreliableStagnation,
}

#[derive(Debug, Clone, Copy)]
pub struct ReliabilityEvidence {
    pub concordance: Option<f64>,
    pub coverage: f64,
    pub rank_evidence: f64,
    pub improvement: f64,
    pub improvement_variance: f64,
    pub nominal_radius: f64,
    pub realized_radius: f64,
    pub center_radius: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ReliabilityTelemetry {
    pub action: ReliabilityAction,
    pub concordance: Option<f64>,
    pub coverage: f64,
    pub reliability_mean: f64,
    pub reliability_lower: f64,
    pub evidence: f64,
    pub progress: f64,
    pub center_radius: Option<f64>,
    pub normalized_step: Option<f64>,
    pub conversion: Option<f64>,
    pub escape_remaining: u32,
    pub escapes: usize,
    pub length: f64,
}

/// Constant-state controller over ENN ranking, progress, and local density.
#[derive(Debug, Clone)]
pub struct ReliabilityController {
    config: ReliabilityControllerConfig,
    bounds: TRLengthConfig,
    length: f64,
    correct: f64,
    incorrect: f64,
    progress: f64,
    log_conversion: Option<f64>,
    stagnation: u32,
    escape_remaining: u32,
    escapes: usize,
    telemetry: ReliabilityTelemetry,
}

impl ReliabilityController {
    pub fn new(
        config: ReliabilityControllerConfig,
        bounds: TRLengthConfig,
    ) -> Result<Self, String> {
        let config = config.validate()?;
        let telemetry = ReliabilityTelemetry {
            action: ReliabilityAction::Hold,
            concordance: None,
            coverage: 0.0,
            reliability_mean: 0.5,
            reliability_lower: 0.5 - config.confidence_z / 12.0f64.sqrt(),
            evidence: 0.0,
            progress: 0.0,
            center_radius: None,
            normalized_step: None,
            conversion: None,
            escape_remaining: 0,
            escapes: 0,
            length: bounds.length_init,
        };
        Ok(Self {
            config,
            bounds,
            length: bounds.length_init,
            correct: PRIOR,
            incorrect: PRIOR,
            progress: 0.0,
            log_conversion: None,
            stagnation: 0,
            escape_remaining: 0,
            escapes: 0,
            telemetry,
        })
    }

    pub const fn config(&self) -> ReliabilityControllerConfig {
        self.config
    }

    pub const fn length(&self) -> f64 {
        self.length
    }

    pub const fn force_fresh(&self) -> bool {
        self.escape_remaining > 0
    }

    pub const fn telemetry(&self) -> ReliabilityTelemetry {
        self.telemetry
    }

    pub fn update(
        &mut self,
        evidence: ReliabilityEvidence,
    ) -> Result<ReliabilityTelemetry, String> {
        validate_evidence(evidence)?;
        self.update_conversion(evidence.nominal_radius, evidence.realized_radius);
        self.update_rank(evidence.concordance, evidence.rank_evidence);
        let decisive_progress =
            self.update_progress(evidence.improvement, evidence.improvement_variance);
        let (mean, lower, amount) = self.reliability();
        let ready = amount >= self.config.minimum_evidence;
        let reliable = ready && lower >= self.config.reliability_threshold;
        let progressing = self.progress >= self.config.progress_threshold;
        self.stagnation = if decisive_progress > 0.0 {
            0
        } else if reliable {
            self.stagnation.saturating_add(1)
        } else {
            self.stagnation
        };
        let regime = match (ready, reliable, progressing) {
            (false, _, _) => FeedbackRegime::Learning,
            (true, true, true) => FeedbackRegime::ReliableProgress,
            (true, true, false) => FeedbackRegime::ReliableStagnation,
            (true, false, true) => FeedbackRegime::UnreliableProgress,
            (true, false, false) => FeedbackRegime::UnreliableStagnation,
        };
        let action = self.choose_action(regime, decisive_progress);
        self.apply_length(action, evidence.center_radius);
        self.telemetry = ReliabilityTelemetry {
            action,
            concordance: evidence.concordance,
            coverage: evidence.coverage,
            reliability_mean: mean,
            reliability_lower: lower,
            evidence: amount,
            progress: self.progress,
            center_radius: evidence.center_radius,
            normalized_step: evidence
                .center_radius
                .filter(|radius| *radius > 0.0)
                .map(|radius| evidence.realized_radius / radius),
            conversion: self.log_conversion.map(f64::exp),
            escape_remaining: self.escape_remaining,
            escapes: self.escapes,
            length: self.length,
        };
        Ok(self.telemetry)
    }

    fn update_conversion(&mut self, nominal: f64, realized: f64) {
        if nominal <= 0.0 || realized <= 0.0 {
            return;
        }
        let sample = (realized / nominal).ln();
        self.log_conversion = Some(self.log_conversion.map_or(sample, |previous| {
            self.config.progress_decay * previous + (1.0 - self.config.progress_decay) * sample
        }));
    }

    fn update_rank(&mut self, concordance: Option<f64>, coverage: f64) {
        self.correct = PRIOR + self.config.evidence_decay * (self.correct - PRIOR);
        self.incorrect = PRIOR + self.config.evidence_decay * (self.incorrect - PRIOR);
        if let Some(value) = concordance {
            self.correct += coverage * value;
            self.incorrect += coverage * (1.0 - value);
        }
    }

    fn update_progress(&mut self, improvement: f64, variance: f64) -> f64 {
        let threshold = 2.0 * variance.sqrt();
        let sample = if improvement > threshold.max(MIN_POSITIVE) {
            1.0
        } else if improvement < -threshold.max(MIN_POSITIVE) {
            -1.0
        } else {
            0.0
        };
        self.progress = self.config.progress_decay * self.progress
            + (1.0 - self.config.progress_decay) * sample;
        sample
    }

    fn reliability(&self) -> (f64, f64, f64) {
        let total = self.correct + self.incorrect;
        let mean = self.correct / total;
        let variance = self.correct * self.incorrect / (total * total * (total + 1.0));
        let lower = (mean - self.config.confidence_z * variance.sqrt()).clamp(0.0, 1.0);
        (mean, lower, (total - 2.0 * PRIOR).max(0.0))
    }

    fn choose_action(
        &mut self,
        regime: FeedbackRegime,
        decisive_progress: f64,
    ) -> ReliabilityAction {
        if self.escape_remaining > 0 {
            if decisive_progress > 0.0 {
                self.escape_remaining = 0;
                return ReliabilityAction::Hold;
            }
            self.escape_remaining -= 1;
            return ReliabilityAction::Escape;
        }
        if matches!(regime, FeedbackRegime::ReliableStagnation)
            && self.stagnation >= self.config.stagnation_patience
        {
            self.stagnation = 0;
            self.escape_remaining = self.config.escape_rounds;
            self.escapes += 1;
            return ReliabilityAction::Escape;
        }
        match regime {
            FeedbackRegime::ReliableProgress => ReliabilityAction::Expand,
            FeedbackRegime::UnreliableStagnation => ReliabilityAction::Shrink,
            FeedbackRegime::Learning
            | FeedbackRegime::ReliableStagnation
            | FeedbackRegime::UnreliableProgress => ReliabilityAction::Hold,
        }
    }

    fn apply_length(&mut self, action: ReliabilityAction, center_radius: Option<f64>) {
        if action == ReliabilityAction::Escape {
            self.length = self.bounds.length_max;
            return;
        }
        let mut log_step = match action {
            ReliabilityAction::Expand => self.config.expand_factor.ln(),
            ReliabilityAction::Shrink => self.config.shrink_factor.ln(),
            ReliabilityAction::Hold | ReliabilityAction::Escape => 0.0,
        };
        if let (Some(radius), Some(log_conversion)) = (center_radius, self.log_conversion)
            && radius > 0.0
        {
            let desired = self.config.target_step_ratio * radius / log_conversion.exp();
            let correction = (desired / self.length)
                .ln()
                .clamp(-2.0f64.ln(), 2.0f64.ln());
            log_step += self.config.density_gain * correction;
        }
        self.length = (self.length.ln() + log_step)
            .exp()
            .clamp(self.bounds.length_min, self.bounds.length_max);
    }
}

fn validate_evidence(evidence: ReliabilityEvidence) -> Result<(), String> {
    let finite = [
        evidence.coverage,
        evidence.rank_evidence,
        evidence.improvement,
        evidence.improvement_variance,
        evidence.nominal_radius,
        evidence.realized_radius,
    ];
    if finite.iter().any(|value| !value.is_finite())
        || evidence.improvement_variance < 0.0
        || evidence.rank_evidence < 0.0
        || (evidence.concordance.is_none() && evidence.rank_evidence != 0.0)
        || evidence.nominal_radius <= 0.0
        || evidence.realized_radius < 0.0
        || !(0.0..=1.0).contains(&evidence.coverage)
        || evidence
            .concordance
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || evidence
            .center_radius
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("invalid reliability-controller evidence".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(concordance: f64, improvement: f64) -> ReliabilityEvidence {
        ReliabilityEvidence {
            concordance: Some(concordance),
            coverage: 1.0,
            rank_evidence: 1.0,
            improvement,
            improvement_variance: 0.0,
            nominal_radius: 0.01,
            realized_radius: 0.01,
            center_radius: Some(0.02),
        }
    }

    #[test]
    fn reliable_progress_expands() {
        let mut controller = ReliabilityController::new(
            ReliabilityControllerConfig {
                minimum_evidence: 1.0,
                reliability_threshold: 0.4,
                ..Default::default()
            },
            TRLengthConfig::new(0.01, 0.0001, 0.1),
        )
        .unwrap();
        let initial = controller.length();
        let state = controller.update(evidence(1.0, 1.0)).unwrap();
        assert_eq!(state.action, ReliabilityAction::Expand);
        assert!(state.length > initial);
    }

    #[test]
    fn unreliable_stagnation_shrinks() {
        let mut controller = ReliabilityController::new(
            ReliabilityControllerConfig {
                minimum_evidence: 1.0,
                reliability_threshold: 0.9,
                ..Default::default()
            },
            TRLengthConfig::new(0.01, 0.0001, 0.1),
        )
        .unwrap();
        let state = controller.update(evidence(0.0, -1.0)).unwrap();
        assert_eq!(state.action, ReliabilityAction::Shrink);
        assert!(state.length < 0.01);
    }

    #[test]
    fn reliable_stagnation_forces_fresh_escape() {
        let config = ReliabilityControllerConfig {
            minimum_evidence: 1.0,
            reliability_threshold: 0.4,
            stagnation_patience: 2,
            escape_rounds: 3,
            ..Default::default()
        };
        let mut controller =
            ReliabilityController::new(config, TRLengthConfig::new(0.01, 0.0001, 0.1)).unwrap();
        controller.update(evidence(1.0, 0.0)).unwrap();
        let state = controller.update(evidence(1.0, 0.0)).unwrap();
        assert_eq!(state.action, ReliabilityAction::Escape);
        assert_eq!(state.length, 0.1);
        assert!(controller.force_fresh());
    }

    #[test]
    fn config_rejects_false_policy() {
        let error = ReliabilityControllerConfig {
            expand_factor: 1.0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("expand_factor"));
    }
}
