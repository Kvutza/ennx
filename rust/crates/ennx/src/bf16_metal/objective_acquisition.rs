//! Independent objective acquisition; scalar control remains caller-supplied.
use super::*;
use crate::Rescalarize;
use crate::mbtrregn::{MorboTRSettings, MorboTrustRegion};
use crate::objective_observation::MAX_OBJECTIVES;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[path = "objective_acquisition/parameters.rs"]
mod parameters;
use parameters::ObjectiveParams;
#[path = "objective_acquisition/hypervolume.rs"]
mod hypervolume;
use hypervolume::{improvement, measured};

/// Every objective is maximized. Negate minimization measurements explicitly.
#[derive(Clone, Debug)]
pub enum ObjectiveAcquisition {
    /// Exact nondominance of objective acquisition vectors, then a seeded tie-break.
    Pareto,
    /// Randomized augmented Chebyshev acquisition, not an implicit objective sum.
    /// Positive preferences multiply Dirichlet(1) weights, then normalize to sum one.
    Morbo {
        weights: Vec<f32>,
        regions: usize,
        alpha: f32,
        seed: u64,
        rescalarize: Rescalarize,
        clip: bool,
    },
}

#[derive(Clone, Debug)]
pub struct ObjectivePolicy {
    pub acquisition: ObjectiveAcquisition,
    /// Explicit per-objective posterior scales, in the measurement's own units.
    pub scales: Vec<f32>,
}

struct MorboRegion {
    trust: MorboTrustRegion,
    center: Option<Buffer>,
    record: crate::objective_observation::ObjectiveRecord,
    trials: u64,
}

pub(super) struct ResidentMorbo {
    regions: Vec<MorboRegion>,
    active: usize,
    preferences: Vec<f32>,
    scales: Vec<f32>,
    reference: Vec<f64>,
    seed: u64,
    observations: usize,
}

impl ResidentMorbo {
    fn new(state: &SearchState, acquisition: &ObjectiveAcquisition) -> Result<Self, String> {
        let ObjectiveAcquisition::Morbo {
            weights,
            regions,
            alpha,
            seed,
            rescalarize,
            clip: _,
        } = acquisition
        else {
            return Err("Resident MORBO requires MORBO acquisition".into());
        };
        if state.reliability.is_some() {
            return Err("MORBO owns trust-region adaptation and cannot be combined with the scalar reliability controller".into());
        }
        if *rescalarize != Rescalarize::OnRestart {
            return Err("Exact resident MORBO requires rescalarize='on_restart'; changing scalarization every proposal would require rematerializing a historical billion-weight center".into());
        }
        if !(2..=8).contains(regions) {
            return Err("Resident MORBO requires 2..=8 trust regions".into());
        }
        if state.implicit_history && !state.latent_history {
            return Err("Multi-region MORBO currently requires latent history geometry so center switches preserve the shared ENN geometry".into());
        }
        let incumbent = *state
            .objective_history
            .incumbent()
            .ok_or("Configure MORBO after the initial vector observation")?;
        let mut resident = Self {
            regions: Vec::with_capacity(*regions),
            active: 0,
            preferences: weights.clone(),
            scales: Vec::new(),
            reference: Vec::new(),
            seed: *seed,
            observations: 0,
        };
        // Objective scales are installed by ObjectiveSelector immediately after
        // construction. Initialize a valid fixed reference until then.
        resident.reference = incumbent
            .observation
            .estimates()
            .iter()
            .map(|estimate| f64::from(estimate.mean) - 1.0)
            .collect();
        for region in 0..*regions {
            let mut rng = StdRng::seed_from_u64(crate::hash::splitmix64(*seed ^ region as u64));
            let mut trust = MorboTrustRegion::new(
                state.dimensions,
                MorboTRSettings {
                    num_metrics: weights.len(),
                    alpha: f64::from(*alpha),
                    length: state.length_config,
                    rescalarize: *rescalarize,
                    noise_aware: true,
                },
                &mut rng,
            )
            .map_err(|error| error.to_string())?;
            trust.set_arms(1);
            trust
                .set_tolerance(state.trust.failure_tolerance() as usize)
                .map_err(|error| error.to_string())?;
            let center = if region == 0 {
                None
            } else {
                let center = state.runtime.buffer::<u16>(state.dimensions);
                state.copy(&state.base, &center)?;
                Some(center)
            };
            resident.regions.push(MorboRegion {
                trust,
                center,
                record: incumbent,
                trials: 0,
            });
            resident.sample_weights(region, 0)?;
            resident.seed(region, incumbent.observation)?;
        }
        Ok(resident)
    }

    fn install_scales(&mut self, scales: &[f32]) {
        self.scales = scales.to_vec();
        let observation = self.regions[0].record.observation;
        self.reference = observation
            .estimates()
            .iter()
            .zip(scales)
            .map(|(estimate, scale)| f64::from(estimate.mean - scale))
            .collect();
    }

    fn sample_weights(&mut self, region: usize, epoch: usize) -> Result<(), String> {
        let stream = ((region as u64) << 32) ^ epoch as u64;
        let mut rng = StdRng::seed_from_u64(crate::hash::splitmix64(self.seed ^ stream));
        let samples = self
            .preferences
            .iter()
            .map(|preference| {
                -rng.r#gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0).ln() * f64::from(*preference)
            })
            .collect::<Vec<_>>();
        self.regions[region]
            .trust
            .set_weights(&samples)
            .map_err(|error| error.to_string())
    }

    fn values(observation: crate::objective_observation::ObjectiveObservation) -> Array1<f64> {
        Array1::from_iter(
            observation
                .estimates()
                .iter()
                .map(|estimate| f64::from(estimate.mean)),
        )
    }

    fn seed(
        &mut self,
        region: usize,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let values = Self::values(observation);
        let row = values.view().insert_axis(ndarray::Axis(0));
        self.regions[region].trust.update_incremental(&row);
        // The trust region deliberately preserves its global observation
        // watermark across a local restart. Seed the first region at one, but
        // never move that watermark backwards when recentering later regions.
        self.observations = self.observations.max(1);
        self.regions[region]
            .trust
            .update_only(&values.view(), self.observations)
            .map_err(|error| error.to_string())
    }

    fn hypervolume<'a>(
        &self,
        rows: impl Iterator<Item = &'a crate::objective_observation::ObjectiveRecord>,
        candidate: Option<crate::objective_observation::ObjectiveObservation>,
    ) -> Result<f64, String> {
        measured(rows, candidate, &self.reference, &self.scales)
    }

    pub(super) fn candidate_improves(
        &self,
        history: &crate::objective_observation::ObjectiveWindow,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(bool, f64), String> {
        let before = self.hypervolume(history.rows(), None)?;
        let after = self.hypervolume(history.rows(), Some(candidate))?;
        Ok(improvement(before, after))
    }

    pub(super) fn observe(
        &mut self,
        candidate: crate::objective_observation::ObjectiveObservation,
        accepted: bool,
        identity: i64,
    ) -> Result<(), String> {
        let region = self.active;
        let observation = candidate;
        let candidate = Self::values(observation);
        self.regions[region]
            .trust
            .update_incremental(&candidate.view().insert_axis(ndarray::Axis(0)));
        self.observations += 1;
        self.regions[region].trials += 1;
        if accepted {
            self.regions[region].record = crate::objective_observation::ObjectiveRecord {
                identity,
                observation,
            };
        }
        let incumbent = Self::values(self.regions[region].record.observation);
        self.regions[region]
            .trust
            .update_only(&incumbent.view(), self.observations)
            .map_err(|error| error.to_string())
    }

    pub(super) fn restart(
        &mut self,
        region: usize,
        epoch: usize,
        incumbent: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        self.regions[region].trust.restart(None);
        self.sample_weights(region, epoch)?;
        self.seed(region, incumbent)
    }

    pub(super) fn trust(&self) -> &MorboTrustRegion {
        &self.regions[self.active].trust
    }

    pub(super) fn set_tolerance(&mut self, failures: usize) -> Result<(), String> {
        for region in &mut self.regions {
            region
                .trust
                .set_tolerance(failures)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn select_region(&self) -> usize {
        if let Some(region) = self.regions.iter().position(|region| region.trials == 0) {
            return region;
        }
        let total_trials = self.regions.iter().map(|region| region.trials).sum::<u64>() as f64;
        let records = self
            .regions
            .iter()
            .map(|region| region.record)
            .collect::<Vec<_>>();
        let total_volume = self.hypervolume(records.iter(), None).unwrap_or(0.0);
        self.regions
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                let left_score = self.exclusive_volume(&records, *left_index, total_volume)
                    + (2.0 * total_trials.ln() / left.trials as f64).sqrt();
                let right_score = self.exclusive_volume(&records, *right_index, total_volume)
                    + (2.0 * total_trials.ln() / right.trials as f64).sqrt();
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap()
    }

    fn region_count(&self) -> usize {
        self.regions.len()
    }

    fn best_region(&self) -> Result<usize, String> {
        let records = self
            .regions
            .iter()
            .map(|region| region.record)
            .collect::<Vec<_>>();
        let total = self.hypervolume(records.iter(), None)?;
        Ok((0..records.len())
            .max_by(|&left, &right| {
                self.exclusive_volume(&records, left, total)
                    .total_cmp(&self.exclusive_volume(&records, right, total))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap_or(0))
    }

    fn restart_source(&self, target: usize) -> Result<usize, String> {
        let records = self
            .regions
            .iter()
            .map(|region| region.record)
            .collect::<Vec<_>>();
        let total = self.hypervolume(records.iter(), None)?;
        Ok((0..records.len())
            .filter(|&region| region != target)
            .max_by(|&left, &right| {
                self.exclusive_volume(&records, left, total)
                    .total_cmp(&self.exclusive_volume(&records, right, total))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap_or(target))
    }

    fn exclusive_volume(
        &self,
        records: &[crate::objective_observation::ObjectiveRecord],
        excluded: usize,
        total: f64,
    ) -> f64 {
        let volume = self
            .hypervolume(
                records
                    .iter()
                    .enumerate()
                    .filter_map(|(index, record)| (index != excluded).then_some(record)),
                None,
            )
            .unwrap_or(0.0);
        total - volume
    }
}

impl ObjectivePolicy {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=MAX_OBJECTIVES).contains(&self.scales.len())
            || self
                .scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(
                "Vector acquisition requires 2..=8 positive finite objective scales".into(),
            );
        }
        if let ObjectiveAcquisition::Morbo {
            weights,
            regions,
            alpha,
            clip,
            ..
        } = &self.acquisition
        {
            if weights.len() != self.scales.len()
                || !(2..=8).contains(regions)
                || weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight <= 0.0)
                || !alpha.is_finite()
                || *alpha < 0.0
                || !*clip
            {
                return Err("MORBO requires one positive finite preference per objective, finite nonnegative alpha and clip=true for incumbent updates".into());
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ObjectiveSelectionReport {
    pub values: [[f32; MAX_OBJECTIVES]; 4],
    pub weights: [f32; MAX_OBJECTIVES],
    pub width: u32,
    pub nondominated_mask: u32,
    pub selected: u32,
    pad: u32,
}

pub(super) struct ObjectiveSelector {
    config: ObjectivePolicy,
    reference: Vec<f64>,
    pub(super) morbo: Option<ResidentMorbo>,
    pub(super) pipeline: ComputePipelineState,
    pub(super) parameters: Buffer,
    pub(super) report: Buffer,
}

impl ObjectiveSelector {
    fn new(state: &SearchState, config: ObjectivePolicy) -> Result<Self, String> {
        config.validate()?;
        parameters::validate_history(state, config.scales.len())?;
        let size = size_of::<ObjectiveParams>() as u64;
        let report_size = size_of::<ObjectiveSelectionReport>() as u64;
        let region_count = match &config.acquisition {
            ObjectiveAcquisition::Morbo { regions, .. } => *regions,
            ObjectiveAcquisition::Pareto => 0,
        };
        let mut sizes = vec![size, report_size];
        sizes.extend(std::iter::repeat_n(
            state.row_bytes(),
            region_count.saturating_sub(1),
        ));
        let resident = sizes.iter().try_fold(0u64, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or("MORBO memory accounting overflow")
        })?;
        preflight(&state.runtime, &sizes, resident)?;
        let mut morbo = matches!(config.acquisition, ObjectiveAcquisition::Morbo { .. })
            .then(|| ResidentMorbo::new(state, &config.acquisition))
            .transpose()?;
        if let Some(morbo) = &mut morbo {
            morbo.install_scales(&config.scales);
        }
        let source = format!(
            "{}\n{}",
            state.procedural_source(state.pool_layout),
            include_str!("../bf16_objectives.metal")
        );
        let reference = state
            .objective_history
            .rows()
            .next()
            .ok_or("Vector acquisition requires an initial observation")?
            .observation
            .estimates()
            .iter()
            .zip(&config.scales)
            .map(|(estimate, scale)| f64::from(estimate.mean - scale))
            .collect();
        Ok(Self {
            config,
            reference,
            morbo,
            pipeline: state
                .runtime
                .precise(&source, "Vector search", "bf16_select_objectives")?,
            parameters: state.runtime.buffer::<ObjectiveParams>(1),
            report: state
                .runtime
                .buffer_with(&[ObjectiveSelectionReport::default()]),
        })
    }

    pub(super) fn upload(&self, state: &SearchState, root: u64, ask: Ask) {
        let parameters = ObjectiveParams::pack(state, &self.config, root, ask);
        unsafe {
            self.parameters
                .contents()
                .cast::<ObjectiveParams>()
                .write(parameters);
        }
    }
}

impl SearchState {
    /// Opt in without changing the scalar UCB/Thompson kernel or its ABI.
    /// Passing None restores scalar acquisition without changing observations.
    pub fn configure_objectives(&mut self, config: Option<ObjectivePolicy>) -> Result<(), String> {
        self.check_idle()?;
        self.objective_selector = config
            .map(|config| ObjectiveSelector::new(self, config))
            .transpose()?;
        if let Some(morbo) = self
            .objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
        {
            self.length = morbo.trust().length();
        }
        Ok(())
    }

    pub(super) fn validate_selector(&self) -> Result<(), String> {
        if let Some(selector) = &self.objective_selector {
            parameters::validate_history(self, selector.config.scales.len())?;
        }
        Ok(())
    }

    pub(super) fn objective_decision(
        &self,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<Option<(bool, f64)>, String> {
        let Some(selector) = self.objective_selector.as_ref() else {
            return Ok(None);
        };
        if let Some(morbo) = selector.morbo.as_ref() {
            return morbo
                .candidate_improves(&self.objective_history, candidate)
                .map(Some);
        }
        let before = measured(
            self.objective_history.rows(),
            None,
            &selector.reference,
            &selector.config.scales,
        )?;
        let after = measured(
            self.objective_history.rows(),
            Some(candidate),
            &selector.reference,
            &selector.config.scales,
        )?;
        Ok(Some(improvement(before, after)))
    }

    pub(super) fn update_morbo(
        &mut self,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok(());
        };
        let Some(mut morbo) = selector.morbo.take() else {
            self.objective_selector = Some(selector);
            return Ok(());
        };
        let result = (|| {
            let region = morbo.active;
            let accepted = self.base_id == self.observation;
            morbo.observe(candidate, accepted, self.observation)?;
            if morbo.trust().needs_restart() {
                self.restart_count += 1;
                let source = morbo.restart_source(region)?;
                let source_center = morbo.regions[source]
                    .center
                    .clone()
                    .ok_or("MORBO inactive region lost its resident center")?;
                let source_record = morbo.regions[source].record;
                morbo.regions[region].record = source_record;
                morbo.restart(region, self.restart_count, source_record.observation)?;
                self.copy(&source_center, &self.base)?;
                self.base_id = source_record.identity;
                self.best = source_record.observation.control().mean;
                self.best_variance = source_record.observation.control().variance;
            }
            self.length = morbo.trust().length();
            Ok(())
        })();
        selector.morbo = Some(morbo);
        self.objective_selector = Some(selector);
        result
    }

    pub(super) fn has_morbo(&self) -> bool {
        self.objective_selector
            .as_ref()
            .is_some_and(|selector| selector.morbo.is_some())
    }

    pub(super) fn activate_morbo(&mut self) -> Result<(), String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok(());
        };
        let Some(morbo) = selector.morbo.as_mut() else {
            self.objective_selector = Some(selector);
            return Ok(());
        };
        let region = morbo.select_region();
        let result = self.switch_region(morbo, region).map(|()| {
            let state = &morbo.regions[region];
            self.base_id = state.record.identity;
            self.best = state.record.observation.control().mean;
            self.best_variance = state.record.observation.control().variance;
            self.length = state.trust.length();
        });
        self.objective_selector = Some(selector);
        result
    }

    pub fn morbo_region(&self) -> Option<usize> {
        self.objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
            .map(|morbo| morbo.active)
    }

    pub fn morbo_regions(&self) -> usize {
        self.objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
            .map_or(0, ResidentMorbo::region_count)
    }

    fn switch_region(&mut self, morbo: &mut ResidentMorbo, region: usize) -> Result<(), String> {
        if region == morbo.active {
            return Ok(());
        }
        if morbo.regions[morbo.active].center.is_some() {
            return Err("MORBO active region unexpectedly owned a stored center".into());
        }
        let center = morbo.regions[region]
            .center
            .take()
            .ok_or("MORBO inactive region lost its resident center")?;
        let previous = std::mem::replace(&mut self.base, center);
        morbo.regions[morbo.active].center = Some(previous);
        morbo.active = region;
        Ok(())
    }

    /// Restore the resident Pareto member with the largest exclusive exact
    /// hypervolume contribution for validation and checkpoint export.
    pub fn finalize_morbo(&mut self) -> Result<Option<usize>, String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok(None);
        };
        let Some(morbo) = selector.morbo.as_mut() else {
            self.objective_selector = Some(selector);
            return Ok(None);
        };
        let result = (|| {
            let region = morbo.best_region()?;
            self.switch_region(morbo, region)?;
            let record = morbo.regions[region].record;
            let length = morbo.regions[region].trust.length();
            self.base_id = record.identity;
            self.best = record.observation.control().mean;
            self.best_variance = record.observation.control().variance;
            self.length = length;
            Ok(Some(region))
        })();
        self.objective_selector = Some(selector);
        result
    }

    /// Read back only bounded candidate acquisition vectors, never model weights.
    pub fn objective_report(&self) -> Result<Option<ObjectiveSelectionReport>, String> {
        self.check_healthy()?;
        if self.async_command.is_some() {
            return Err(
                "Finish the pending GPU selection before reading its objective report".into(),
            );
        }
        Ok(self.objective_selector.as_ref().and_then(|selector| {
            let report = read::<ObjectiveSelectionReport>(&selector.report, 1)[0];
            (report.width != 0).then_some(report)
        }))
    }
}

#[cfg(test)]
#[path = "objective_acquisition/tests.rs"]
mod tests;
