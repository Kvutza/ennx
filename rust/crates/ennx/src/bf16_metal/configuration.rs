use super::*;
use crate::params::ENNParams;

impl SearchState {
    /// Supply the measured incumbent before the first proposal or controller update.
    pub(crate) fn observe_initial(&mut self, value: f32, variance: f32) -> Result<(), String> {
        self.observe_objectives(
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub(crate) fn set_tolerance(&mut self, failures: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started {
            return Err("Set Metal BF16 failure tolerance before any asks".into());
        }
        self.trust
            .set_tolerance(failures)
            .map_err(|error| error.to_string())?;
        if let Some(morbo) = self
            .objective_selector
            .as_mut()
            .and_then(|selector| selector.morbo.as_mut())
        {
            morbo.set_tolerance(failures)?;
        }
        Ok(())
    }

    pub(crate) fn configure_enn(
        &mut self,
        config: crate::config::ResidentEnnConfig,
    ) -> Result<(), String> {
        self.check_idle()?;
        check_ask(config.ask)?;
        self.configure_pool(config.pool)?;
        if !self.implicit_history || self.history != 0 {
            return Err("Configure implicit ENN before its first observation".into());
        }
        if config.ask.neighbors > MAX_HISTORY {
            return Err(format!(
                "Implicit ENN k={} exceeds logical history capacity {MAX_HISTORY}",
                config.ask.neighbors
            ));
        }
        let k =
            i32::try_from(config.ask.neighbors).map_err(|_| "Implicit ENN k does not fit i32")?;
        let mut fitter = ENNFitter::new(k, true);
        fitter.set_params(
            ENNParams::new(
                k,
                f64::from(config.ask.epistemic_scale),
                f64::from(config.ask.aleatoric_scale),
            )
            .map_err(|error| error.to_string())?,
        );
        self.initial_observations = config.ask.neighbors;
        self.fit_candidates = config.num_candidates;
        self.fit_samples = config.num_samples;
        self.fit_neighbors = config.fit_neighbors;
        self.fit_seed = config.ask.seed;
        self.distance_scaling = config.distance_scaling;
        self.latent_history = config.history_geometry == crate::config::HistoryGeometry::Latent;
        if self.latent_history && self.perturbation != Perturbation::Rademacher {
            return Err("Latent history geometry requires Rademacher perturbations".into());
        }
        self.local_scale_neighbors = config.local_scale_neighbors;
        self.proposal_method = config.proposal_method;
        self.threshold = match config.proposal_method {
            crate::procedural_pool::ProposalMethod::Independent => None,
            method @ (crate::procedural_pool::ProposalMethod::SpectralBasis
            | crate::procedural_pool::ProposalMethod::PolynomialThreshold) => {
                if self.perturbation != Perturbation::Rademacher || !self.latent_history {
                    return Err("Spectral proposals require Rademacher coordinates and latent history geometry".into());
                }
                Some(threshold::ThresholdSearch::new(
                    f64::from(config.ask.aleatoric_scale.max(1.0e-6)),
                    method == crate::procedural_pool::ProposalMethod::PolynomialThreshold,
                )?)
            }
        };
        self.fitter = Some(fitter);
        Ok(())
    }

    pub(crate) fn configure_controller(
        &mut self,
        config: reliability_region::ReliabilityPolicy,
    ) -> Result<(), String> {
        self.check_idle()?;
        if !self.implicit_history || self.started || self.history != 0 {
            return Err("Configure reliability control before the initial observation".into());
        }
        if self.fitter.is_none() {
            return Err("Configure the resident ENN before reliability control".into());
        }
        self.reliability = Some(reliability_region::ReliabilityController::new(
            config,
            self.length_config,
        )?);
        Ok(())
    }

    pub(crate) fn fitted_enn(&self) -> Option<(usize, f32, f32, f32)> {
        self.fitted_enn
    }

    pub(crate) fn enable_family(&mut self, groups: Vec<usize>) -> Result<(), String> {
        self.check_idle()?;
        if !self.independent_fp16
            || !self.implicit_history
            || self.history != 0
            || self.family.is_some()
        {
            return Err("Configure learned family shape once before initial observation".into());
        }
        let family = FamilyHistory::new(groups, &self.blocks)?;
        let metric = metric::MetricGpu::new(Arc::clone(&self.runtime))?;
        self.resident_bytes = self
            .resident_bytes
            .checked_add(metric.bytes())
            .ok_or("Metric scratch memory accounting overflow")?;
        unsafe {
            for (index, (&group, block)) in family.groups.iter().zip(&family.original).enumerate() {
                self.family_groups
                    .contents()
                    .cast::<u32>()
                    .add(index)
                    .write(group as u32);
                self.family_base_weights
                    .contents()
                    .cast::<f32>()
                    .add(index)
                    .write(block.weight);
            }
        }
        self.family = Some(family);
        self.metric_gpu = Some(metric);
        Ok(())
    }

    pub(crate) fn family_shape(&self) -> Option<([f32; FAMILIES], [f32; FAMILIES])> {
        self.family.as_ref().map(|f| (f.weights, f.scales()))
    }

    pub(super) fn apply_family(&mut self) -> Result<(), String> {
        let Some(family) = &self.family else {
            return Ok(());
        };
        let scales = family.scales();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let g = family.groups[i];
            block.scale = family.original[i].scale * scales[g];
            block.weight = family.original[i].weight * family.weights[g];
        }
        let (_, _, leaves) = make_layout(&self.blocks, self.dimensions)?;
        // No ask is in flight: fitting occurs after completed objective/tell work.
        unsafe {
            std::ptr::copy_nonoverlapping(
                leaves.as_ptr(),
                self.leaves_gpu.contents().cast::<Leaf>(),
                leaves.len(),
            );
        }
        let distances = family.aggregate(family.weights, self.history);
        for i in 0..self.history {
            for j in 0..self.history {
                self.pairwise_distances[i * MAX_HISTORY + j] = distances[[i, j]] as f32;
            }
        }
        Ok(())
    }

    /// Records the seed; the model-sized reference allocation waits until first ask.
    pub fn correlate(&mut self, reference_seed: u64) -> Result<(), String> {
        self.check_idle()?;
        if self.independent_fp16 || self.started || self.reference_seed.is_some() {
            return Err("Enable correlated Metal sampling once before any asks".into());
        }
        self.reference_seed = Some(reference_seed);
        Ok(())
    }

    /// Build the correlated direction before a latency-sensitive ask.
    pub(crate) fn prepare_reference(&mut self) -> Result<(), String> {
        self.check_idle()?;
        self.ensure_ref()
    }

    pub fn enable_relative(&mut self, failure_tolerance: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started
            || self.relative
            || self.history == 0
            || self.reference_seed.is_none()
            || failure_tolerance != 4
            || self.history_rows.len() != 2
        {
            return Err("Metal paired-relative mode requires fresh correlated search and failure_tolerance=4".into());
        }
        self.relative = true;
        self.outcomes.fill(0.0);
        self.variances.fill(0.0);
        Ok(())
    }
}
