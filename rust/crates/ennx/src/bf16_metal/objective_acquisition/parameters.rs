use super::*;

#[repr(C)]
pub(super) struct ObjectiveParams {
    outcomes: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    variances: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    draws: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    scales: [f32; MAX_OBJECTIVES],
    weights: [f32; MAX_OBJECTIVES],
    minima: [f32; MAX_OBJECTIVES],
    maxima: [f32; MAX_OBJECTIVES],
    priorities: [u64; 4],
    width: u32,
    mode: u32,
    alpha: f32,
    clip: u32,
}

pub(super) fn validate_history(state: &SearchState, width: usize) -> Result<(), String> {
    if state.relative || state.history == 0 {
        return Err("Vector acquisition requires measured absolute objective observations".into());
    }
    let records = state.objective_history.rows();
    if records.len() != state.history {
        return Err("Vector acquisition history is missing an objective row".into());
    }
    for (identity, record) in state.identities[..state.history].iter().zip(records) {
        if record.identity != *identity {
            return Err("Vector acquisition history order differs from scalar geometry".into());
        }
        if record.observation.estimates().len() != width {
            return Err("Vector acquisition scales do not match the fixed objective schema".into());
        }
        record.observation.validate()?;
    }
    Ok(())
}

impl ObjectiveParams {
    pub(super) fn pack(state: &SearchState, config: &ObjectivePolicy, root: u64, ask: Ask) -> Self {
        let mut params = Self {
            outcomes: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            variances: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            draws: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            scales: [1.0; MAX_OBJECTIVES],
            weights: [0.0; MAX_OBJECTIVES],
            minima: [f32::INFINITY; MAX_OBJECTIVES],
            maxima: [f32::NEG_INFINITY; MAX_OBJECTIVES],
            priorities: std::array::from_fn(|candidate| {
                crate::hash::splitmix64(
                    root ^ (candidate as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                )
            }),
            width: config.scales.len() as u32,
            mode: 0,
            alpha: 0.0,
            clip: 0,
        };
        params.scales[..config.scales.len()].copy_from_slice(&config.scales);
        params.fill_observations(state, ask.seed);
        params.configure_mode(state, config);
        params
    }

    fn fill_observations(&mut self, state: &SearchState, seed: u64) {
        for (row, record) in state.objective_history.rows().enumerate() {
            for (metric, estimate) in record.observation.estimates().iter().enumerate() {
                self.outcomes[metric][row] = estimate.mean;
                self.variances[metric][row] = estimate.variance;
                self.draws[metric][row] =
                    crate::hash::normal_metric(seed, record.identity, metric) as f32;
                self.minima[metric] = self.minima[metric].min(estimate.mean);
                self.maxima[metric] = self.maxima[metric].max(estimate.mean);
            }
        }
    }

    fn configure_mode(&mut self, state: &SearchState, config: &ObjectivePolicy) {
        let ObjectiveAcquisition::Morbo { alpha, .. } = &config.acquisition else {
            return;
        };
        let morbo = state
            .objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
            .expect("validated MORBO selector requires its trust region");
        let trust = morbo.trust();
        self.mode = 1;
        self.alpha = *alpha;
        // MORBO acquisition extrapolates beyond observed ranges. Incumbent
        // updates are clipped separately by MorboTrustRegion.
        self.clip = 0;
        for (metric, &weight) in trust.weights().iter().enumerate() {
            self.weights[metric] = weight as f32;
        }
        if let (Some(minima), Some(maxima)) = (trust.y_min(), trust.y_max()) {
            for metric in 0..trust.num_metrics() {
                self.minima[metric] = minima[metric] as f32;
                self.maxima[metric] = maxima[metric] as f32;
            }
        }
    }
}
