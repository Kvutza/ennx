use super::*;
use crate::objective_observation::ObjectiveRecord;

impl SearchState {
    /// Initialize independent objective measurements while retaining an
    /// explicitly supplied scalar control policy; vector selection is opt-in.
    pub fn new_objectives(
        base: &[u16],
        observation: crate::objective_observation::ObjectiveObservation,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            observation.validate()?;
            let mut state = Self::new_inner(
                base,
                None,
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                Perturbation::Gaussian,
            )?;
            state.observe_objectives(observation)?;
            Ok(state)
        })
    }
    pub fn observe_objectives(
        &mut self,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        self.check_idle()?;
        if self.started || self.history != 0 {
            return Err("Initial Metal BF16 observation is already set".into());
        }
        check_scores(&[value], &[variance])?;
        self.objective_history.validate(observation)?;
        self.observed.push(f64::from(value));
        self.trust
            .update(&ndarray::ArrayView1::from(&self.observed), 1)
            .map_err(|error| error.to_string())?;
        self.outcomes[0] = value;
        self.variances[0] = variance;
        self.best = value;
        self.best_variance = variance;
        self.history = 1;
        self.resident_history = 1;
        self.resident_identities[0] = self.base_id;
        self.objective_history
            .record(self.base_id, observation, true);
        Ok(())
    }

    /// Borrow raw, independently measured objectives in retained observation order.
    /// No scalarization, posterior fitting, or candidate selection is performed.
    pub fn objective_observations(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = &ObjectiveRecord>, String> {
        self.check_healthy()?;
        if self.relative {
            return Err(
                "Paired-relative histories do not contain absolute objective vectors".into(),
            );
        }
        Ok(self.objective_history.rows())
    }

    pub fn incumbent_objectives(&self) -> Result<Option<&ObjectiveRecord>, String> {
        self.check_healthy()?;
        if self.relative {
            return Err(
                "Paired-relative histories do not contain absolute objective vectors".into(),
            );
        }
        Ok(self.objective_history.incumbent())
    }
}
