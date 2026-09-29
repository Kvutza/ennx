//! Bounded outcome vectors, separate from the scalar decision/control policy.
//!
//! Objective positions belong to the caller's fixed schema. No normalization,
//! scalarization, covariance estimate, or preference is inferred here.

/// Compile-time bound on independently measured objective channels.
pub const MAX_OBJECTIVES: usize = 8;
/// The resident optimizer's bounded observation window, not model parameters.
pub const OBJECTIVE_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectiveEstimate {
    pub mean: f32,
    pub variance: f32,
}

impl ObjectiveEstimate {
    pub fn validate(self) -> Result<(), String> {
        if !self.mean.is_finite() || !self.variance.is_finite() || self.variance < 0.0 {
            return Err(
                "Objective means must be finite and variances finite and nonnegative".into(),
            );
        }
        Ok(())
    }
}

/// A measured vector and an explicitly supplied scalar control observation.
/// The control is not computed from the vector by this representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveObservation {
    estimates: [ObjectiveEstimate; MAX_OBJECTIVES],
    width: u8,
    control: ObjectiveEstimate,
}

impl ObjectiveObservation {
    pub fn new(
        estimates: &[ObjectiveEstimate],
        control: ObjectiveEstimate,
    ) -> Result<Self, String> {
        if estimates.is_empty() || estimates.len() > MAX_OBJECTIVES {
            return Err(format!("Objective count must be in 1..={MAX_OBJECTIVES}"));
        }
        let mut observation = Self {
            estimates: [ObjectiveEstimate::default(); MAX_OBJECTIVES],
            width: estimates.len() as u8,
            control,
        };
        observation.estimates[..estimates.len()].copy_from_slice(estimates);
        observation.validate()?;
        Ok(observation)
    }

    pub fn scalar(mean: f32, variance: f32) -> Result<Self, String> {
        let observation = Self::scalar_adapter(mean, variance);
        observation.validate()?;
        Ok(observation)
    }

    /// Internal adapter defers validation to the existing tell boundary, so
    /// invalid scalar calls retain their established state/error ordering.
    pub(crate) fn scalar_adapter(mean: f32, variance: f32) -> Self {
        let control = ObjectiveEstimate { mean, variance };
        let mut estimates = [ObjectiveEstimate::default(); MAX_OBJECTIVES];
        estimates[0] = control;
        Self {
            estimates,
            width: 1,
            control,
        }
    }

    pub fn estimates(&self) -> &[ObjectiveEstimate] {
        &self.estimates[..usize::from(self.width)]
    }

    pub(crate) fn is_vector(&self) -> bool {
        self.width > 1
    }

    pub fn control(self) -> ObjectiveEstimate {
        self.control
    }

    pub fn validate(self) -> Result<(), String> {
        self.control.validate()?;
        for estimate in self.estimates() {
            estimate.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveRecord {
    pub identity: i64,
    pub observation: ObjectiveObservation,
}

/// Fixed storage owned by the resident search. Recording and borrowing rows
/// allocate nothing; FIFO retention never retains dense parameter vectors.
pub(crate) struct ObjectiveWindow {
    rows: [Option<ObjectiveRecord>; OBJECTIVE_CAPACITY],
    len: usize,
    capacity: usize,
    width: usize,
    incumbent: Option<ObjectiveRecord>,
}

impl ObjectiveWindow {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!((1..=OBJECTIVE_CAPACITY).contains(&capacity));
        Self {
            rows: [None; OBJECTIVE_CAPACITY],
            len: 0,
            capacity,
            width: 0,
            incumbent: None,
        }
    }

    pub(crate) fn validate(&self, observation: ObjectiveObservation) -> Result<(), String> {
        observation.validate()?;
        if self.width != 0 && self.width != observation.estimates().len() {
            return Err("Objective schema width cannot change within a resident search".into());
        }
        Ok(())
    }

    pub(crate) fn record(
        &mut self,
        identity: i64,
        observation: ObjectiveObservation,
        accepted: bool,
    ) {
        let record = ObjectiveRecord {
            identity,
            observation,
        };
        self.width = observation.estimates().len();
        if self.len == self.capacity {
            self.rows[..self.capacity].rotate_left(1);
        } else {
            self.len += 1;
        }
        self.rows[self.len - 1] = Some(record);
        if accepted {
            self.incumbent = Some(record);
        }
    }

    pub(crate) fn rows(&self) -> impl ExactSizeIterator<Item = &ObjectiveRecord> {
        self.rows[..self.len]
            .iter()
            .map(|row| row.as_ref().unwrap())
    }

    pub(crate) fn incumbent(&self) -> Option<&ObjectiveRecord> {
        self.incumbent.as_ref()
    }

    pub(crate) fn retain_incumbent(&mut self) {
        self.rows.fill(None);
        self.rows[0] = self.incumbent;
        self.len = usize::from(self.incumbent.is_some());
    }
}

#[cfg(test)]
#[path = "objective_observation/tests.rs"]
mod tests;
