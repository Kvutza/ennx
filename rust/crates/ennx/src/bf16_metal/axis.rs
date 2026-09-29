use super::*;

pub(super) struct SearchAxis {
    pub(super) lower: f32,
    pub(super) span: f32,
    pub(super) base: f32,
    pub(super) history: [f32; MAX_HISTORY],
    pub(super) weight: f32,
    pub(super) step: f32,
}

impl SearchAxis {
    pub(super) fn value(&self, coordinate: f32) -> f32 {
        self.lower + coordinate * self.span
    }
}

impl SearchState {
    /// Add one bounded scalar policy block to the ENN geometry. Its metric mass
    /// matches the complete normalized weight block, rather than vanishing as
    /// one coordinate among billions.
    pub fn configure_axis(
        &mut self,
        initial: f32,
        bounds: [f32; 2],
        step: f32,
    ) -> Result<(), String> {
        self.check_idle()?;
        let [lower, upper] = bounds;
        if self.started
            || self.history != 0
            || self.axis.is_some()
            || !initial.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
            || lower >= upper
            || !(lower..=upper).contains(&initial)
            || !step.is_finite()
            || step <= 0.0
            || step > upper - lower
        {
            return Err(
                "Configure one finite bounded search axis before the initial observation".into(),
            );
        }
        let coordinate = (initial - lower) / (upper - lower);
        let normalized_step = step / (upper - lower);
        let mut history = [0.0; MAX_HISTORY];
        history[0] = coordinate;
        self.axis = Some(SearchAxis {
            lower,
            span: upper - lower,
            base: coordinate,
            history,
            weight: self.blocks.len() as f32
                * (self.length_config.length_init as f32 / normalized_step).powi(2),
            step: normalized_step,
        });
        Ok(())
    }

    pub(super) fn axis_candidate(&self, root: u64, candidate: usize) -> Option<f32> {
        let axis = self.axis.as_ref()?;
        let pair = candidate / 2;
        let seed = crate::hash::splitmix64(
            root ^ (pair as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x7465_6d70_2d61_7869,
        );
        let magnitude = match self.perturbation {
            Perturbation::Rademacher => 1.0,
            Perturbation::Gaussian => crate::hash::normal_metric(seed, pair as i64, 0).abs() as f32,
        };
        let first_positive = crate::hash::splitmix64(seed ^ 0x7369_676e) & 1 == 0;
        let positive = if candidate & 1 == 0 {
            first_positive
        } else {
            !first_positive
        };
        let direction = if positive { magnitude } else { -magnitude };
        let trust_ratio = self.radius(candidate) / self.length_config.length_init as f32;
        Some((axis.base + axis.step * trust_ratio * direction).clamp(0.0, 1.0))
    }

    pub(super) fn axis_distance(&self, left: f32, right: f32) -> f32 {
        self.axis
            .as_ref()
            .map_or(0.0, |axis| (left - right).powi(2) * axis.weight)
    }

    pub(super) fn total_distance(&self, row: usize, column: usize) -> f32 {
        let tensor = self.pairwise_distances[row * MAX_HISTORY + column];
        self.axis.as_ref().map_or(tensor, |axis| {
            tensor + self.axis_distance(axis.history[row], axis.history[column])
        })
    }

    pub(super) fn record_axis(&mut self, round: &Proposals, slot: usize, accept: bool) {
        let Some(axis) = &mut self.axis else {
            return;
        };
        let coordinate = round.axis_coordinate.unwrap();
        if self.implicit_history {
            axis.history[self.history] = coordinate;
        } else {
            axis.history[slot] = coordinate;
            if self.history == self.history_rows.len() {
                axis.history[..self.history].rotate_left(1);
            }
        }
        if accept {
            axis.base = coordinate;
        }
    }

    pub(super) fn retain_axis(&mut self) {
        if let Some(axis) = &mut self.axis {
            axis.history.fill(0.0);
            axis.history[0] = axis.base;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_axis() -> Result<(), String> {
        autoreleasepool(|| {
            let base = vec![0x3400; 257];
            let length = TRLengthConfig::new(0.001, 0.0001, 0.1);
            let mut search = SearchState::new_fp16(
                &base,
                vec![ParamBlock::new(7, 0, base.len(), 0.25, 1.0)?],
                2,
                length,
                Perturbation::Rademacher,
            )?;
            search.configure_axis(0.6457, [0.0001, 0.9997], 0.2)?;
            let axis = search.axis.as_ref().unwrap();
            let expected = search.blocks.len() as f32 * length.length_init.powi(2) as f32;
            let actual = axis.weight * axis.step.powi(2);
            assert!((actual - expected).abs() < 1.0e-10);
            search.observe_initial(0.0, 0.0)?;
            let proposal = search.ask_round(1, 4, 17, Ask::default())?;
            assert!(
                proposal
                    .axis_value()
                    .is_some_and(|value| { (0.0001..=0.9997).contains(&value) && value != 0.6457 })
            );
            Ok(())
        })
    }
}
