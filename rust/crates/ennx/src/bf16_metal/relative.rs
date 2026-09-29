use super::*;

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    /// Inconclusive rejections still update history. Set `reject_is_failure` only
    /// when the external screening rule counts a rejection toward contraction.
    pub fn tell_relative(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        autoreleasepool(|| {
            self.tell_inner(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn tell_inner(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        self.check_round(round)?;
        if !self.relative {
            return Err("Enable legacy paired-relative mode before relative tells".into());
        }
        check_scores(
            &[value, incumbent_value, improvement],
            &[variance, incumbent_variance, improvement_variance],
        )?;
        let selected = self.pending.as_ref().unwrap();
        let radius = selected.length;
        let command = self.runtime.queue.new_command_buffer();
        if accept {
            self.encode_ref(
                command,
                Params {
                    seed: selected.seed,
                    candidate: selected.index as u32,
                    ..Params::default()
                },
            );
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.proposal, 0, &self.base, 0, self.row_bytes());
            blit.copy_from_buffer(&self.proposal, 0, &self.anchor, 0, self.row_bytes());
            blit.end_encoding();
            if let Err(error) = finish(command) {
                self.poisoned = true;
                return Err(error);
            }
            if let Err(error) = self.check_reference() {
                self.poisoned = true;
                return Err(error);
            }
            self.best = value;
            self.best_variance = variance;
            self.outcomes.fill(0.0);
            self.variances.fill(0.0);
            self.history = 1;
            self.length = f64::from(radius);
            self.failures = 0;
        } else {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.proposal, 0, &self.rejected, 0, self.row_bytes());
            blit.end_encoding();
            if let Err(error) = finish(command) {
                self.poisoned = true;
                return Err(error);
            }
            self.best = incumbent_value;
            self.best_variance = incumbent_variance;
            self.outcomes[1] = improvement;
            self.variances[1] = improvement_variance;
            self.history = 2;
            if reject_is_failure {
                self.failures += 1;
                if self.failures == 4 {
                    self.length = (self.length * 0.5).max(self.length_config.length_min);
                    self.failures = 0;
                }
            }
        }
        self.pending = None;
        self.queued = Some(accept);
        Ok(())
    }
}
