use super::*;
use crate::apple_gpu::gpu_interval;

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tell_absolute(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        self.absolute_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
            incumbent_value,
            incumbent_variance,
            accept,
            trust_incumbent,
            trust_outcome,
            reliability_evidence,
            adapt_region,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn absolute_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        self.tell_profile = None;
        self.check_round(round)?;
        if self.relative {
            return Err("Cannot mix absolute and paired-relative observations".into());
        }
        check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
        self.objective_history.validate(observation)?;
        let identity = self
            .observation
            .checked_add(1)
            .ok_or("Observation ID overflow")?;
        if self.implicit_history && self.history == MAX_HISTORY {
            return Err("Metal implicit history reached its 128-observation capacity".into());
        }
        let implicit_distances = self.observation_distances(round)?;
        let slot = if self.resident_history == self.history_rows.len() {
            0
        } else {
            self.resident_history
        };
        let command = self.runtime.queue.new_command_buffer().to_owned();
        let zero_copy = self.commit_weights(round, accept, &command)?;
        self.archive_proposal(slot, zero_copy);
        if self.profiling {
            let reference_ms = if accept {
                gpu_interval(&command)
                    .map(|(start, end)| ((end - start) * 1000.0) as f32)
                    .ok_or_else(|| "Metal did not expose a tell GPU interval".to_string())?
            } else {
                0.0
            };
            self.tell_profile = Some(TellProfile {
                reference_ms,
                history_copy_ms: 0.0,
                total_ms: reference_ms,
            });
        }
        self.observation = identity;
        if accept {
            self.base_id = identity;
        }
        self.resident_identities[slot] = identity;
        if self.resident_history == self.history_rows.len() {
            self.history_rows.rotate_left(1);
            self.resident_identities.rotate_left(1);
        } else {
            self.resident_history += 1;
        }
        self.record_axis(round, slot, accept);
        self.record_observation(
            round,
            identity,
            slot,
            (value, variance),
            observation,
            implicit_distances,
            accept,
        )?;
        if let (Some(threshold), Some(table)) = (&mut self.threshold, &round.threshold_table) {
            threshold.observe(
                identity,
                table,
                round.length,
                round.direction_norm,
                f64::from(value) - f64::from(incumbent_value),
                f64::from(variance) + f64::from(incumbent_variance),
            )?;
        }
        self.best = if accept { value } else { incumbent_value };
        self.best_variance = if accept { variance } else { incumbent_variance };
        let update = self.adapt_observation(
            observation,
            value,
            trust_incumbent,
            trust_outcome,
            reliability_evidence,
            adapt_region,
        );
        if let Err(error) = update {
            self.poisoned = true;
            return Err(error);
        }
        self.pending = None;
        self.queued = Some(accept);
        Ok(())
    }

    fn commit_weights(
        &mut self,
        round: &Proposals,
        accept: bool,
        command: &CommandBufferRef,
    ) -> Result<bool, String> {
        if accept {
            if !self.independent_fp16 {
                self.encode_ref(
                    &command,
                    Params {
                        seed: round.seed,
                        candidate: round.index as u32,
                        ..Params::default()
                    },
                );
            }
            if self.exact_history {
                let blit = command.new_blit_command_encoder();
                blit.copy_from_buffer(&self.proposal, 0, &self.base, 0, self.row_bytes());
                blit.end_encoding();
            }
            if let Err(error) = finish(&command) {
                self.poisoned = true;
                return Err(error);
            }
            if !self.exact_history {
                std::mem::swap(&mut self.base, &mut self.proposal);
            }
            if let Err(error) = self.check_reference() {
                self.poisoned = true;
                return Err(error);
            }
        }
        Ok(accept && !self.exact_history)
    }

    fn archive_proposal(&mut self, slot: usize, zero_copy: bool) {
        if zero_copy {
            // `base` now owns the accepted candidate. Archive that allocation
            // and recycle only the FIFO allocation being replaced as proposal
            // scratch. The previous incumbent may still be referenced by an
            // older history row, so writing through `proposal` would corrupt
            // replay history after the next ask.
            self.proposal = std::mem::replace(&mut self.history_rows[slot], self.base.clone());
            if std::ptr::eq::<metal::BufferRef>(&*self.proposal, &*self.base) {
                self.proposal = self.runtime.buffer::<u16>(self.dimensions);
            }
            return;
        }
        std::mem::swap(&mut self.proposal, &mut self.history_rows[slot]);
        if std::ptr::eq::<metal::BufferRef>(&*self.proposal, &*self.base) {
            // The overwritten FIFO slot was the current incumbent's shared
            // history view. Do not reuse that allocation as proposal scratch.
            self.proposal = self.runtime.buffer::<u16>(self.dimensions);
        }
    }

    fn observation_distances(
        &self,
        round: &Proposals,
    ) -> Result<Option<Vec<(usize, f32)>>, String> {
        let distances = if self.implicit_history {
            if round.history_distances.len() != self.history {
                return Err("Metal implicit history distance row has the wrong length".into());
            }
            Some(
                round
                    .history_distances
                    .iter()
                    .map(|&(previous_identity, distance)| {
                        self.identities[..self.history]
                            .iter()
                            .position(|&candidate| candidate == previous_identity)
                            .map(|previous| {
                                let tensor_distance = if let (Some(axis), Some(coordinate)) =
                                    (&self.axis, round.axis_coordinate)
                                {
                                    (distance
                                        - self.axis_distance(coordinate, axis.history[previous]))
                                    .max(0.0)
                                } else {
                                    distance
                                };
                                (previous, tensor_distance)
                            })
                            .ok_or_else(|| {
                                "Metal implicit history distance has an unknown identity"
                                    .to_string()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(distances)
    }
    fn record_observation(
        &mut self,
        round: &Proposals,
        identity: i64,
        slot: usize,
        (value, variance): (f32, f32),
        observation: crate::objective_observation::ObjectiveObservation,
        distances: Option<Vec<(usize, f32)>>,
        accept: bool,
    ) -> Result<(), String> {
        if self.implicit_history {
            let next = self.history;
            if self.exact_history {
                if round.block_scales.len() != self.blocks.len() {
                    return Err("Exact replay proposal has the wrong block-scale count".into());
                }
                let step = ReplayStep {
                    seed: round.seed,
                    radius: round.length,
                    accepted: u32::from(accept),
                    candidate: round.index as u32,
                    mode: self.perturbation.shader(),
                };
                // Shared Metal buffers are CPU-visible and no command is using
                // these future-history slots while tell owns the state.
                unsafe {
                    self.replay_steps
                        .contents()
                        .cast::<ReplayStep>()
                        .add(next)
                        .write(step);
                    std::ptr::copy_nonoverlapping(
                        round.block_scales.as_ptr(),
                        self.replay_scales
                            .contents()
                            .cast::<f32>()
                            .add(next * self.blocks.len()),
                        self.blocks.len(),
                    );
                }
            }
            self.outcomes[next] = value;
            self.variances[next] = variance;
            self.identities[next] = identity;
            for &(previous, distance) in distances.as_ref().unwrap() {
                self.pairwise_distances[next * MAX_HISTORY + previous] = distance;
                self.pairwise_distances[previous * MAX_HISTORY + next] = distance;
            }
            self.history += 1;
            if let Some(family) = &mut self.family {
                let components = round
                    .family_distances
                    .as_ref()
                    .ok_or("Missing family components")?;
                for (previous, &values) in components.iter().enumerate() {
                    family.components[next * MAX_HISTORY + previous] = values;
                    family.components[previous * MAX_HISTORY + next] = values;
                }
            }
            self.objective_history.record(identity, observation, accept);
            if self.history >= self.initial_observations {
                self.fit_enn()?;
            }
        } else {
            self.outcomes[slot] = value;
            self.variances[slot] = variance;
            self.identities[slot] = identity;
            self.objective_history.record(identity, observation, accept);
            if self.history == self.history_rows.len() {
                self.outcomes.rotate_left(1);
                self.variances.rotate_left(1);
                self.identities.rotate_left(1);
            } else {
                self.history += 1;
            }
        }
        Ok(())
    }
    fn adapt_observation(
        &mut self,
        observation: crate::objective_observation::ObjectiveObservation,
        value: f32,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let update = if self.has_morbo() {
            self.update_morbo(observation)
        } else if adapt_region && self.reliability.is_some() {
            reliability_evidence
                .ok_or_else(|| "Reliability control requires a model-aware tell".to_string())
                .and_then(|evidence| self.update_reliability(value, evidence))
        } else if adapt_region {
            match (trust_incumbent, trust_outcome) {
                (Some(incumbent), Some(outcome)) => self.update_outcome(value, incumbent, outcome),
                (Some(incumbent), None) => self.update_noisy(value, incumbent),
                (None, None) => self.update_region(value),
                (None, Some(_)) => Err("Trust outcome requires an incumbent".into()),
            }
        } else {
            self.observed.push(f64::from(value));
            Ok(())
        };
        update
    }
}
