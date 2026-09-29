use super::*;

impl SearchState {
    /// Keep absolute measurements across incumbent changes. The paired decision
    /// controls weight acceptance; radius adaptation uses the shared CPU controller.
    pub fn tell_paired(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
    ) -> Result<(), String> {
        autoreleasepool(|| {
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accept,
                None,
                None,
                None,
                true,
            )
        })
    }

    pub(crate) fn tell_initial(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.initial_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub fn initial_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            if !self.implicit_history
                || self.initial_observations == 0
                || self.history >= self.initial_observations
            {
                return Err("Metal ENN initialization is complete or not configured".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let rewarming = self.observation as usize > self.history;
            let control_threshold = rewarming
                .then(|| 2.0 * (f64::from(variance) + f64::from(incumbent_variance)).sqrt())
                .unwrap_or(0.0);
            let (accepted, improvement) = self
                .objective_decision(observation)?
                .unwrap_or((control_improvement > control_threshold, control_improvement));
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                None,
                None,
                None,
                false,
            )?;
            if self.history == self.initial_observations && !self.has_morbo() {
                if let Some(controller) = &self.reliability {
                    self.length = controller.length();
                } else {
                    let failure_tolerance = self.trust.failure_tolerance() as usize;
                    self.trust = TurboTrustRegion::new(self.dimensions, self.length_config);
                    self.trust.set_arms(1);
                    self.trust
                        .set_tolerance(failure_tolerance)
                        .map_err(|error| error.to_string())?;
                    self.trust
                        .update(
                            &ndarray::ArrayView1::from(&self.observed),
                            self.observed.len(),
                        )
                        .map_err(|error| error.to_string())?;
                    self.length = self.trust.length();
                }
            }
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold: control_threshold,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome: TrustRegionOutcome::Inconclusive,
            })
        })
    }

    /// Add one independent noisy observation. The candidate becomes the new
    /// incumbent only when its reward exceeds the stored incumbent by two
    /// combined standard errors.
    pub fn tell_noisy(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.noisy_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub fn noisy_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let combined_variance = f64::from(variance) + f64::from(incumbent_variance);
            if !combined_variance.is_finite() {
                return Err("Combined noisy-observation variance is not finite".into());
            }
            let control_threshold = 2.0 * combined_variance.sqrt();
            let objectives = self.objective_decision(observation)?;
            let (accepted, improvement, threshold) = objectives.map_or(
                (
                    control_improvement > control_threshold,
                    control_improvement,
                    control_threshold,
                ),
                |(accepted, improvement)| (accepted, improvement, 0.0),
            );
            let next_incumbent = if accepted { value } else { incumbent_value };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                None,
                None,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome: if accepted {
                    TrustRegionOutcome::Success
                } else {
                    TrustRegionOutcome::Failure
                },
            })
        })
    }

    /// Add one stochastic-objective observation using the local ENN posterior
    /// for the incumbent baseline and trust-region model agreement.
    pub fn tell_modeled(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.modeled_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub fn modeled_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            check_scores(
                &[value, round.predicted_mean, round.incumbent_mean],
                &[
                    variance,
                    round.predicted_standard_error.powi(2),
                    round.incumbent_standard_error.powi(2),
                ],
            )?;
            let incumbent_value = round.incumbent_mean;
            let incumbent_variance = round.incumbent_standard_error.powi(2);
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let control_threshold =
                2.0 * (f64::from(variance) + f64::from(incumbent_variance)).sqrt();
            let objectives = self.objective_decision(observation)?;
            let (accepted, improvement, threshold) = objectives.map_or(
                (
                    control_improvement > control_threshold,
                    control_improvement,
                    control_threshold,
                ),
                |(accepted, improvement)| (accepted, improvement, 0.0),
            );
            let control_prediction = f64::from(round.predicted_mean) - f64::from(incumbent_value);
            let predicted_improvement = (!self.has_morbo()).then_some(control_prediction);
            let agreement_ratio = predicted_improvement
                .filter(|prediction| *prediction > f64::EPSILON)
                .map(|prediction| improvement / prediction);
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            let reliability = if self.has_morbo() {
                None
            } else {
                self.reliability_evidence(
                    round,
                    value,
                    variance,
                    improvement,
                    f64::from(variance) + f64::from(incumbent_variance),
                )?
            };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                Some(trust_outcome),
                reliability,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement,
                agreement_ratio,
                trust_outcome,
            })
        })
    }

    /// Add an absolute cumulative outcome measured through a paired
    /// candidate/incumbent comparison on the same stochastic objective.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paired_modeled(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        paired_improvement: f32,
        improvement_variance: f32,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            self.check_round(round)?;
            check_scores(
                &[
                    value,
                    incumbent_value,
                    round.predicted_mean,
                    round.incumbent_mean,
                ],
                &[
                    variance,
                    incumbent_variance,
                    round.predicted_standard_error.powi(2),
                    round.incumbent_standard_error.powi(2),
                ],
            )?;
            if !paired_improvement.is_finite()
                || !improvement_variance.is_finite()
                || improvement_variance < 0.0
            {
                return Err(
                    "Paired improvement must be finite and variance must be finite and nonnegative"
                        .into(),
                );
            }
            let improvement = f64::from(paired_improvement);
            let threshold = 2.0 * f64::from(improvement_variance).sqrt();
            let accepted = improvement > threshold;
            let predicted_improvement =
                f64::from(round.predicted_mean) - f64::from(round.incumbent_mean);
            let agreement_ratio = (predicted_improvement > f64::EPSILON)
                .then_some(improvement / predicted_improvement);
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let reliability = self.reliability_evidence(
                round,
                value,
                variance,
                improvement,
                f64::from(improvement_variance),
            )?;
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(incumbent_value),
                Some(trust_outcome),
                reliability,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: Some(predicted_improvement),
                agreement_ratio,
                trust_outcome,
            })
        })
    }
}
