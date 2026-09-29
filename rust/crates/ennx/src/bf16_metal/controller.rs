use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

impl SearchState {
    pub(super) fn update_reliability(
        &mut self,
        value: f32,
        mut evidence: reliability_region::ReliabilityEvidence,
    ) -> Result<(), String> {
        self.observed.push(f64::from(value));
        evidence.center_radius = self.center_radius();
        let controller = self
            .reliability
            .as_mut()
            .ok_or("Reliability controller is not configured")?;
        controller.update(evidence)?;
        self.length = controller.length();
        Ok(())
    }

    pub(super) fn fit_enn(&mut self) -> Result<(), String> {
        let n = self.history;
        let fit_family = n == self.initial_observations || (n - self.initial_observations) % 4 == 0;
        if fit_family {
            if let Some(family) = &mut self.family {
                let outcomes =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.outcomes[i]));
                let variances =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.variances[i]));
                let params = *self
                    .fitter
                    .as_ref()
                    .and_then(|f| f.params())
                    .ok_or("Family fit needs ENN parameters")?;
                let candidates = family.candidates();
                // Paired warm measurements cross over at H=64 on this Apple GPU.
                // A tiny history must not pay more launch latency than CPU work.
                let local = (self.distance_scaling == crate::config::DistanceScaling::SelfTuning)
                    .then_some(self.local_scale_neighbors);
                let scores = if n < 64 {
                    family.reference(
                        &candidates,
                        &outcomes.view(),
                        &variances.view(),
                        params,
                        self.fit_samples,
                        self.fit_seed.wrapping_add(n as u64),
                        local,
                    )?
                } else {
                    self.metric_gpu
                        .as_mut()
                        .ok_or("Family metric GPU is missing")?
                        .score(
                            family,
                            &candidates,
                            &outcomes.view(),
                            &variances.view(),
                            params,
                            self.fit_samples,
                            self.fit_seed.wrapping_add(n as u64),
                            local,
                        )?
                };
                family.select(&candidates, &scores, self.fit_samples.min(n))?;
                self.apply_family()?;
            }
        }
        let raw_distances = ndarray::Array2::from_shape_fn((n, n), |(row, column)| {
            f64::from(self.total_distance(row, column))
        });
        let distances = if self.distance_scaling == crate::config::DistanceScaling::SelfTuning {
            crate::fit::tuned_distances(&raw_distances.view(), self.local_scale_neighbors)
                .map_err(|error| error.to_string())?
        } else {
            raw_distances
        };
        let outcomes =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.outcomes[row]));
        let variances =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.variances[row]));
        let mut rng = StdRng::seed_from_u64(self.fit_seed.wrapping_add(n as u64));
        let fitter = self
            .fitter
            .as_mut()
            .ok_or("Metal implicit ENN is not configured")?;
        let params = if self.fit_neighbors {
            fitter.ask_adaptive(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                i32::try_from(self.initial_observations)
                    .map_err(|_| "Initial ENN neighbor bound does not fit i32")?,
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        } else {
            fitter.ask_distances(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        }
        .map_err(|error| error.to_string())?;
        let y_scale = fitter.y_std()[0];
        self.fitted_enn = Some((
            usize::try_from(params.k_neighbors)
                .map_err(|_| "Fitted ENN neighbor count does not fit usize")?,
            params.epistemic_scale as f32,
            params.aleatoric_scale as f32,
            y_scale as f32,
        ));
        Ok(())
    }

    pub(super) fn update_region(&mut self, value: f32) -> Result<(), String> {
        self.observed.push(f64::from(value));
        self.trust
            .update(
                &ndarray::ArrayView1::from(&self.observed),
                self.observed.len(),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            self.trust
                .update(&ndarray::ArrayView1::from(&self.observed), 1)
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    pub(super) fn update_noisy(&mut self, value: f32, incumbent: f32) -> Result<(), String> {
        self.observed.push(f64::from(value));
        self.trust
            .update_history(
                &ndarray::ArrayView1::from(&self.observed),
                self.observed.len(),
                f64::from(incumbent),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            self.trust
                .update_history(
                    &ndarray::ArrayView1::from(&self.observed),
                    1,
                    f64::from(self.best),
                )
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    pub(super) fn update_outcome(
        &mut self,
        value: f32,
        incumbent: f32,
        outcome: TrustRegionOutcome,
    ) -> Result<(), String> {
        self.observed.push(f64::from(value));
        let y_new = [f64::from(value)];
        self.trust
            .update_outcome(
                &ndarray::ArrayView1::from(&y_new),
                self.observed.len(),
                f64::from(incumbent),
                outcome,
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            let seed = [f64::from(self.best)];
            self.trust
                .update_outcome(
                    &ndarray::ArrayView1::from(&seed),
                    1,
                    f64::from(self.best),
                    TrustRegionOutcome::Inconclusive,
                )
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }
}
