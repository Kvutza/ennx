use super::*;

impl SearchState {
    pub(super) fn history_scales(&self) -> Vec<f64> {
        (0..self.history)
            .map(|row| {
                let mut values = (0..self.history)
                    .filter(|&column| column != row)
                    .map(|column| f64::from(self.pairwise_distances[row * MAX_HISTORY + column]))
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let k = self.local_scale_neighbors.min(values.len());
                values.get(k.saturating_sub(1)).copied().unwrap_or(1.0)
            })
            .collect()
    }

    pub(super) fn ranked_neighbors(&self, round: &Proposals) -> Vec<usize> {
        let scales = (self.distance_scaling == crate::config::DistanceScaling::SelfTuning)
            .then(|| self.history_scales());
        let mut ranked = round
            .history_distances
            .iter()
            .enumerate()
            .map(|(index, &(_, distance))| {
                let scaled = scales.as_ref().map_or(f64::from(distance), |values| {
                    f64::from(distance) / values[index].max(1.0e-12).sqrt()
                });
                (index, scaled)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
        ranked
            .into_iter()
            .take(self.initial_observations.min(self.history))
            .map(|(index, _)| index)
            .collect()
    }

    pub(super) fn rank_concordance(
        &self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> (Option<f64>, f64, f64) {
        let Some(controller) = &self.reliability else {
            return (None, 0.0, 0.0);
        };
        let z = controller.config().confidence_z;
        let predicted_variance = f64::from(round.predicted_standard_error).powi(2);
        let mut decisive = 0usize;
        let mut correct = 0usize;
        let neighbors = self.ranked_neighbors(round);
        for index in neighbors.iter().copied() {
            let outcome = f64::from(self.outcomes[index]);
            let history_variance = f64::from(self.variances[index]);
            let predicted_delta = f64::from(round.predicted_mean) - outcome;
            let observed_delta = f64::from(value) - outcome;
            let predicted_threshold = z * (predicted_variance + history_variance).sqrt();
            let observed_threshold = z * (f64::from(variance) + history_variance).sqrt();
            if predicted_delta.abs() <= predicted_threshold
                || observed_delta.abs() <= observed_threshold
            {
                continue;
            }
            decisive += 1;
            correct += usize::from(predicted_delta.signum() == observed_delta.signum());
        }
        let coverage = decisive as f64 / neighbors.len().max(1) as f64;
        let concordance = (decisive > 0).then_some(correct as f64 / decisive as f64);
        (concordance, coverage, decisive as f64)
    }

    pub(super) fn reliability_evidence(
        &self,
        round: &Proposals,
        value: f32,
        variance: f32,
        improvement: f64,
        improvement_variance: f64,
    ) -> Result<Option<reliability_region::ReliabilityEvidence>, String> {
        if self.reliability.is_none() {
            return Ok(None);
        }
        let realized_radius = f64::from(
            *round
                .pool_radii
                .get(round.index)
                .ok_or("Selected pool radius is missing")?,
        );
        let (concordance, coverage, rank_evidence) = self.rank_concordance(round, value, variance);
        Ok(Some(reliability_region::ReliabilityEvidence {
            concordance,
            coverage,
            rank_evidence,
            improvement,
            improvement_variance,
            nominal_radius: f64::from(round.length),
            realized_radius,
            center_radius: None,
        }))
    }

    pub(super) fn center_radius(&self) -> Option<f64> {
        let config = self.reliability.as_ref()?.config();
        let center = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)?;
        let mut distances = (0..self.history)
            .filter(|&column| column != center)
            .map(|column| f64::from(self.pairwise_distances[center * MAX_HISTORY + column]))
            .filter(|distance| distance.is_finite() && *distance >= 0.0)
            .collect::<Vec<_>>();
        distances.sort_by(f64::total_cmp);
        let k = config.local_scale_neighbors.min(distances.len());
        distances
            .get(k.checked_sub(1)?)
            .copied()
            .map(|squared| squared.max(1.0e-12).sqrt())
    }
}
