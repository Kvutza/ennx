use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct BoundReport {
    pub history: u16,
    pub neighbors: u16,
    pub raw_survivors: [u16; 4],
    pub final_survivors: [u16; 4],
    pub bound_violations: [u16; 4],
    pub rank_violations: [u16; 4],
}

fn kth(values: impl Iterator<Item = f64>, rank: usize) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[rank.min(values.len()) - 1]
}

fn row_scales(state: &SearchState) -> Vec<f64> {
    (0..state.history)
        .map(|row| {
            kth(
                (0..state.history)
                    .filter(|&column| column != row)
                    .map(|column| f64::from(state.total_distance(row, column))),
                state.local_scale_neighbors.min(state.history - 1),
            )
            .max(1.0e-12)
        })
        .collect()
}

fn top_mask(values: &[f64], count: usize) -> Vec<bool> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then(left.cmp(&right))
    });
    let mut selected = vec![false; values.len()];
    for index in order.into_iter().take(count) {
        selected[index] = true;
    }
    selected
}

impl SearchState {
    pub(super) fn inspect_bounds(
        &self,
        distances: &[f32],
        components: &[f32],
        root: u64,
    ) -> BoundReport {
        let family = self.family.as_ref().unwrap();
        let base = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)
            .unwrap();
        let neighbors = self
            .fitted_enn
            .map_or(self.initial_observations, |fitted| fitted.0)
            .clamp(1, self.history);
        let self_tuning =
            self.distance_scaling == crate::config::DistanceScaling::SelfTuning && self.history > 1;
        let history_scales = self_tuning.then(|| row_scales(self));
        let mut report = BoundReport {
            history: self.history as u16,
            neighbors: neighbors as u16,
            ..BoundReport::default()
        };
        for candidate in 0..4 {
            let query = &components[(candidate * self.history + base) * FAMILIES..][..FAMILIES];
            let mut lower = vec![0.0f64; self.history];
            let mut upper = vec![0.0f64; self.history];
            for row in 0..self.history {
                let anchor = family.components[base * MAX_HISTORY + row];
                for group in 0..FAMILIES {
                    let weight = f64::from(family.weights[group]);
                    let left = (f64::from(anchor[group]) * weight).max(0.0).sqrt();
                    let right = (f64::from(query[group]) * weight).max(0.0).sqrt();
                    lower[row] += (left - right).powi(2);
                    upper[row] += (left + right).powi(2);
                }
                if let (Some(axis), Some(candidate_axis)) =
                    (&self.axis, self.axis_candidate(root, candidate))
                {
                    let scale = f64::from(axis.weight).sqrt();
                    let left = f64::from((axis.history[row] - axis.base).abs()) * scale;
                    let right = f64::from((candidate_axis - axis.base).abs()) * scale;
                    lower[row] += (left - right).powi(2);
                    upper[row] += (left + right).powi(2);
                }
            }
            let actual = distances[candidate * self.history..][..self.history]
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>();
            let raw_cutoff = kth(upper.iter().copied(), neighbors);
            let raw_keep = lower
                .iter()
                .map(|&bound| bound <= raw_cutoff)
                .collect::<Vec<_>>();
            report.raw_survivors[candidate] = raw_keep.iter().filter(|&&keep| keep).count() as u16;
            let mut final_keep = raw_keep;
            let ranked = if let Some(history_scales) = &history_scales {
                let query_scale =
                    kth(actual.iter().copied(), self.local_scale_neighbors).max(1.0e-12);
                let scaled_lower = lower
                    .iter()
                    .zip(history_scales)
                    .map(|(&bound, &scale)| bound / (query_scale * scale).sqrt())
                    .collect::<Vec<_>>();
                let scaled_upper = upper
                    .iter()
                    .zip(history_scales)
                    .map(|(&bound, &scale)| bound / (query_scale * scale).sqrt())
                    .collect::<Vec<_>>();
                let cutoff = kth(scaled_upper.iter().copied(), neighbors);
                for (keep, bound) in final_keep.iter_mut().zip(scaled_lower) {
                    *keep |= bound <= cutoff;
                }
                actual
                    .iter()
                    .zip(history_scales)
                    .map(|(&value, &scale)| value / (query_scale * scale).sqrt())
                    .collect::<Vec<_>>()
            } else {
                actual.clone()
            };
            report.final_survivors[candidate] =
                final_keep.iter().filter(|&&keep| keep).count() as u16;
            report.bound_violations[candidate] = actual
                .iter()
                .zip(&lower)
                .zip(&upper)
                .filter(|&((&value, &low), &high)| value < low || value > high)
                .count() as u16;
            let exact = top_mask(&ranked, neighbors);
            report.rank_violations[candidate] = exact
                .iter()
                .zip(final_keep)
                .filter(|&(selected, kept)| *selected && !kept)
                .count() as u16;
        }
        report
    }
}
