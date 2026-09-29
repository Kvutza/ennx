//! Cheap measurements over complete generated token trajectories.

use super::decode;
use ennx_wire::json::{Value, json};
use std::collections::{HashSet, VecDeque};

const BLOCK_TOKENS: usize = 128;

fn distinct_fraction(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    tokens.iter().collect::<HashSet<_>>().len() as f64 / tokens.len() as f64
}

fn repeat_fraction(tokens: &[u32]) -> f64 {
    let grams = tokens.len().saturating_sub(3);
    if grams == 0 {
        return 0.0;
    }
    let unique = tokens.windows(4).collect::<HashSet<_>>().len();
    1.0 - unique as f64 / grams as f64
}

const RANK_CAPACITY: usize = 128;

/// Exact pairwise ordering audit over the same bounded history used by the
/// resident optimizer. Likelihood and realized sequence quality remain
/// separate; this diagnostic never participates in acceptance.
pub(super) struct RankAudit {
    rows: VecDeque<(f32, f32)>,
}

impl RankAudit {
    pub(super) fn new(likelihood: f32, quality: f32) -> Self {
        Self {
            rows: VecDeque::from([(likelihood, quality)]),
        }
    }

    pub(super) fn observe(&mut self, likelihood: f32, quality: f32) -> Value {
        if self.rows.len() == RANK_CAPACITY {
            self.rows.pop_front();
        }
        self.rows.push_back((likelihood, quality));
        let mut concordant = 0usize;
        let mut inverted = 0usize;
        let mut tied = 0usize;
        for left in 0..self.rows.len() {
            for right in left + 1..self.rows.len() {
                let likelihood_order = self.rows[left].0.total_cmp(&self.rows[right].0);
                let quality_order = self.rows[left].1.total_cmp(&self.rows[right].1);
                if likelihood_order.is_eq() || quality_order.is_eq() {
                    tied += 1;
                } else if likelihood_order == quality_order {
                    concordant += 1;
                } else {
                    inverted += 1;
                }
            }
        }
        let comparable = concordant + inverted;
        json!({
            "schema": "ennx.model_selection_audit.v1",
            "history": self.rows.len(),
            "capacity": RANK_CAPACITY,
            "likelihood": likelihood,
            "realized_quality": quality,
            "comparable_pairs": comparable,
            "concordant_pairs": concordant,
            "inverted_pairs": inverted,
            "tied_pairs": tied,
            "inversion_fraction": if comparable == 0 {
                0.0
            } else {
                inverted as f64 / comparable as f64
            },
            "used_for_acceptance": false,
        })
    }
}

fn longest_run(tokens: &[u32]) -> usize {
    let mut longest = 0;
    let mut current = 0;
    let mut previous = None;
    for &token in tokens {
        current = if previous == Some(token) {
            current + 1
        } else {
            1
        };
        longest = longest.max(current);
        previous = Some(token);
    }
    longest
}

fn distribution(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({
            "values": values,
            "mean": 0.0,
            "population_variance": 0.0,
            "min": 0.0,
            "max": 0.0,
        });
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    json!({
        "values": values,
        "mean": mean,
        "population_variance": variance,
        "min": min,
        "max": max,
    })
}

pub(super) fn rollout_diagnostics(rollout: &decode::Rollout) -> Value {
    let tokens = &rollout.tokens;
    let longest = longest_run(tokens);
    let distinct = tokens
        .chunks(BLOCK_TOKENS)
        .map(distinct_fraction)
        .collect::<Vec<_>>();
    let repeated = tokens
        .chunks(BLOCK_TOKENS)
        .map(repeat_fraction)
        .collect::<Vec<_>>();
    json!({
        "schema": "ennx.sequence_metrics.v1",
        "generated_tokens": tokens.len(),
        "unique_tokens": tokens.iter().collect::<HashSet<_>>().len(),
        "distinct_token_fraction": distinct_fraction(tokens),
        "longest_identical_token_run": longest,
        "longest_identical_token_run_fraction": if tokens.is_empty() {
            0.0
        } else {
            longest as f64 / tokens.len() as f64
        },
        "repeated_fourgram_fraction": repeat_fraction(tokens),
        "block_tokens": BLOCK_TOKENS,
        "block_count": distinct.len(),
        "block_distinct_token_fraction": distribution(&distinct),
        "block_repeated_fourgram_fraction": distribution(&repeated),
        "target_quality": rollout.target_quality,
        "variance_interpretation": "Within-rollout block population variance; not an independent-sample or aleatoric variance estimate.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rollout(tokens: Vec<u32>) -> decode::Rollout {
        decode::Rollout {
            draft: None,
            committed_tokens: tokens.len(),
            tokens,
            finish_reason: "length",
            wall_seconds: 0.0,
            gpu_seconds: 0.0,
            evaluated_positions: 0,
            broad_passes: 0,
            correction_waves: 0,
            repair_batches: 0,
            accepted_tokens: 0,
            first_mismatch: None,
            evaluated_lengths: Vec::new(),
            accepted_lengths: Vec::new(),
            committed_lengths: Vec::new(),
            route_samples: Vec::new(),
            free_running_target_nll: None,
            target_quality: None,
        }
    }

    #[test]
    fn collapse_blocks() {
        let tokens = (0..256)
            .map(|index| if index < 128 { 7 } else { index as u32 })
            .collect();
        let report = rollout_diagnostics(&rollout(tokens));
        assert_eq!(report["schema"], "ennx.sequence_metrics.v1");
        assert_eq!(report["block_count"], 2);
        assert_eq!(
            report["block_distinct_token_fraction"]["values"][0],
            1.0 / 128.0
        );
        assert_eq!(report["block_distinct_token_fraction"]["values"][1], 1.0);
        assert_eq!(
            report["block_repeated_fourgram_fraction"]["values"][0],
            124.0 / 125.0
        );
        assert_eq!(report["block_repeated_fourgram_fraction"]["values"][1], 0.0);
    }

    #[test]
    fn short_metrics() {
        let report = rollout_diagnostics(&rollout(vec![1, 2, 3]));
        assert_eq!(report["distinct_token_fraction"], 1.0);
        assert_eq!(report["repeated_fourgram_fraction"], 0.0);
        assert_eq!(report["longest_identical_token_run"], 1);
    }

    #[test]
    fn rank_inversions() {
        let mut audit = RankAudit::new(-2.0, 0.25);
        let report = audit.observe(-1.0, 0.20);
        assert_eq!(report["inverted_pairs"], 1);
        assert_eq!(report["inversion_fraction"], 1.0);
        let report = audit.observe(-0.5, 0.30);
        assert_eq!(report["comparable_pairs"], 3);
        assert_eq!(report["inverted_pairs"], 1);
    }
}
