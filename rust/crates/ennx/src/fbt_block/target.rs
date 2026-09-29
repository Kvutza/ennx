use super::*;

pub(super) struct TargetStats {
    sum: f64,
    pub(super) count: usize,
    maximum: f32,
    window: Vec<f32>,
    window_sum: f64,
    window_count: usize,
    window_cursor: usize,
    worst_window_sum: f64,
    matches: usize,
    mismatch_sum: f64,
    mismatches: usize,
    first_mismatch: Option<usize>,
}

impl Default for TargetStats {
    fn default() -> Self {
        Self::new(1)
    }
}

impl TargetStats {
    pub(super) fn new(window_tokens: usize) -> Self {
        assert!(window_tokens > 0);
        Self {
            sum: 0.0,
            count: 0,
            maximum: f32::NEG_INFINITY,
            window: vec![0.0; window_tokens],
            window_sum: 0.0,
            window_count: 0,
            window_cursor: 0,
            worst_window_sum: f64::NEG_INFINITY,
            matches: 0,
            mismatch_sum: 0.0,
            mismatches: 0,
            first_mismatch: None,
        }
    }

    pub(super) fn push(&mut self, position: usize, loss: f32, matched: bool) {
        self.sum += f64::from(loss);
        self.count += 1;
        self.maximum = self.maximum.max(loss);
        if self.window_count == self.window.len() {
            self.window_sum -= f64::from(self.window[self.window_cursor]);
        } else {
            self.window_count += 1;
        }
        self.window[self.window_cursor] = loss;
        self.window_cursor = (self.window_cursor + 1) % self.window.len();
        self.window_sum += f64::from(loss);
        if self.window_count == self.window.len() {
            self.worst_window_sum = self.worst_window_sum.max(self.window_sum);
        }
        if matched {
            self.matches += 1;
        } else {
            self.mismatch_sum += f64::from(loss);
            self.mismatches += 1;
            self.first_mismatch.get_or_insert(position);
        }
    }

    pub(super) fn finish(&self) -> Result<decode::TargetQuality, String> {
        if self.count == 0 {
            return Err("target quality requires at least one scored token".into());
        }
        let window_tokens = self.count.min(self.window.len());
        let worst_window_sum = if self.count < self.window.len() {
            self.sum
        } else {
            self.worst_window_sum
        };
        Ok(decode::TargetQuality {
            mean_nll: (self.sum / self.count as f64) as f32,
            maximum_nll: self.maximum,
            worst_window_nll: (worst_window_sum / window_tokens as f64) as f32,
            window_tokens,
            positional_accuracy: self.matches as f32 / self.count as f32,
            positional_mismatches: self.mismatches,
            first_positional_mismatch: self.first_mismatch,
            mismatch_target_nll: (self.mismatches > 0)
                .then_some((self.mismatch_sum / self.mismatches as f64) as f32),
        })
    }
}
