use super::decode::RouteSample;
use super::target::TargetStats;

#[derive(Default)]
pub(super) struct VerificationProgress {
    pub(super) cursor: usize,
    pub(super) broad_passes: usize,
    pub(super) correction_waves: usize,
    pub(super) repair_batches: usize,
    pub(super) parallel_positions: usize,
    pub(super) repair_positions: usize,
    pub(super) first_mismatch: Option<usize>,
    pub(super) evaluated_lengths: Vec<usize>,
    pub(super) accepted_lengths: Vec<usize>,
    pub(super) committed_lengths: Vec<usize>,
    pub(super) route_samples: Vec<RouteSample>,
    pub(super) gpu_seconds: f64,
    pub(super) target: TargetStats,
}

pub(super) struct CommitStats {
    pub(super) accepted: usize,
    pub(super) committed: usize,
    pub(super) mismatch: Option<usize>,
}

#[derive(Default)]
pub(super) struct RepairStats {
    pub(super) gpu_seconds: f64,
    pub(super) evaluated_positions: usize,
    pub(super) correction_waves: usize,
    pub(super) repair_batches: usize,
    pub(super) first_mismatch: Option<usize>,
    pub(super) evaluated_lengths: Vec<usize>,
    pub(super) accepted_lengths: Vec<usize>,
    pub(super) committed_lengths: Vec<usize>,
    pub(super) route_samples: Vec<RouteSample>,
}

impl RepairStats {
    pub(super) fn record_waves(&mut self, window: u32, waves: u32) {
        self.evaluated_positions += window as usize * waves as usize;
        self.correction_waves += waves as usize;
        self.repair_batches += 1;
        self.evaluated_lengths
            .extend(std::iter::repeat_n(window as usize, waves as usize));
    }

    pub(super) fn report(&self, cursor: usize, count: usize, next: &mut usize) {
        if count > 4096 && cursor >= *next {
            eprintln!(
                "ENNX_GENERATION_PROGRESS committed={} target={} correction_waves={} gpu_ms={:.3}",
                cursor,
                count,
                self.correction_waves,
                self.gpu_seconds * 1000.0
            );
            *next = (cursor / 16_384 + 1) * 16_384;
        }
    }
}
