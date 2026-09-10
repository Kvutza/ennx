use std::sync::atomic::{AtomicU64, Ordering};

// IDs never repeat across search instances, including after a restart.
static NEXT_TRIAL: AtomicU64 = AtomicU64::new(0);

pub(super) fn trial_id() -> Result<u64, String> {
    NEXT_TRIAL
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "trial identity space exhausted".to_string())
}

/// Identity of an evaluation. Diagnostics do not determine ownership or equality.
#[derive(Debug, Clone, Copy)]
pub struct Trial {
    pub(super) id: u64,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
}

impl PartialEq for Trial {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Trial {}
