//! Disk backend shape and search status.

use super::enn_backend::DiskBpannEnnBackend;
use crate::index::IndexDriver;

impl DiskBpannEnnBackend {
    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn defers_search(&self) -> bool {
        true
    }

    pub fn index_stale(&self) -> bool {
        false
    }
}
