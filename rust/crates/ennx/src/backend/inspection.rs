//! Backend shape and storage inspection.

use super::*;

impl EnnBackend {
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_dim(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.num_dim()).unwrap_or(0),
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_metrics(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.num_metrics()).unwrap_or(0),
        }
    }

    pub fn driver(&self) -> IndexDriver {
        match self {
            Self::InMemory(b) => b.driver(),
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.driver())
                .unwrap_or(IndexDriver::BpAnnDisk),
        }
    }
}
