//! Disk-backed ENN backend wrapping the standalone `bpann` crate.

use std::path::PathBuf;

use bpann::BpannBackend;
use ndarray::{Array1, Array2, ArrayView2};

use crate::backend::TrainRows;
use crate::error::ENNError;
use crate::file_config::bpann_config;
use crate::index::IndexDriver;

fn bpann_err(e: bpann::BpannError) -> ENNError {
    match e {
        bpann::BpannError::InvalidShape { expected, got } => {
            ENNError::InvalidShape { expected, got }
        }
        bpann::BpannError::InvalidParameter(s) => ENNError::InvalidParameter(s),
    }
}

fn apply_threshold(inner: BpannBackend) -> BpannBackend {
    let t = bpann::current_tuning();
    inner
        .with_soft(t.soft_threshold)
        .with_hard(t.hard_threshold)
}

pub struct DiskBpannEnnBackend {
    inner: BpannBackend,
    driver: IndexDriver,
    num_metrics: usize,
}

impl DiskBpannEnnBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::BpAnnDisk {
            return Err(ENNError::InvalidParameter(
                "DiskBpannEnnBackend requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        bpann_config().map_err(ENNError::InvalidParameter)?;
        let inner = if train_x.nrows() == 0
            && train_y.nrows() == 0
            && work_dir.join("metadata.json").exists()
        {
            BpannBackend::reopen(work_dir.clone()).map_err(bpann_err)?
        } else {
            BpannBackend::new(work_dir, train_x, train_y, train_yvar, scale_x, x_scale)
                .map_err(bpann_err)?
        };
        let inner = apply_threshold(inner);
        let num_metrics = inner.num_metrics();
        Ok(Self {
            inner,
            driver,
            num_metrics,
        })
    }

    pub fn new_empty(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        bpann_config().map_err(ENNError::InvalidParameter)?;
        let inner = apply_threshold(
            BpannBackend::new_empty(work_dir, num_dim, num_metrics).map_err(bpann_err)?,
        );
        Ok(Self {
            inner,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
        })
    }

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

    pub fn mark_stale(&mut self) {
        self.inner.mark_stale();
    }

    pub fn persist_index(&mut self) -> Result<(), ENNError> {
        self.inner.persist_index().map_err(bpann_err)
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        self.inner.append_rows(x, y, yvar).map_err(bpann_err)
    }

    pub fn ensure_sync(&mut self, scale_x: bool, x_scale: &Array1<f64>) -> Result<(), ENNError> {
        self.inner.ensure_scale(scale_x, x_scale).map_err(bpann_err)
    }

    pub fn release_pages(&mut self) -> Result<(), ENNError> {
        self.inner.release_pages().map_err(bpann_err)
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, ENNError> {
        self.inner.train_rows(indices).map_err(bpann_err)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(
            self.inner.row_slice(i).map_err(bpann_err)?.to_vec(),
        ))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        // Do not use train_rows here: that gathers train_x too and faults Θ(N·D)
        // when callers iterate all rows (y_obs / incumbent rebuild).
        let (y, _) = self.inner.y_yvar(i).map_err(bpann_err)?;
        Ok(Array1::from(y.to_vec()))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        let (_, yvar) = self.inner.y_yvar(i).map_err(bpann_err)?;
        Ok(yvar.map(|row| Array1::from(row.to_vec())))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.inner
            .search(x, search_k as usize, exclude_nearest)
            .map_err(bpann_err)
    }

    pub fn index_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.inner.index_bytes())
    }
}

impl DiskBpannEnnBackend {
    pub fn new_threshold(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        soft_threshold: usize,
    ) -> Result<Self, ENNError> {
        let hard = bpann::current_tuning().hard_threshold;
        Self::new_thresholds(
            work_dir,
            num_dim,
            num_metrics,
            soft_threshold,
            hard.max(soft_threshold),
        )
    }

    pub fn new_thresholds(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        soft_threshold: usize,
        hard_threshold: usize,
    ) -> Result<Self, ENNError> {
        bpann_config().map_err(ENNError::InvalidParameter)?;
        if soft_threshold == 0 {
            return Err(ENNError::InvalidParameter(
                "soft_threshold must be >= 1".to_string(),
            ));
        }
        if hard_threshold == 0 {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= 1".to_string(),
            ));
        }
        if hard_threshold < soft_threshold {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= soft_threshold".to_string(),
            ));
        }
        let inner = BpannBackend::new_empty(work_dir, num_dim, num_metrics)
            .map_err(bpann_err)?
            .with_soft(soft_threshold)
            .with_hard(hard_threshold)
            .defer_indexing(true);
        Ok(Self {
            inner,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
        })
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_rows()
    }

    pub fn soft_threshold(&self) -> usize {
        self.inner.soft_threshold()
    }

    pub fn hard_threshold(&self) -> usize {
        self.inner.hard_threshold()
    }

    pub fn set_thresholds(&mut self, soft: usize, hard: usize) {
        self.inner.set_thresholds(soft, hard);
    }

    pub fn append_threshold(&self) -> bool {
        !self.inner.index_deferred()
    }

    pub(crate) fn defer_flush(&self) -> bool {
        self.inner.index_deferred()
    }

    /// Build soft-sync fragments without mutating the published index.
    pub(crate) fn build_detached(&self) -> Result<Option<bpann::IncrementalIndex>, ENNError> {
        bpann::soft_build(&self.inner).map_err(bpann_err)
    }

    /// Publish a detached soft-sync result (short exclusive critical section).
    pub(crate) fn publish_detached(
        &mut self,
        built: bpann::IncrementalIndex,
    ) -> Result<(), ENNError> {
        bpann::soft_publish(&mut self.inner, built);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn new_sync() {
        let dir = TempDir::new().expect("tempdir");
        let backend =
            DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 5).expect("backend");
        assert_eq!(backend.soft_threshold(), 5);
        assert!(!backend.append_threshold());
        assert_eq!(backend.pending_count(), 0);
    }

    #[test]
    fn bpann_driver() {
        let dir = TempDir::new().expect("tempdir");
        let result = DiskBpannEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::Exact,
        );
        assert!(matches!(result, Err(ENNError::InvalidParameter(_))));
    }

    #[test]
    fn append_maps() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend =
            DiskBpannEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).expect("backend");
        let err = backend
            .append_rows(&array![[0.0, 0.0, 0.0]].view(), &array![[0.0]].view(), None)
            .unwrap_err();
        assert!(matches!(err, ENNError::InvalidShape { .. }));
    }

    #[test]
    fn reopen_dir() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn y_train() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.num_metrics(), 2);
    }

    #[test]
    fn pending_sync() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 100)
            .expect("backend");
        backend
            .append_rows(&array![[0.0, 0.0]].view(), &array![[1.0]].view(), None)
            .expect("append");
        assert!(backend.pending_count() > 0);
    }

    #[test]
    fn soft_pages2() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend =
            DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 5).expect("backend");
        assert!(backend.defer_flush());
        backend
            .append_rows(
                &array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].view(),
                &array![[0.0], [1.0], [2.0]].view(),
                None,
            )
            .expect("append");
        assert!(backend.pending_count() > 0);
        if let Some(built) = backend.build_detached().expect("build") {
            backend.publish_detached(built).expect("publish");
        }
        assert_eq!(backend.pending_count(), 0);
        assert!(!dir.path().join("index/pages.bin").exists());
        let _mapped = bpann_err(bpann::BpannError::InvalidParameter("x".into()));
        assert!(matches!(_mapped, ENNError::InvalidParameter(_)));
        let _tuned = apply_threshold(
            bpann::BpannBackend::new_empty(dir.path().join("t"), 2, 1).expect("tuned"),
        );
        assert!(_tuned.soft_threshold() >= 1);
    }
}
