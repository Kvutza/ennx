//! Focused accessors for ENN row and index operations.

use ndarray::{Array1, Array2, ArrayView2};

use super::ENN;
use crate::backend::TrainRows;
use crate::error::ENNError;

/// Index search and sync operations on an ENN model.
pub struct EnnIndexAccess<'a> {
    model: &'a ENN,
}

impl<'a> EnnIndexAccess<'a> {
    pub(crate) fn new(model: &'a ENN) -> Self {
        Self { model }
    }

    pub fn ensure_sync(&self) -> Result<(), ENNError> {
        self.model
            .backend
            .ensure_sync(self.model.scale_x, &self.model.x_scale)
    }

    pub fn memory_bytes(&self) -> Result<usize, ENNError> {
        if !self.model.backend.defers_search() {
            self.ensure_sync()?;
        }
        self.model.backend.index_bytes()
    }

    pub fn is_stale(&self) -> bool {
        self.model.backend.index_stale()
    }

    pub fn release_pages(&self) -> Result<(), ENNError> {
        crate::backend::enn_pages(&self.model.backend)
    }

    pub fn len(&self) -> usize {
        self.model.backend.index_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nearest_neighbors(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        if !self.model.backend.defers_search() {
            self.ensure_sync()?;
        }
        self.model.backend.search(x, search_k, exclude_nearest)
    }

    pub fn posterior_neighbors(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
        tie_neighbors: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let _ = tie_neighbors;
        crate::posterior::index_search(self.model, x, search_k, exclude_nearest, tie_neighbors)
    }
}

/// Row gather operations on an ENN model.
pub struct EnnRowAccess<'a> {
    model: &'a ENN,
}

impl<'a> EnnRowAccess<'a> {
    pub(crate) fn new(model: &'a ENN) -> Self {
        Self { model }
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, ENNError> {
        self.model.backend.train_rows(indices)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.model.backend.row_x(i)
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.model.backend.row_y(i)
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        self.model.backend.row_yvar(i)
    }
}

impl ENN {
    pub fn index_access(&self) -> EnnIndexAccess<'_> {
        EnnIndexAccess::new(self)
    }

    pub fn rows(&self) -> EnnRowAccess<'_> {
        EnnRowAccess::new(self)
    }

    pub(crate) fn ensure_sync(&self) -> Result<(), ENNError> {
        self.index_access().ensure_sync()
    }
}

#[cfg(test)]
mod access_tests {
    use crate::IndexDriver;
    use crate::ENN;
    use ndarray::array;

    #[test]
    fn neighbor_distance() {
        let model = ENN::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let query = array![[0.1, 0.1]];
        let access = model.index_access();
        let _ = access.nearest_neighbors(&query.view(), 1, false).unwrap();
        let _ = access
            .posterior_neighbors(&query.view(), 1, false, true)
            .unwrap();
    }
}
