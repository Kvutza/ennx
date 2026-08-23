#[cfg(all(feature = "usearch", feature = "usearch-native"))]
compile_error!("features \"usearch\" and \"usearch-native\" are mutually exclusive");

use ndarray::{Array2, ArrayView1, ArrayView2};

use super::pad_neighbor_cols_to_search_k;
use crate::index::IndexError;

fn usearch_error(error: impl std::fmt::Display) -> IndexError {
    IndexError::InvalidParameter(error.to_string())
}

fn squared_l2(left: ArrayView1<'_, f64>, right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn rerank(
    query: ArrayView1<'_, f64>,
    keys: &[u64],
    rows: &[f64],
    num_dim: usize,
    k: usize,
) -> Result<Vec<(f64, i64)>, IndexError> {
    let mut neighbors = Vec::with_capacity(keys.len());
    for &key in keys {
        let row = usize::try_from(key)
            .ok()
            .and_then(|key| key.checked_mul(num_dim))
            .and_then(|start| start.checked_add(num_dim).map(|end| (start, end)))
            .and_then(|(start, end)| rows.get(start..end))
            .ok_or_else(|| {
                IndexError::InvalidParameter(format!("USearch returned unknown key {key}"))
            })?;
        neighbors.push((squared_l2(query, row), key as i64));
    }
    neighbors.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    neighbors.truncate(k);
    Ok(neighbors)
}

#[cfg(feature = "usearch")]
mod binding {
    use super::{usearch_error, IndexError};
    use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

    pub(super) struct BackendIndex {
        index: Index,
    }

    fn options(num_dim: usize) -> IndexOptions {
        IndexOptions {
            dimensions: num_dim,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        }
    }

    impl BackendIndex {
        pub(super) fn new(num_dim: usize) -> Result<Self, IndexError> {
            let index = new_index(&options(num_dim)).map_err(usearch_error)?;
            Ok(Self { index })
        }

        pub(super) fn reserve(&mut self, capacity: usize) -> Result<(), IndexError> {
            self.index.reserve(capacity).map_err(usearch_error)
        }

        pub(super) fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), IndexError> {
            self.index.add(key, vector).map_err(usearch_error)
        }

        pub(super) fn search(
            &self,
            query: &[f32],
            wanted: usize,
            exact: bool,
        ) -> Result<Vec<u64>, IndexError> {
            let matches = if exact {
                self.index.exact_search(query, wanted)
            } else {
                self.index.search(query, wanted)
            }
            .map_err(usearch_error)?;
            Ok(matches.keys)
        }

        pub(super) fn expansion_search(&self) -> usize {
            self.index.expansion_search()
        }

        pub(super) fn memory_usage_bytes(&self) -> usize {
            self.index.memory_usage()
        }
    }
}

#[cfg(feature = "usearch-native")]
mod binding {
    use super::{usearch_error, IndexError};
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr;

    const ERROR_BUFFER_LEN: usize = 512;

    extern "C" {
        fn ennx_usearch_new(
            num_dim: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn ennx_usearch_destroy(handle: *mut c_void);
        fn ennx_usearch_reserve(
            handle: *mut c_void,
            capacity: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        fn ennx_usearch_add(
            handle: *mut c_void,
            key: u64,
            vector: *const f32,
            num_dim: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        fn ennx_usearch_search(
            handle: *const c_void,
            query: *const f32,
            num_dim: usize,
            wanted: usize,
            exact: bool,
            out_keys: *mut u64,
            out_capacity: usize,
            out_count: *mut usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        fn ennx_usearch_expansion_search(handle: *const c_void) -> usize;
        fn ennx_usearch_memory_usage(handle: *const c_void) -> usize;
    }

    fn error_from_buffer(buffer: &[c_char], fallback: &str) -> IndexError {
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned();
        if message.is_empty() {
            usearch_error(fallback)
        } else {
            usearch_error(message)
        }
    }

    fn call_error_bool(ok: bool, error: &[c_char], fallback: &str) -> Result<(), IndexError> {
        if ok {
            Ok(())
        } else {
            Err(error_from_buffer(error, fallback))
        }
    }

    pub(super) struct BackendIndex {
        handle: *mut c_void,
    }

    unsafe impl Send for BackendIndex {}

    impl Drop for BackendIndex {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { ennx_usearch_destroy(self.handle) };
                self.handle = ptr::null_mut();
            }
        }
    }

    impl BackendIndex {
        pub(super) fn new(num_dim: usize) -> Result<Self, IndexError> {
            let mut error = [0 as c_char; ERROR_BUFFER_LEN];
            let handle = unsafe { ennx_usearch_new(num_dim, error.as_mut_ptr(), error.len()) };
            if handle.is_null() {
                return Err(error_from_buffer(
                    &error,
                    "failed to construct the native USearch index",
                ));
            }
            Ok(Self { handle })
        }

        pub(super) fn reserve(&mut self, capacity: usize) -> Result<(), IndexError> {
            let mut error = [0 as c_char; ERROR_BUFFER_LEN];
            let ok = unsafe {
                ennx_usearch_reserve(self.handle, capacity, error.as_mut_ptr(), error.len())
            };
            call_error_bool(ok, &error, "failed to reserve native USearch capacity")
        }

        pub(super) fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), IndexError> {
            let mut error = [0 as c_char; ERROR_BUFFER_LEN];
            let ok = unsafe {
                ennx_usearch_add(
                    self.handle,
                    key,
                    vector.as_ptr(),
                    vector.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            call_error_bool(ok, &error, "failed to add a native USearch vector")
        }

        pub(super) fn search(
            &self,
            query: &[f32],
            wanted: usize,
            exact: bool,
        ) -> Result<Vec<u64>, IndexError> {
            let mut error = [0 as c_char; ERROR_BUFFER_LEN];
            let mut keys = vec![0u64; wanted];
            let mut count = 0usize;
            let ok = unsafe {
                ennx_usearch_search(
                    self.handle,
                    query.as_ptr(),
                    query.len(),
                    wanted,
                    exact,
                    keys.as_mut_ptr(),
                    keys.len(),
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            call_error_bool(ok, &error, "native USearch search failed")?;
            keys.truncate(count);
            Ok(keys)
        }

        pub(super) fn expansion_search(&self) -> usize {
            unsafe { ennx_usearch_expansion_search(self.handle) }
        }

        pub(super) fn memory_usage_bytes(&self) -> usize {
            unsafe { ennx_usearch_memory_usage(self.handle) }
        }
    }
}

#[cfg(feature = "usearch")]
use binding::BackendIndex;
#[cfg(feature = "usearch-native")]
use binding::BackendIndex;

pub(crate) struct USearchBackend {
    index: BackendIndex,
    rows: Vec<f64>,
    num_dim: usize,
}

impl USearchBackend {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        if num_dim == 0 {
            return Err(IndexError::InvalidParameter(
                "USearch requires at least one dimension".to_string(),
            ));
        }
        let index = BackendIndex::new(num_dim)?;
        let mut backend = Self {
            index,
            rows: Vec::new(),
            num_dim,
        };
        backend.add(train_scaled, 0)?;
        Ok(backend)
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len() / self.num_dim
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.index
            .memory_usage_bytes()
            .saturating_add(self.rows.len().saturating_mul(std::mem::size_of::<f64>()))
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        *self = Self::new(self.num_dim, train_scaled)?;
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if start_key != self.len() as u64 {
            return Err(IndexError::InvalidParameter(format!(
                "USearch keys must be contiguous: expected {}, got {start_key}",
                self.len()
            )));
        }
        if rows_scaled.is_empty() {
            return Ok(());
        }

        self.index
            .reserve(self.len() + rows_scaled.nrows())
            .map_err(usearch_error)?;
        for (offset, row) in rows_scaled.rows().into_iter().enumerate() {
            let vector: Vec<f32> = row.iter().map(|value| *value as f32).collect();
            self.index
                .add(start_key + offset as u64, &vector)
                .map_err(usearch_error)?;
            self.rows.extend(row);
        }
        Ok(())
    }

    fn shortlist_len(&self, k: usize) -> usize {
        k.max(self.index.expansion_search()).min(self.len())
    }

    fn shortlist(&self, query: &[f32], count: usize) -> Result<Vec<u64>, IndexError> {
        let exact = count == self.len();
        self.index.search(query, count, exact)
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if queries_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: queries_scaled.ncols(),
            });
        }

        let mut distances = Array2::from_elem((queries_scaled.nrows(), k_eff), f64::INFINITY);
        let mut indices = Array2::zeros((queries_scaled.nrows(), k_eff));
        let shortlist_len = self.shortlist_len(k_eff);
        for (query_index, query) in queries_scaled.rows().into_iter().enumerate() {
            let query_f32: Vec<f32> = query.iter().map(|value| *value as f32).collect();
            let keys = self.shortlist(&query_f32, shortlist_len)?;
            let neighbors = rerank(query, &keys, &self.rows, self.num_dim, k_eff)?;
            if neighbors.len() != k_eff {
                return Err(IndexError::InvalidParameter(format!(
                    "USearch returned {} candidates for requested k={k_eff}",
                    neighbors.len()
                )));
            }
            for (column, (distance, key)) in neighbors.into_iter().enumerate() {
                distances[[query_index, column]] = distance;
                indices[[query_index, column]] = key;
            }
        }
        Ok(pad_neighbor_cols_to_search_k(distances, indices, search_k))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn search_reranks_with_original_f64_rows() {
        let train = array![[1.0 + 5.0e-8], [1.0 + 4.0e-8], [3.0]];
        let backend = USearchBackend::new(1, &train.view()).unwrap();
        let (distances, indices) = backend.search(&array![[0.0]].view(), 2, 2).unwrap();
        assert_eq!(indices.row(0).to_vec(), vec![1, 0]);
        assert!(distances[[0, 0]] < distances[[0, 1]]);
    }

    #[test]
    fn add_and_rebuild_preserve_contiguous_keys() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let mut backend = USearchBackend::new(2, &train.view()).unwrap();
        backend.add(&array![[0.0, 1.0]].view(), 2).unwrap();
        assert_eq!(backend.len(), 3);
        let (_, indices) = backend.search(&array![[0.0, 0.9]].view(), 2, 2).unwrap();
        assert_eq!(indices[[0, 0]], 2);

        backend.rebuild(&train.view()).unwrap();
        assert_eq!(backend.len(), 2);
    }

    #[test]
    fn rejects_non_contiguous_keys() {
        let mut backend = USearchBackend::new(2, &array![[0.0, 0.0]].view()).unwrap();
        let error = backend.add(&array![[1.0, 1.0]].view(), 3).unwrap_err();
        assert!(error.to_string().contains("contiguous"));
    }
}
