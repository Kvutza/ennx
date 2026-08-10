use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::{
    ffi::{c_void, CStr},
    ptr::NonNull,
};

use memmap2::MmapMut;
use ndarray::{Array2, ArrayView2};

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k, unpack_batch_search};
use crate::error::ENNError;
use crate::index::{IndexDriver, IndexError};

unsafe extern "C" {
    fn enn_faiss_new(num_dim: usize) -> *mut c_void;
    fn enn_faiss_free(handle: *mut c_void);
    fn enn_faiss_reset(handle: *mut c_void) -> i32;
    fn enn_faiss_add(handle: *mut c_void, num_rows: usize, rows: *const f32) -> i32;
    fn enn_faiss_len(handle: *const c_void) -> usize;
    fn enn_faiss_search(
        handle: *mut c_void,
        num_queries: usize,
        k: usize,
        queries: *const f32,
        distances: *mut f32,
        labels: *mut i64,
    ) -> i32;
    fn enn_faiss_last_error() -> *const std::ffi::c_char;
}

pub(crate) struct FaissBackend {
    handle: NonNull<c_void>,
    num_dim: usize,
}

// Faiss is protected by the mutex in KnnBackend.
unsafe impl Send for FaissBackend {}

fn faiss_error() -> IndexError {
    let message = unsafe {
        let ptr = enn_faiss_last_error();
        if ptr.is_null() {
            "unknown Faiss error".to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    IndexError::InvalidParameter(message)
}

#[cfg(test)]
fn faiss_spec(driver: IndexDriver) -> &'static str {
    match driver {
        IndexDriver::Exact => "Flat",
        IndexDriver::Auto
        | IndexDriver::Agx
        | IndexDriver::USearch
        | IndexDriver::BpAnnDisk
        | IndexDriver::Metal
        | IndexDriver::OpenCl => {
            panic!("non-Faiss driver must not be routed to FaissBackend")
        }
    }
}

impl FaissBackend {
    pub(crate) fn new(
        num_dim: usize,
        _driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        if train_scaled.ncols() != num_dim {
            return Err(IndexError::InvalidShape {
                expected: num_dim,
                got: train_scaled.ncols(),
            });
        }
        let mut backend = Self {
            handle: NonNull::new(unsafe { enn_faiss_new(num_dim) }).ok_or_else(faiss_error)?,
            num_dim,
        };
        if !train_scaled.is_empty() {
            backend.add(train_scaled, 0)?;
        }
        Ok(backend)
    }

    pub(crate) fn len(&self) -> usize {
        unsafe { enn_faiss_len(self.handle.as_ptr()) }
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.len()
            .saturating_mul(self.num_dim)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        if train_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: train_scaled.ncols(),
            });
        }
        {
            if unsafe { enn_faiss_reset(self.handle.as_ptr()) } != 0 {
                return Err(faiss_error());
            }
        }
        if !train_scaled.is_empty() {
            self.add(train_scaled, 0)?;
        }
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        let rows = arr2_rows_to_f32(rows_scaled);
        {
            if unsafe { enn_faiss_add(self.handle.as_ptr(), rows_scaled.nrows(), rows.as_ptr()) }
                != 0
            {
                return Err(faiss_error());
            }
        }
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
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
        let n_query = queries_scaled.nrows();
        let queries = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.faiss.prepare"));
            span.emit_value(n_query as u64);
            arr2_rows_to_f32(queries_scaled)
        };
        let (distances, labels) = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.faiss.search"));
            span.emit_value(n_query as u64);
            let mut distances = vec![0.0_f32; n_query.saturating_mul(k_eff)];
            let mut labels = vec![0_i64; n_query.saturating_mul(k_eff)];
            if k_eff > 0 && n_query > 0 {
                if unsafe {
                    enn_faiss_search(
                        self.handle.as_ptr(),
                        n_query,
                        k_eff,
                        queries.as_ptr(),
                        distances.as_mut_ptr(),
                        labels.as_mut_ptr(),
                    )
                } != 0
                {
                    return Err(faiss_error());
                }
            }
            (distances, labels)
        };
        let output = {
            let span = crate::tracy::zone(tracy_client::span_location!("knn.faiss.unpack"));
            span.emit_value(n_query as u64);
            let (distances, labels) = unpack_batch_search(n_query, k_eff, &distances, &labels);
            pad_neighbor_cols_to_search_k(distances, labels, search_k)
        };
        Ok(output)
    }
}

impl Drop for FaissBackend {
    fn drop(&mut self) {
        unsafe { enn_faiss_free(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
pub(crate) fn faiss_spec_for_test(driver: IndexDriver) -> &'static str {
    faiss_spec(driver)
}

#[cfg(test)]
pub(crate) fn make_faiss_for_test(
    num_dim: usize,
    driver: IndexDriver,
    train_scaled: &ArrayView2<f64>,
) -> Result<FaissBackend, IndexError> {
    FaissBackend::new(num_dim, driver, train_scaled)
}

/// Grow the backing file to the exact row count needed (no pre-allocation tail).
const MMAP_GROW_ROWS: usize = 64;

pub struct MmapColumnStore {
    #[allow(dead_code)]
    pub(crate) path: PathBuf,
    pub(crate) ncols: usize,
    pub(crate) nrows: usize,
    file: File,
    mmap: MmapMut,
}

impl MmapColumnStore {
    fn row_bytes(&self) -> usize {
        self.ncols * std::mem::size_of::<f64>()
    }

    fn bytes_for_rows(&self, nrows: usize) -> usize {
        nrows.saturating_mul(self.row_bytes())
    }

    fn ensure_capacity(&mut self, need_rows: usize) -> Result<(), ENNError> {
        let need_bytes = self.bytes_for_rows(need_rows);
        if need_bytes <= self.mmap.len() {
            return Ok(());
        }
        let grow_rows = (need_rows - self.nrows).max(MMAP_GROW_ROWS);
        let new_len = self.bytes_for_rows(self.nrows + grow_rows);
        if !self.mmap.is_empty() {
            self.mmap
                .flush()
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        self.file
            .set_len(new_len as u64)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        self.mmap = unsafe {
            MmapMut::map_mut(&self.file).map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        Ok(())
    }

    pub fn mmap_open_or_create(
        path: PathBuf,
        ncols: usize,
        known_nrows: Option<usize>,
    ) -> Result<Self, ENNError> {
        if !path.exists() {
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            drop(file);
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let len = file
            .metadata()
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
            .len();
        let row_bytes = ncols * std::mem::size_of::<f64>();
        let nrows = known_nrows.unwrap_or_else(|| {
            if row_bytes > 0 {
                (len as usize) / row_bytes
            } else {
                0
            }
        });
        if known_nrows.is_some() && nrows * row_bytes > len as usize {
            return Err(ENNError::InvalidParameter(format!(
                "known_nrows {nrows} exceeds train file bytes {len}"
            )));
        }
        let mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        Ok(Self {
            path,
            ncols,
            nrows,
            file,
            mmap,
        })
    }

    pub fn mmap_append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
        if rows.nrows() == 0 {
            return Ok(());
        }
        if rows.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![rows.nrows(), rows.ncols()],
            });
        }
        let new_nrows = self.nrows + rows.nrows();
        self.ensure_capacity(new_nrows)?;
        let offset = self.nrows * self.row_bytes();
        let n = rows.nrows() * self.ncols;
        let byte_len = n * std::mem::size_of::<f64>();
        let dst = &mut self.mmap[offset..offset + byte_len];
        // Always materialize C-order first, then one memcpy. Correct for Fortran /
        // strided views without sharing bpann's as_slice + axis_iter dual path.
        let contiguous = rows.as_standard_layout();
        let src = contiguous
            .as_slice()
            .expect("as_standard_layout yields a contiguous f64 slice");
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, byte_len) };
        dst.copy_from_slice(src_bytes);
        self.nrows = new_nrows;
        Ok(())
    }

    pub fn mmap_row_slice(&self, i: usize) -> Result<&[f64], ENNError> {
        if i >= self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "row {i} out of range [0, {})",
                self.nrows
            )));
        }
        let start = i * self.ncols;
        let row_bytes = self.ncols * std::mem::size_of::<f64>();
        let byte_start = start * std::mem::size_of::<f64>();
        let byte_end = byte_start + row_bytes;
        let bytes = &self.mmap[byte_start..byte_end];
        let slice: &[f64] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.ncols) };
        Ok(slice)
    }

    pub(crate) fn mmap_gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
            let row = self.mmap_row_slice(old_i)?;
            for j in 0..self.ncols {
                out[[new_i, j]] = row[j];
            }
        }
        Ok(out)
    }

    /// Copy rows `[start, end)` into a dense buffer (does not materialize the full store).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mmap_row_range(&self, start: usize, end: usize) -> Result<Array2<f64>, ENNError> {
        if start > end {
            return Err(ENNError::InvalidParameter(format!(
                "mmap_row_range: start {start} > end {end}"
            )));
        }
        if end > self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "mmap_row_range end {end} out of range [0, {})",
                self.nrows
            )));
        }
        let n = end - start;
        if n == 0 {
            return Ok(Array2::zeros((0, self.ncols)));
        }
        let mut out = Array2::zeros((n, self.ncols));
        for (new_i, old_i) in (start..end).enumerate() {
            let row = self.mmap_row_slice(old_i)?;
            for j in 0..self.ncols {
                out[[new_i, j]] = row[j];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod faiss_backend_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn faiss_backend_exact_roundtrip() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut backend = FaissBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(backend.len(), 3);
        backend.add(&array![[1.0, 1.0]].view(), 3).unwrap();
        assert_eq!(backend.len(), 4);
        let (d, i) = backend.search(&array![[0.0, 0.0]].view(), 2, 2).unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < 1e-5);
        backend.rebuild(&train.view()).unwrap();
        assert_eq!(backend.len(), 3);
    }

    #[test]
    fn faiss_spec_is_flat() {
        assert_eq!(faiss_spec(IndexDriver::Exact), "Flat");
    }

    #[test]
    fn make_faiss_backend() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let index = make_faiss_for_test(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn MmapColumnStore() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.ncols, 2);
    }

    #[test]
    fn row_bytes() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.row_bytes(), 16);
    }

    #[test]
    fn bytes_for_rows() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.bytes_for_rows(4), 64);
    }

    #[test]
    fn ensure_capacity() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        store.ensure_capacity(3).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.nrows, 2);
    }

    #[test]
    fn mmap_column_store_single_row_append_without_remap_churn() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("col.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(path, 2, None).unwrap();
        let n = 400usize;
        store.ensure_capacity(n).unwrap();
        for i in 0..n {
            store
                .mmap_append(&array![[i as f64, (i + 1) as f64]].view())
                .unwrap();
        }
        assert_eq!(store.nrows, n);
        assert_eq!(store.mmap_row_slice(n - 1).unwrap()[0], (n - 1) as f64);
    }

    #[test]
    fn mmap_append_fortran_order_preserves_rows() {
        use ndarray::{Array2, ShapeBuilder};
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 3, None).unwrap();
        let mut f = Array2::<f64>::zeros((2, 3).f());
        f[[0, 0]] = 1.0;
        f[[0, 1]] = 2.0;
        f[[0, 2]] = 3.0;
        f[[1, 0]] = 4.0;
        f[[1, 1]] = 5.0;
        f[[1, 2]] = 6.0;
        assert!(!f.is_standard_layout());
        store.mmap_append(&f.view()).unwrap();
        assert_eq!(
            store.mmap_row_slice(0).unwrap(),
            &[1.0, 2.0, 3.0],
            "row0 corrupted under Fortran layout"
        );
        assert_eq!(store.mmap_row_slice(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn mmap_column_store_direct_api() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("col.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(path, 2, None).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.mmap_row_slice(1).unwrap()[0], 3.0);
        let gathered = store.mmap_gather(&[0, 1]).unwrap();
        assert_eq!(gathered.nrows(), 2);
        assert_eq!(store.mmap_row_range(0, store.nrows).unwrap().nrows(), 2);
        let mid = store.mmap_row_range(1, 2).unwrap();
        assert_eq!(mid[[0, 0]], 3.0);
    }

    #[test]
    fn faiss_backend_memory_usage() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let backend = FaissBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(
            backend.memory_usage_bytes(),
            2 * 2 * std::mem::size_of::<f32>()
        );
    }
}
