//! Shared disk observation metadata helpers (mmap stores + metadata.json).

use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::ENNError;
use crate::knn::MmapColumnStore;

pub type MmapRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub const FORMAT_VERSION: u32 = 1;
pub const MAX_DIM: usize = 100_000;
pub const MAX_STRIDE: usize = 8 * 1024 * 1024;

pub fn check_rows(new_n: usize) -> Result<(), ENNError> {
    if new_n >= u32::MAX as usize {
        return Err(ENNError::InvalidParameter(
            "disk ENN row count exceeds u32::MAX".to_string(),
        ));
    }
    Ok(())
}

pub fn set_stale(index_stale: &Mutex<bool>) {
    *index_stale.lock().expect("index_stale mutex poisoned") = true;
}

pub fn read_stale(index_stale: &Mutex<bool>) -> bool {
    *index_stale.lock().expect("index_stale mutex poisoned")
}

pub fn mmap_at(
    n: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<MmapRows, ENNError> {
    for &i in indices {
        if i >= n {
            return Err(ENNError::InvalidParameter(format!(
                "train_rows index {i} out of range [0, {n})"
            )));
        }
    }
    let x = train_x.mmap_gather(indices)?;
    let y = train_y.mmap_gather(indices)?;
    let yvar = train_yvar.map(|s| s.mmap_gather(indices)).transpose()?;
    Ok((x, y, yvar))
}

pub fn mmap_yvar(
    train_yvar: Option<&MmapColumnStore>,
    i: usize,
) -> Result<Option<Array1<f64>>, ENNError> {
    Ok(train_yvar.map(|s| Array1::from(s.row_slice(i).unwrap().to_vec())))
}

pub fn open_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: Option<&Array2<f64>>,
) -> Result<Option<MmapColumnStore>, ENNError> {
    if let Some(yv) = train_yvar {
        let yv_path = work_dir.join("train_yvar.bin");
        let known_nrows = load_obs(work_dir);
        let mut store = MmapColumnStore::open_mmap(yv_path, num_metrics, known_nrows)?;
        if store.nrows == 0 {
            store.mmap_append(&yv.view())?;
        }
        Ok(Some(store))
    } else {
        Ok(None)
    }
}

pub fn check_backend(work_dir: &Path, expected: &str) -> Result<(), ENNError> {
    if let Some(backend) = load_backend(work_dir) {
        if backend != expected {
            return Err(ENNError::InvalidParameter(format!(
                "work_dir index_backend is {backend}, expected {expected}"
            )));
        }
    }
    Ok(())
}

pub fn append_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: &mut Option<MmapColumnStore>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    match (train_yvar.as_mut(), yvar) {
        (Some(store), Some(yv)) => store.mmap_append(yv)?,
        (None, Some(yv)) => {
            let yv_path = work_dir.join("train_yvar.bin");
            let known_nrows = load_obs(work_dir);
            let mut store = MmapColumnStore::open_mmap(yv_path, num_metrics, known_nrows)?;
            store.mmap_append(yv)?;
            *train_yvar = Some(store);
        }
        _ => {}
    }
    Ok(())
}

pub struct DiskAppendContext<'a> {
    pub work_dir: &'a Path,
    pub num_metrics: usize,
    pub train_x: &'a mut MmapColumnStore,
    pub train_y: &'a mut MmapColumnStore,
    pub train_yvar: &'a mut Option<MmapColumnStore>,
    pub index_dirty: &'a Mutex<bool>,
    pub current_len: usize,
}

pub fn append_rows(
    ctx: &mut DiskAppendContext<'_>,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    if x.nrows() == 0 {
        return Ok(());
    }
    check_rows(ctx.current_len + x.nrows())?;
    ctx.train_x.mmap_append(x)?;
    ctx.train_y.mmap_append(y)?;
    append_yvar(ctx.work_dir, ctx.num_metrics, ctx.train_yvar, yvar)?;
    Ok(())
}

pub fn mark_dirty(index_dirty: &Mutex<bool>) {
    *index_dirty.lock().expect("index_dirty mutex poisoned") = true;
}

pub fn disk_at(
    n: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<MmapRows, ENNError> {
    mmap_at(n, train_x, train_y, train_yvar, indices)
}

pub fn append_dirty(
    ctx: &mut DiskAppendContext<'_>,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    if x.nrows() == 0 {
        return Ok(());
    }
    append_rows(ctx, x, y, yvar)?;
    mark_dirty(ctx.index_dirty);
    Ok(())
}

pub fn train_backend(
    len: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<MmapRows, ENNError> {
    disk_at(len, train_x, train_y, train_yvar, indices)
}

/// Shared `append_rows` / `train_rows` for disk backends with mmap observation stores.
#[macro_export]
macro_rules! api_observation {
    ($backend:ty) => {
        impl $backend {
            pub fn append_rows(
                &mut self,
                x: &ndarray::ArrayView2<f64>,
                y: &ndarray::ArrayView2<f64>,
                yvar: Option<&ndarray::ArrayView2<f64>>,
            ) -> Result<(), $crate::error::ENNError> {
                let current_len = self.train_x.nrows;
                $crate::backend::disk_observation::append_dirty(
                    &mut $crate::backend::disk_observation::DiskAppendContext {
                        work_dir: &self.work_dir,
                        num_metrics: self.num_metrics,
                        train_x: &mut self.train_x,
                        train_y: &mut self.train_y,
                        train_yvar: &mut self.train_yvar,
                        index_dirty: &self.index_dirty,
                        current_len,
                    },
                    x,
                    y,
                    yvar,
                )
            }

            pub fn train_rows(
                &self,
                indices: &[usize],
            ) -> Result<$crate::backend::TrainRows, $crate::error::ENNError> {
                $crate::backend::disk_observation::train_backend(
                    self.train_x.nrows,
                    &self.train_x,
                    &self.train_y,
                    self.train_yvar.as_ref(),
                    indices,
                )
            }
        }
    };
}

pub fn load_rows(work_dir: &Path) -> Option<usize> {
    let meta_path = work_dir.join("metadata.json");
    let text = fs::read_to_string(meta_path).ok()?;
    parse_number(&text, "indexed_rows")
}

#[allow(dead_code)]
pub fn load_obs(work_dir: &Path) -> Option<usize> {
    let meta_path = work_dir.join("metadata.json");
    let text = fs::read_to_string(meta_path).ok()?;
    parse_number(&text, "num_obs")
}

pub fn load_backend(work_dir: &Path) -> Option<String> {
    let meta_path = work_dir.join("metadata.json");
    let text = fs::read_to_string(meta_path).ok()?;
    parse_string(&text, "index_backend")
}

pub fn write_metadata(
    work_dir: &Path,
    num_obs: usize,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    indexed_rows: usize,
    index_backend: &str,
) -> Result<(), ENNError> {
    let meta_path = work_dir.join("metadata.json");
    let json = format!(
        "{{\"format_version\":{FORMAT_VERSION},\"num_obs\":{num_obs},\"num_dim\":{num_dim},\"num_metrics\":{num_metrics},\"scale_x\":{scale_x},\"index_backend\":\"{index_backend}\",\"indexed_rows\":{indexed_rows}}}"
    );
    fs::write(meta_path, json).map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn check_dims(num_dim: usize, record_stride: usize) -> Result<(), ENNError> {
    if num_dim > MAX_DIM {
        return Err(ENNError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_DIM}"
        )));
    }
    if record_stride > MAX_STRIDE {
        return Err(ENNError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_STRIDE}"
        )));
    }
    Ok(())
}

pub fn parse_number(text: &str, field: &str) -> Option<usize> {
    let key = format!("\"{field}\":");
    let pos = text.find(&key)? + key.len();
    let tail = text[pos..].trim_start();
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end].parse().ok()
}

pub fn parse_string(text: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":\"");
    let pos = text.find(&key)? + key.len();
    let tail = &text[pos..];
    let end = tail.find('"')?;
    Some(tail[..end].to_string())
}

#[cfg(test)]
mod disk_observation {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn bpann_disk() {
        let dir = TempDir::new().expect("tempdir");
        write_metadata(dir.path(), 7, 3, 2, true, 5, "bpann_disk").unwrap();
        assert_eq!(load_rows(dir.path()), Some(5));
        assert_eq!(load_obs(dir.path()), Some(7));
        assert_eq!(load_backend(dir.path()).as_deref(), Some("bpann_disk"));
    }

    #[test]
    fn dim_above() {
        check_dims(MAX_DIM, 100).unwrap();
        let err = check_dims(MAX_DIM + 1, 100).unwrap_err();
        assert!(err.to_string().contains(&MAX_DIM.to_string()));
    }

    #[test]
    fn dim_dim() {
        let err = check_dims(MAX_DIM + 1, 100).unwrap_err();
        assert!(err.to_string().contains("num_dim"));
        let err2 = check_dims(8, MAX_STRIDE + 1).unwrap_err();
        assert!(err2.to_string().contains("record_stride"));
    }

    #[test]
    fn load_cases() {
        let dir = TempDir::new().expect("tempdir");
        std::fs::write(
            dir.path().join("metadata.json"),
            "{\"indexed_rows\":not_a_number,\"index_backend\":\"bpann_disk\"}",
        )
        .unwrap();
        assert_eq!(load_rows(dir.path()), None);
        assert_eq!(load_backend(dir.path()), Some("bpann_disk".to_string()));
        std::fs::write(dir.path().join("metadata.json"), "{\"indexed_rows\":5}").unwrap();
        assert_eq!(load_backend(dir.path()), None);
        std::fs::write(dir.path().join("metadata.json"), "{\"format_version\":1}").unwrap();
        assert_eq!(load_rows(dir.path()), None);
    }

    #[test]
    fn parse_branches() {
        let text = "{\"indexed_rows\":42,\"index_backend\":\"bpann_disk\"}";
        assert_eq!(parse_number(text, "indexed_rows"), Some(42));
        assert_eq!(parse_number(text, "missing"), None);
        assert_eq!(
            parse_string(text, "index_backend").as_deref(),
            Some("bpann_disk")
        );
        assert_eq!(parse_string(text, "missing"), None);
        assert_eq!(parse_string("{\"index_backend\":", "index_backend"), None);
    }

    #[test]
    fn shared_wrappers() {
        use ndarray::array;
        let dir = TempDir::new().expect("tempdir");
        write_metadata(dir.path(), 0, 2, 1, false, 0, "bpann_disk").unwrap();
        check_backend(dir.path(), "bpann_disk").unwrap();
        let err = check_backend(dir.path(), "flat").unwrap_err();
        assert!(err.to_string().contains("index_backend"));

        let err_limit = check_rows(u32::MAX as usize).unwrap_err();
        assert!(err_limit.to_string().contains("u32::MAX"));
        check_rows(0).unwrap();

        let yv = array![[0.1], [0.2]];
        let store = open_yvar(dir.path(), 1, Some(&yv)).unwrap();
        assert!(store.is_some());
        assert_eq!(store.unwrap().nrows, 2);
        assert!(open_yvar(dir.path(), 1, None).unwrap().is_none());

        let x_path = dir.path().join("train_x.bin");
        let y_path = dir.path().join("train_y.bin");
        let mut x_store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let mut y_store = MmapColumnStore::open_mmap(y_path, 1, None).unwrap();
        x_store.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
        y_store.mmap_append(&array![[0.0]].view()).unwrap();
        let (tx, ty, _) = train_backend(1, &x_store, &y_store, None, &[0]).unwrap();
        assert_eq!(tx[[0, 0]], 0.0);
        assert_eq!(ty[[0, 0]], 0.0);
    }

    #[test]
    fn shared_direct() {
        use ndarray::array;
        let dir = TempDir::new().expect("tempdir");
        let x_path = dir.path().join("x.bin");
        let y_path = dir.path().join("y.bin");
        let mut x_store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let mut y_store = MmapColumnStore::open_mmap(y_path, 1, None).unwrap();
        x_store
            .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        y_store.mmap_append(&array![[0.0], [1.0]].view()).unwrap();
        let stale = Mutex::new(false);
        set_stale(&stale);
        assert!(*stale.lock().unwrap());
        let dirty = Mutex::new(false);
        let mut yvar_slot = None;
        append_dirty(
            &mut DiskAppendContext {
                work_dir: dir.path(),
                num_metrics: 1,
                train_x: &mut x_store,
                train_y: &mut y_store,
                train_yvar: &mut yvar_slot,
                index_dirty: &dirty,
                current_len: 2,
            },
            &array![[2.0, 2.0]].view(),
            &array![[2.0]].view(),
            Some(&array![[0.5]].view()),
        )
        .unwrap();
        assert!(*dirty.lock().unwrap());
        assert_eq!(x_store.nrows, 3);
        let (gx, gy, _) = disk_at(3, &x_store, &y_store, None, &[1]).unwrap();
        assert_eq!(gx[[0, 0]], 1.0);
        assert_eq!(gy[[0, 0]], 1.0);
        assert!(mmap_yvar(None, 0).unwrap().is_none());
        assert!(mmap_at(3, &x_store, &y_store, None, &[99]).is_err());
        let dirty2 = Mutex::new(false);
        let empty_x = Array2::<f64>::zeros((0, 2));
        let empty_y = Array2::<f64>::zeros((0, 1));
        append_dirty(
            &mut DiskAppendContext {
                work_dir: dir.path(),
                num_metrics: 1,
                train_x: &mut x_store,
                train_y: &mut y_store,
                train_yvar: &mut yvar_slot,
                index_dirty: &dirty2,
                current_len: 3,
            },
            &empty_x.view(),
            &empty_y.view(),
            None,
        )
        .unwrap();
        assert!(!*dirty2.lock().unwrap());
        append_yvar(dir.path(), 1, &mut yvar_slot, Some(&array![[0.6]].view())).unwrap();
        assert_eq!(yvar_slot.as_ref().unwrap().nrows, 2);
        mark_dirty(&dirty2);
        assert!(*dirty2.lock().unwrap());
        let mut x3 = x_store;
        let mut y3 = y_store;
        let mut yv3 = yvar_slot;
        let dirty3 = Mutex::new(false);
        let err = append_rows(
            &mut DiskAppendContext {
                work_dir: dir.path(),
                num_metrics: 1,
                train_x: &mut x3,
                train_y: &mut y3,
                train_yvar: &mut yv3,
                index_dirty: &dirty3,
                current_len: u32::MAX as usize - 1,
            },
            &array![[1.0, 1.0], [2.0, 2.0]].view(),
            &array![[1.0], [2.0]].view(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("u32::MAX"));
    }
}

#[cfg(test)]
mod behavior_tests {
    use super::*;

    #[test]
    fn disk_called() {
        use ndarray::array;
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        crate::backend::disk_observation::check_rows(10).unwrap();
        crate::backend::disk_observation::write_metadata(
            dir.path(),
            0,
            4,
            1,
            false,
            0,
            "bpann_disk",
        )
        .unwrap();
        crate::backend::disk_observation::check_backend(dir.path(), "bpann_disk").unwrap();
        crate::backend::disk_observation::check_dims(4, 1).unwrap();
        assert!(crate::backend::disk_observation::check_dims(
            crate::backend::disk_observation::MAX_DIM + 1,
            1
        )
        .is_err());
        assert_eq!(
            crate::backend::disk_observation::load_rows(dir.path()),
            Some(0)
        );
        assert_eq!(
            crate::backend::disk_observation::load_obs(dir.path()),
            Some(0)
        );
        assert_eq!(
            crate::backend::disk_observation::load_backend(dir.path()).as_deref(),
            Some("bpann_disk")
        );
        let mut yvar = crate::backend::disk_observation::open_yvar(dir.path(), 1, None).unwrap();
        crate::backend::disk_observation::append_yvar(
            dir.path(),
            1,
            &mut yvar,
            Some(&array![[0.2]].view()),
        )
        .unwrap();
        let dirty = Mutex::new(false);
        crate::backend::disk_observation::mark_dirty(&dirty);
        let text = r#"{"indexed_rows":42,"index_backend":"bpann_disk"}"#;
        assert_eq!(
            crate::backend::disk_observation::parse_number(text, "indexed_rows"),
            Some(42)
        );
        assert_eq!(
            crate::backend::disk_observation::parse_string(text, "index_backend").as_deref(),
            Some("bpann_disk")
        );
    }
}
