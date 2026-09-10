//! Disk ENN persist-on-close: multi-fragment ingest, persist, fast reopen.

use std::fs;
use std::time::Instant;

use ennx::backend::EnnStorage;
use ennx::index::IndexDriver;
use ennx::ENN;
use ndarray::{Array2, ArrayView2};
use tempfile::TempDir;

fn append_rows(model: &mut ENN, x: &ArrayView2<f64>, y: &ArrayView2<f64>) {
    model.add(x, y, None).expect("add");
}

fn build_model(work_dir: &std::path::Path, dim: usize) -> ENN {
    let rows = 2500usize;
    let x = Array2::from_shape_fn((rows, dim), |(i, j)| (i + j) as f64);
    let y = Array2::from_shape_fn((rows, 1), |(i, _)| i as f64);
    let model = ENN::new_storage(
        x,
        y,
        None,
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(work_dir.to_path_buf()),
    )
    .expect("reference new");
    model.index_access().ensure_sync().expect("reference sync");
    model
}
fn build_model2(work_dir: &std::path::Path, dim: usize) -> ENN {
    let mut model = ENN::new_empty(
        dim,
        1,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(work_dir.to_path_buf()),
        Some(1000),
    )
    .expect("new_empty");
    for (start, count) in [(0, 1000usize), (1000, 1000usize), (2000, 500usize)] {
        let x = Array2::from_shape_fn((count, dim), |(i, j)| (start + i + j) as f64);
        let y = Array2::from_shape_fn((count, 1), |(i, _)| (start + i) as f64);
        append_rows(&mut model, &x.view(), &y.view());
        model.schedule_flush().expect("flush");
    }
    model.index_access().ensure_sync().expect("sync");
    model
}

fn pages_checksum(work_dir: &std::path::Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let bytes = fs::read(work_dir.join("index/pages.bin")).expect("pages.bin");
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn disk_correct() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().to_path_buf();
    let dim = 32usize;
    let rows = 2500usize;
    let query = Array2::from_shape_fn((1, dim), |(_, j)| j as f64 * 0.01);

    let pre_idx = {
        let model = build_model2(&work_dir, dim);
        let pre_idx = model
            .neighbors(&query.view(), 5, false)
            .expect("neighbors pre-persist");
        model.persist_index().expect("persist");
        let post_persist_idx = model
            .neighbors(&query.view(), 5, false)
            .expect("neighbors post-persist");
        assert_eq!(
            pre_idx, post_persist_idx,
            "in-session neighbors must not change on persist"
        );
        pre_idx
    };

    let header_text = fs::read_to_string(work_dir.join("index/header.json")).expect("header.json");
    assert!(header_text.contains(&format!("\"indexed_rows\": {rows}")));

    let ref_dir = TempDir::new().expect("ref tempdir");
    let ref_model = build_model(ref_dir.path(), dim);
    let ref_idx = ref_model
        .neighbors(&query.view(), 5, false)
        .expect("reference neighbors");

    let t0 = Instant::now();
    let reopened = ENN::new_storage(
        Array2::zeros((0, dim)),
        Array2::zeros((0, 1)),
        None,
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(work_dir.clone()),
    )
    .expect("reopen");
    let reopen_s = t0.elapsed().as_secs_f64();
    assert!(
        reopen_s < 1.0,
        "reopen took {reopen_s:.3}s; expected fast mmap open after persist"
    );
    reopened
        .index_access()
        .ensure_sync()
        .expect("post-reopen sync");
    let post_idx = reopened
        .neighbors(&query.view(), 5, false)
        .expect("neighbors");
    assert_eq!(
        ref_idx, post_idx,
        "post-reopen must match reference disk model"
    );
    let _ = pre_idx;
}

#[test]
fn disk_index2() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().to_path_buf();
    let dim = 32usize;
    let query = Array2::from_shape_fn((1, dim), |(_, j)| j as f64 * 0.01);

    let model = build_model2(&work_dir, dim);
    model.persist_index().expect("persist");
    let checksum_after_persist = pages_checksum(&work_dir);
    let neighbors_after_persist = model.neighbors(&query.view(), 5, false).expect("neighbors");

    model.index_access().ensure_sync().expect("tell sync");

    assert_eq!(
        checksum_after_persist,
        pages_checksum(&work_dir),
        "ensure_sync must not rewrite pages.bin"
    );
    let neighbors_after_sync = model.neighbors(&query.view(), 5, false).expect("neighbors");
    assert_eq!(neighbors_after_persist, neighbors_after_sync);
}

#[test]
fn schedule_persist() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().to_path_buf();
    let dim = 8usize;
    let mut model = ENN::new_empty(
        dim,
        1,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(work_dir.clone()),
        Some(50),
    )
    .expect("new_empty");
    let pages = work_dir.join("index/pages.bin");
    assert!(!pages.exists());

    for start in [0usize, 50, 100] {
        let x = Array2::from_shape_fn((60, dim), |(i, j)| (start + i + j) as f64);
        let y = Array2::from_shape_fn((60, 1), |(i, _)| (start + i) as f64);
        append_rows(&mut model, &x.view(), &y.view());
        model.schedule_flush().expect("schedule");
        model.index_access().ensure_sync().expect("wait");
    }
    model.index_access().ensure_sync().expect("soft drain");
    assert!(
        !pages.exists(),
        "schedule/wait soft sync must not write pages.bin"
    );

    let query = Array2::from_shape_fn((1, dim), |(_, j)| j as f64 * 0.01);
    let idx = model
        .neighbors(&query.view(), 3, false)
        .expect("neighbors searchable after soft sync");
    assert_eq!(idx.ncols(), 3);
    assert!(idx[[0, 0]] < 180);

    model.persist_index().expect("hard persist");
    assert!(pages.exists());
}
