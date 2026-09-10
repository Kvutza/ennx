//! Integration tests for EnnBackend dispatch and disk storage.

use ennx::{EnnStorage, IndexDriver, ENN};
use ndarray::array;
use tempfile::TempDir;

#[test]
fn memory_accessors() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let train_y = array![[0.0], [1.0], [2.0]];
    let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    assert_eq!(model.index_access().len(), 3);
    assert_eq!(model.num_dim(), 2);
    let x0 = model.rows().row_x(0).unwrap();
    assert!((x0[0] - 0.0).abs() < 1e-12);
    let y1 = model.rows().row_y(1).unwrap();
    assert!((y1[0] - 1.0).abs() < 1e-12);
    let (x_at, y_at, _) = model.rows().train_rows(&[0, 2]).unwrap();
    assert_eq!(x_at.nrows(), 2);
    assert_eq!(y_at.nrows(), 2);
    let mem = model.index_access().memory_bytes().unwrap();
    assert!(mem > 0);
    let query = array![[0.1, 0.1]];
    let idx = model.neighbors(&query.view(), 2, false).unwrap();
    assert_eq!(idx.nrows(), 1);
}

#[test]
fn new_add() {
    let mut model =
        ENN::new_empty(2, 1, IndexDriver::Exact, EnnStorage::InMemory, None, None).unwrap();
    model
        .add(&array![[1.0, 2.0]].view(), &array![[3.0]].view(), None)
        .unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.rows().row_x(0).unwrap()[0], 1.0);
}

#[test]
fn disk_search() {
    let dir = TempDir::new().expect("tempdir");
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let model = ENN::new_storage(
        train_x.clone(),
        train_y,
        None,
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .unwrap();
    assert_eq!(model.len(), 4);
    let (_, y_all, _) = model
        .rows()
        .train_rows(&(0..4).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(y_all.nrows(), 4);
    let query = array![[0.9, 0.9]];
    let idx = model.neighbors(&query.view(), 1, false).unwrap();
    assert_eq!(idx[[0, 0]], 3);
    let mem = ENN::new(
        train_x,
        array![[0.0], [1.0], [1.0], [2.0]],
        None,
        false,
        IndexDriver::Exact,
    )
    .unwrap();
    let exact = mem.neighbors(&query.view(), 1, false).unwrap();
    assert_eq!(idx[[0, 0]], exact[[0, 0]]);
}

#[test]
fn disk_driver2() {
    let dir = TempDir::new().expect("tempdir");
    match ENN::new_storage(
        array![[0.0, 0.0]],
        array![[0.0]],
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    ) {
        Ok(_) => panic!("expected disk + Exact to error"),
        Err(e) => assert!(e.to_string().contains("BpAnnDisk")),
    }
}

#[test]
fn x_disk() {
    let dir = TempDir::new().expect("tempdir");
    match ENN::new_storage(
        array![[0.0, 0.0]],
        array![[0.0]],
        None,
        true,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    ) {
        Ok(_) => panic!("expected scale_x + BpAnnDisk to error"),
        Err(e) => assert!(
            e.to_string()
                .contains("scale_x=True is not compatible with BPANN_DISK"),
            "unexpected error: {e}"
        ),
    }
}
