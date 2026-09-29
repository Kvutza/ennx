use super::*;
use ndarray::array;
use tempfile::TempDir;

#[test]
fn disk_metric() {
    let dir = TempDir::new().unwrap();
    let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]];
    let y = array![[0.0], [1.0], [2.0], [3.0]];
    let mut model = ENN::new_storage(
        x.clone(),
        y,
        None,
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .unwrap();
    model
        .neighbors(&array![[0.2, 0.7]].view(), 4, false)
        .unwrap(); // Warm the unscaled flat cache.
    model.learn_metric(vec![], 719).unwrap();
    model.metric_weights(&[1.2, 0.8]).unwrap();
    model.metric_weights(&[9.0, 0.25]).unwrap();
    model.metric_weights(&[4.0, 0.25]).unwrap();
    let query = array![[0.2, 0.7]];
    let (distances, ids) = model
        .index_access()
        .nearest_neighbors(&query.view(), 4, false)
        .unwrap();
    for j in 0..4 {
        let id = ids[[0, j]] as usize;
        let expected = 4.0 * (query[[0, 0]] - x[[id, 0]]).powi(2)
            + 0.25 * (query[[0, 1]] - x[[id, 1]]).powi(2);
        assert!((distances[[0, j]] - expected).abs() < 1e-5);
    }
    model
        .add(&array![[3.0, 2.0]].view(), &array![[4.0]].view(), None)
        .unwrap();
    assert_eq!(model.x_scale(), array![[0.5, 2.0]]); // Appending cannot overwrite the learned metric with input std.
    let snapshot = model.metric_snapshot().unwrap();
    assert_eq!(snapshot.num_seen, 5);
    assert!(snapshot.counters.num_rescales > 0 && snapshot.counters.num_rebuilds > 0);
    model.persist_index().unwrap();
    drop(model);
    // Reopening without AUTO must rebuild unit geometry, not reuse scaled pages.
    let reopened = ENN::new_empty(
        2,
        1,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
        None,
    )
    .unwrap();
    assert_eq!(reopened.len(), 5);
    assert_eq!(reopened.x_scale(), array![[1.0, 1.0]]);
    assert!(reopened.metric_snapshot().is_none());
    let (d, ids) = reopened
        .index_access()
        .nearest_neighbors(&query.view(), 5, false)
        .unwrap();
    for j in 0..5 {
        let row = reopened.rows().row_x(ids[[0, j]] as usize).unwrap();
        let expected = (query[[0, 0]] - row[0]).powi(2) + (query[[0, 1]] - row[1]).powi(2);
        assert!((d[[0, j]] - expected).abs() < 1e-5);
    }
}

#[test]
fn metric_rejection() {
    let mut model =
        ENN::new_empty(2, 1, IndexDriver::Exact, EnnStorage::InMemory, None, None).unwrap();
    assert!(model.learn_metric(vec![], 719).is_err());
    let dir = TempDir::new().unwrap();
    let mut model = ENN::new_empty(
        2,
        1,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
        None,
    )
    .unwrap();
    assert!(model.learn_metric(vec![vec![2]], 719).is_err());
    model.learn_metric(vec![], 719).unwrap();
    assert!(model.metric_weights(&[1.0, 0.0]).is_err());
    assert!(
        model
            .add(&array![[f64::NAN, 0.0]].view(), &array![[1.0]].view(), None)
            .is_err()
    );
    assert_eq!(model.len(), 0);
    assert_eq!(model.metric_snapshot().unwrap().num_seen, 0);
}
