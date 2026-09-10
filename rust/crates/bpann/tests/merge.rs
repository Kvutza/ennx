use bpann::mmap_store::MmapColumnStore;
use ndarray::array;
use tempfile::TempDir;

#[test]
fn merge_candidates() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let merged = bpann::merge::merge_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0), (1, 1.0)],
        &[],
        1,
        2,
        true,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    assert_eq!(merged[0].0, 1);
}

#[test]
fn merge_nearest() {
    let merged = bpann::merge::merge_dist(&[(0, 0.0), (1, 1.0), (2, 4.0)], &[], 1, 3, true);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].0, 1);
}
