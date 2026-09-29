use super::*;
use crate::index::BpannIndex;
use tempfile::TempDir;

#[test]
fn forest_metric() {
    let dir = TempDir::new().unwrap();
    let vectors = (0..32)
        .map(|i| vec![i as f32 / 8.0, (i % 5) as f32])
        .collect::<Vec<_>>();
    let tree = BpannIndex::build_vectors(&vectors, 2, 4, 719, dir.path().to_path_buf()).unwrap();
    let original = tree.pages.clone();
    let mut forest = IncrementalIndex::new(dir.path().to_path_buf());
    forest.indices.push(tree);
    forest.check_rescale(&[0.5, 2.0]).unwrap();
    forest.rescale(&[0.5, 2.0]);
    for (page, old) in forest.indices[0].pages.iter().zip(&original) {
        assert_eq!(page.page_id(), old.page_id());
        for (row, old) in page_vectors(page).iter().zip(page_vectors(old)) {
            assert_eq!(row[0], old[0] * 0.5);
            assert_eq!(row[1], old[1] * 2.0);
        }
    }
    assert!(forest.check_rescale(&[f64::MAX, 2.0]).is_err());
    assert!(forest.check_rescale(&[-1.0, 2.0]).is_err());
}
