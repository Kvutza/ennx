use ennx::{EnnIndexAccess, EnnRowAccess, EpistemicNearestNeighbors, IndexDriver};
use ndarray::array;

#[test]
fn model_access() {
    let model = EpistemicNearestNeighbors::new(
        array![[0.0]],
        array![[1.0]],
        None,
        false,
        IndexDriver::Exact,
    )
    .unwrap();
    let _: EnnIndexAccess<'_> = model.index_access();
    let _: EnnRowAccess<'_> = model.rows();
    model.index_access().ensure_sync().unwrap();
    assert!(model.index_access().memory_bytes().unwrap() > 0);
    assert!(!model.index_access().is_stale());
    assert_eq!(model.index_access().len(), 1);
    assert_eq!(model.rows().train_rows_at(&[0]).unwrap().0.nrows(), 1);
    assert_eq!(model.rows().row_x(0).unwrap(), array![0.0]);
    assert_eq!(model.rows().row_y(0).unwrap(), array![1.0]);
    assert!(model.rows().row_yvar(0).unwrap().is_none());
}
