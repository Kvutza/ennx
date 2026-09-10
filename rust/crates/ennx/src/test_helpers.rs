use crate::index::IndexDriver;
use crate::model::ENN;
use ndarray::array;

pub(crate) fn test_model() -> ENN {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];

    ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap()
}
