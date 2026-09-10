use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum BpannError {
    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

impl From<ndarray::ShapeError> for BpannError {
    fn from(e: ndarray::ShapeError) -> Self {
        BpannError::InvalidParameter(e.to_string())
    }
}
