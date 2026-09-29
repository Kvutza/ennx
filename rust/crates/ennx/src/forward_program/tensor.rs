#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F16,
    Bf16,
    F32,
    U32,
    Packed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    Scalar,
    Sequence,
    SequenceHidden,
    HeadSequenceWidth,
    RoutedRows,
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorStorage {
    Host,
    Device,
    WeightArena,
    Overlay,
    State,
    Scratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLifetime {
    Invocation,
    Round,
    Sequence,
    Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    Fixed(usize),
    Symbol(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub shape: Vec<Extent>,
    pub dtype: TensorDType,
    pub layout: TensorLayout,
    pub storage: TensorStorage,
    pub lifetime: TensorLifetime,
}
