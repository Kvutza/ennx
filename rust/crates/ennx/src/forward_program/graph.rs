#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticOp {
    Embed,
    Normalize,
    ProjectQkv,
    PisaSummaries,
    PisaSelect,
    PisaAttention,
    ProjectAttention,
    RouteTopK,
    PackRoutes,
    RoutedGateUp,
    RoutedDown,
    CombineRoutes,
    Residual,
    Feedback,
    Readout,
    Sample,
    Draft,
    Verify,
    ScoreGeneration,
    Objective,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exactness {
    Exact,
    Approximate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionBoundary {
    Open,
    CloseAfter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendCapability {
    Fp16,
    Fp32Accumulate,
    PackedWeightOverlay,
    ResidentState,
    IndirectDispatch,
    SubgroupMatrix,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticNode {
    pub name: &'static str,
    pub op: SemanticOp,
    pub inputs: Vec<&'static str>,
    pub outputs: Vec<&'static str>,
    pub exactness: Exactness,
    pub fusion: FusionBoundary,
}
