use ennx::experimental::{
    quantize_fp4_e2m1, quantize_int4, ComputeBackend, ForwardProgram, KnnPlan, PackedModel,
    SharingPolicy, WeightAsk, WeightBlock,
};
use ennx::prelude::{
    create_optimizer_zero, standardize_y, CandidateRV, ENNParams, EpistemicNearestNeighbors,
    IndexDriver, OptimizerConfig, PosteriorFlags,
};

#[test]
fn prelude_exports_core() {
    let _ = (
        std::mem::size_of::<EpistemicNearestNeighbors>(),
        std::mem::size_of::<ENNParams>(),
        std::mem::size_of::<PosteriorFlags>(),
        std::mem::size_of::<IndexDriver>(),
        std::mem::size_of::<CandidateRV>(),
        std::mem::size_of::<OptimizerConfig>(),
        create_optimizer_zero,
        standardize_y,
    );
}

#[test]
fn quantization_is_experimental() {
    assert_eq!(quantize_int4([0.0, 1.0, 2.0], 1.0), vec![0x10, 0x02]);
    assert_eq!(quantize_fp4_e2m1([0.0, 1.0, 2.0], 1.0), vec![0x20, 0x04]);

    const LIB: &str = include_str!("../src/lib.rs");
    assert!(
        !LIB.contains("pub use quantization"),
        "quantization should not be a root re-export"
    );
}

#[test]
fn frontier_is_experimental() {
    let _ = (
        std::mem::size_of::<ComputeBackend>(),
        std::mem::size_of::<ForwardProgram>(),
        std::mem::size_of::<KnnPlan>(),
        std::mem::size_of::<PackedModel>(),
        std::mem::size_of::<SharingPolicy>(),
        std::mem::size_of::<WeightAsk>(),
        std::mem::size_of::<WeightBlock>(),
    );

    const LIB: &str = include_str!("../src/lib.rs");
    for root in [
        "pub mod forward_program",
        "pub mod forward_weights",
        "pub mod knn",
        "pub mod trials",
        "pub mod weights",
        "pub use forward_program",
        "pub use forward_weights",
        "pub use trials",
        "pub use weights",
    ] {
        assert!(!LIB.contains(root), "{root} should not be root API");
    }
}
