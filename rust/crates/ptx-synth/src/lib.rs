//! `ptx-synth`: Type-driven symbolic PTX code synthesizer for NVIDIA GPUs.
//!
//! Emits PTX recipes for hardware validation. Timing, occupancy, and physical
//! register use must be measured after assembly on the target device.

pub mod emit;
pub mod gemm;
pub mod hierarchy;
pub mod ir;
pub mod layout;
pub mod reg;
pub mod turing;

pub use emit::PtxEmitter;
pub use hierarchy::{
    HierarchicalAttentionConfig, HierarchicalAttentionOutput, build_leaf_summaries,
    fine_attention_reference, mass_preserving_hierarchical_attention,
};
pub use ir::{PtxKernelBuilder, RegAlloc, TargetArch};
pub use layout::{BankConflictReport, SwizzleMode, TileLayout};
pub use turing::{
    ProbeOp, ProbeShape, TuringGemmConfig, synthesize_turing_fp16_gemm,
    synthesize_turing_fused_rademacher, synthesize_turing_probe, synthesize_turing_vector_scale,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turing_vector_scale_emission() {
        let ptx = synthesize_turing_vector_scale("turing_vector_scale_test");
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains("ld.global.nc.v4.f32"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("st.global.v4.f32"));
        assert!(ptx.contains(".maxnreg 32"));
    }

    #[test]
    fn test_turing_gemm_emission() {
        let config = TuringGemmConfig::default();
        let ptx = synthesize_turing_fp16_gemm(&config);
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"));
        assert!(ptx.contains(".reqntid 256, 1, 1"));
    }

    #[test]
    fn test_turing_fused_rademacher_emission() {
        let ptx = synthesize_turing_fused_rademacher("turing_rademacher_test");
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains("ld.global.nc.v4.f32"));
        assert!(ptx.contains("st.global.v4.f32"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn test_mass_preserving_hierarchy_parity() {
        let temp_dir = std::env::temp_dir();
        let test_artifact = temp_dir.join("hierarchy_parity_test.json");
        let result = hierarchy::run_hierarchy_parity_check(&test_artifact);
        assert!(
            result.is_ok(),
            "Hierarchy parity check failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_zero_bank_conflicts() {
        let layout = TileLayout::new(32, 32, 4, SwizzleMode::None);
        let warp_indices = (0..32).map(|i| (0, i));
        let result = layout.verify_warp_access(warp_indices);
        assert!(
            result.is_ok(),
            "Expected 0 bank conflicts for contiguous row access"
        );
    }
}
