use super::graph::{BackendCapability, SemanticOp};
use super::model::{ModelProgram, ProgramKind};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    CudaOxide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compiler {
    Native,
    CudaOxide,
    Tvm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledKernel {
    pub nodes: Vec<&'static str>,
    pub kernel: &'static str,
    pub tile: [usize; 3],
    pub threads: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSchedule {
    pub backend: Backend,
    pub compiler: Compiler,
    pub capabilities: Vec<BackendCapability>,
    pub kernels: Vec<ScheduledKernel>,
}

impl BackendSchedule {
    pub fn existing_metal() -> Self {
        Self {
            backend: Backend::Metal,
            compiler: Compiler::Native,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["embed"], "fbt_moe_embed", [1, 512, 1], [256, 1, 1]),
                kernel(&["attention-norm"], "fbt_moe_rms", [1, 512, 1], [128, 1, 1]),
                kernel(
                    &["qkv"],
                    "fbt_model_tensorops_qkv",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(
                    &["pisa-summaries"],
                    "fbt_pisa1_leaf_means",
                    [64, 64, 1],
                    [64, 1, 1],
                ),
                kernel(
                    &["pisa-select", "pisa-attention"],
                    "fbt_pisa1_select_attention",
                    [1, 512, 64],
                    [128, 1, 1],
                ),
                kernel(
                    &["attention-project"],
                    "fbt_model_tensorops_output_projection",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(
                    &["attention-residual", "moe-norm"],
                    "fbt_moe_residual_rms",
                    [1, 512, 1],
                    [128, 1, 1],
                ),
                kernel(&["route"], "fbt_moe_select_top3", [1, 128, 1], [128, 1, 1]),
                kernel(
                    &["pack-routes"],
                    "fbt_moe_route_pack",
                    [64, 512, 1],
                    [128, 1, 1],
                ),
                kernel(
                    &["gate-up"],
                    "fbt_moe_routed_gate",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(
                    &["down"],
                    "fbt_moe_routed_down",
                    [128, 64, 216],
                    [128, 1, 1],
                ),
                kernel(
                    &["combine-routes"],
                    "fbt_moe_combine",
                    [1, 512, 3],
                    [128, 1, 1],
                ),
                kernel(
                    &["moe-residual"],
                    "fbt_moe_residual",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["feedback"],
                    "fbt_moe_feedback_fuse",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["readout"],
                    "fbt_model_tensorops_readout",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(&["verify"], "fbt_block_verify", [128, 4, 1], [128, 1, 1]),
            ],
        }
    }

    pub fn cuda_oxide() -> Self {
        Self {
            backend: Backend::CudaOxide,
            compiler: Compiler::CudaOxide,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["embed"], "embed", [1, 512, 1], [256, 1, 1]),
                kernel(
                    &["attention-norm", "moe-norm"],
                    "rms",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["qkv", "attention-project", "readout"],
                    "matmul",
                    [16, 16, 16],
                    [16, 16, 1],
                ),
                kernel(
                    &["pisa-summaries"],
                    "pisaleaves+pisaupper",
                    [64, 64, 1],
                    [64, 1, 1],
                ),
                kernel(&["pisa-select"], "pisa_select", [1, 8, 64], [64, 1, 1]),
                kernel(
                    &["pisa-attention"],
                    "pisa_attention",
                    [1, 8, 512],
                    [256, 1, 1],
                ),
                kernel(&["route"], "routetopk", [1, 625, 1], [256, 1, 1]),
                kernel(
                    &["pack-routes"],
                    "route_counts+route_layout+route_pack",
                    [64, 512, 3],
                    [256, 1, 1],
                ),
                kernel(
                    &["gate-up"],
                    "routed_project+routed_activate",
                    [64, 16, 8],
                    [256, 1, 1],
                ),
                kernel(&["down"], "routed_project", [64, 16, 8], [256, 1, 1]),
                kernel(
                    &["combine-routes"],
                    "combine_routes",
                    [1, 512, 3],
                    [256, 1, 1],
                ),
                kernel(
                    &["attention-residual", "moe-residual"],
                    "residual",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(&["feedback"], "feedback_shift", [1, 512, 1], [256, 1, 1]),
                kernel(&["feedback"], "feedback_norm", [1, 512, 1], [256, 1, 1]),
                kernel(
                    &["feedback"],
                    "matmul(state,gate)",
                    [16, 16, 16],
                    [16, 16, 1],
                ),
                kernel(&["feedback"], "feedback_combine", [1, 512, 1], [256, 1, 1]),
            ],
        }
    }

    /// TVM lowering boundary for stateless matrix operations on Metal.
    /// Stateful PISA, routing, recurrence, verification, and objectives remain
    /// visible as missing coverage until a parity-qualified lowering exists.
    pub fn tvm_metal() -> Self {
        Self::tvm(Backend::Metal, [32, 8, 16], [32, 1, 1])
    }

    /// TVM lowering boundary for stateless matrix operations on CUDA.
    pub fn tvm_cuda() -> Self {
        Self::tvm(Backend::CudaOxide, [16, 16, 16], [256, 1, 1])
    }

    fn tvm(backend: Backend, tile: [usize; 3], threads: [usize; 3]) -> Self {
        Self {
            backend,
            compiler: Compiler::Tvm,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["qkv"], "tvm.project-qkv", tile, threads),
                kernel(
                    &["attention-project"],
                    "tvm.project-attention",
                    tile,
                    threads,
                ),
                kernel(&["gate-up"], "tvm.routed-gate-up", tile, threads),
                kernel(&["down"], "tvm.routed-down", tile, threads),
                kernel(&["readout"], "tvm.readout", tile, threads),
            ],
        }
    }

    pub fn validate(&self, program: &ModelProgram) -> Result<(), String> {
        program.validate()?;
        let capabilities = self.capabilities.iter().copied().collect::<HashSet<_>>();
        if program
            .requirements
            .iter()
            .any(|required| !capabilities.contains(required))
        {
            return Err("backend schedule does not satisfy the model program".into());
        }
        for kernel in &self.kernels {
            if kernel.kernel.is_empty()
                || kernel.nodes.is_empty()
                || kernel.tile.contains(&0)
                || kernel.threads.contains(&0)
                || kernel.nodes.iter().any(|name| program.node(name).is_none())
            {
                return Err(format!("kernel {} is invalid", kernel.kernel));
            }
        }
        Ok(())
    }

    pub fn coverage(
        &self,
        program: &ModelProgram,
        kind: ProgramKind,
    ) -> Result<Vec<SemanticOp>, String> {
        self.validate(program)?;
        let scheduled = self
            .kernels
            .iter()
            .flat_map(|kernel| kernel.nodes.iter().copied())
            .collect::<HashSet<_>>();
        let function = program
            .function(kind)
            .ok_or("model program function is missing")?;
        Ok(function
            .nodes
            .iter()
            .filter(|name| !scheduled.contains(*name))
            .filter_map(|name| program.node(name).map(|node| node.op))
            .collect())
    }

    pub fn partitions(&self) -> HashMap<&'static str, Vec<&'static str>> {
        self.kernels
            .iter()
            .map(|kernel| (kernel.kernel, kernel.nodes.clone()))
            .collect()
    }
}

fn kernel(
    nodes: &[&'static str],
    name: &'static str,
    tile: [usize; 3],
    threads: [usize; 3],
) -> ScheduledKernel {
    ScheduledKernel {
        nodes: nodes.to_vec(),
        kernel: name,
        tile,
        threads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules() {
        let program = ModelProgram::fbt_pisa();
        BackendSchedule::existing_metal()
            .validate(&program)
            .unwrap();
        BackendSchedule::cuda_oxide().validate(&program).unwrap();
        BackendSchedule::tvm_metal().validate(&program).unwrap();
        BackendSchedule::tvm_cuda().validate(&program).unwrap();
        let missing = BackendSchedule::cuda_oxide()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(!missing.contains(&SemanticOp::Feedback));
        assert!(!missing.contains(&SemanticOp::PackRoutes));
        assert!(!missing.contains(&SemanticOp::CombineRoutes));
        assert!(!missing.contains(&SemanticOp::PisaSelect));
        assert!(!missing.contains(&SemanticOp::RouteTopK));
        let metal_missing = BackendSchedule::existing_metal()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(metal_missing.is_empty());
        let tvm_missing = BackendSchedule::tvm_cuda()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(tvm_missing.contains(&SemanticOp::PisaAttention));
        assert!(!tvm_missing.contains(&SemanticOp::ProjectQkv));
    }
}
