//! Experimental contract for resident, fused model evaluation.
//!
//! Academic References & Citations:
//! - Epistemic Neural Networks (ENN): Osband et al. (2021) "Epistemic Neural Networks", NeurIPS.
//! - Linear Attention & Recurrence: Gu & Dao (2023) "Mamba", De et al. (2024) "Griffin".
//! - Sparsely Gated MoE: Shazeer et al. (2017) "Outrageously Large Neural Networks".
//! - Hardware-Resident BO: Bafna et al. (2026) "Hardware-Resident Bayesian Optimization in ENNX".

use crate::trials::{Ask, Leaf, Search, Trial};
use crate::weights::ComputeDevice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardOp {
    EmbeddingLookup,
    PerturbPackedWeights,
    Dequantize,
    Matmul,
    KdaStateUpdate,
    RouteExperts,
    Activation,
    Normalize,
    Residual,
    ReduceObjective,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardProgram {
    ops: Vec<ForwardOp>,
    version: u32,
}

impl ForwardProgram {
    pub const VERSION: u32 = 1;

    pub fn new(ops: impl Into<Vec<ForwardOp>>) -> Result<Self, String> {
        let ops = ops.into();
        if !ops.contains(&ForwardOp::PerturbPackedWeights) {
            return Err("forward program must perturb packed weights".to_string());
        }
        if !ops.contains(&ForwardOp::Matmul) {
            return Err("forward program must contain a matrix multiply".to_string());
        }
        if !ops.contains(&ForwardOp::ReduceObjective) {
            return Err("forward program must reduce an objective".to_string());
        }
        Ok(Self {
            ops,
            version: Self::VERSION,
        })
    }

    pub fn kda() -> Result<Self, String> {
        Self::new([
            ForwardOp::EmbeddingLookup,
            ForwardOp::PerturbPackedWeights,
            ForwardOp::Dequantize,
            ForwardOp::Matmul,
            ForwardOp::KdaStateUpdate,
            ForwardOp::Activation,
            ForwardOp::Residual,
            ForwardOp::ReduceObjective,
        ])
    }

    pub fn ops(&self) -> &[ForwardOp] {
        &self.ops
    }

    pub const fn version(&self) -> u32 {
        self.version
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidentRound {
    pub trial: Trial,
    pub program_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkAxis {
    Token,
    Output,
    Input,
    Head,
    Expert,
    State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkTile {
    pub axis: WorkAxis,
    pub items: usize,
}

impl WorkTile {
    pub fn new(axis: WorkAxis, items: usize) -> Result<Self, String> {
        if items == 0 {
            return Err("kernel tile size must be positive".to_string());
        }
        Ok(Self { axis, items })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkGrid {
    pub items: usize,
    pub threads_per_group: usize,
}

impl WorkGrid {
    pub fn new(items: usize, threads_per_group: usize) -> Result<Self, String> {
        if items == 0 || threads_per_group == 0 {
            return Err("kernel work grid dimensions must be positive".to_string());
        }
        Ok(Self {
            items,
            threads_per_group,
        })
    }

    pub fn groups(&self) -> usize {
        self.items.div_ceil(self.threads_per_group)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub kernel: &'static str,
    pub grid: WorkGrid,
    pub tiles: Vec<WorkTile>,
    pub scratch_bytes: usize,
}

impl KernelPlan {
    pub fn new(
        kernel: &'static str,
        items: usize,
        threads_per_group: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            kernel,
            grid: WorkGrid::new(items, threads_per_group)?,
            tiles: Vec::new(),
            scratch_bytes: 0,
        })
    }

    pub fn with_tile(mut self, axis: WorkAxis, items: usize) -> Result<Self, String> {
        self.tiles.push(WorkTile::new(axis, items)?);
        Ok(self)
    }

    pub fn with_scratch_bytes(mut self, bytes: usize) -> Self {
        self.scratch_bytes = bytes;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedAffinePlan {
    pub token_tile: usize,
    pub output_tile: usize,
    pub input_tile: usize,
    pub threads_per_group: usize,
}

impl PackedAffinePlan {
    pub fn apple_simd() -> Self {
        Self {
            token_tile: 1,
            output_tile: 16,
            input_tile: 64,
            threads_per_group: 256,
        }
    }

    pub fn validate(&self, linear: KdaPackedLinear) -> Result<(), String> {
        linear.validate()?;
        if self.token_tile == 0
            || self.output_tile == 0
            || self.input_tile == 0
            || self.threads_per_group == 0
        {
            return Err("packed affine schedule dimensions must be positive".to_string());
        }
        if !linear.input_width.is_multiple_of(self.input_tile) {
            return Err("packed affine input width must be divisible by input tile".to_string());
        }
        Ok(())
    }

    pub fn kernel_plan(
        &self,
        kernel: &'static str,
        tokens: usize,
        linear: KdaPackedLinear,
    ) -> Result<KernelPlan, String> {
        self.validate(linear)?;
        let items = tokens
            .checked_mul(linear.output_width)
            .ok_or("packed affine dispatch size overflow")?;
        KernelPlan::new(kernel, items, self.threads_per_group)?
            .with_tile(WorkAxis::Token, self.token_tile)?
            .with_tile(WorkAxis::Output, self.output_tile)?
            .with_tile(WorkAxis::Input, self.input_tile)
    }
}

pub trait ForwardEvaluator {
    fn evaluate(&mut self, round: &ResidentRound, program: &ForwardProgram) -> Result<f32, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaTensorLayout {
    pub batch: usize,
    pub sequence_length: usize,
    pub heads: usize,
    pub key_width: usize,
    pub value_width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaPackedLinear {
    pub byte_offset: usize,
    pub scale_offset: usize,
    pub bias_offset: usize,
    pub input_width: usize,
    pub output_width: usize,
    pub bits: u8,
    pub group_size: usize,
    pub element_offset: usize,
    pub perturb_whole: u32,
    pub perturb_threshold: u32,
}

impl KdaPackedLinear {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_width == 0 || self.output_width == 0 || self.group_size == 0 {
            return Err("packed KDA linear dimensions must be positive".to_string());
        }
        if self.bits != 4 && self.bits != 8 {
            return Err(format!(
                "packed KDA linear bits must be 4 or 8, got {}",
                self.bits
            ));
        }
        Ok(())
    }

    pub fn packed_bytes(&self) -> usize {
        let bytes_per_row = (self.input_width * self.bits as usize).div_ceil(8);
        bytes_per_row * self.output_width
    }

    pub fn groups_per_row(&self) -> usize {
        self.input_width.div_ceil(self.group_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaForwardRequest {
    pub tensor: KdaTensorLayout,
    pub qkv: KdaPackedLinear,
    pub control: KdaPackedLinear,
    pub output: KdaPackedLinear,
    pub seed: u64,
}

/// Remaining packed projections and vectors needed to turn a KDA recurrence
/// into a complete attention update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaControlRequest {
    pub qkv_conv: KdaPackedLinear,
    pub control: KdaPackedLinear,
    pub forget: KdaPackedLinear,
    pub output_gate: KdaPackedLinear,
    pub decay: KdaPackedLinear,
    pub time_bias: KdaPackedLinear,
    pub output_norm: KdaPackedLinear,
    pub output: KdaPackedLinear,
    pub gate_rank: usize,
}

impl KdaControlRequest {
    pub fn validate(&self, tensor: KdaTensorLayout, hidden_width: usize) -> Result<(), String> {
        for linear in [
            self.qkv_conv,
            self.control,
            self.forget,
            self.output_gate,
            self.decay,
            self.time_bias,
            self.output_norm,
            self.output,
        ] {
            linear.validate()?;
        }
        let width = tensor.heads * tensor.value_width;
        if self.qkv_conv.input_width != 4
            || self.qkv_conv.output_width != 3 * width
            || self.qkv_conv.bits != 8
            || self.qkv_conv.group_size != 4
        {
            return Err("KDA QKV convolution must contain four INT8 taps per channel".to_string());
        }
        if self.control.input_width != hidden_width
            || self.control.output_width != 2 * self.gate_rank + tensor.heads
        {
            return Err("KDA control projection shape is invalid".to_string());
        }
        if self.forget.input_width != self.gate_rank || self.forget.output_width != width {
            return Err("KDA forget projection shape is invalid".to_string());
        }
        if self.output_gate.input_width != self.gate_rank || self.output_gate.output_width != width
        {
            return Err("KDA output-gate projection shape is invalid".to_string());
        }
        if self.output.input_width != width || self.output.output_width != hidden_width {
            return Err("KDA output projection shape is invalid".to_string());
        }
        if self.decay.input_width != tensor.heads
            || self.decay.output_width != 1
            || self.time_bias.input_width != width
            || self.time_bias.output_width != 1
            || self.output_norm.input_width != tensor.value_width
            || self.output_norm.output_width != 1
        {
            return Err("KDA recurrent vector shapes are invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaDispatch {
    ProjectQkv { threads: usize },
    SplitQkv { threads: usize },
    Recur { threads: usize },
}

/// Packed descriptors for one complete KDA-MoE decoder layer.
///
/// This is deliberately a data-only ABI: the Rust Metal device owns buffers
/// and command encoding, while Python only supplies the model layout once.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaMoeLayerRequest {
    pub kda: KdaForwardRequest,
    pub attention_norm: KdaPackedLinear,
    pub moe_norm: KdaPackedLinear,
    pub router: KdaPackedLinear,
    pub expert_gate: Vec<KdaPackedLinear>,
    pub expert_up: Vec<KdaPackedLinear>,
    pub expert_down: Vec<KdaPackedLinear>,
    pub top_k: usize,
    pub residual_scale: f32,
    pub rms_epsilon: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaMoeDispatch {
    AttentionNorm { threads: usize },
    Kda(KdaDispatch),
    AttentionResidual { threads: usize },
    MoeNorm { threads: usize },
    RouteExperts { threads: usize },
    ExpertGateUp { threads: usize },
    ExpertDown { threads: usize },
    MoeResidual { threads: usize },
}

impl KdaMoeDispatch {
    pub fn kernel_plan(&self, affine: PackedAffinePlan) -> Result<KernelPlan, String> {
        match *self {
            KdaMoeDispatch::AttentionNorm { threads } => {
                KernelPlan::new("decoder_rms_norm", threads, 256)?
                    .with_tile(WorkAxis::Token, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::Kda(dispatch) => dispatch.kernel_plan(affine),
            KdaMoeDispatch::AttentionResidual { threads }
            | KdaMoeDispatch::MoeResidual { threads } => {
                KernelPlan::new("decoder_residual", threads, 256)?.with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::MoeNorm { threads } => {
                KernelPlan::new("decoder_rms_norm", threads, 256)?
                    .with_tile(WorkAxis::Token, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::RouteExperts { threads } => {
                KernelPlan::new("moe_router_topk", threads, 1)?.with_tile(WorkAxis::Token, 1)
            }
            KdaMoeDispatch::ExpertGateUp { threads } => {
                KernelPlan::new("moe_gate_up", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Expert, 1)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
            KdaMoeDispatch::ExpertDown { threads } => {
                KernelPlan::new("moe_down", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Output, affine.output_tile)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
        }
    }
}

impl KdaMoeLayerRequest {
    pub fn validate(&self) -> Result<(), String> {
        self.kda.validate()?;
        for linear in [self.attention_norm, self.moe_norm, self.router] {
            linear.validate()?;
        }
        let hidden = self.kda.qkv.input_width;
        if self.attention_norm.input_width != hidden || self.attention_norm.output_width != 1 {
            return Err("attention norm must be a hidden-width vector".to_string());
        }
        if self.moe_norm.input_width != hidden || self.moe_norm.output_width != 1 {
            return Err("MoE norm must be a hidden-width vector".to_string());
        }
        let experts = self.expert_gate.len();
        if experts == 0 || self.expert_up.len() != experts || self.expert_down.len() != experts {
            return Err(
                "KDA-MoE expert descriptor lists must be non-empty and equal in length".to_string(),
            );
        }
        if self.top_k == 0 || self.top_k > experts || self.top_k > 8 {
            return Err("KDA-MoE top_k must be in 1..=min(experts, 8)".to_string());
        }
        if !self.residual_scale.is_finite() || !self.rms_epsilon.is_finite() {
            return Err("KDA-MoE numeric parameters must be finite".to_string());
        }
        if self.rms_epsilon <= 0.0 {
            return Err("KDA-MoE RMS epsilon must be positive".to_string());
        }
        if self.router.input_width != hidden || self.router.output_width != experts {
            return Err("router dimensions must be hidden_width by num_experts".to_string());
        }
        for ((gate, up), down) in self
            .expert_gate
            .iter()
            .zip(&self.expert_up)
            .zip(&self.expert_down)
        {
            gate.validate()?;
            up.validate()?;
            down.validate()?;
            if gate.input_width != hidden
                || up.input_width != hidden
                || gate.output_width != up.output_width
            {
                return Err(
                    "expert gate and up projections must share hidden input and expert width"
                        .to_string(),
                );
            }
            if down.input_width != gate.output_width || down.output_width != hidden {
                return Err(
                    "expert down projection must map expert width to hidden width".to_string(),
                );
            }
        }
        Ok(())
    }

    pub fn hidden_elements(&self) -> usize {
        self.kda.tensor.batch * self.kda.tensor.sequence_length * self.kda.qkv.input_width
    }

    pub fn expert_activation_elements(&self) -> usize {
        self.kda.tensor.batch
            * self.kda.tensor.sequence_length
            * self.top_k
            * self.expert_gate[0].output_width
    }

    pub fn encode(&self) -> Result<Vec<KdaMoeDispatch>, String> {
        self.validate()?;
        let kda = KdaEncoder::new(self.kda)?;
        let tokens = self.kda.tensor.batch * self.kda.tensor.sequence_length;
        let hidden = self.hidden_elements();
        Ok(vec![
            KdaMoeDispatch::AttentionNorm { threads: tokens },
            KdaMoeDispatch::Kda(kda.encode()[0]),
            KdaMoeDispatch::Kda(kda.encode()[1]),
            KdaMoeDispatch::Kda(kda.encode()[2]),
            KdaMoeDispatch::AttentionResidual { threads: hidden },
            KdaMoeDispatch::MoeNorm { threads: tokens },
            KdaMoeDispatch::RouteExperts { threads: tokens },
            KdaMoeDispatch::ExpertGateUp {
                threads: self.expert_activation_elements(),
            },
            KdaMoeDispatch::ExpertDown { threads: hidden },
            KdaMoeDispatch::MoeResidual { threads: hidden },
        ])
    }

    pub fn kernel_plans(&self, affine: PackedAffinePlan) -> Result<Vec<KernelPlan>, String> {
        self.validate()?;
        affine.validate(self.kda.qkv)?;
        affine.validate(self.router)?;
        for linear in self
            .expert_gate
            .iter()
            .chain(&self.expert_up)
            .chain(&self.expert_down)
        {
            affine.validate(*linear)?;
        }
        self.encode()?
            .into_iter()
            .map(|dispatch| dispatch.kernel_plan(affine))
            .collect()
    }
}

pub struct KdaEncoder {
    request: KdaForwardRequest,
}

impl KdaEncoder {
    pub fn new(request: KdaForwardRequest) -> Result<Self, String> {
        request.validate()?;
        Ok(Self { request })
    }

    pub fn request(&self) -> KdaForwardRequest {
        self.request
    }

    pub fn encode(&self) -> [KdaDispatch; 3] {
        [
            KdaDispatch::ProjectQkv {
                threads: self.request.projection_dispatch_threads(),
            },
            KdaDispatch::SplitQkv {
                threads: self.request.tensor.qkv_elements(),
            },
            KdaDispatch::Recur {
                threads: self.request.recurrence_dispatch_threads(),
            },
        ]
    }
}

impl KdaDispatch {
    pub fn kernel_plan(&self, affine: PackedAffinePlan) -> Result<KernelPlan, String> {
        match *self {
            KdaDispatch::ProjectQkv { threads } => {
                KernelPlan::new("kda_project_packed_16k", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Output, affine.output_tile)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
            KdaDispatch::SplitQkv { threads } => {
                KernelPlan::new("kda_split_qkv_16k", threads, 256)?
                    .with_tile(WorkAxis::Head, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaDispatch::Recur { threads } => KernelPlan::new("kda_recurrence_16k", threads, 256)?
                .with_tile(WorkAxis::State, 1)?
                .with_tile(WorkAxis::Token, 16_384)
                .map(|plan| plan.with_scratch_bytes(0)),
        }
    }
}

impl KdaForwardRequest {
    pub fn qkv_output_width(&self) -> usize {
        3 * self.tensor.heads * self.tensor.key_width
    }

    pub fn projection_dispatch_threads(&self) -> usize {
        self.tensor.batch * self.tensor.sequence_length * self.qkv_output_width()
    }

    pub fn recurrence_dispatch_threads(&self) -> usize {
        self.tensor.batch * self.tensor.heads * self.tensor.value_width
    }

    pub fn validate(&self) -> Result<(), String> {
        for linear in [self.qkv, self.control, self.output] {
            linear.validate()?;
        }
        let expected_qkv = self.qkv_output_width();
        if self.qkv.input_width != self.output.input_width {
            return Err("KDA projection input widths do not match".to_string());
        }
        if self.qkv.output_width != expected_qkv {
            return Err(format!(
                "KDA QKV output width must be {expected_qkv}, got {}",
                self.qkv.output_width
            ));
        }
        Ok(())
    }
}

impl KdaTensorLayout {
    pub fn new(
        batch: usize,
        sequence_length: usize,
        heads: usize,
        key_width: usize,
        value_width: usize,
    ) -> Result<Self, String> {
        if batch == 0 || sequence_length == 0 || heads == 0 || key_width == 0 || value_width == 0 {
            return Err("KDA tensor dimensions must be positive".to_string());
        }
        Ok(Self {
            batch,
            sequence_length,
            heads,
            key_width,
            value_width,
        })
    }

    pub fn qkv_elements(&self) -> usize {
        self.batch * self.sequence_length * self.heads * self.key_width
    }

    pub fn value_elements(&self) -> usize {
        self.batch * self.sequence_length * self.heads * self.value_width
    }

    pub fn state_elements(&self) -> usize {
        self.batch * self.heads * self.key_width * self.value_width
    }
}

/// Rust-owned BO session for a resident forward evaluator.
///
/// `Search` owns the device model slots and history. The forward device binds
/// the selected slot directly, so candidate weights are materialized once and
/// are not regenerated or perturbed a second time during evaluation.
pub struct ResidentBoState {
    search: Search,
    program: ForwardProgram,
    pending: Option<ResidentRound>,
    rewards: Vec<f32>,
}

impl ResidentBoState {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        history_capacity: usize,
        device: ComputeDevice,
        program: ForwardProgram,
    ) -> Result<Self, String> {
        Ok(Self {
            search: Search::new(base, base_value, leaves, history_capacity, device)?,
            program,
            pending: None,
            rewards: Vec::with_capacity(history_capacity),
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<ResidentRound, String> {
        if self.pending.is_some() {
            return Err("tell must finish the resident forward round before ask".to_string());
        }
        let trial = self.search.ask(seeds, config)?;
        let round = ResidentRound {
            trial,
            program_version: self.program.version(),
        };
        self.pending = Some(round.clone());
        Ok(round)
    }

    pub fn tell(&mut self, round: ResidentRound, reward: f32, accept: bool) -> Result<(), String> {
        let pending = self
            .pending
            .take()
            .ok_or("no resident forward round is pending")?;
        if pending != round {
            self.pending = Some(pending);
            return Err("resident forward round does not match pending trial".to_string());
        }
        self.search.tell(round.trial, reward, accept)?;
        self.rewards.push(reward);
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn bind_pending_metal_row(
        &self,
        round: &ResidentRound,
        executor: &mut crate::forward_metal::KdaMoeMetalExecutor,
    ) -> Result<(), String> {
        let pending = self
            .pending
            .as_ref()
            .ok_or("no resident forward round is pending")?;
        if pending != round {
            return Err("resident forward round does not match pending trial".to_string());
        }
        let (rows, offset) = self.search.pending_metal_row(round.trial)?;
        executor.bind_resident_row(&rows, offset, self.search.row_bytes())
    }

    pub fn ask_evaluate<E: ForwardEvaluator>(
        &mut self,
        seeds: &[u64],
        config: Ask,
        evaluator: &mut E,
        accept: bool,
    ) -> Result<(ResidentRound, f32), String> {
        let round = self.ask(seeds, config)?;
        let reward = match evaluator.evaluate(&round, &self.program) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        self.tell(round.clone(), reward, accept)?;
        Ok((round, reward))
    }

    pub fn program(&self) -> &ForwardProgram {
        &self.program
    }

    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ForwardEvaluator, ForwardOp, ForwardProgram, KdaDispatch, KdaEncoder, KdaForwardRequest,
        KdaMoeDispatch, KdaMoeLayerRequest, KdaPackedLinear, KdaTensorLayout, PackedAffinePlan,
        ResidentBoState, ResidentRound, WorkAxis,
    };
    use crate::trials::{Ask, Leaf};
    use crate::weights::ComputeDevice;

    fn leaves() -> Vec<Leaf> {
        vec![Leaf::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()]
    }

    #[test]
    fn kda_program_is_a_valid_fused_program() {
        let program = ForwardProgram::kda().unwrap();
        assert_eq!(program.version(), ForwardProgram::VERSION);
        assert!(program.ops().contains(&ForwardOp::KdaStateUpdate));
    }

    #[test]
    fn kda_16k_layout_has_resident_state_shape() {
        let layout = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        assert_eq!(layout.qkv_elements(), 16_777_216);
        assert_eq!(layout.value_elements(), 16_777_216);
        assert_eq!(layout.state_elements(), 131_072);
    }

    #[test]
    fn kda_forward_request_validates_packed_projection_layout() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let linear = KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width: 1024,
            output_width: 1024,
            bits: 8,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let request = KdaForwardRequest {
            tensor,
            qkv: KdaPackedLinear {
                output_width: 3 * 8 * 128,
                ..linear
            },
            control: linear,
            output: linear,
            seed: 7,
        };
        request.validate().unwrap();
        assert_eq!(request.qkv.packed_bytes(), 3 * 8 * 128 * 1024);
        let dispatches = KdaEncoder::new(request).unwrap().encode();
        assert_eq!(
            dispatches[0],
            KdaDispatch::ProjectQkv {
                threads: 50_331_648
            }
        );
        assert_eq!(
            dispatches[1],
            KdaDispatch::SplitQkv {
                threads: 16_777_216
            }
        );
        assert_eq!(dispatches[2], KdaDispatch::Recur { threads: 1_024 });
    }

    #[test]
    fn kda_moe_layer_encodes_a_resident_decoder_schedule() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let hidden = 1_024;
        let expert = 576;
        let linear = |input_width, output_width| KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width,
            output_width,
            bits: 4,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let kda = KdaForwardRequest {
            tensor,
            qkv: linear(hidden, hidden * 3),
            control: linear(hidden, 264),
            output: linear(hidden, hidden),
            seed: 7,
        };
        let layer = KdaMoeLayerRequest {
            kda,
            attention_norm: linear(hidden, 1),
            moe_norm: linear(hidden, 1),
            router: linear(hidden, 32),
            expert_gate: vec![linear(hidden, expert); 32],
            expert_up: vec![linear(hidden, expert); 32],
            expert_down: vec![linear(expert, hidden); 32],
            top_k: 4,
            residual_scale: 0.22,
            rms_epsilon: 1.0e-6,
        };
        let dispatches = layer.encode().unwrap();
        assert_eq!(dispatches.len(), 10);
        assert!(matches!(
            dispatches[6],
            KdaMoeDispatch::RouteExperts { threads: 16_384 }
        ));
        assert!(matches!(dispatches[7], KdaMoeDispatch::ExpertGateUp { .. }));
    }

    #[test]
    fn kda_moe_layer_exposes_backend_kernel_plans() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let hidden = 1_024;
        let expert = 576;
        let linear = |input_width, output_width| KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width,
            output_width,
            bits: 4,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let layer = KdaMoeLayerRequest {
            kda: KdaForwardRequest {
                tensor,
                qkv: linear(hidden, hidden * 3),
                control: linear(hidden, 264),
                output: linear(hidden, hidden),
                seed: 7,
            },
            attention_norm: linear(hidden, 1),
            moe_norm: linear(hidden, 1),
            router: linear(hidden, 32),
            expert_gate: vec![linear(hidden, expert); 32],
            expert_up: vec![linear(hidden, expert); 32],
            expert_down: vec![linear(expert, hidden); 32],
            top_k: 4,
            residual_scale: 0.22,
            rms_epsilon: 1.0e-6,
        };
        let plans = layer.kernel_plans(PackedAffinePlan::apple_simd()).unwrap();
        assert_eq!(plans.len(), 10);
        assert_eq!(plans[1].kernel, "kda_project_packed_16k");
        assert_eq!(plans[1].grid.threads_per_group, 256);
        assert!(plans[1]
            .tiles
            .iter()
            .any(|tile| tile.axis == WorkAxis::Input && tile.items == 64));
        assert_eq!(plans[3].kernel, "kda_recurrence_16k");
        assert!(plans[3]
            .tiles
            .iter()
            .any(|tile| tile.axis == WorkAxis::Token && tile.items == 16_384));
    }

    #[test]
    fn resident_state_preserves_ask_tell_contract() {
        let mut state = ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::Cpu,
            ForwardProgram::kda().unwrap(),
        )
        .unwrap();
        let round = state
            .ask(
                &[7],
                Ask {
                    neighbors: 1,
                    ..Ask::default()
                },
            )
            .unwrap();
        state.tell(round, 1.25, true).unwrap();
        assert_eq!(state.rewards(), &[1.25]);
    }

    struct ConstantEvaluator;

    impl ForwardEvaluator for ConstantEvaluator {
        fn evaluate(
            &mut self,
            _round: &ResidentRound,
            _program: &ForwardProgram,
        ) -> Result<f32, String> {
            Ok(2.5)
        }
    }

    #[test]
    fn resident_state_can_evaluate_and_tell_one_round() {
        let mut state = ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::Cpu,
            ForwardProgram::kda().unwrap(),
        )
        .unwrap();
        let mut evaluator = ConstantEvaluator;
        let (_, reward) = state
            .ask_evaluate(
                &[11],
                Ask {
                    neighbors: 1,
                    ..Ask::default()
                },
                &mut evaluator,
                true,
            )
            .unwrap();
        assert_eq!(reward, 2.5);
        assert_eq!(state.rewards(), &[2.5]);
    }
}
