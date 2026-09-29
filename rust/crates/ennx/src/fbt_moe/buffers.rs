use super::*;

impl Buffers {
    pub(super) fn interleaved_gateup(&self) -> Result<&BufferRef, String> {
        self.interleaved_gateup
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<u16>(MODEL_LAYERS as usize * routing::GATE_PARAMETERS)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn quantized_gateup(&self) -> Result<&BufferRef, String> {
        self.quantized_gateup
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<i8>(MODEL_LAYERS as usize * routing::GATE_PARAMETERS)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn logits(&self) -> Result<&BufferRef, String> {
        self.logits
            .get_or_init(|| {
                Runtime::shared().map(|runtime| runtime.buffer::<u16>((ROWS * VOCAB) as usize))
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn new(runtime: &Runtime) -> Self {
        let half = |n: u32| n as usize;
        let residual = residual_buffers(runtime);
        Self {
            fine_grained: routing::FineGrainedMoeBuffers::new(runtime),
            quantized_gateup: std::cell::OnceCell::new(),
            interleaved_gateup: std::cell::OnceCell::new(),
            input: patterned(runtime, half(ROWS * WIDTH), 0x243f_6a88, 0x2c00),
            router: patterned(runtime, half(WIDTH * EXPERTS), 0x85a3_08d3, 0x1800),
            gates: runtime.buffer::<u16>(half(ROWS)),
            grouped: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            gate_up_base: patterned(
                runtime,
                half(EXPERTS * WIDTH * GATE_UP),
                0x1319_8a2e,
                0x2000,
            ),
            gate_up: canary(runtime, (ROWS * GATE_UP) as usize),
            gate_inner: factor(runtime, EXPERTS, 32, 32),
            gate_outer: factor(runtime, EXPERTS, 16, 54),
            activation: canary(runtime, (ROWS * EXPERT_WIDTH) as usize),
            down_base: patterned(
                runtime,
                half(EXPERTS * EXPERT_WIDTH * WIDTH),
                0x0370_7344,
                0x2000,
            ),
            down: canary(runtime, (ROWS * WIDTH) as usize),
            down_inner: factor(runtime, EXPERTS, 27, 16),
            down_outer: factor(runtime, EXPERTS, 32, 32),
            output: canary(runtime, (ROWS * WIDTH) as usize),
            mps_gate_up: canary(runtime, (ROWS * GATE_UP) as usize),
            mps_down: canary(runtime, (ROWS * WIDTH) as usize),
            materialized_gate_up: canary(runtime, (EXPERTS * WIDTH * GATE_UP) as usize),
            materialized_down: canary(runtime, (EXPERTS * EXPERT_WIDTH * WIDTH) as usize),
            qkv_weights: patterned(runtime, half(WIDTH * QKV_WIDTH), 0xa409_3822, 0x2000),
            qkv: canary(runtime, (ROWS * QKV_WIDTH) as usize),
            output_projection_weights: patterned(runtime, half(WIDTH * WIDTH), 0x299f_31d0, 0x2000),
            projected: canary(runtime, (ROWS * WIDTH) as usize),
            mps_qkv: canary(runtime, (ROWS * QKV_WIDTH) as usize),
            mps_projected: canary(runtime, (ROWS * WIDTH) as usize),
            feedback_state_weights: patterned(runtime, half(WIDTH * WIDTH), 0x082e_fa98, 0x2000),
            feedback_gate_weights: patterned(runtime, half(WIDTH * WIDTH), 0xec4e_6c89, 0x2000),
            feedback_state: residual.feedback_state,
            feedback_gate: residual.feedback_gate,
            feedback: residual.feedback,
            readout_weights: patterned(runtime, half(WIDTH * VOCAB), 0x4528_21e6, 0x1800),
            logits: std::cell::OnceCell::new(),
            loss_partials: runtime.buffer::<[f32; 4]>(half(ROWS * (VOCAB / 64))),
            proposal_loss_partials: runtime.buffer::<[f32; 4]>(half(ROWS * (VOCAB / 64))),
            labels: synthetic_buffer(runtime, 1),
            losses: runtime.buffer_with(&vec![f32::NAN; ROWS as usize + 64]),
            score_mask: runtime.buffer_with(&vec![1u8; ROWS as usize]),
            sequence_scores: runtime.buffer_with(&[f32::NAN; BATCH as usize]),
            tokens: synthetic_buffer(runtime, 0),
            unit_norm: residual.unit_norm,
            normalized: residual.normalized,
            attention_state: residual.attention_state,
            mhc_streams: residual.mhc_streams,
            mhc_coefficients: residual.mhc_coefficients,
            rope: rope_table(runtime, MAX_CONTEXT),
        }
    }

    pub(super) fn for_context(runtime: &Runtime, context: u32) -> Self {
        let mut buffers = Self::new(runtime);
        if context > ROWS {
            buffers.tokens = runtime.buffer_with(&vec![0u32; context as usize]);
            buffers.labels = runtime.buffer_with(&vec![0u32; context as usize]);
            buffers.losses = runtime.buffer_with(&vec![f32::NAN; context as usize + 64]);
            buffers.rope = rope_table(runtime, context);
        }
        buffers
    }
}

struct ResidualBuffers {
    feedback_state: Buffer,
    feedback_gate: Buffer,
    feedback: Buffer,
    unit_norm: Buffer,
    normalized: Buffer,
    attention_state: Buffer,
    mhc_streams: [Buffer; 2],
    mhc_coefficients: Buffer,
}

fn residual_buffers(runtime: &Runtime) -> ResidualBuffers {
    ResidualBuffers {
        feedback_state: runtime.buffer::<u16>((ROWS * WIDTH) as usize),
        feedback_gate: runtime.buffer::<u16>((ROWS * WIDTH) as usize),
        feedback: runtime.buffer::<u16>((ROWS * WIDTH) as usize),
        unit_norm: filled(runtime, WIDTH as usize, 0x3c00),
        normalized: runtime.buffer::<u16>((ROWS * WIDTH) as usize),
        attention_state: runtime.buffer::<u16>((ROWS * WIDTH) as usize),
        mhc_streams: [
            runtime.buffer::<u16>((ROWS * MHC_STREAMS * WIDTH) as usize),
            runtime.buffer::<u16>((ROWS * MHC_STREAMS * WIDTH) as usize),
        ],
        mhc_coefficients: runtime.buffer::<f32>((ROWS * MHC_COEFFICIENTS) as usize),
    }
}
