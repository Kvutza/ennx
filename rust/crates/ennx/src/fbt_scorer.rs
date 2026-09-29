//! Production candidate scoring in one Metal compute pass, with optional
//! encoder-boundary timestamp instrumentation.

use super::architecture::LayerStep;
use super::*;
use metal::MTLDispatchType;
use std::cell::RefCell;

#[path = "fbt_scorer/input.rs"]
mod input;
#[path = "fbt_scorer/mhc.rs"]
mod mhc;

pub(super) use super::scorer_trace::ScorerStageTrace;

pub(super) struct Scorer<'a> {
    pub(super) command: &'a CommandBufferRef,
    pub(super) encoder: RefCell<Option<&'a ComputeCommandEncoderRef>>,
    pub(super) pipelines: &'a Pipelines,
    pub(super) tensorops: &'a TensorOpsPipelines,
    pub(super) pisa1: &'a Pisa1,
    pub(super) buffers: &'a Buffers,
    pub(super) weights: CandidateRow<'a>,
    pub(super) row_start: u32,
    pub(super) rows: u32,
    pub(super) cache: Option<&'a [decode::Cache]>,
    pub(super) dirty: bool,
    pub(super) draft: Option<&'a draft::Context>,
    pub(super) trace: Option<&'a ScorerStageTrace>,
    pub(super) feedback: crate::config::FeedbackTransition,
    pub(super) diffusion: Option<diffusion::Input<'a>>,
    pub(super) cache_only: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ProposalReadout<'a> {
    pub output: &'a BufferRef,
    pub seeds: &'a BufferRef,
    pub temperature: f32,
    pub score_targets: bool,
    pub feedback: crate::config::FeedbackTransition,
}

struct ProposalPlan<'a> {
    readout: ProposalReadout<'a>,
    row_start: u32,
    rows: u32,
    cache: Option<&'a [decode::Cache]>,
    dirty: bool,
    draft: Option<&'a draft::Context>,
    trace: Option<&'a ScorerStageTrace>,
    hidden_only: bool,
    diffusion: Option<diffusion::Input<'a>>,
}

pub(super) fn objective_fused(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
) -> Result<(), String> {
    objective_trace(command, pipelines, tensorops, pisa1, buffers, weights, None)
}

pub(super) fn encode_proposals(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    rows: u32,
    cache: Option<&[decode::Cache]>,
    draft: Option<&draft::Context>,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if rows == 0 || rows > ROWS || (cache.is_none() && rows % CONTEXT != 0) || rows % 128 != 0 {
        return Err(format!("invalid active scorer row count {rows}"));
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: 0,
            rows,
            cache,
            dirty: false,
            draft,
            trace,
            hidden_only: false,
            diffusion: None,
        },
    )
}

pub(super) fn suffix_proposals(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    row_start: u32,
    rows: u32,
    cache: &[decode::Cache],
) -> Result<(), String> {
    if row_start % 4 != 0 || rows == 0 || rows % 128 != 0 || row_start + rows > pisa1.context() {
        return Err(format!(
            "invalid candidate suffix rows {row_start}..{}",
            row_start + rows
        ));
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start,
            rows,
            cache: Some(cache),
            dirty: true,
            draft: None,
            trace: None,
            hidden_only: false,
            diffusion: None,
        },
    )
}

/// Chunk-local activations with absolute token positions and persistent KV.
pub(super) fn context_chunk(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    start: u32,
    rows: u32,
    cache: &[decode::Cache],
    hidden_only: bool,
) -> Result<(), String> {
    if rows == 0
        || rows > 4096
        || rows % 128 != 0
        || start % 4 != 0
        || start
            .checked_add(rows)
            .is_none_or(|end| end > pisa.context())
        || cache.first().is_none_or(|entry| entry.kv.is_none())
    {
        return Err("invalid compact context chunk".into());
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: start,
            rows,
            cache: Some(cache),
            dirty: true,
            draft: None,
            trace: None,
            hidden_only,
            diffusion: None,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn denoise_chunk(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    start: u32,
    rows: u32,
    cache: &[decode::Cache],
    hidden_only: bool,
    input: diffusion::Input<'_>,
) -> Result<(), String> {
    if rows == 0
        || rows > 4096
        || rows % 128 != 0
        || start % 128 != 0
        || start
            .checked_add(rows)
            .is_none_or(|end| end > pisa.context())
        || cache.first().is_none_or(|entry| entry.kv.is_none())
    {
        return Err("invalid diffusion chunk".into());
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: start,
            rows,
            cache: Some(cache),
            dirty: true,
            draft: None,
            trace: input.trace,
            hidden_only,
            diffusion: Some(input),
        },
    )
}

fn proposals_at(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    plan: ProposalPlan<'_>,
) -> Result<(), String> {
    if let Some(trace) = plan.trace {
        trace.reset();
    }
    let scorer = Scorer {
        command,
        encoder: RefCell::new(plan.trace.is_none().then(|| {
            command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
        })),
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        row_start: plan.row_start,
        rows: plan.rows,
        cache: plan.cache,
        dirty: plan.dirty,
        draft: plan.draft,
        trace: plan.trace,
        feedback: weights
            .architecture
            .feedback_transition(plan.readout.feedback),
        diffusion: plan.diffusion,
        cache_only: plan.hidden_only && plan.diffusion.is_some(),
    };
    let result = scorer.encode_hidden().and_then(|()| {
        if plan.hidden_only {
            return Ok(());
        }
        scorer.start_stage("proposal_readout")?;
        scorer.readout_proposals(
            plan.readout.output,
            plan.readout.seeds,
            plan.readout.temperature,
            plan.readout.score_targets,
        )
    });
    if let Some(encoder) = scorer.encoder.borrow_mut().take() {
        encoder.end_encoding();
    }
    result
}

pub(super) fn objective_trace(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if let Some(trace) = trace {
        trace.reset();
    }
    let scorer = Scorer {
        command,
        encoder: RefCell::new(trace.is_none().then(|| {
            command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
        })),
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        row_start: 0,
        rows: ROWS,
        cache: None,
        dirty: false,
        draft: None,
        trace,
        feedback: weights
            .architecture
            .feedback_transition(crate::config::FeedbackTransition::ProjectedSigmoid),
        diffusion: None,
        cache_only: false,
    };
    let result = scorer.encode();
    if let Some(encoder) = scorer.encoder.borrow_mut().take() {
        encoder.end_encoding();
    }
    result
}

impl Scorer<'_> {
    fn scratch_start(&self) -> u32 {
        if self.cache.is_some_and(|entries| entries[0].kv.is_some()) {
            0
        } else {
            self.row_start
        }
    }

    pub(super) fn activation_offset(&self) -> u64 {
        half_bytes(u64::from(self.scratch_start()) * u64::from(WIDTH))
    }

    fn qkv_offset(&self) -> u64 {
        half_bytes(u64::from(self.scratch_start()) * u64::from(QKV_WIDTH))
    }

    fn qkv(&self, execution: u32) -> &BufferRef {
        self.layer_cache(execution)
            .map_or(&self.buffers.qkv, |cache| &cache.qkv)
    }

    fn layer_cache(&self, execution: u32) -> Option<&decode::Cache> {
        self.cache.map(|cache| &cache[execution as usize])
    }

    fn next_norm(&self, layer: u32, pass: u32) -> (&BufferRef, u64) {
        if layer + 1 < MODEL_LAYERS {
            (
                self.weights.buffer,
                self.weights.attention_norm + half_bytes(u64::from(layer + 1) * u64::from(WIDTH)),
            )
        } else if pass + 1 < FEEDBACK_PASSES {
            if self.feedback == crate::config::FeedbackTransition::ProjectedSigmoid {
                (&self.buffers.unit_norm, 0)
            } else {
                (self.weights.buffer, self.weights.attention_norm)
            }
        } else {
            (self.weights.buffer, self.weights.final_norm)
        }
    }

    fn rotate_qk(&self, qkv: &BufferRef, qkv_offset: u64) {
        let encoder = self.active_encoder();
        let shape = [self.rows, self.row_start, self.pisa1.context(), ROPE_PAIRS];
        encoder.set_compute_pipeline_state(&self.pipelines.rope);
        encoder.set_buffer(0, Some(qkv), qkv_offset);
        encoder.set_buffer(1, Some(&self.buffers.rope), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(self.rows * ROPE_HEADS * ROPE_PAIRS)),
            thread_group(
                self.pipelines
                    .rope
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
        );
        encoder.memory_barrier_with_resources(&[qkv]);
    }

    fn attention_offsets(layer: u32) -> (u64, u64) {
        (
            half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(QKV_WIDTH)),
            half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(WIDTH)),
        )
    }

    pub(super) fn start_stage(&self, stage: &'static str) -> Result<(), String> {
        if let Some(trace) = self.trace {
            if let Some(encoder) = self.encoder.borrow_mut().take() {
                encoder.end_encoding();
            }
            self.encoder
                .replace(Some(trace.encoder(self.command, stage)?));
        }
        Ok(())
    }

    pub(super) fn active_encoder(&self) -> &ComputeCommandEncoderRef {
        self.encoder
            .borrow()
            .expect("scorer stage encoder initialized")
    }

    fn encode(&self) -> Result<(), String> {
        self.encode_hidden()?;
        self.start_stage("readout")?;
        self.readout()
    }

    fn encode_hidden(&self) -> Result<(), String> {
        self.prepare_gateup()?;
        self.start_stage("embed")?;
        self.embed()?;
        let mut input: &BufferRef = &self.buffers.input;
        let diffusion_steps = self
            .diffusion
            .map(|input| {
                crate::forward_program::RecurrentCore::selective_fbt()
                    .layer_visits(MODEL_LAYERS as usize, input.visits as usize)
            })
            .transpose()?;
        let dynamic_steps = diffusion_steps.map(|steps| {
            steps
                .into_iter()
                .map(|step| LayerStep {
                    layer: step.layer as u32,
                    pass: step.visit as u32,
                    execution: step.execution as u32,
                })
                .collect::<Vec<_>>()
        });
        let steps = dynamic_steps
            .as_deref()
            .unwrap_or_else(|| self.weights.architecture.layer_steps());
        for step in steps {
            // KV for the final execution depends only on its attention input.
            // Its residual update, FFN and final readout cannot affect any cache.
            if self.cache_only && step.execution as usize + 1 == steps.len() {
                self.start_stage("mhc_attention_input")?;
                self.mhc_prepare(step.layer, step.execution, true)?;
                self.attention(step.layer, step.execution)?;
                return Ok(());
            }
            self.layer(input, step.layer, step.pass, step.execution)?;
            if let Some(draft) = self.draft {
                draft.capture(
                    self.active_encoder(),
                    &self.buffers.output,
                    step.execution,
                    steps.len() as u32,
                    self.row_start,
                    self.rows,
                );
            }
            input = &self.buffers.output;
            if !self.weights.architecture.is_multistream()
                && step.layer + 1 == MODEL_LAYERS
                && step.pass + 1 < FEEDBACK_PASSES
            {
                if self.feedback == crate::config::FeedbackTransition::ProjectedSigmoid {
                    self.start_stage("feedback")?;
                    self.feedback(input)?;
                    input = &self.buffers.feedback;
                }
            }
        }
        if self.weights.architecture.is_multistream() {
            self.start_stage("mhc_readout")?;
            self.mhc_finalize()?;
        }
        Ok(())
    }

    fn prepare_gateup(&self) -> Result<(), String> {
        let fine = &self.pipelines.fine_grained;
        let (pipeline, output, elements_per_thread, stage) = if fine.int8_gate() {
            (
                &self.pipelines.quantize_gate_up_int8,
                self.buffers.quantized_gateup()?,
                1,
                "quantize_gate",
            )
        } else if fine.interleaved_gate() {
            (
                &self.pipelines.interleave_gate_up,
                self.buffers.interleaved_gateup()?,
                4,
                "interleave_gate",
            )
        } else {
            return Ok(());
        };
        self.start_stage(stage)?;
        let elements = u64::from(MODEL_LAYERS) * routing::GATE_PARAMETERS as u64;
        let encoder = self.active_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.weights.buffer), self.weights.gate_up);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&elements) as u64,
            (&elements as *const u64).cast(),
        );
        encoder.dispatch_threads(
            thread_group(elements / elements_per_thread),
            thread_group(pipeline.max_total_threads_per_threadgroup().min(256)),
        );
        encoder.memory_barrier_with_resources(&[output]);
        Ok(())
    }

    fn attention(&self, layer: u32, execution: u32) -> Result<(), String> {
        self.start_stage("qkv")?;
        let encoder = self.active_encoder();
        let tensorops = self.tensorops;
        let pisa1 = self.pisa1;
        let buffers = self.buffers;
        let weights = self.weights;
        let qkv = self.qkv(execution);
        let activation_offset = self.activation_offset();
        let qkv_offset = self.qkv_offset();
        let (matrix_offset, _) = Self::attention_offsets(layer);
        tensor_workspace(
            &encoder,
            &tensorops.qkv_wide,
            &[
                (&buffers.normalized, activation_offset),
                (weights.buffer, weights.qkv + matrix_offset),
                (qkv, qkv_offset),
            ],
            MTLSize {
                width: u64::from(QKV_WIDTH / 128),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            8,
        )?;
        encoder.memory_barrier_with_resources(&[qkv]);
        self.rotate_qk(qkv, qkv_offset);
        if let Some(cache) = self
            .layer_cache(execution)
            .filter(|entry| entry.kv.is_some())
        {
            self.start_stage("pisa_attention")?;
            let encoder = self.active_encoder();
            if let Some(input) = self.diffusion {
                let step = crate::forward_program::RecurrentCore::selective_fbt()
                    .layer_visits(MODEL_LAYERS as usize, input.visits as usize)?
                    [execution as usize];
                let last_cache = self.cache_only && layer + 1 == MODEL_LAYERS;
                pisa1.denoise(
                    &encoder,
                    qkv,
                    cache.kv.as_ref().unwrap(),
                    &cache.pyramid,
                    self.row_start,
                    self.rows,
                    crate::context_metal::IndexPolicy {
                        weights: weights.buffer,
                        offset: weights.index_query,
                        block: input.config.block,
                        mode: input.config.index,
                        layer,
                        fresh: input.fresh && step.visit == 0,
                        reuse: input.config.reuse,
                    },
                    !last_cache,
                )?;
                if last_cache {
                    return Ok(());
                }
                return self.project_attention(layer);
            }
            pisa1.cached_attention(
                &encoder,
                qkv,
                cache.kv.as_ref().unwrap(),
                &cache.pyramid,
                self.row_start,
                self.rows,
            )?;
            return self.project_attention(layer);
        }
        self.start_stage("pisa_pyramid")?;
        let encoder = self.active_encoder();
        if let Some(cache) = self.layer_cache(execution) {
            if self.dirty {
                pisa1.pyramid_range(&encoder, qkv, &cache.pyramid, self.row_start, self.rows);
            } else {
                pisa1.pyramid_output(&encoder, qkv, &cache.pyramid, pisa1.context());
            }
        } else {
            pisa1.pyramid_layer(&encoder, qkv, self.rows);
        }
        self.start_stage("pisa_attention")?;
        let encoder = self.active_encoder();
        if let Some(cache) = self.layer_cache(execution) {
            pisa1.attention_from(&encoder, qkv, &cache.pyramid, self.row_start, self.rows);
        } else {
            pisa1.attention_range(&encoder, qkv, self.row_start, self.rows);
        }
        self.project_attention(layer)
    }

    fn project_attention(&self, layer: u32) -> Result<(), String> {
        self.start_stage("projection")?;
        let encoder = self.active_encoder();
        let activation_offset = self.activation_offset();
        let (_, output_offset) = Self::attention_offsets(layer);
        tensor_workspace(
            &encoder,
            &self.tensorops.output_projection_wide,
            &[
                (self.pisa1.output(), activation_offset),
                (self.weights.buffer, self.weights.output + output_offset),
                (&self.buffers.projected, activation_offset),
            ],
            MTLSize {
                width: u64::from(WIDTH / 128),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            8,
        )?;
        encoder.memory_barrier_with_resources(&[&self.buffers.projected]);
        Ok(())
    }

    fn feed_forward(&self, input: &BufferRef, layer: u32, pass: u32) -> Result<(), String> {
        self.start_stage("pre_ffn")?;
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        let shape = model_rows(self.rows);
        encoder.set_compute_pipeline_state(&pipelines.residual_rms);
        encoder.set_buffer(0, Some(input), activation_offset);
        encoder.set_buffer(1, Some(&buffers.projected), activation_offset);
        encoder.set_buffer(
            2,
            Some(weights.buffer),
            weights.ffn_norm + half_bytes(u64::from(layer) * u64::from(WIDTH)),
        );
        encoder.set_buffer(3, Some(&buffers.attention_state), activation_offset);
        encoder.set_buffer(4, Some(&buffers.normalized), activation_offset);
        encoder.set_bytes(
            5,
            std::mem::size_of::<MoeShape>() as u64,
            (&shape as *const MoeShape).cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.rows)),
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.memory_barrier_with_resources(&[&buffers.attention_state, &buffers.normalized]);
        let next_norm = self.next_norm(layer, pass);
        if self.trace.is_some() {
            let (router_offset, gate_up_offset, down_offset) = ffn_offsets(layer);
            let fine = &pipelines.fine_grained;
            let fine_buffers = &buffers.fine_grained;
            self.start_stage("route")?;
            fine.route_rows(
                self.active_encoder(),
                fine_buffers,
                &buffers.normalized,
                activation_offset,
                weights.buffer,
                weights.router + router_offset,
                self.rows,
            )?;
            self.start_stage("gate")?;
            let int8_gate = fine.int8_gate();
            let gate_weights = if int8_gate {
                buffers.quantized_gateup()?
            } else if fine.interleaved_gate() {
                buffers.interleaved_gateup()?
            } else {
                weights.buffer
            };
            let gate_offset = if int8_gate {
                gate_up_offset / 2
            } else if fine.interleaved_gate() {
                gate_up_offset
            } else {
                weights.gate_up + gate_up_offset
            };
            fine.gate_rows(
                self.active_encoder(),
                fine_buffers,
                &buffers.normalized,
                activation_offset,
                gate_weights,
                gate_offset,
                self.rows,
            )?;
            self.start_stage("down")?;
            fine.down_rows(
                self.active_encoder(),
                fine_buffers,
                weights.buffer,
                weights.down + down_offset,
                self.rows,
            )?;
            self.start_stage("combine_post_ffn")?;
            fine.combine_offsets(
                self.active_encoder(),
                fine_buffers,
                (&buffers.attention_state, activation_offset),
                next_norm,
                (&buffers.output, activation_offset),
                (&buffers.normalized, activation_offset),
                self.rows,
            );
        } else {
            self.start_stage("ffn")?;
            ffn_rows(
                self.active_encoder(),
                pipelines,
                &buffers.normalized,
                &buffers.attention_state,
                layer,
                buffers,
                weights,
                next_norm,
                self.scratch_start(),
                self.rows,
            )?;
        }
        Ok(())
    }

    fn feedback(&self, input: &BufferRef) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let tensorops = self.tensorops;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        tensor_workspace(
            &encoder,
            &tensorops.output_projection_wide,
            &[
                (input, activation_offset),
                (weights.buffer, weights.feedback_state),
                (&buffers.feedback_state, activation_offset),
            ],
            MTLSize {
                width: u64::from(WIDTH / 128),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            8,
        )?;
        tensor_workspace(
            &encoder,
            &tensorops.output_projection_wide,
            &[
                (&buffers.normalized, activation_offset),
                (weights.buffer, weights.feedback_gate),
                (&buffers.feedback_gate, activation_offset),
            ],
            MTLSize {
                width: u64::from(WIDTH / 128),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            8,
        )?;
        encoder.memory_barrier_with_resources(&[&buffers.feedback_state, &buffers.feedback_gate]);
        let shape = model_rows(self.rows);
        encoder.set_compute_pipeline_state(&pipelines.feedback_fuse_rms);
        encoder.set_buffer(0, Some(&buffers.feedback_state), activation_offset);
        encoder.set_buffer(1, Some(&buffers.feedback_gate), activation_offset);
        encoder.set_buffer(2, Some(weights.buffer), weights.attention_norm);
        encoder.set_buffer(3, Some(&buffers.feedback), activation_offset);
        encoder.set_buffer(4, Some(&buffers.normalized), activation_offset);
        encoder.set_bytes(
            5,
            std::mem::size_of::<MoeShape>() as u64,
            (&shape as *const MoeShape).cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.rows)),
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.memory_barrier_with_resources(&[&buffers.feedback, &buffers.normalized]);
        Ok(())
    }
}
