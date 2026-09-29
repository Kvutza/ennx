//! Resident checkpoint execution. Physical weights are shared across recurrent visits.
use crate::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig2D,
};
use ennx_cuda_kernels::{
    FbtShape, MatmulShape, MoeShape, PisaShape, RouteShape, RoutedTile, fbt_model,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
mod cached;
mod readout;
#[path = "../../rust/crates/ennx/src/forward_program/recurrence.rs"]
pub mod recurrence;
#[path = "model/routing.rs"]
mod routing;
pub(crate) mod run;
#[path = "model/weights.rs"]
pub(crate) mod weights;
use weights::{Layer, Weights};
#[path = "model/diffusion.rs"]
mod diffusion;
#[path = "model/diffusion_run.rs"]
pub(crate) mod diffusion_run;
use diffusion::DiffusionState;

const WIDTH: usize = 512;
const VOCAB: usize = 8192;
const EXPERTS: usize = 625;

#[derive(Clone, Copy)]
enum Stage {
    Embed,
    Prepare,
    Qkv,
    Index,
    Pisa,
    Output,
    Update,
    Moe,
    Readout,
}

fn mark(
    events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    stream: &CudaStream,
    stage: Stage,
) -> CudaResult<()> {
    if let Some(events) = events.as_deref_mut() {
        events.push((stage, timing_event(stream)?));
    }
    Ok(())
}

pub struct ModelOutput {
    pub tokens: Vec<u32>,
    pub hidden: Vec<u16>,
    pub stream_probe: Vec<u16>,
    pub device_ms: f32,
    pub wall_ms: f32,
    pub visits: usize,
}

pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub frontiers: Vec<usize>,
    pub prompt_tokens: usize,
    pub passes: usize,
    pub visits: usize,
    pub device_ms: f32,
    pub wall_ms: f32,
    pub first_wave_ms: Option<f32>,
    pub chunk_profiles: Vec<(usize, WaveProfile)>,
}

pub struct WaveProfile {
    pub embed_ms: f32,
    pub mhc_ms: f32,
    pub qkv_ms: f32,
    pub index_ms: f32,
    pub pisa_ms: f32,
    pub output_ms: f32,
    pub attention_ms: f32,
    pub moe_ms: f32,
    pub readout_ms: f32,
    pub total_ms: f32,
}

pub struct FbtModel {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: fbt_model::LoadedModule,
    pub(crate) weights: Weights,
    pub(crate) scratch: Scratch,
    pub(crate) rows: usize,
    pub(crate) sequence: usize,
    pub(crate) diffusion: Option<DiffusionState>,
    synth: Option<crate::gemm::Gemm>,
    parallel_mhc: bool,
    cache: Option<cached::State>,
    first: usize,
}

struct Scratch {
    tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<u16>,
    streams: [DeviceBuffer<u16>; 2],
    coefficients: DeviceBuffer<f32>,
    normalized: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    pyramid: DeviceBuffer<u16>,
    blocks: DeviceBuffer<u32>,
    attended: DeviceBuffer<u16>,
    branch: DeviceBuffer<u16>,
    route_scores: DeviceBuffer<u16>,
    experts: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<u16>,
    margins: DeviceBuffer<f32>,
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    prefixes: DeviceBuffer<u32>,
    mapping: DeviceBuffer<u32>,
    packed: DeviceBuffer<u16>,
    tiles: DeviceBuffer<RoutedTile>,
    shared_tiles: DeviceBuffer<RoutedTile>,
    projection: DeviceBuffer<u16>,
    shared_projection: DeviceBuffer<u16>,
    activation: DeviceBuffer<u16>,
    routed: DeviceBuffer<u16>,
    shared_activation: DeviceBuffer<u16>,
    shared: DeviceBuffer<u16>,
    logits: DeviceBuffer<u16>,
    sampled: DeviceBuffer<u32>,
    frontier: DeviceBuffer<u32>,
    rope: DeviceBuffer<f32>,
    tile_capacity: usize,
}

impl Scratch {
    fn new(stream: &CudaStream, rows: usize, sequence: usize) -> CudaResult<Self> {
        fn buffer<T: cuda_core::DeviceCopy>(
            stream: &CudaStream,
            len: usize,
        ) -> CudaResult<DeviceBuffer<T>> {
            DeviceBuffer::zeroed(stream, len).map_err(cuda_error)
        }
        let tile_capacity = (rows * 3).div_ceil(64) + EXPERTS;
        let shared_tiles = (0..rows.div_ceil(64))
            .map(|tile| RoutedTile {
                expert: 0,
                first_row: (tile * 64) as u32,
                valid_rows: (rows - tile * 64).min(64) as u32,
                reserved: 1,
            })
            .collect::<Vec<_>>();
        let mut rope = Vec::with_capacity(sequence * 64);
        for position in 0..sequence {
            for pair in 0..32 {
                let angle = position as f32 * 10000.0_f32.powf(-2.0 * pair as f32 / 64.0);
                rope.extend([angle.cos(), angle.sin()]);
            }
        }
        Ok(Self {
            tokens: buffer(stream, rows)?,
            embedded: buffer(stream, rows * WIDTH)?,
            streams: [
                buffer(stream, rows * WIDTH * 4)?,
                buffer(stream, rows * WIDTH * 4)?,
            ],
            coefficients: buffer(stream, rows * 24)?,
            normalized: buffer(stream, rows * WIDTH)?,
            qkv: buffer(stream, rows * 640)?,
            pyramid: buffer(stream, (rows / sequence * (sequence / 32 - 1) * 64).max(1))?,
            blocks: buffer(stream, rows * 8)?,
            attended: buffer(stream, rows * WIDTH)?,
            branch: buffer(stream, rows * WIDTH)?,
            route_scores: buffer(stream, rows * EXPERTS)?,
            experts: buffer(stream, rows * 3)?,
            route_weights: buffer(stream, rows * 3)?,
            margins: buffer(stream, rows)?,
            counts: buffer(stream, EXPERTS)?,
            offsets: buffer(stream, EXPERTS)?,
            prefixes: buffer(stream, EXPERTS * (rows * 3).div_ceil(1024))?,
            mapping: buffer(stream, rows * 3)?,
            packed: buffer(stream, rows * 3 * WIDTH)?,
            tiles: buffer(stream, tile_capacity)?,
            shared_tiles: DeviceBuffer::from_host(stream, &shared_tiles).map_err(cuda_error)?,
            projection: buffer(stream, rows * 3 * 432)?,
            shared_projection: buffer(stream, rows * 432)?,
            activation: buffer(stream, rows * 3 * 224)?,
            routed: buffer(stream, rows * 3 * WIDTH)?,
            shared_activation: buffer(stream, rows * 224)?,
            shared: buffer(stream, rows * WIDTH)?,
            logits: buffer(stream, rows.min(512) * VOCAB)?,
            sampled: buffer(stream, rows)?,
            frontier: buffer(stream, 1)?,
            rope: DeviceBuffer::from_host(stream, &rope).map_err(cuda_error)?,
            tile_capacity,
        })
    }
}

impl FbtModel {
    pub fn fixture() -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        // SAFETY: the generated binding loads the matching embedded module.
        let module = unsafe { fbt_model::load(&context) }.map_err(cuda_error)?;
        let weights = Weights::fixture(&stream)?;
        let synth = crate::gemm::Gemm::from_env(&context)?;
        let scratch = Scratch::new(&stream, 4096, 4096)?;
        Ok(Self {
            context,
            stream,
            module,
            weights,
            scratch,
            rows: 4096,
            sequence: 4096,
            diffusion: None,
            synth,
            parallel_mhc: true,
            cache: None,
            first: 0,
        })
    }
    pub fn open(path: &Path, rows: usize, sequence: usize) -> CudaResult<Self> {
        if rows == 0
            || rows > 65536
            || sequence < 4096
            || sequence > 65536
            || !sequence.is_power_of_two()
            || rows % sequence != 0
        {
            return Err("model requires 4096..65536 power-of-two context and complete sequences, at most 65536 rows".into());
        }
        PisaShape::fbt(rows as u32, sequence as u32).map_err(str::to_string)?;
        Self::load(path, rows, sequence)
    }

    /// Allocate bounded activations directly, without a full-context workspace.
    pub fn open_streamed(path: &Path, sequence: usize, chunk: usize) -> CudaResult<Self> {
        if !(4096..=1_048_576).contains(&sequence)
            || !sequence.is_power_of_two()
            || !(64..=4096).contains(&chunk)
            || !chunk.is_power_of_two()
            || sequence % chunk != 0
        {
            return Err(
                "streamed model requires power-of-two context 4096..1048576 and chunk 64..4096"
                    .into(),
            );
        }
        PisaShape::cached(chunk as u32, sequence as u32, 0).map_err(str::to_string)?;
        Self::load(path, chunk, sequence)
    }

    fn load(path: &Path, rows: usize, sequence: usize) -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        // SAFETY: generated binding loads the matching embedded module.
        let module = unsafe { fbt_model::load(&context) }.map_err(cuda_error)?;
        let weights = Weights::open(path, &stream)?;
        let synth = crate::gemm::Gemm::from_env(&context)?;
        let scratch = Scratch::new(&stream, rows, sequence)?;
        Ok(Self {
            context,
            stream,
            module,
            weights,
            scratch,
            rows,
            sequence,
            diffusion: None,
            synth,
            parallel_mhc: true,
            cache: None,
            first: 0,
        })
    }

    pub fn run(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
    ) -> CudaResult<ModelOutput> {
        if self.weights.diffusion.is_some()
            || self.cache.is_some()
            || tokens.len() != self.rows
            || tokens.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
        {
            return Err("model token count, vocabulary, or temperature is invalid".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        let wall = Instant::now();
        copy_prefix(&self.scratch.tokens, tokens, &self.stream)?;
        let start = timing_event(&self.stream)?;
        let visits = self.enqueue(&plan, temperature, seed)?;
        let end = timing_event(&self.stream)?;
        let tokens = read_prefix(&self.scratch.sampled, &self.stream, self.rows)?;
        if tokens.iter().any(|&token| token >= VOCAB as u32) {
            return Err("readout contains no finite logit in at least one row".into());
        }
        let hidden = read_prefix(&self.scratch.normalized, &self.stream, self.rows * WIDTH)?;
        if hidden.iter().any(|&value| value & 0x7c00 == 0x7c00) {
            return Err("model produced non-finite hidden state".into());
        }
        let stream_probe = read_prefix(&self.scratch.streams[0], &self.stream, 2048)?;
        self.context.check_err().map_err(cuda_error)?;
        Ok(ModelOutput {
            tokens,
            hidden,
            stream_probe,
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            visits,
        })
    }

    fn enqueue(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
    ) -> CudaResult<usize> {
        self.enqueue_events(plan, temperature, seed, None)
    }

    fn enqueue_events(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
        mut events: Option<&mut Vec<(Stage, CudaEvent)>>,
    ) -> CudaResult<usize> {
        let rows = self.rows as u32;
        let shape = FbtShape::new(rows, 512, 8192, 1.0e-5).map_err(str::to_string)?;
        if let Some(state) = &mut self.diffusion {
            let weights = self
                .weights
                .diffusion
                .as_ref()
                .ok_or("diffusion checkpoint missing mask/index tensors")?;
            let launch = state
                .module
                .prepare_diffusion_embed(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            state
                .module
                .diffusion_embed(
                    &self.stream,
                    &launch,
                    &self.weights.embedding,
                    &self.scratch.tokens,
                    &weights.mask,
                    &state.confidence,
                    &mut self.scratch.embedded,
                    rows,
                )
                .map_err(cuda_error)?;
        } else {
            let launch = self
                .module
                .prepare_embed_packed(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .embed_packed(
                    &self.stream,
                    &launch,
                    &self.weights.embedding,
                    &self.scratch.tokens,
                    &mut self.scratch.embedded,
                    shape,
                )
                .map_err(cuda_error)?;
        }
        let launch = self
            .module
            .prepare_mhc_replicate(LaunchConfig1D::new((rows * 512).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_replicate(
                &self.stream,
                &launch,
                &self.scratch.embedded,
                &mut self.scratch.streams[0],
                rows,
            )
            .map_err(cuda_error)?;
        mark(&mut events, &self.stream, Stage::Embed)?;
        for (ordinal, visit) in plan.iter().enumerate() {
            self.prepare(visit.layer, true)?;
            mark(&mut events, &self.stream, Stage::Prepare)?;
            if self.cache.is_some() {
                self.cached_attention(visit.layer, ordinal, &mut events)?;
            } else {
                self.attention(visit.layer, &mut events)?;
            }
            self.update()?;
            mark(&mut events, &self.stream, Stage::Update)?;
            self.prepare(visit.layer, false)?;
            mark(&mut events, &self.stream, Stage::Prepare)?;
            self.moe(visit.layer)?;
            mark(&mut events, &self.stream, Stage::Moe)?;
            self.update()?;
            mark(&mut events, &self.stream, Stage::Update)?;
        }
        let launch = self
            .module
            .prepare_mhc_mix(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_mix(
                &self.stream,
                &launch,
                &self.scratch.streams[0],
                &self.scratch.coefficients,
                &self.weights.final_norm,
                &mut self.scratch.normalized,
                rows,
                1,
            )
            .map_err(cuda_error)?;
        self.readout(temperature, seed)?;
        mark(&mut events, &self.stream, Stage::Readout)?;
        Ok(plan.len())
    }

    pub fn profile_wave(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
    ) -> CudaResult<WaveProfile> {
        if self.cache.is_some()
            || tokens.len() != self.rows
            || tokens.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
        {
            return Err("wave profile token count, vocabulary, or temperature is invalid".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        copy_prefix(&self.scratch.tokens, tokens, &self.stream)?;
        let start = timing_event(&self.stream)?;
        let mut events = Vec::with_capacity(plan.len() * 9 + 2);
        self.enqueue_events(&plan, temperature, seed, Some(&mut events))?;
        if events.len() != plan.len() * 9 + 2 {
            return Err("wave profile event count is invalid".into());
        }
        let profile = WaveProfile::from_events(&start, &events)?;
        self.context.check_err().map_err(cuda_error)?;
        Ok(profile)
    }
}

impl WaveProfile {
    fn from_events(start: &CudaEvent, events: &[(Stage, CudaEvent)]) -> CudaResult<Self> {
        let mut embed_ms = 0.0;
        let mut mhc_ms = 0.0;
        let mut qkv_ms = 0.0;
        let mut index_ms = 0.0;
        let mut pisa_ms = 0.0;
        let mut output_ms = 0.0;
        let mut moe_ms = 0.0;
        let mut readout_ms = 0.0;
        let mut previous = start;
        for (stage, event) in events {
            let elapsed = previous.elapsed_ms(event).map_err(cuda_error)?;
            match stage {
                Stage::Embed => embed_ms += elapsed,
                Stage::Prepare | Stage::Update => mhc_ms += elapsed,
                Stage::Qkv => qkv_ms += elapsed,
                Stage::Index => index_ms += elapsed,
                Stage::Pisa => pisa_ms += elapsed,
                Stage::Output => output_ms += elapsed,
                Stage::Moe => moe_ms += elapsed,
                Stage::Readout => readout_ms += elapsed,
            }
            previous = event;
        }
        let attention_ms = qkv_ms + index_ms + pisa_ms + output_ms;
        let end = &events.last().ok_or("wave profile is empty")?.1;
        Ok(WaveProfile {
            embed_ms,
            mhc_ms,
            qkv_ms,
            index_ms,
            pisa_ms,
            output_ms,
            attention_ms,
            moe_ms,
            readout_ms,
            total_ms: start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}

impl FbtModel {
    /// Verify and repair an accepted prefix on-device. Sampling is indexed by
    /// absolute position, so repairs reuse the same random draw at each row.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
        unroll: usize,
    ) -> CudaResult<GenerationOutput> {
        if self.weights.diffusion.is_some()
            || self.cache.is_some()
            || self.rows != self.sequence
            || prompt.is_empty()
            || prompt.len() >= self.rows
            || prompt.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
            || !(1..=32).contains(&unroll)
        {
            return Err("generation requires one complete context, a nonempty in-vocabulary prompt shorter than context, nonnegative finite temperature, and 1..32 unroll".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        let wall = Instant::now();
        let mut tokens = vec![0; self.rows];
        tokens[..prompt.len()].copy_from_slice(prompt);
        copy_prefix(&self.scratch.tokens, &tokens, &self.stream)?;
        copy_prefix(&self.scratch.frontier, &[prompt.len() as u32], &self.stream)?;
        let start = timing_event(&self.stream)?;
        let launch = self
            .module
            .prepare_verify_prefix(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let mut passes = 0;
        let mut frontier = prompt.len();
        let mut frontiers = vec![frontier];
        while frontier < self.rows {
            for _ in 0..unroll {
                self.enqueue(&plan, temperature, seed)?;
                self.module
                    .verify_prefix(
                        &self.stream,
                        &launch,
                        &self.scratch.sampled,
                        &mut self.scratch.tokens,
                        &mut self.scratch.frontier,
                        self.rows as u32,
                    )
                    .map_err(cuda_error)?;
                passes += 1;
            }
            let next = read_prefix(&self.scratch.frontier, &self.stream, 1)?[0] as usize;
            if next <= frontier || next > self.rows {
                return Err(
                    "accepted-prefix repair failed to advance or produced invalid logits".into(),
                );
            }
            frontier = next;
            frontiers.push(frontier);
        }
        let end = timing_event(&self.stream)?;
        let tokens = read_prefix(&self.scratch.tokens, &self.stream, self.rows)?;
        if tokens.iter().any(|&token| token >= VOCAB as u32) || tokens[..prompt.len()] != *prompt {
            return Err("generation produced invalid tokens or changed the prompt".into());
        }
        self.context.check_err().map_err(cuda_error)?;
        Ok(GenerationOutput {
            tokens,
            frontiers,
            prompt_tokens: prompt.len(),
            passes,
            visits: plan.len(),
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            first_wave_ms: None,
            chunk_profiles: Vec::new(),
        })
    }

    fn prepare(&mut self, layer: usize, attention: bool) -> CudaResult<()> {
        let weights = &self.weights.layers[layer];
        let site = if attention {
            &weights.attention
        } else {
            &weights.moe
        };
        let rows = self.rows as u32;
        let launch = self
            .module
            .prepare_mhc_predict(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_predict(
                &self.stream,
                &launch,
                &self.scratch.streams[0],
                &site.predictor,
                &site.bias,
                &site.control,
                &mut self.scratch.coefficients,
                rows,
                u32::from(self.parallel_mhc),
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_mhc_mix(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_mix(
                &self.stream,
                &launch,
                &self.scratch.streams[0],
                &self.scratch.coefficients,
                &site.norm,
                &mut self.scratch.normalized,
                rows,
                0,
            )
            .map_err(cuda_error)
    }

    fn update(&mut self) -> CudaResult<()> {
        let rows = self.rows as u32;
        let launch = self
            .module
            .prepare_mhc_update(LaunchConfig1D::new((rows * 2048).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        let [input, output] = &mut self.scratch.streams;
        self.module
            .mhc_update(
                &self.stream,
                &launch,
                input,
                &self.scratch.branch,
                &self.scratch.coefficients,
                output,
                rows,
            )
            .map_err(cuda_error)?;
        self.scratch.streams.swap(0, 1);
        Ok(())
    }

    fn attention(
        &mut self,
        layer: usize,
        events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    ) -> CudaResult<()> {
        let rows = self.rows as u32;
        let mut shape = PisaShape::fbt(rows, self.sequence as u32).map_err(str::to_string)?;
        if let Some(state) = &self.diffusion {
            shape.visibility = state.block;
        }
        let weights = &self.weights.layers[layer];
        let s = &mut self.scratch;
        matmul(
            self.synth.as_ref(),
            &self.module,
            &self.stream,
            &s.normalized,
            &weights.qkv,
            &mut s.qkv,
            self.rows,
            640,
            WIDTH,
        )?;
        mark(events, &self.stream, Stage::Qkv)?;
        let launch = self
            .module
            .prepare_rope(LaunchConfig1D::new((rows * 9 * 32).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .rope(
                &self.stream,
                &launch,
                &mut s.qkv,
                &s.rope,
                rows,
                self.sequence as u32,
                0,
            )
            .map_err(cuda_error)?;
        let sequences = rows / self.sequence as u32;
        let launch = self
            .module
            .prepare_pisaleaves(LaunchConfig2D::new((shape.leaves(), sequences), (64, 1), 0))
            .map_err(cuda_error)?;
        self.module
            .pisaleaves(
                &self.stream,
                &launch,
                &s.qkv,
                cuda_host::RowWidth::new(&mut s.pyramid, 64),
                shape,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_pisaupper(LaunchConfig1D::new(sequences, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .pisaupper(&self.stream, &launch, &mut s.pyramid, shape)
            .map_err(cuda_error)?;
        if let Some(state) = &mut self.diffusion {
            let weights = self
                .weights
                .diffusion
                .as_ref()
                .ok_or("missing diffusion index weights")?;
            let launch = state
                .module
                .prepare_project_index(LaunchConfig1D::new(rows, 64, 0))
                .map_err(cuda_error)?;
            state
                .module
                .project_index(
                    &self.stream,
                    &launch,
                    &s.qkv,
                    &weights.index,
                    &mut state.index,
                    rows,
                )
                .map_err(cuda_error)?;
        }
        let launch = self
            .module
            .prepare_pisa_select(LaunchConfig1D::new(rows, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_select(
                &self.stream,
                &launch,
                self.diffusion.as_ref().map_or(&s.qkv, |state| &state.index),
                &s.pyramid,
                &mut s.blocks,
                shape,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::Index)?;
        let launch = self
            .module
            .prepare_pisa_multihead(LaunchConfig1D::new(rows * 2, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_multihead(
                &self.stream,
                &launch,
                &s.qkv,
                &s.qkv,
                &s.blocks,
                &mut s.attended,
                shape,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::Pisa)?;
        matmul(
            self.synth.as_ref(),
            &self.module,
            &self.stream,
            &s.attended,
            &weights.output,
            &mut s.branch,
            self.rows,
            WIDTH,
            WIDTH,
        )?;
        mark(events, &self.stream, Stage::Output)
    }

    fn moe(&mut self, layer: usize) -> CudaResult<()> {
        let s = &mut self.scratch;
        let weights = &self.weights.layers[layer];
        let rows = self.rows as u32;
        let route = RouteShape {
            rows,
            experts: EXPERTS as u32,
            top_k: 3,
        };
        let moe = MoeShape {
            row_tiles: 1,
            ..MoeShape::fbt()
        };
        matmul(
            self.synth.as_ref(),
            &self.module,
            &self.stream,
            &s.normalized,
            &weights.router,
            &mut s.route_scores,
            self.rows,
            EXPERTS,
            WIDTH,
        )?;
        let launch = self
            .module
            .prepare_routetopk(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .routetopk(
                &self.stream,
                &launch,
                &s.route_scores,
                &mut s.experts,
                &mut s.route_weights,
                &mut s.margins,
                route,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_counts(LaunchConfig1D::new(EXPERTS as u32, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .route_counts(
                &self.stream,
                &launch,
                &s.experts,
                &mut s.counts,
                &mut s.prefixes,
                rows * 3,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_layout(LaunchConfig1D::new(1, 32, 0))
            .map_err(cuda_error)?;
        self.module
            .route_layout(
                &self.stream,
                &launch,
                &s.counts,
                &mut s.offsets,
                &mut s.tiles,
                EXPERTS as u32,
                s.tile_capacity as u32,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_pack(LaunchConfig1D::new(rows * 3, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .route_pack(
                &self.stream,
                &launch,
                &s.normalized,
                &s.experts,
                &s.offsets,
                &s.prefixes,
                &mut s.mapping,
                &mut s.packed,
                rows * 3,
            )
            .map_err(cuda_error)?;
        experts(
            &self.module,
            &self.stream,
            weights,
            &s.packed,
            &s.tiles,
            &mut s.projection,
            &mut s.activation,
            &mut s.routed,
            s.tile_capacity,
            rows * 3,
            self.rows <= 2048,
        )?;
        experts(
            &self.module,
            &self.stream,
            weights,
            &s.normalized,
            &s.shared_tiles,
            &mut s.shared_projection,
            &mut s.shared_activation,
            &mut s.shared,
            self.rows.div_ceil(64),
            rows,
            false,
        )?;
        let launch = self
            .module
            .prepare_combine_routes(LaunchConfig1D::new((rows * 512).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .combine_routes(
                &self.stream,
                &launch,
                &s.route_weights,
                &s.mapping,
                &s.routed,
                &s.shared,
                &mut s.branch,
                route,
                moe,
            )
            .map_err(cuda_error)
    }
}

pub(crate) fn matmul(
    synth: Option<&crate::gemm::Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    columns: usize,
    inner: usize,
) -> CudaResult<()> {
    matmul_at(
        synth, module, stream, input, weights, output, rows, columns, inner, 0,
    )
}

fn matmul_at(
    synth: Option<&crate::gemm::Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    columns: usize,
    inner: usize,
    first: usize,
) -> CudaResult<()> {
    if first
        .checked_add(rows)
        .and_then(|v| v.checked_mul(inner))
        .is_none_or(|v| v > input.len())
    {
        return Err("matmul input range exceeds allocation".into());
    }
    // The measured recipe benefits the wide vocabulary readout. Retain the
    // CUDA-Oxide schedule for smaller projections, where synthesis regressed.
    if columns == VOCAB {
        if let Some(kernel) = synth {
            return kernel.range(stream, input, weights, output, rows, columns, inner, first);
        }
    }
    let shape = MatmulShape {
        rows: rows as u32,
        columns: columns as u32,
        inner: inner as u32,
        input_stride: inner as u32,
        weight_stride: columns as u32,
        output_stride: columns as u32,
        input_offset: (first * inner) as u64,
        weight_offset: 0,
        output_offset: 0,
    };
    let launch = module
        .prepare_matmul_turing(LaunchConfig2D::new(
            (columns.div_ceil(16) as u32, rows.div_ceil(64) as u32),
            (256, 1),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .matmul_turing(
            stream,
            &launch,
            input,
            weights,
            cuda_host::RowWidth::new(output, columns as u32),
            shape,
        )
        .map_err(cuda_error)
}

fn experts(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    weights: &Layer,
    input: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    projection: &mut DeviceBuffer<u16>,
    activation: &mut DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    capacity: usize,
    rows: u32,
    compact: bool,
) -> CudaResult<()> {
    expert_project(
        module,
        stream,
        input,
        &weights.gate,
        tiles,
        projection,
        capacity,
        0,
        compact,
    )?;
    let launch = module
        .prepare_routed_activate(LaunchConfig1D::new((rows * 216).div_ceil(256), 256, 0))
        .map_err(cuda_error)?;
    module
        .routed_activate(stream, &launch, projection, activation, rows)
        .map_err(cuda_error)?;
    expert_project(
        module,
        stream,
        activation,
        &weights.down,
        tiles,
        output,
        capacity,
        1,
        compact,
    )
}

fn expert_project(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    output: &mut DeviceBuffer<u16>,
    capacity: usize,
    mode: u32,
    compact: bool,
) -> CudaResult<()> {
    let columns: u32 = if mode == 0 { 432 } else { 512 };
    if compact {
        let launch = module
            .prepare_routed_compact(LaunchConfig2D::new(
                (columns.div_ceil(16), capacity as u32 * 4),
                (64, 1),
                0,
            ))
            .map_err(cuda_error)?;
        module
            .routed_compact(
                stream,
                &launch,
                input,
                weights,
                tiles,
                cuda_host::RowWidth::new(output, columns),
                mode,
            )
            .map_err(cuda_error)
    } else {
        let launch = module
            .prepare_routed_project(LaunchConfig2D::new(
                (columns.div_ceil(16), capacity as u32),
                (256, 1),
                0,
            ))
            .map_err(cuda_error)?;
        module
            .routed_project(
                stream,
                &launch,
                input,
                weights,
                tiles,
                cuda_host::RowWidth::new(output, columns),
                mode,
            )
            .map_err(cuda_error)
    }
}
