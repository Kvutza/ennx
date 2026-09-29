use super::*;
use ennx_cuda_kernels::diffusion_model;

pub(super) struct DiffusionState {
    pub module: diffusion_model::LoadedModule,
    pub confidence: DeviceBuffer<f32>,
    pub index: DeviceBuffer<u16>,
    pub block: u32,
}

impl DiffusionState {
    fn new(model: &FbtModel, block: usize) -> CudaResult<Self> {
        // SAFETY: the generated module contains these exact kernel bindings.
        let module = unsafe { diffusion_model::load(&model.context) }.map_err(cuda_error)?;
        Ok(Self {
            module,
            confidence: DeviceBuffer::zeroed(&model.stream, model.rows).map_err(cuda_error)?,
            index: DeviceBuffer::zeroed(&model.stream, model.rows * 640).map_err(cuda_error)?,
            block: block as u32,
        })
    }
}

impl FbtModel {
    pub fn denoise(
        &mut self,
        prompt: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
        steps: usize,
        block: usize,
        soft: bool,
    ) -> CudaResult<GenerationOutput> {
        if self.weights.diffusion.is_none()
            || self.rows != self.sequence
            || prompt.is_empty()
            || prompt.len() >= self.rows
            || prompt.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
            || !(1..=64).contains(&steps)
            || !(128..=4096).contains(&block)
            || !block.is_power_of_two()
            || prompt.len() % block != 0
        {
            return Err("diffusion requires its own checkpoint, one complete context, a block-aligned prompt, block=128..4096, steps=1..64, and finite nonnegative temperature".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        self.diffusion = Some(DiffusionState::new(self, block)?);
        let wall = Instant::now();
        let mut tokens = vec![u32::MAX; self.rows];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let mut confidence = vec![0.0; self.rows];
        confidence[..prompt.len()].fill(1.0);
        copy_prefix(&self.scratch.tokens, &tokens, &self.stream)?;
        copy_prefix(
            &self.diffusion.as_ref().unwrap().confidence,
            &confidence,
            &self.stream,
        )?;
        let start = timing_event(&self.stream)?;
        for step in 0..steps {
            let step_seed = mix(seed ^ step as u64 ^ 0x6469_6666);
            self.enqueue(&plan, temperature, step_seed)?;
            let proposed = read_prefix(&self.scratch.sampled, &self.stream, self.rows)?;
            confidence = read_prefix(
                &self.diffusion.as_ref().unwrap().confidence,
                &self.stream,
                self.rows,
            )?;
            for position in prompt.len()..self.rows {
                if proposed[position] >= VOCAB as u32 || !confidence[position].is_finite() {
                    return Err("diffusion produced an invalid token or probability".into());
                }
                tokens[position] = proposed[position];
                confidence[position] = if step + 1 == steps || !soft {
                    1.0
                } else {
                    confidence[position].clamp(0.0, 1.0)
                };
            }
            confidence[..prompt.len()].fill(1.0);
            copy_prefix(&self.scratch.tokens, &tokens, &self.stream)?;
            copy_prefix(
                &self.diffusion.as_ref().unwrap().confidence,
                &confidence,
                &self.stream,
            )?;
        }
        let end = timing_event(&self.stream)?;
        self.stream.synchronize().map_err(cuda_error)?;
        self.context.check_err().map_err(cuda_error)?;
        self.diffusion = None;
        Ok(GenerationOutput {
            tokens,
            frontiers: vec![prompt.len(), self.rows],
            prompt_tokens: prompt.len(),
            passes: steps,
            visits: plan.len(),
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            first_wave_ms: None,
            chunk_profiles: Vec::new(),
        })
    }
}

pub(super) fn mix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
