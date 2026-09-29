use super::*;

impl QwenEvaluator {
    pub(crate) fn new_backend(max_tokens: u32, backend: &str, tile_attn: bool) -> Result<Self> {
        let mode = match backend {
            "reference" => MpsMode::Off,
            "fp32" => MpsMode::F32,
            "fp16" => MpsMode::F16,
            _ => return Err("Qwen backend must be reference, fp32, or fp16".into()),
        };
        autoreleasepool(|| {
            Self::backend_options(QwenConfig::default(), max_tokens, mode, tile_attn)
        })
    }

    pub fn new(max_tokens: u32) -> Result<Self> {
        autoreleasepool(|| Self::new_inner(max_tokens))
    }

    pub(super) fn new_inner(max_tokens: u32) -> Result<Self> {
        Self::with_config(QwenConfig::default(), max_tokens)
    }

    pub(super) fn with_config(config: QwenConfig, max_tokens: u32) -> Result<Self> {
        Self::with_backend(config, max_tokens, mps_mode()?)
    }

    pub(super) fn with_backend(config: QwenConfig, max_tokens: u32, mode: MpsMode) -> Result<Self> {
        let tile_attn = std::env::var_os("ENNX_QWEN_TILE_ATTN").is_some();
        Self::backend_options(config, max_tokens, mode, tile_attn)
    }

    pub(super) fn backend_options(
        config: QwenConfig,
        max_tokens: u32,
        mode: MpsMode,
        tile_attn: bool,
    ) -> Result<Self> {
        config.validate(max_tokens)?;
        if tile_attn && config.head_dim() != 128 {
            return Err("Qwen tiled attention requires head dimension 128".into());
        }
        let layout = Layout::new(config)?;
        let prefill_chunk = if mode == MpsMode::Off {
            PREFILL_CHUNK
        } else {
            MPS_CHUNK
        };
        let sizes = workspace_sizes(config, max_tokens, prefill_chunk)?;
        let mut workspace_bytes = sizes.iter().try_fold(0u64, |sum, &size| {
            sum.checked_add(size as u64)
                .ok_or_else(|| "Qwen workspace size overflow".to_string())
        })?;
        let runtime = Runtime::shared()?;
        if mode != MpsMode::Off && !metal::mps::mps_supports_device(&runtime.device) {
            return Err("MPS does not support the Qwen Metal device".into());
        }
        let (weight_elements, half_elements, mps_bytes) =
            mps_storage(config, max_tokens, prefill_chunk, mode)?;
        workspace_bytes = workspace_bytes
            .checked_add(mps_bytes)
            .ok_or("Qwen MPS workspace size overflow")?;
        trace("Qwen runtime acquired");
        let rope_table_values = rope_table(config, max_tokens)?;
        let rope_table = runtime.buffer_with(&rope_table_values);
        let rope_table_bytes = std::mem::size_of_val(rope_table_values.as_slice()) as u64;
        let weight_bytes = product(&[layout.len, 2])? as u64;
        let limit = runtime.device.recommended_max_working_set_size();
        let allocated = runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(workspace_bytes)
            .and_then(|value| value.checked_add(rope_table_bytes))
            .and_then(|value| value.checked_add(weight_bytes))
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen Metal evaluation exceeds the conservative working-set budget".into());
        }
        if weight_bytes > runtime.device.max_buffer_length() {
            return Err("Qwen BF16 weights exceed Metal maxBufferLength".into());
        }
        if rope_table_bytes > runtime.device.max_buffer_length() || rope_table.contents().is_null()
        {
            return Err("Qwen RoPE table exceeds Metal buffer limits".into());
        }
        let pipelines = compile_pipelines(&runtime)?;
        let mut workspace = Vec::with_capacity(sizes.len());
        trace("allocating Qwen workspace");
        for size in sizes {
            let buffer = runtime.buffer::<u8>(size);
            if buffer.contents().is_null() {
                return Err(format!(
                    "Metal could not allocate {size} Qwen workspace bytes"
                ));
            }
            workspace.push(buffer);
        }
        let mps = allocate_mps(&runtime, mode, weight_elements, half_elements)?;
        trace("Qwen evaluator ready");
        Ok(Self {
            runtime,
            config,
            max_tokens,
            layout,
            workspace,
            rope_table,
            workspace_bytes,
            pipelines,
            blocks: Vec::new(),
            loss_cache: None,
            loss_profile: None,
            mps,
            prefill_chunk,
            tile_attn,
            frozen_readout: None,
        })
    }

    pub fn weights_len(&self) -> usize {
        self.layout.len
    }

    /// Only for weights whose contents will remain immutable for this evaluator.
    pub(crate) fn prepare_readout(&mut self, weights: &Buffer) -> Result<()> {
        if self.frozen_readout.is_some() {
            return Err("frozen readout has already been prepared".into());
        }
        let readout = readout::Readout::new(self, weights)?;
        self.workspace_bytes += readout.bytes();
        self.frozen_readout = Some(readout);
        Ok(())
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub fn device_name(&self) -> &str {
        &self.runtime.info().name
    }

    pub fn vocab(&self) -> usize {
        self.config.vocab as usize
    }

    pub fn blocks(&self) -> Vec<(u64, usize, usize, f32, f32)> {
        self.blocks
            .iter()
            .map(|block| {
                (
                    block.key,
                    block.offset,
                    block.length,
                    block.scale,
                    block.weight,
                )
            })
            .collect()
    }

    pub fn loss_profile(&self) -> Option<QwenLossProfile> {
        self.loss_profile
    }

    pub fn load_weights(&mut self, path: &Path) -> Result<Buffer> {
        autoreleasepool(|| self.load_file(path))
    }

    pub(super) fn load_file(&mut self, path: &Path) -> Result<Buffer> {
        let file = File::open(path).map_err(|error| format!("Qwen weights open: {error}"))?;
        let mapping =
            unsafe { Mmap::map(&file) }.map_err(|error| format!("Qwen weights mmap: {error}"))?;
        if mapping.len() < 8 {
            return Err("Qwen weights are missing a safetensors header".into());
        }
        let header_size = usize::try_from(u64::from_le_bytes(
            mapping[..8]
                .try_into()
                .map_err(|_| "Invalid Qwen safetensors header length")?,
        ))
        .map_err(|_| "Qwen safetensors header is too large")?;
        let payload_offset = 8usize
            .checked_add(header_size)
            .ok_or("Qwen safetensors header offset overflow")?;
        if payload_offset > mapping.len() {
            return Err("Qwen safetensors header exceeds the file".into());
        }
        let header: BTreeMap<String, ennx_wire::json::Value> =
            ennx_wire::json::from_slice(&mapping[8..payload_offset])
                .map_err(|error| format!("Qwen safetensors header: {error}"))?;
        let mut records = BTreeMap::<String, SafeTensorRecord>::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let record: SafeTensorRecord = ennx_wire::json::from_value(value)
                .map_err(|error| format!("Qwen tensor {name} metadata: {error}"))?;
            records.insert(name, record);
        }
        let expected = &self.layout.tensors;
        for name in records.keys() {
            if !expected.contains_key(name) && name != "lm_head.weight" {
                return Err(format!("Unexpected Qwen tensor: {name}"));
            }
        }
        for name in expected.keys() {
            if !records.contains_key(name) {
                return Err(format!("Missing Qwen tensor: {name}"));
            }
        }
        let embedding_range = records
            .get("model.embed_tokens.weight")
            .ok_or("Missing Qwen embedding tensor")?
            .data_offsets;
        let embedding_bytes = tensor_bytes(
            &mapping,
            payload_offset,
            embedding_range,
            "model.embed_tokens.weight",
        )?;
        if let Some(record) = records.get("lm_head.weight") {
            let lm_head = tensor_bytes(
                &mapping,
                payload_offset,
                record.data_offsets,
                "lm_head.weight",
            )?;
            if record.dtype != "BF16"
                || product(&record.shape)? != expected["model.embed_tokens.weight"].1
                || lm_head != embedding_bytes
            {
                return Err("Tied Qwen embedding and lm_head weights differ".into());
            }
        }

        let bytes = product(&[self.layout.len, 2])? as u64;
        let limit = self.runtime.device.recommended_max_working_set_size();
        let allocated = self.runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(bytes)
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen Metal evaluation exceeds the conservative working-set budget".into());
        }
        let buffer = self.runtime.buffer::<u16>(self.layout.len);
        if buffer.contents().is_null() {
            return Err("Metal could not allocate Qwen weights".into());
        }
        let tensor_count = expected.len();
        let mut blocks = Vec::with_capacity(tensor_count);
        for (name, &(offset, length)) in expected {
            let record = records
                .get(name)
                .ok_or_else(|| format!("Missing Qwen tensor: {name}"))?;
            if record.dtype != "BF16" || product(&record.shape)? != length {
                return Err(format!("Unexpected dtype or shape for Qwen tensor {name}"));
            }
            let bytes = tensor_bytes(&mapping, payload_offset, record.data_offsets, name)?;
            let (scale, nonzero) = bf16_stats(bytes, name)?;
            if !nonzero {
                return Err(format!("Qwen tensor {name} is all zero"));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>().add(offset * 2),
                    length * 2,
                );
            }
            blocks.push(QwenBlock {
                key: crate::hash::tensor_key(name),
                offset,
                length,
                scale,
                weight: 1.0 / (tensor_count as f32 * length as f32 * scale * scale),
            });
        }
        self.blocks = blocks;
        Ok(buffer)
    }

    pub fn check_weights(&self, weights: &Buffer) -> Result<()> {
        if weights.length() != self.weights_len() as u64 * 2 {
            return Err(format!(
                "Qwen requires {} contiguous Metal BF16 weights",
                self.weights_len()
            ));
        }
        if weights.device().registry_id() != self.runtime.device.registry_id() {
            return Err("Qwen weights must belong to the evaluator's Metal device".into());
        }
        Ok(())
    }

    pub(super) fn check_tokens(&self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.max_tokens as usize
            || tokens
                .iter()
                .any(|&token| token < 0 || token as u32 >= self.config.vocab)
        {
            return Err(
                "Qwen tokens must be nonempty, in vocabulary, and within max_tokens".into(),
            );
        }
        Ok(())
    }
}

fn mps_storage(
    config: QwenConfig,
    max_tokens: u32,
    prefill_chunk: u32,
    mode: MpsMode,
) -> Result<(usize, usize, u64)> {
    let mps_elements = if mode != MpsMode::Off {
        product(&[config.hidden as usize, config.intermediate as usize])?
    } else {
        0
    };
    let weight_elements = if mode == MpsMode::F16 {
        mps_elements
            .checked_mul(2)
            .ok_or("Qwen MPS weight workspace overflow")?
    } else {
        mps_elements
    };
    let half_elements = product(&[
        prefill_chunk.min(max_tokens) as usize,
        config.intermediate as usize,
    ])?;
    let half_storage = weight_elements
        .checked_add(
            half_elements
                .checked_mul(2)
                .ok_or("Qwen MPS half workspace overflow")?,
        )
        .ok_or("Qwen MPS half workspace overflow")?;
    let mps_bytes = match mode {
        MpsMode::Off => 0,
        MpsMode::F32 => product(&[mps_elements, size_of::<f32>()])?,
        MpsMode::F16 => product(&[half_storage, size_of::<u16>()])?,
    } as u64;
    Ok((weight_elements, half_elements, mps_bytes))
}

fn compile_pipelines(runtime: &Runtime) -> Result<BTreeMap<&'static str, ComputePipelineState>> {
    let mut pipelines = BTreeMap::new();
    for name in [
        "flame_linear",
        "qwen_widen",
        "qwen_bf16_f16",
        "qwen_f32_to_f16",
        "qwen_f16_to_f32",
        "qwen_gemv",
        "qwen_gemv_rows",
        "qwen_mlp_rows",
        "qwen_simd_gemm",
        "qwen_qkv",
        "flame_matmul",
        "flame_softmax",
        "flame_xent",
        "flame_mean",
        "qwen_embedding",
        "qwen_rms",
        "qwen_bias",
        "qwen_rope",
        "qwen_cache",
        "qwen_arow",
        "qwen_atile",
        "qwen_attn16",
        "qwen_drope",
        "qwen_dattn",
        "qwen_argmax",
        "qwen_argn",
        "qwen_repkv",
        "qwen_aout",
        "qwen_silu",
        "qwen_silu16",
        "qwen_residual",
    ] {
        trace(&format!("Qwen pipeline {name}"));
        let pipeline = runtime.precise(source_for(name), "Qwen", name)?;
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < 256
        {
            return Err("Qwen requires Apple GPU 32-lane SIMD groups and 256-thread groups".into());
        }
        pipelines.insert(name, pipeline);
    }
    if pipelines["qwen_attn16"].static_threadgroup_memory_length()
        > runtime.device.max_threadgroup_memory_length()
    {
        return Err("Qwen tiled attention exceeds threadgroup memory".into());
    }
    Ok(pipelines)
}

fn allocate_mps(
    runtime: &Runtime,
    mode: MpsMode,
    weight_elements: usize,
    half_elements: usize,
) -> Result<Option<MpsPrefill>> {
    let mps = if mode != MpsMode::Off {
        let weights = match mode {
            MpsMode::F32 => runtime.buffer::<f32>(weight_elements),
            MpsMode::F16 => runtime.buffer::<u16>(weight_elements),
            MpsMode::Off => unreachable!(),
        };
        if weights.contents().is_null() {
            return Err("Metal could not allocate Qwen MPS weight scratch".into());
        }
        let input = (mode == MpsMode::F16).then(|| runtime.buffer::<u16>(half_elements));
        let output = (mode == MpsMode::F16).then(|| runtime.buffer::<u16>(half_elements));
        if input
            .as_ref()
            .into_iter()
            .chain(output.as_ref())
            .any(|buffer| buffer.contents().is_null())
        {
            return Err("Metal could not allocate Qwen MPS half scratch".into());
        }
        Some(MpsPrefill {
            weights,
            input,
            output,
            matmul: RefCell::new(MpsMatmul::default()),
            mode,
        })
    } else {
        None
    };
    Ok(mps)
}
