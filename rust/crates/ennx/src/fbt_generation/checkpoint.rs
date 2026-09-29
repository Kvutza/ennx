use super::generation::*;

pub(super) fn verify_policy(config: &GenerationConfig) -> Result<(), String> {
    if config.purpose != crate::config::GenerationPurpose::CodingOptimization {
        return Ok(());
    }
    crate::BasePolicyManifest::load_verified(
        config
            .qualification_manifest
            .as_deref()
            .ok_or("missing base-policy qualification manifest")?,
        config
            .checkpoint
            .as_deref()
            .ok_or("missing qualified checkpoint")?,
    )?;
    Ok(())
}

pub(super) fn checkpoint_layout(
    architecture: ResidualArchitecture,
) -> Vec<(&'static str, Vec<usize>)> {
    let layers = MODEL_LAYERS as usize;
    let width = WIDTH as usize;
    let experts = routing::MOE_EXPERTS as usize;
    let mut layout = vec![
        (
            "router",
            vec![layers, width, routing::ROUTED_EXPERTS as usize],
        ),
        ("qkv", vec![layers, width, QKV_WIDTH as usize]),
        ("attention_output", vec![layers, width, width]),
        (
            "expert_gate_up",
            vec![layers, experts, width, routing::MOE_GATEUP as usize],
        ),
        (
            "expert_down",
            vec![layers, experts, routing::ROUTED_WIDTH as usize, width],
        ),
        ("attention_norm", vec![layers, width]),
        ("ffn_norm", vec![layers, width]),
        ("embedding_readout", vec![width, VOCAB as usize]),
    ];
    if architecture.is_multistream() {
        layout.extend([
            (
                "mhc_attention_predictor",
                vec![layers, MHC_INPUT as usize, MHC_COEFFICIENTS as usize],
            ),
            (
                "mhc_attention_bias",
                vec![layers, MHC_COEFFICIENTS as usize],
            ),
            ("mhc_attention_control", vec![layers, 4]),
            (
                "mhc_moe_predictor",
                vec![layers, MHC_INPUT as usize, MHC_COEFFICIENTS as usize],
            ),
            ("mhc_moe_bias", vec![layers, MHC_COEFFICIENTS as usize]),
            ("mhc_moe_control", vec![layers, 4]),
        ]);
    } else {
        layout.extend([
            ("feedback_state", vec![width, width]),
            ("feedback_gate", vec![width, width]),
        ]);
    }
    layout.push(("final_norm", vec![width]));
    if architecture == ResidualArchitecture::DiffusionMhc4 {
        layout.push(("mask_embed", vec![width]));
        layout.push(("index_query", vec![64, 64]));
    }
    layout
}

pub(super) fn load_checkpoint(weights: &CandidateWeights, path: &Path) -> Result<(), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len()
        == weights.architecture.parameter_count() as u64 * 2
    {
        return Err("raw v1 checkpoint has no positional-encoding metadata; use a RoPE v2 Safetensors checkpoint".into());
    }
    drop(file);
    let checkpoint = crate::tensor_store::SafeTensors::open(path)?;
    let metadata = checkpoint.metadata();
    let format = match weights.architecture {
        ResidualArchitecture::Legacy
        | ResidualArchitecture::Residual1
        | ResidualArchitecture::ProjectedBoundary => "ennx.fbt-pisa1-rope.v2",
        ResidualArchitecture::Hc4 => "ennx.fbt-pisa1-hc4-rope.v1",
        ResidualArchitecture::Mhc4 => "ennx.fbt-pisa1-mhc4-rope.v1",
        ResidualArchitecture::LoopedMhc4 => "ennx.fbt-pisa1-looped-mhc4-rope.v1",
        ResidualArchitecture::DiffusionMhc4 => "ennx.fbt-pisa1-diffusion-mhc4-rope.v1",
    };
    if metadata.get("format").map(String::as_str) != Some(format)
        || metadata.get("position_encoding").map(String::as_str) != Some("rope")
        || metadata.get("rope_base").map(String::as_str) != Some("10000")
        || metadata.get("rope_dimensions").map(String::as_str) != Some("64")
    {
        return Err("checkpoint architecture is not ENNX PISA-1 RoPE v2".into());
    }
    let layout = checkpoint_layout(weights.architecture);
    for (((name, buffer, _), (expected_name, expected_shape)), index) in
        weights.tensors().into_iter().zip(&layout).zip(0usize..)
    {
        if name != *expected_name {
            return Err(format!("checkpoint layout mismatch at tensor {index}"));
        }
        let (tensor, bytes) = checkpoint
            .tensor(name)
            .ok_or_else(|| format!("checkpoint is missing tensor {name:?}"))?;
        if tensor.dtype != crate::tensor_store::DType::F16
            || tensor.shape != *expected_shape
            || bytes.len() != buffer.length() as usize
        {
            return Err(format!(
                "checkpoint tensor {name:?} has the wrong dtype or shape"
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>(),
                bytes.len(),
            );
        }
    }
    Ok(())
}

pub(super) fn save_checkpoint(
    row: &BufferRef,
    path: &Path,
    architecture: ResidualArchitecture,
) -> Result<(), String> {
    let bytes =
        unsafe { std::slice::from_raw_parts(row.contents().cast::<u8>(), row.length() as usize) };
    let layout = checkpoint_layout(architecture);
    let mut offset = 0usize;
    let mut tensors = Vec::with_capacity(layout.len());
    for (name, shape) in &layout {
        let length = shape
            .iter()
            .try_fold(2usize, |bytes, &dimension| bytes.checked_mul(dimension))
            .ok_or("checkpoint tensor length overflow")?;
        let end = offset
            .checked_add(length)
            .ok_or("checkpoint tensor range overflow")?;
        if end > bytes.len() {
            return Err(format!(
                "checkpoint tensor {name:?} exceeds the candidate row"
            ));
        }
        tensors.push(crate::tensor_store::TensorData {
            name,
            dtype: crate::tensor_store::DType::F16,
            shape,
            bytes: &bytes[offset..end],
        });
        offset = end;
    }
    if offset != bytes.len() {
        return Err("checkpoint layout does not cover the candidate row".into());
    }
    let format = match architecture {
        ResidualArchitecture::Legacy
        | ResidualArchitecture::Residual1
        | ResidualArchitecture::ProjectedBoundary => "ennx.fbt-pisa1-rope.v2",
        ResidualArchitecture::Hc4 => "ennx.fbt-pisa1-hc4-rope.v1",
        ResidualArchitecture::Mhc4 => "ennx.fbt-pisa1-mhc4-rope.v1",
        ResidualArchitecture::LoopedMhc4 => "ennx.fbt-pisa1-looped-mhc4-rope.v1",
        ResidualArchitecture::DiffusionMhc4 => "ennx.fbt-pisa1-diffusion-mhc4-rope.v1",
    };
    crate::tensor_store::write_safetensors(
        path,
        BTreeMap::from([
            ("format".into(), format.into()),
            (
                "parameters".into(),
                architecture.parameter_count().to_string(),
            ),
            ("position_encoding".into(), "rope".into()),
            ("rope_base".into(), "10000".into()),
            ("rope_dimensions".into(), HEAD_DIM.to_string()),
        ]),
        &tensors,
    )
}
