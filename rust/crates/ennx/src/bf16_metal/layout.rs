use super::*;

pub(super) fn finish(command: &CommandBufferRef) -> Result<(), String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        Err(format!("Metal BF16 command failed: {:?}", command.status()))
    } else {
        Ok(())
    }
}

pub(super) fn read<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
    assert!(std::mem::size_of::<T>() * count <= buffer.length() as usize);
    // All callers use initialized shared buffers after command completion.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), count).to_vec() }
}

pub(super) fn make_layout(
    blocks: &[ParamBlock],
    elements: usize,
) -> Result<(Vec<Tile>, Vec<u32>, Vec<Leaf>), String> {
    let mut covered = 0usize;
    let mut tiles = Vec::new();
    let mut offsets = vec![0u32];
    let mut leaves = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        ParamBlock::new(
            block.key,
            block.offset,
            block.len,
            block.scale,
            block.weight,
        )?;
        if block.offset != covered || block.len > u32::MAX as usize {
            return Err("BF16 blocks must be contiguous with each length fitting u32".into());
        }
        covered = covered
            .checked_add(block.len)
            .ok_or("BF16 layout overflow")?;
        let leaf = u32::try_from(index).map_err(|_| "too many BF16 blocks")?;
        for start in (0..block.len).step_by(TILE_ELEMENTS) {
            tiles.push(Tile {
                leaf,
                start: start as u32,
                length: (block.len - start).min(TILE_ELEMENTS) as u32,
                pad: 0,
            });
        }
        offsets.push(u32::try_from(tiles.len()).map_err(|_| "too many BF16 tiles")?);
        leaves.push(Leaf {
            key: block.key,
            offset: block.offset as u64,
            length: block.len as u64,
            scale: block.scale,
            weight: block.weight,
            address: u32::from(block.address.bits()),
            pad: 0,
        });
    }
    if covered == 0 || covered != elements {
        return Err("BF16 blocks must cover the complete nonempty base".into());
    }
    Ok((tiles, offsets, leaves))
}

pub(super) fn bytes<T>(count: usize) -> Result<u64, String> {
    count
        .checked_mul(size_of::<T>())
        .and_then(|x| u64::try_from(x).ok())
        .ok_or_else(|| "Metal BF16 allocation size overflow".into())
}

pub(super) fn preflight(runtime: &Runtime, sizes: &[u64], additional: u64) -> Result<(), String> {
    check_memory(
        sizes,
        additional,
        runtime.device.max_buffer_length(),
        runtime.device.current_allocated_size(),
        runtime.device.recommended_max_working_set_size(),
    )
}

pub(super) fn check_memory(
    sizes: &[u64],
    additional: u64,
    max_buffer: u64,
    current: u64,
    recommended: u64,
) -> Result<(), String> {
    if sizes.iter().any(|size| *size > max_buffer) {
        return Err(format!(
            "Metal BF16 allocation exceeds maxBufferLength={max_buffer}; each row must fit one buffer"
        ));
    }
    let total = current
        .checked_add(additional)
        .ok_or("Metal BF16 memory estimate overflow")?;
    let ceiling = (recommended * 115) / 100;
    if total > ceiling {
        return Err(format!(
            "Metal BF16 requires {additional} additional bytes; currentAllocatedSize={current}, recommendedMaxWorkingSetSize={recommended}"
        ));
    }
    Ok(())
}

pub(super) fn checked_length(length: TRLengthConfig) -> Result<TRLengthConfig, String> {
    if [length.length_min, length.length_init, length.length_max]
        .iter()
        .any(|value| !value.is_finite() || !(*value as f32).is_finite() || *value as f32 <= 0.0)
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err(
            "trust-region radii must be ordered, positive and representable as FP32".into(),
        );
    }
    let mut min = length.length_min as f32;
    let mut max = length.length_max as f32;
    if f64::from(min) < length.length_min {
        min = min.next_up();
    }
    if f64::from(max) > length.length_max {
        max = max.next_down();
    }
    if !min.is_finite() || min >= max {
        return Err("trust-region bounds must contain two distinct FP32 radii".into());
    }
    let (min, max) = (f64::from(min), f64::from(max));
    let initial = length.length_init.clamp(min, max);
    Ok(TRLengthConfig::new(initial, min, max))
}

pub(super) fn check_scores(values: &[f32], variances: &[f32]) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) || variances.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        Err("BF16 measurements must be finite with nonnegative finite variances".into())
    } else {
        Ok(())
    }
}

pub(super) fn check_ask(config: Ask) -> Result<(), String> {
    if config.neighbors == 0
        || config.neighbors > MAX_HISTORY
        || [
            config.epistemic_scale,
            config.aleatoric_scale,
            config.y_scale,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
        || !config.beta.is_finite()
    {
        return Err("Metal BF16 acquisition requires neighbors in 1..=128, finite nonnegative scales and finite beta".into());
    }
    Ok(())
}

pub(super) fn candidate_seed(root: u64, candidate: usize) -> u64 {
    crate::procedural_pool::legacy_seed(root, candidate as u32)
}

/// The same FP32 ENNX weighting and shared-per-observation Thompson draw as CUDA.
#[cfg(test)]
pub(super) fn acquisition(
    distances: &[f32],
    outcomes: &[f32],
    variances: &[f32],
    config: Ask,
) -> f32 {
    let mut indices = (0..distances.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| distances[*a].total_cmp(&distances[*b]).then(a.cmp(b)));
    indices.truncate(config.neighbors.min(distances.len()));
    let y_scale_sq = (config.y_scale * config.y_scale).max(1.0e-12);
    let weight = |index: usize| {
        let variance = 1.0e-9
            + config.epistemic_scale * distances[index]
            + config.aleatoric_scale
            + variances[index] / y_scale_sq;
        1.0 / variance.max(1.0e-12)
    };
    let (mut sum, mut value, mut reference) = (0.0f32, 0.0f32, f32::MIN_POSITIVE);
    for &index in &indices {
        let w = weight(index);
        sum += w;
        value += w * outcomes[index];
        reference = reference.max(w);
    }
    let mean = value / sum.max(1.0e-12);
    let aleatoric = indices
        .iter()
        .map(|&index| {
            (weight(index) / sum.max(1.0e-12))
                * (config.aleatoric_scale + variances[index] / y_scale_sq)
        })
        .sum::<f32>();
    let se = (1.0 / sum.max(1.0e-12) + aleatoric).sqrt() * config.y_scale;
    match config.acquisition {
        AcquisitionKind::Thompson => {
            let (mut noise, mut squared) = (0.0f32, 0.0f32);
            for &index in &indices {
                let w = weight(index) / reference;
                // Capacity two pins anchor slot 1 and reuses rejected slot 2.
                noise += w * crate::hash::normal_metric(config.seed, (index + 1) as i64, 0) as f32;
                squared += w * w;
            }
            mean + se * (noise / squared.sqrt().max(1.0e-12))
        }
        AcquisitionKind::Pareto => mean + se,
        AcquisitionKind::Ucb => mean + config.beta * se,
    }
}
