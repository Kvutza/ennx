use cuda_device::{thread, warp};

use super::{DenseTerm, DenseTile, Seed, THREADS, WARPS, dense_hash, dense_mix64, dense_sign};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bf16Leaf {
    pub key: u64,
    pub offset: u64,
    pub length: u64,
    pub scale: f32,
    pub weight: f32,
}

// SAFETY: Bf16Leaf is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Bf16Leaf {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bf16Change {
    pub squared: f32,
    pub changed: u32,
}

// SAFETY: Bf16Change is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Bf16Change {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bf16Score {
    pub row_stride: u64,
    pub coefficient: f32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub history: u32,
    pub candidates: u32,
    pub base_slot: u32,
    pub neighbors: u32,
    pub acquisition: u32,
    pub tiles: u32,
    pub resident: u32,
    pub correlated: u32,
}

// SAFETY: Bf16Score is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Bf16Score {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SearchState {
    pub length: f64,
    pub length_init: f64,
    pub length_min: f64,
    pub length_max: f64,
    pub trust_best: f64,
    pub hist_min: f64,
    pub hist_max: f64,
    pub best: f32,
    pub best_variance: f32,
    pub prev_obs: u64,
    pub successes: u32,
    pub failures: u32,
    pub restarts: u32,
    pub history: u32,
    pub status: u32,
}

// SAFETY: SearchState is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for SearchState {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TellSummary {
    pub length: f64,
    pub best: f32,
    pub best_variance: f32,
    pub history: u32,
    pub restarts: u32,
    pub restarted: u32,
    pub status: u32,
}

// SAFETY: TellSummary is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for TellSummary {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TellParams {
    pub row_stride: u64,
    pub row_len: u64,
    pub trials: u32,
    pub capacity: u32,
    pub failure_tolerance: u32,
    pub status_count: u32,
    pub correlated: u32,
    pub paired: u32,
    pub accept: u32,
    pub incumbent_value: f32,
    pub incumbent_variance: f32,
    pub improvement: f32,
    pub improvement_variance: f32,
    pub reject_is_failure: u32,
}

// SAFETY: TellParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for TellParams {}

#[inline(always)]
pub(super) fn bf16_decode(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

#[inline(always)]
pub(super) fn bf16_finite(value: u16) -> bool {
    value & 0x7f80 != 0x7f80
}

#[inline(always)]
pub(super) fn bf16_encode(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
fn bf16_next(value: u16, positive: bool) -> u16 {
    if value & 0x7fff == 0 {
        return if positive { 1 } else { 0x8001 };
    }
    let grows = (value & 0x8000 == 0) == positive;
    let candidate = if grows {
        value.wrapping_add(1)
    } else {
        value.wrapping_sub(1)
    };
    if bf16_decode(candidate).is_finite() {
        candidate
    } else if grows {
        value.wrapping_sub(1)
    } else {
        value.wrapping_add(1)
    }
}

#[inline(always)]
pub(super) fn bf16_value(
    value: u16,
    leaf_key: u64,
    element: u64,
    scale: f32,
    terms: &[DenseTerm],
    term_count: u32,
) -> u16 {
    let mut sum = 0.0_f32;
    let mut strongest = 0.0_f32;
    let mut positive = true;
    let mut term_index = 0;
    while term_index < term_count {
        let term = terms[term_index as usize];
        if term.coefficient != 0.0 {
            let direction = dense_sign(term.seed, leaf_key, element);
            sum += term.coefficient * direction;
            if term.coefficient.abs() > strongest {
                strongest = term.coefficient.abs();
                positive = (term.coefficient > 0.0) == (direction > 0.0);
            }
        }
        term_index += 1;
    }
    let candidate = bf16_encode(bf16_decode(value) + scale * sum);
    if sum == 0.0 || candidate == value {
        bf16_next(value, positive)
    } else {
        candidate
    }
}

#[inline(always)]
pub(super) fn bf16_seed(
    value: u16,
    leaf: Bf16Leaf,
    element: u64,
    seed: Seed,
    coefficient: f32,
) -> u16 {
    let seed = u64::from(seed.low) | (u64::from(seed.high) << 32);
    let direction = dense_sign(seed, leaf.key, element);
    bf16_encode(bf16_decode(value) + leaf.scale * coefficient * direction)
}

#[inline(always)]
pub(super) fn candidate_radius(state: SearchState, candidate: u32) -> f32 {
    let factor = if candidate & 1 == 0 { 0.5 } else { 2.0 };
    (state.length * factor)
        .max(state.length_min)
        .min(state.length_max) as f32
}

#[inline(always)]
pub(super) fn paired_radius(
    state: &mut SearchState,
    selected_radius: f32,
    accept: bool,
    failure_tolerance: u32,
    reject_is_failure: bool,
) {
    if accept {
        state.length = f64::from(selected_radius);
        state.failures = 0;
    } else if reject_is_failure {
        state.failures += 1;
        if state.failures >= failure_tolerance {
            state.length = (state.length * 0.5).max(state.length_min);
            state.failures = 0;
        }
    }
}

#[inline(always)]
pub(super) fn dense_normal(seed: u64, key: u64, element: u64) -> f32 {
    let first = dense_hash(seed, key, element / 2);
    let second = dense_mix64(first ^ 0xd2b7_4407_b1ce_6e93);
    let u1 = (((first >> 11) as f64 + 1.0) * (1.0 / 9007199254740992.0)) as f32;
    let u2 = (((second >> 11) as f64) * (1.0 / 9007199254740992.0)) as f32;
    let radius = (-2.0 * u1.max(1e-12).min(0.99999994).ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    radius
        * if element & 1 == 0 {
            angle.cos()
        } else {
            angle.sin()
        }
}

#[inline(always)]
pub(super) fn correlated_value(
    reference: f32,
    inverse_rms: f32,
    noise: f32,
    candidate: u32,
) -> f32 {
    if candidate < 2 {
        0.75 * (reference * inverse_rms) + 0.4375_f32.sqrt() * noise
    } else {
        noise
    }
}

#[inline(always)]
pub(super) fn bf16_candidate(
    value: u16,
    leaf: Bf16Leaf,
    element: u64,
    seed: Seed,
    coefficient: f32,
    reference: &[u16],
    inverse_rms: f32,
    correlated: u32,
    candidate: u32,
) -> u16 {
    if correlated == 0 {
        return bf16_seed(value, leaf, element, seed, coefficient);
    }
    let index = leaf.offset + element;
    let seed = u64::from(seed.low) | (u64::from(seed.high) << 32);
    let noise = dense_normal(seed ^ 0x8ebc_6af0_9c88_c6e3, leaf.key, element);
    let direction = if correlated == 1 && candidate < 2 {
        correlated_value(
            bf16_decode(reference[index as usize]),
            inverse_rms,
            noise,
            candidate,
        )
    } else {
        noise
    };
    bf16_encode(bf16_decode(value) + leaf.scale * coefficient * direction)
}

#[inline(always)]
fn insert_nearest(
    distances: *mut f32,
    indices: *mut u32,
    neighbors: u32,
    observation: u32,
    distance: f32,
) {
    let mut insert_at = neighbors;
    let mut nearest = 0u32;
    while nearest < neighbors {
        let nearest_distance = unsafe { distances.add(nearest as usize).read() };
        let nearest_index = unsafe { indices.add(nearest as usize).read() };
        if distance < nearest_distance
            || (distance == nearest_distance && observation < nearest_index)
        {
            insert_at = nearest;
            break;
        }
        nearest += 1;
    }
    if insert_at >= neighbors {
        return;
    }
    let mut move_index = neighbors - 1;
    while move_index > insert_at {
        unsafe {
            distances
                .add(move_index as usize)
                .write(distances.add((move_index - 1) as usize).read());
            indices
                .add(move_index as usize)
                .write(indices.add((move_index - 1) as usize).read());
        }
        move_index -= 1;
    }
    unsafe {
        distances.add(insert_at as usize).write(distance);
        indices.add(insert_at as usize).write(observation);
    }
}

#[inline(always)]
pub(super) fn acquisition_score(
    distances: *const f32,
    nearest_distances: *mut f32,
    nearest_indices: *mut u32,
    outcomes: &[f32],
    variances: &[f32],
    draws: &[f32],
    params: Bf16Score,
) -> f32 {
    let mut nearest = 0u32;
    while nearest < params.neighbors {
        unsafe {
            nearest_distances.add(nearest as usize).write(f32::INFINITY);
            nearest_indices.add(nearest as usize).write(0);
        }
        nearest += 1;
    }
    let mut observation = 0u32;
    while observation < params.history {
        let distance = unsafe { distances.add(observation as usize).read() };
        insert_nearest(
            nearest_distances,
            nearest_indices,
            params.neighbors,
            observation,
            distance,
        );
        observation += 1;
    }

    let mut weight_sum = 0.0f32;
    let mut weighted_value = 0.0f32;
    let mut reference_weight = f32::MIN_POSITIVE;
    nearest = 0;
    while nearest < params.neighbors {
        let distance = unsafe { nearest_distances.add(nearest as usize).read() };
        let index = unsafe { nearest_indices.add(nearest as usize).read() } as usize;
        let variance =
            1.0e-9 + params.epistemic_scale * distance + params.aleatoric_scale + variances[index];
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * outcomes[index];
        reference_weight = reference_weight.max(weight);
        nearest += 1;
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * params.y_scale;
    match params.acquisition {
        1 => {
            // Observation variances can change weight order; use the largest weight.
            let mut weighted_noise = 0.0f32;
            let mut squared_weight_sum = 0.0f32;
            nearest = 0;
            while nearest < params.neighbors {
                let distance = unsafe { nearest_distances.add(nearest as usize).read() };
                let index = unsafe { nearest_indices.add(nearest as usize).read() } as usize;
                let variance = 1.0e-9
                    + params.epistemic_scale * distance
                    + params.aleatoric_scale
                    + variances[index];
                let draw_weight = (1.0 / variance.max(1.0e-12)) / reference_weight;
                weighted_noise += draw_weight * draws[index];
                squared_weight_sum += draw_weight * draw_weight;
                nearest += 1;
            }
            mean + se * (weighted_noise / squared_weight_sum.sqrt().max(1.0e-12))
        }
        2 => mean + se,
        _ => mean + params.beta * se,
    }
}

#[inline(always)]
pub(super) fn tile_distances(
    rows: &[u16],
    history_slots: &[u32],
    reference: &[u16],
    inverse_rms: f32,
    candidate: u32,
    seed: Seed,
    leaf: Bf16Leaf,
    tile: DenseTile,
    values: *mut f32,
    distances: *mut f32,
    warp_status: *mut u32,
    params: Bf16Score,
) {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let base_offset = params.base_slot as usize * params.row_stride as usize;
    let mut tile_offset = 0u32;
    while tile_offset < tile.length {
        let local = tile_offset + thread_index;
        let mut invalid = false;
        let mut changed = false;
        if local < tile.length {
            let element = u64::from(tile.start + local);
            let index = leaf.offset + element;
            let value = bf16_candidate(
                rows[base_offset + index as usize],
                leaf,
                element,
                seed,
                params.coefficient,
                reference,
                inverse_rms,
                params.correlated,
                candidate,
            );
            invalid = !bf16_finite(value);
            changed = value != rows[base_offset + index as usize];
            unsafe { values.add(thread_index as usize).write(bf16_decode(value)) };
        }
        let flags = u32::from(warp::any(invalid)) | (u32::from(warp::any(changed)) << 1);
        if lane == 0 {
            unsafe {
                let status = warp_status.add(warp_index as usize);
                status.write(status.read() | flags);
            }
        }
        thread::sync_threads();

        let tile_length = (tile.length - tile_offset).min(THREADS);
        let mut history_base = 0u32;
        while history_base < params.history {
            let observation = history_base + warp_index;
            let mut sum = 0.0f32;
            if observation < params.history {
                let row_offset =
                    history_slots[observation as usize] as usize * params.row_stride as usize;
                let mut item = lane;
                while item < tile_length {
                    let element = u64::from(tile.start + tile_offset + item);
                    let observed = rows[row_offset + (leaf.offset + element) as usize];
                    let delta = unsafe { values.add(item as usize).read() } - bf16_decode(observed);
                    sum += delta * delta * leaf.weight;
                    item += 32;
                }
            }
            let partial = warp::reduce_sum_f32(sum);
            if lane == 0 && observation < params.history {
                unsafe {
                    let slot = distances.add(observation as usize);
                    slot.write(slot.read() + partial);
                }
            }
            history_base += WARPS as u32;
        }
        thread::sync_threads();
        tile_offset += THREADS;
    }
}

#[inline(always)]
pub(super) fn warp_invalid(status: *const u32) -> bool {
    let mut invalid = 0u32;
    let mut warp_index = 0usize;
    while warp_index < WARPS {
        invalid |= unsafe { status.add(warp_index).read() };
        warp_index += 1;
    }
    invalid & 1 != 0
}
