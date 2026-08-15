use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};

pub const MODULE_NAME: &str = env!("CARGO_PKG_NAME");
pub const THREADS: u32 = 256;
pub const MAX_HISTORY: usize = 128;
pub const MAX_CENTER_DEPTH: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Seed {
    pub low: u32,
    pub high: u32,
}

// SAFETY: Seed is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Seed {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CenterStep {
    pub parent: u32,
    pub seed: Seed,
}

// SAFETY: CenterStep is repr(C) and contains only DeviceCopy fields.
unsafe impl cuda_core::DeviceCopy for CenterStep {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Selection {
    pub index: u32,
    pub score: f32,
}

// SAFETY: Selection is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Selection {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    pub byte_offset: u32,
    pub element_offset: u32,
    pub length: u32,
    pub bits: u32,
    pub encoding: u32,
    pub scale: f32,
    pub weight: f32,
    pub whole: u32,
    pub threshold: u32,
}

// SAFETY: Leaf is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Leaf {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

// SAFETY: Tile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Tile {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseLeaf {
    pub key: u64,
    pub offset: u64,
    pub length: u64,
    pub scale: f32,
    pub pad: u32,
}

// SAFETY: DenseLeaf is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseLeaf {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTerm {
    pub seed: u64,
    pub coefficient: f32,
    pub pad: u32,
}

// SAFETY: DenseTerm is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseTerm {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

// SAFETY: DenseTile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseTile {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseLinearParams {
    pub rows: u32,
    pub columns: u32,
    pub has_bias: u32,
    pub term_count: u32,
    pub weight_key: u64,
    pub weight_start: u64,
    pub bias_key: u64,
    pub bias_start: u64,
    pub weight_scale: f32,
    pub bias_scale: f32,
    pub pad0: u32,
    pub pad1: u32,
}

// SAFETY: DenseLinearParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseLinearParams {}

const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline(always)]
fn trial_hash(seed_low: u32, seed_high: u32, element: u32) -> u32 {
    let mut value = seed_low ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= seed_high;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

#[inline(always)]
fn decode_fp8_e4m3(code: u32) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2.0_f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

#[inline(always)]
fn decode_fp8_e5m2(code: u32) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 2) & 0x1f;
    let mant = code & 0x03;
    if exp == 0 {
        sign * (mant as f32 / 4.0) * 2.0_f32.powi(-14)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant as f32 / 4.0) * 2.0_f32.powi(exp as i32 - 15)
    }
}

#[inline(always)]
fn decode_code(code: u32, leaf: Leaf) -> f32 {
    let value = match leaf.encoding {
        2 => FP4_E2M1[(code & 0x0f) as usize],
        3 => decode_fp8_e4m3(code),
        4 => decode_fp8_e5m2(code),
        _ => code as f32,
    };
    value * leaf.scale
}

#[inline(always)]
fn code_at(rows: &[u8], row_offset: usize, leaf: Leaf, element: u32) -> u32 {
    if leaf.bits == 4 {
        let byte = rows[row_offset + leaf.byte_offset as usize + (element / 2) as usize];
        u32::from((byte >> ((element & 1) * 4)) & 0x0f)
    } else {
        u32::from(rows[row_offset + leaf.byte_offset as usize + element as usize])
    }
}

#[inline(always)]
unsafe fn code_at_ptr(rows: *const u8, row_offset: usize, leaf: Leaf, element: u32) -> u32 {
    let index = row_offset
        + leaf.byte_offset as usize
        + if leaf.bits == 4 {
            (element / 2) as usize
        } else {
            element as usize
        };
    let byte = unsafe { rows.add(index).read() };
    if leaf.bits == 4 {
        u32::from((byte >> ((element & 1) * 4)) & 0x0f)
    } else {
        u32::from(byte)
    }
}

#[inline(always)]
fn perturb_code(
    code: u32,
    seed_low: u32,
    seed_high: u32,
    element: u32,
    bits: u32,
    whole: u32,
    threshold: u32,
) -> u32 {
    let random = trial_hash(seed_low, seed_high, element);
    let amount = whole + u32::from((random >> 1) < (threshold >> 1));
    if amount == 0 {
        return code;
    }

    let max_code = (1_u32 << bits) - 1;
    if random & 1 == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

#[inline(always)]
fn resolve_center(
    mut code: u32,
    centers: &[CenterStep],
    mut center: u32,
    element: u32,
    leaf: Leaf,
) -> u32 {
    let mut chain = [Seed { low: 0, high: 0 }; MAX_CENTER_DEPTH];
    let mut depth = 0_usize;
    while center != u32::MAX && depth < MAX_CENTER_DEPTH {
        let step = centers[center as usize];
        chain[depth] = step.seed;
        center = step.parent;
        depth += 1;
    }
    while depth > 0 {
        depth -= 1;
        let seed = chain[depth];
        code = perturb_code(
            code,
            seed.low,
            seed.high,
            element,
            leaf.bits,
            leaf.whole,
            leaf.threshold,
        );
    }
    code
}

#[inline(always)]
fn dense_mix64(input: u64) -> u64 {
    let mut value = input.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline(always)]
fn dense_sign(seed: u64, leaf: u64, element: u64) -> f32 {
    let leaf = dense_mix64(leaf ^ 0xd6e8_feb8_6659_fd93);
    let element = dense_mix64(element ^ 0xa076_1d64_78bd_642f);
    if dense_mix64(seed ^ leaf ^ element) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

#[inline(always)]
fn dense_next_finite(value: f32, positive: bool) -> f32 {
    if value == 0.0 {
        return f32::from_bits(if positive { 1 } else { 0x8000_0001 });
    }
    let bits = if (value > 0.0) == positive {
        value.to_bits().wrapping_add(1)
    } else {
        value.to_bits().wrapping_sub(1)
    };
    let candidate = f32::from_bits(bits);
    if candidate.is_finite() {
        candidate
    } else if (value > 0.0) == positive {
        f32::from_bits(value.to_bits().wrapping_sub(1))
    } else {
        f32::from_bits(value.to_bits().wrapping_add(1))
    }
}

#[inline(always)]
fn dense_value(
    value: f32,
    leaf_key: u64,
    element: u64,
    scale: f32,
    terms: &[DenseTerm],
    term_count: u32,
) -> f32 {
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
    let candidate = value + scale * sum;
    if sum == 0.0 || candidate == value {
        dense_next_finite(value, positive)
    } else {
        candidate
    }
}

#[cuda_module]
pub mod trials {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score_trials(
        rows: &[u8],
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        draws: &[f32],
        leaves: &[Leaf],
        centers: &[CenterStep],
        candidate_centers: &[u32],
        mut scores: DisjointSlice<f32>,
        row_bytes: u32,
        history: u32,
        candidates: u32,
        base_slot: u32,
        center_count: u32,
        neighbors: u32,
        acquisition: u32,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
    ) {
        static mut CANDIDATE_VALUES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;

        let candidate = thread::blockIdx_x();
        if candidate >= candidates {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;

        if thread_index < history {
            unsafe {
                DISTANCES[thread_index as usize] = 0.0;
            }
        }
        thread::sync_threads();

        let base_offset = base_slot as usize * row_bytes as usize;
        let seed = seeds[candidate as usize];
        let center = if center_count == 0 {
            u32::MAX
        } else {
            candidate_centers[candidate as usize]
        };

        let mut leaf_index = 0;
        while leaf_index < leaves.len() {
            let leaf = leaves[leaf_index];
            if leaf.bits == 4 {
                let bytes = leaf.length.div_ceil(2);
                let mut tile_byte = 0;
                while tile_byte < bytes {
                    let local_byte = thread_index;
                    let byte = tile_byte + local_byte;
                    if byte < bytes {
                        let first = byte * 2;
                        let base_byte =
                            rows[base_offset + leaf.byte_offset as usize + byte as usize];
                        let low_base = resolve_center(
                            u32::from(base_byte & 0x0f),
                            centers,
                            center,
                            leaf.element_offset + first,
                            leaf,
                        );
                        let low = perturb_code(
                            low_base,
                            seed.low,
                            seed.high,
                            leaf.element_offset + first,
                            leaf.bits,
                            leaf.whole,
                            leaf.threshold,
                        );
                        unsafe {
                            CANDIDATE_VALUES[(local_byte * 2) as usize] = decode_code(low, leaf);
                        }
                        if first + 1 < leaf.length {
                            let high_base = resolve_center(
                                u32::from(base_byte >> 4),
                                centers,
                                center,
                                leaf.element_offset + first + 1,
                                leaf,
                            );
                            let high = perturb_code(
                                high_base,
                                seed.low,
                                seed.high,
                                leaf.element_offset + first + 1,
                                leaf.bits,
                                leaf.whole,
                                leaf.threshold,
                            );
                            unsafe {
                                CANDIDATE_VALUES[(local_byte * 2 + 1) as usize] =
                                    decode_code(high, leaf);
                            }
                        }
                    }
                    thread::sync_threads();

                    let tile_elements = (leaf.length - tile_byte * 2).min(THREADS * 2);
                    let tile_bytes = tile_elements.div_ceil(2);
                    let mut history_base = 0;
                    while history_base < history {
                        let h = history_base + warp_index;
                        let mut sum = 0.0_f32;
                        if h < history {
                            let observation_offset =
                                history_slots[h as usize] as usize * row_bytes as usize;
                            let mut local = lane;
                            while local < tile_bytes {
                                let observed = rows[observation_offset
                                    + leaf.byte_offset as usize
                                    + tile_byte as usize
                                    + local as usize];
                                let first = local * 2;
                                let low_delta = unsafe { CANDIDATE_VALUES[first as usize] }
                                    - decode_code(u32::from(observed & 0x0f), leaf);
                                sum += low_delta * low_delta * leaf.weight;
                                if first + 1 < tile_elements {
                                    let high_delta =
                                        unsafe { CANDIDATE_VALUES[(first + 1) as usize] }
                                            - decode_code(u32::from(observed >> 4), leaf);
                                    sum += high_delta * high_delta * leaf.weight;
                                }
                                local += 32;
                            }
                        }
                        let partial = warp::reduce_sum_f32(sum);
                        if lane == 0 && h < history {
                            unsafe {
                                DISTANCES[h as usize] += partial;
                            }
                        }
                        history_base += 8;
                    }
                    thread::sync_threads();
                    tile_byte += THREADS;
                }
            } else {
                let mut tile_element = 0;
                while tile_element < leaf.length {
                    let local = thread_index;
                    let element = tile_element + local;
                    if element < leaf.length {
                        let base_code = resolve_center(
                            code_at(rows, base_offset, leaf, element),
                            centers,
                            center,
                            leaf.element_offset + element,
                            leaf,
                        );
                        let value = perturb_code(
                            base_code,
                            seed.low,
                            seed.high,
                            leaf.element_offset + element,
                            leaf.bits,
                            leaf.whole,
                            leaf.threshold,
                        );
                        unsafe {
                            CANDIDATE_VALUES[local as usize] = decode_code(value, leaf);
                        }
                    }
                    thread::sync_threads();

                    let tile_elements = (leaf.length - tile_element).min(THREADS);
                    let mut history_base = 0;
                    while history_base < history {
                        let h = history_base + warp_index;
                        let mut sum = 0.0_f32;
                        if h < history {
                            let observation_offset =
                                history_slots[h as usize] as usize * row_bytes as usize;
                            let mut local = lane;
                            while local < tile_elements {
                                let observed_code =
                                    code_at(rows, observation_offset, leaf, tile_element + local);
                                let delta = unsafe { CANDIDATE_VALUES[local as usize] }
                                    - decode_code(observed_code, leaf);
                                sum += delta * delta * leaf.weight;
                                local += 32;
                            }
                        }
                        let partial = warp::reduce_sum_f32(sum);
                        if lane == 0 && h < history {
                            unsafe {
                                DISTANCES[h as usize] += partial;
                            }
                        }
                        history_base += 8;
                    }
                    thread::sync_threads();
                    tile_element += THREADS;
                }
            }
            leaf_index += 1;
        }

        if thread_index == 0 {
            let mut k = 0;
            while k < neighbors {
                unsafe {
                    NEAREST_DISTANCES[k as usize] = f32::INFINITY;
                    NEAREST_INDICES[k as usize] = 0;
                }
                k += 1;
            }
            let mut h = 0;
            while h < history {
                let distance = unsafe { DISTANCES[h as usize] };
                let mut insert_at = neighbors;
                let mut nearest = 0;
                while nearest < neighbors {
                    let nearest_distance = unsafe { NEAREST_DISTANCES[nearest as usize] };
                    let nearest_index = unsafe { NEAREST_INDICES[nearest as usize] };
                    if distance < nearest_distance
                        || (distance == nearest_distance && h < nearest_index)
                    {
                        insert_at = nearest;
                        break;
                    }
                    nearest += 1;
                }
                if insert_at < neighbors {
                    let mut move_index = neighbors - 1;
                    while move_index > insert_at {
                        unsafe {
                            NEAREST_DISTANCES[move_index as usize] =
                                NEAREST_DISTANCES[(move_index - 1) as usize];
                            NEAREST_INDICES[move_index as usize] =
                                NEAREST_INDICES[(move_index - 1) as usize];
                        }
                        move_index -= 1;
                    }
                    unsafe {
                        NEAREST_DISTANCES[insert_at as usize] = distance;
                        NEAREST_INDICES[insert_at as usize] = h;
                    }
                }
                h += 1;
            }
            let mut weight_sum = 0.0_f32;
            let mut weighted_value = 0.0_f32;
            k = 0;
            while k < neighbors {
                let distance = unsafe { NEAREST_DISTANCES[k as usize] };
                let outcome = outcomes[unsafe { NEAREST_INDICES[k as usize] } as usize];
                let variance = 1.0e-9 + epistemic_scale * distance + aleatoric_scale;
                let weight = 1.0 / variance.max(1.0e-12);
                weight_sum += weight;
                weighted_value += weight * outcome;
                k += 1;
            }
            let mean = weighted_value / weight_sum.max(1.0e-12);
            let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * y_scale;
            let score = match acquisition {
                1 => mean + se * draws[candidate as usize],
                2 => mean + se,
                _ => mean + beta * se,
            };
            unsafe {
                *scores.get_unchecked_mut(candidate as usize) = score;
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pick_trial(
        scores: &[f32],
        mut selection: DisjointSlice<Selection>,
        regions: u32,
        candidates_per_region: u32,
    ) {
        static mut WARP_BEST_SCORES: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut WARP_BEST_INDICES: SharedArray<u32, 8> = SharedArray::UNINIT;

        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let region = thread::blockIdx_x();
        if region >= regions {
            return;
        }
        let first_candidate = region * candidates_per_region;
        let end_candidate = first_candidate + candidates_per_region;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_index = u32::MAX;
        let mut candidate = first_candidate + thread_index;
        while candidate < end_candidate {
            let score = scores[candidate as usize];
            if score > best_score || (score == best_score && candidate < best_index) {
                best_score = score;
                best_index = candidate;
            }
            candidate += THREADS;
        }

        let mut offset = 16_u32;
        while offset > 0 {
            let other_score = warp::shuffle_xor_f32(best_score, offset);
            let other_index = warp::shuffle_xor(best_index, offset);
            if other_score > best_score || (other_score == best_score && other_index < best_index) {
                best_score = other_score;
                best_index = other_index;
            }
            offset /= 2;
        }

        if lane == 0 {
            unsafe {
                WARP_BEST_SCORES[warp_index as usize] = best_score;
                WARP_BEST_INDICES[warp_index as usize] = best_index;
            }
        }
        thread::sync_threads();

        if warp_index == 0 {
            let mut warp0_score = if lane < 8 {
                unsafe { WARP_BEST_SCORES[lane as usize] }
            } else {
                f32::NEG_INFINITY
            };
            let mut warp0_index = if lane < 8 {
                unsafe { WARP_BEST_INDICES[lane as usize] }
            } else {
                u32::MAX
            };

            let mut offset = 4_u32;
            while offset > 0 {
                let other_score = warp::shuffle_xor_f32(warp0_score, offset);
                let other_index = warp::shuffle_xor(warp0_index, offset);
                if other_score > warp0_score
                    || (other_score == warp0_score && other_index < warp0_index)
                {
                    warp0_score = other_score;
                    warp0_index = other_index;
                }
                offset /= 2;
            }

            if lane == 0 {
                unsafe {
                    *selection.get_unchecked_mut(region as usize) = Selection {
                        index: warp0_index,
                        score: warp0_score,
                    };
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn write_trial(
        mut rows: DisjointSlice<u8>,
        seeds: &[Seed],
        selection: &[Selection],
        leaves: &[Leaf],
        tiles: &[Tile],
        row_bytes: u32,
        base_slot: u32,
        trial_slot: u32,
    ) {
        let tile_index = thread::blockIdx_x();
        if tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let seed = seeds[selection[0].index as usize];
        let thread_index = thread::threadIdx_x();
        let base_offset = base_slot as usize * row_bytes as usize;
        let trial_offset = trial_slot as usize * row_bytes as usize;
        let rows_ptr = rows.as_mut_ptr() as *const u8;

        if leaf.bits == 4 {
            let first_byte = tile.start / 2;
            let bytes = tile.length.div_ceil(2);
            let mut local_byte = thread_index;
            while local_byte < bytes {
                let first = tile.start + local_byte * 2;
                let low = perturb_code(
                    unsafe { code_at_ptr(rows_ptr, base_offset, leaf, first) },
                    seed.low,
                    seed.high,
                    leaf.element_offset + first,
                    leaf.bits,
                    leaf.whole,
                    leaf.threshold,
                );
                let high = if first + 1 < leaf.length {
                    perturb_code(
                        unsafe { code_at_ptr(rows_ptr, base_offset, leaf, first + 1) },
                        seed.low,
                        seed.high,
                        leaf.element_offset + first + 1,
                        leaf.bits,
                        leaf.whole,
                        leaf.threshold,
                    )
                } else {
                    0
                };
                let output = trial_offset
                    + leaf.byte_offset as usize
                    + first_byte as usize
                    + local_byte as usize;
                unsafe {
                    *rows.get_unchecked_mut(output) = (low | (high << 4)) as u8;
                }
                local_byte += THREADS;
            }
        } else {
            let end = tile.start + tile.length;
            let mut element = tile.start + thread_index;
            while element < end {
                let value = perturb_code(
                    unsafe { code_at_ptr(rows_ptr, base_offset, leaf, element) },
                    seed.low,
                    seed.high,
                    leaf.element_offset + element,
                    leaf.bits,
                    leaf.whole,
                    leaf.threshold,
                );
                let output = trial_offset + leaf.byte_offset as usize + element as usize;
                unsafe {
                    *rows.get_unchecked_mut(output) = value as u8;
                }
                element += THREADS;
            }
        }
    }

    #[kernel]
    pub fn materialize(
        base: &[u8],
        mut output: DisjointSlice<u8>,
        byte_offset: u32,
        element_offset: u32,
        length: u32,
        bits: u32,
        seed_low: u32,
        seed_high: u32,
        whole: u32,
        threshold: u32,
    ) {
        let thread_index = thread::index_1d().get();
        let local = thread_index as u32;

        if bits == 4 {
            let byte_count = length.div_ceil(2);
            if local >= byte_count {
                return;
            }
            let first = local * 2;
            let byte_index = byte_offset as usize + thread_index;
            let base_byte = base[byte_index];
            let low = perturb_code(
                u32::from(base_byte & 0x0f),
                seed_low,
                seed_high,
                element_offset + first,
                bits,
                whole,
                threshold,
            );
            let high = if first + 1 < length {
                perturb_code(
                    u32::from(base_byte >> 4),
                    seed_low,
                    seed_high,
                    element_offset + first + 1,
                    bits,
                    whole,
                    threshold,
                )
            } else {
                0
            };

            // SAFETY: one thread writes one byte and each leaf owns a disjoint range.
            unsafe {
                *output.get_unchecked_mut(byte_index) = (low | (high << 4)) as u8;
            }
        } else {
            if local >= length {
                return;
            }
            let byte_index = byte_offset as usize + thread_index;
            let value = perturb_code(
                u32::from(base[byte_index]),
                seed_low,
                seed_high,
                element_offset + local,
                bits,
                whole,
                threshold,
            );

            // SAFETY: one thread writes one byte and each leaf owns a disjoint range.
            unsafe {
                *output.get_unchecked_mut(byte_index) = value as u8;
            }
        }
    }

    #[kernel]
    pub fn apply_dense(
        base: &[f32],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
        tiles: &[DenseTile],
        mut output: DisjointSlice<f32>,
        term_count: u32,
    ) {
        let tile_index = thread::blockIdx_x();
        if tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let mut item = thread::threadIdx_x();
        while item < tile.length {
            let coordinate = u64::from(tile.start + item);
            let index = leaf.offset + coordinate;
            unsafe {
                *output.get_unchecked_mut(index as usize) = dense_value(
                    base[index as usize],
                    leaf.key,
                    coordinate,
                    leaf.scale,
                    terms,
                    term_count,
                );
            }
            item += THREADS;
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn dense_linear(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        terms: &[DenseTerm],
        mut output: DisjointSlice<f32>,
        params: DenseLinearParams,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= params.rows {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let row_start = u64::from(row) * u64::from(params.columns);
        let mut sum = 0.0_f32;
        let mut column = thread_index;
        while column < params.columns {
            let index = row_start + u64::from(column);
            let value = dense_value(
                weight[index as usize],
                params.weight_key,
                params.weight_start + index,
                params.weight_scale,
                terms,
                params.term_count,
            );
            sum = input[column as usize].mul_add(value, sum);
            column += THREADS;
        }

        let warp_sum = warp::reduce_sum_f32(sum);
        if lane == 0 {
            unsafe {
                WARP_SUMS[warp_index as usize] = warp_sum;
            }
        }
        thread::sync_threads();
        if warp_index == 0 {
            let part = if lane < 8 {
                unsafe { WARP_SUMS[lane as usize] }
            } else {
                0.0
            };
            let mut total = warp::reduce_sum_f32(part);
            if lane == 0 {
                if params.has_bias != 0 {
                    total += dense_value(
                        bias[row as usize],
                        params.bias_key,
                        params.bias_start + u64::from(row),
                        params.bias_scale,
                        terms,
                        params.term_count,
                    );
                }
                unsafe {
                    *output.get_unchecked_mut(row as usize) = total;
                }
            }
        }
    }
}
