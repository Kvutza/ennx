use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};

mod bf16;
pub use bf16::{Bf16Leaf, Bf16Score, SearchState, TellParams, TellSummary};
use bf16::{acquisition_score, bf16_finite, bf16_seed, bf16_value, tile_distances, warp_invalid};
mod knn;
pub use knn::{
    BatchParams, BatchValue, DrawParams, KNN_MAX_K, KNN_ROW_TILE, KNN_WARP_TILE, KnnParams,
    MergeParams, PosteriorParams, WeightedParams,
};
use knn::{block_sum, init_pairs, row_distance, sort_pairs, warp_distance, write_list};

pub const MODULE_NAME: &str = env!("CARGO_PKG_NAME");
pub const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;
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
pub struct SparseEdit {
    pub leaf: u32,
    pub element: u32,
}

// SAFETY: SparseEdit is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for SparseEdit {}

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
fn sparse_code(code: u32, seed: Seed, element: u32, leaf: Leaf) -> u32 {
    if leaf.whole == 0 && leaf.threshold == 0 {
        return code;
    }
    perturb_code(
        code,
        seed.low,
        seed.high,
        element,
        leaf.bits,
        leaf.whole.max(1),
        0,
    )
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
fn draw_normal(seed: u64, index: u32, metric: u32) -> f64 {
    const SEED_PRIME: u64 = 1_000_003;
    const XOR_OFFSET: u64 = 0xd2b7_4407_b1ce_6e93;
    const INV_2P53: f64 = 1.0 / 9_007_199_254_740_992.0;
    const CLIP_MIN: f64 = 1.0e-12;
    const CLIP_MAX: f64 = 1.0 - 1.0e-12;
    const TAU: f64 = 6.283_185_307_179_586;

    let base = seed
        .wrapping_mul(SEED_PRIME)
        .wrapping_add(index as u64)
        .wrapping_mul(SEED_PRIME);
    let combined = base.wrapping_add(metric as u64);
    let first = dense_mix64(combined);
    let second = dense_mix64(combined ^ XOR_OFFSET);
    let raw = ((first >> 11) as f64) * INV_2P53;
    let u1 = if raw < CLIP_MIN {
        CLIP_MIN
    } else if raw > CLIP_MAX {
        CLIP_MAX
    } else {
        raw
    };
    let u2 = ((second >> 11) as f64) * INV_2P53;
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
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

macro_rules! tell_validate {
    ($values:ident, $trial_variances:ident, $accepted:ident, $trials:expr) => {{
        let mut status = 0_u32;
        let mut trial = 0_u32;
        while trial < $trials {
            let value = $values[trial as usize];
            let variance = $trial_variances[trial as usize];
            if !value.is_finite() || !variance.is_finite() || variance < 0.0 {
                status = 1;
            }
            unsafe { *$accepted.get_unchecked_mut(trial as usize) = 0 };
            trial += 1;
        }
        status
    }};
}

macro_rules! tell_history {
    ($local:ident, $history_slots:ident, $outcomes:ident, $variances:ident,
     $trial_slots:ident, $values:ident, $trial_variances:ident, $accepted:ident,
     $params:ident) => {{
        let slots_ptr = $history_slots.as_mut_ptr() as *const u32;
        let outcomes_ptr = $outcomes.as_mut_ptr() as *const f32;
        let variances_ptr = $variances.as_mut_ptr() as *const f32;
        let mut best_slot = u32::MAX;
        let mut trial = 0_u32;
        while trial < $params.trials {
            let value = $values[trial as usize];
            let variance = $trial_variances[trial as usize];
            let slot = $trial_slots[trial as usize];
            if value > $local.best {
                $local.best = value;
                $local.best_variance = variance;
                best_slot = slot;
                unsafe { *$accepted.get_unchecked_mut(trial as usize) = 1 };
            }
            if $local.history == $params.capacity {
                let mut index = 1_u32;
                while index < $local.history {
                    unsafe {
                        *$history_slots.get_unchecked_mut((index - 1) as usize) =
                            slots_ptr.add(index as usize).read();
                        *$outcomes.get_unchecked_mut((index - 1) as usize) =
                            outcomes_ptr.add(index as usize).read();
                        *$variances.get_unchecked_mut((index - 1) as usize) =
                            variances_ptr.add(index as usize).read();
                    }
                    index += 1;
                }
                $local.history -= 1;
            }
            let index = $local.history as usize;
            unsafe {
                *$history_slots.get_unchecked_mut(index) = slot;
                *$outcomes.get_unchecked_mut(index) = value;
                *$variances.get_unchecked_mut(index) = variance;
            }
            $local.history += 1;
            trial += 1;
        }
        best_slot
    }};
}

macro_rules! tell_adapt {
    ($local:ident, $values:ident, $history_slots:ident, $outcomes:ident,
     $variances:ident, $params:ident) => {{
        let previous_best = $local.trust_best;
        if previous_best.is_finite() {
            let scale = if $local.prev_obs >= 2 {
                ($local.hist_max - $local.hist_min).max(1.0e-6)
            } else {
                0.0
            };
            if f64::from($local.best) > previous_best + 1.0e-3 * scale {
                $local.successes += 1;
                $local.failures = 0;
            } else {
                $local.failures += 1;
                $local.successes = 0;
            }
            if $local.successes >= 3 {
                $local.length = ($local.length * 2.0).min($local.length_max);
                $local.successes = 0;
            } else if $local.failures >= $params.failure_tolerance {
                $local.length *= 0.5;
                $local.failures = 0;
            }
        }
        $local.trust_best = f64::from($local.best);
        let mut trial = 0_u32;
        while trial < $params.trials {
            let value = f64::from($values[trial as usize]);
            $local.hist_min = $local.hist_min.min(value);
            $local.hist_max = $local.hist_max.max(value);
            trial += 1;
        }
        $local.prev_obs += u64::from($params.trials);
        let mut restarted = 0_u32;
        if $local.length < $local.length_min {
            $local.length = $local.length_init;
            $local.successes = 0;
            $local.failures = 0;
            $local.trust_best = f64::NEG_INFINITY;
            $local.hist_min = f64::INFINITY;
            $local.hist_max = f64::NEG_INFINITY;
            $local.history = 1;
            $local.restarts += 1;
            unsafe {
                *$history_slots.get_unchecked_mut(0) = 1;
                *$outcomes.get_unchecked_mut(0) = $local.best;
                *$variances.get_unchecked_mut(0) = $local.best_variance;
            }
            restarted = 1;
        }
        restarted
    }};
}

#[cuda_module]
pub mod trials {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_rows(
        rows: &[f32],
        queries: &[f32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        params: KnnParams,
    ) {
        static mut VALUES: SharedArray<f32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;

        let list = thread::blockIdx_x();
        let total_lists = params.queries * params.lists;
        if list >= total_lists {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
        unsafe { init_pairs(values, local_indices, KNN_ROW_TILE) };

        let query = list / params.lists;
        let tile = list - query * params.lists;
        let row = tile * KNN_ROW_TILE + thread_index;
        if row < params.rows {
            let distance = row_distance(rows, queries, row, query, params.dims);
            unsafe {
                values.add(thread_index as usize).write(distance);
                local_indices.add(thread_index as usize).write(row);
            }
        }
        thread::sync_threads();

        unsafe {
            sort_pairs(values, local_indices, KNN_ROW_TILE);
            write_list(
                values,
                local_indices,
                &mut distances,
                &mut indices,
                list,
                params.neighbors,
            );
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_warps(
        rows: &[f32],
        queries: &[f32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        params: KnnParams,
    ) {
        static mut VALUES: SharedArray<f32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;

        let list = thread::blockIdx_x();
        let total_lists = params.queries * params.lists;
        if list >= total_lists {
            return;
        }
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
        unsafe { init_pairs(values, local_indices, KNN_ROW_TILE) };
        thread::sync_threads();

        let thread_index = thread::threadIdx_x();
        let warp_index = thread_index / 32;
        let query = list / params.lists;
        let tile = list - query * params.lists;
        let row = tile * KNN_WARP_TILE + warp_index;
        if row < params.rows {
            let distance = warp_distance(rows, queries, row, query, params.dims);
            if warp::lane_id() == 0 {
                unsafe {
                    values.add(warp_index as usize).write(distance);
                    local_indices.add(warp_index as usize).write(row);
                }
            }
        }
        thread::sync_threads();

        unsafe {
            sort_pairs(values, local_indices, KNN_WARP_TILE);
            write_list(
                values,
                local_indices,
                &mut distances,
                &mut indices,
                list,
                params.neighbors,
            );
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn merge_topk(
        input_distances: &[f32],
        input_indices: &[u32],
        mut output_distances: DisjointSlice<f32>,
        mut output_indices: DisjointSlice<u32>,
        params: MergeParams,
    ) {
        static mut VALUES: SharedArray<f32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, { KNN_ROW_TILE as usize }> = SharedArray::UNINIT;

        let output_list = thread::blockIdx_x();
        let total_lists = params.queries * params.output_lists;
        if output_list >= total_lists {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
        unsafe { init_pairs(values, local_indices, KNN_ROW_TILE) };

        let query = output_list / params.output_lists;
        let local_list = output_list - query * params.output_lists;
        let left_list = query * params.input_lists + 2 * local_list;
        let right_local = 2 * local_list + 1;
        if thread_index < params.neighbors {
            let source = left_list as usize * params.neighbors as usize + thread_index as usize;
            unsafe {
                values
                    .add(thread_index as usize)
                    .write(input_distances[source]);
                local_indices
                    .add(thread_index as usize)
                    .write(input_indices[source]);
            }
        } else if thread_index < 2 * params.neighbors && right_local < params.input_lists {
            let neighbor = thread_index - params.neighbors;
            let right_list = left_list + 1;
            let source = right_list as usize * params.neighbors as usize + neighbor as usize;
            unsafe {
                values
                    .add(thread_index as usize)
                    .write(input_distances[source]);
                local_indices
                    .add(thread_index as usize)
                    .write(input_indices[source]);
            }
        }
        thread::sync_threads();

        let mut width = 2_u32;
        while width < 2 * params.neighbors {
            width *= 2;
        }
        unsafe {
            sort_pairs(values, local_indices, width);
            write_list(
                values,
                local_indices,
                &mut output_distances,
                &mut output_indices,
                output_list,
                params.neighbors,
            );
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_light(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        scales: &[f32],
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut selected: DisjointSlice<u32>,
        params: PosteriorParams,
    ) {
        static mut WEIGHTS: SharedArray<f32, KNN_MAX_K> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

        let query = thread::blockIdx_x();
        if query >= params.queries {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let weights = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS) };
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let mut weight = 0.0_f32;
        if thread_index < params.used_k {
            let source = query as usize * params.input_k as usize
                + params.skip as usize
                + thread_index as usize;
            let variance = params.epsilon
                + params.epistemic_scale * distances[source]
                + params.aleatoric_scale;
            weight = 1.0 / variance.max(params.epsilon);
            unsafe {
                weights.add(thread_index as usize).write(weight);
                *selected.get_unchecked_mut(
                    query as usize * params.used_k as usize + thread_index as usize,
                ) = indices[source];
            }
        }
        unsafe { sums.add(thread_index as usize).write(weight) };
        thread::sync_threads();

        let mut stride = THREADS / 2;
        while stride > 0 {
            if thread_index < stride {
                unsafe {
                    let value = sums.add(thread_index as usize).read()
                        + sums.add((thread_index + stride) as usize).read();
                    sums.add(thread_index as usize).write(value);
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse = 1.0 / unsafe { sums.read() }.max(params.epsilon);
        let error_base = inverse.max(params.epsilon).sqrt();
        let mut metric = thread_index;
        while metric < params.metrics {
            let mut weighted = 0.0_f32;
            let mut neighbor = 0_u32;
            while neighbor < params.used_k {
                let source = query as usize * params.input_k as usize
                    + params.skip as usize
                    + neighbor as usize;
                let row = indices[source] as usize;
                weighted += unsafe { weights.add(neighbor as usize).read() }
                    * outcomes[row * params.metrics as usize + metric as usize];
                neighbor += 1;
            }
            let output = query as usize * params.metrics as usize + metric as usize;
            unsafe {
                *means.get_unchecked_mut(output) = weighted * inverse;
                *errors.get_unchecked_mut(output) = error_base * scales[metric as usize];
            }
            metric += THREADS;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_full(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        scales: &[f32],
        mut weights_out: DisjointSlice<f32>,
        mut l2_out: DisjointSlice<f32>,
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut epistemic: DisjointSlice<f32>,
        mut aleatoric: DisjointSlice<f32>,
        mut selected: DisjointSlice<u32>,
        params: WeightedParams,
    ) {
        static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let total = params.queries * params.metrics;
        if block >= total {
            return;
        }
        let query = block / params.metrics;
        let metric = block - query * params.metrics;
        let thread_index = thread::threadIdx_x();
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let scale = scales[metric as usize];
        let scale_sq = scale * scale;
        let mut weight = 0.0_f32;
        let mut row = 0_u32;
        let mut variance = 0.0_f32;
        if thread_index < params.used_k {
            let source = query as usize * params.input_k as usize
                + params.skip as usize
                + thread_index as usize;
            row = indices[source];
            if params.has_yvar != 0 {
                variance =
                    variances[row as usize * params.metrics as usize + metric as usize] / scale_sq;
            }
            let total_variance = params.epsilon
                + params.epistemic_scale * distances[source]
                + params.aleatoric_scale
                + variance;
            weight = 1.0 / total_variance.max(params.epsilon);
            if metric == 0 {
                unsafe {
                    *selected.get_unchecked_mut(
                        query as usize * params.used_k as usize + thread_index as usize,
                    ) = row;
                }
            }
        }
        unsafe { sums.add(thread_index as usize).write(weight) };
        let inverse = 1.0 / unsafe { block_sum(sums) }.max(params.epsilon);

        let normalized = if thread_index < params.used_k {
            weight * inverse
        } else {
            0.0
        };
        if thread_index < params.used_k {
            let output = (query as usize * params.used_k as usize + thread_index as usize)
                * params.metrics as usize
                + metric as usize;
            unsafe { *weights_out.get_unchecked_mut(output) = normalized };
        }

        unsafe {
            sums.add(thread_index as usize)
                .write(normalized * normalized)
        };
        let l2 = unsafe { block_sum(sums) }.sqrt();

        let weighted = if thread_index < params.used_k {
            normalized * outcomes[row as usize * params.metrics as usize + metric as usize]
        } else {
            0.0
        };
        unsafe { sums.add(thread_index as usize).write(weighted) };
        let mean = unsafe { block_sum(sums) };

        let noise = if thread_index < params.used_k && params.observation_noise != 0 {
            normalized * (params.aleatoric_scale + variance)
        } else {
            0.0
        };
        unsafe { sums.add(thread_index as usize).write(noise) };
        let aleatoric_var = unsafe { block_sum(sums) };
        let epistemic_var = inverse;
        let total_variance = epistemic_var + aleatoric_var;
        let (error, epi, ale) = if total_variance < params.epsilon {
            let value = params.epsilon.sqrt() * scale;
            (value, value, 0.0)
        } else {
            (
                total_variance.sqrt() * scale,
                epistemic_var.sqrt() * scale,
                aleatoric_var.sqrt() * scale,
            )
        };
        if thread_index == 0 {
            unsafe {
                *l2_out.get_unchecked_mut(block as usize) = l2;
                *means.get_unchecked_mut(block as usize) = mean;
                *errors.get_unchecked_mut(block as usize) = error;
                *epistemic.get_unchecked_mut(block as usize) = epi;
                *aleatoric.get_unchecked_mut(block as usize) = ale;
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_batch(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        scales: &[f32],
        values: &[BatchValue],
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut epistemic: DisjointSlice<f32>,
        mut aleatoric: DisjointSlice<f32>,
        params: BatchParams,
    ) {
        static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let query_values = params.queries * params.metrics;
        let total = params.param_count * query_values;
        if block >= total {
            return;
        }
        let param_index = block / query_values;
        let query_metric = block - param_index * query_values;
        let query = query_metric / params.metrics;
        let metric = query_metric - query * params.metrics;
        let value = values[param_index as usize];
        let thread_index = thread::threadIdx_x();
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let scale = scales[metric as usize];
        let scale_sq = scale * scale;
        let mut row = 0_u32;
        let mut variance = 0.0_f32;
        let mut weight = 0.0_f32;
        if thread_index < value.used_k {
            let source = query as usize * params.input_k as usize
                + value.skip as usize
                + thread_index as usize;
            row = indices[source];
            if params.has_yvar != 0 {
                variance =
                    variances[row as usize * params.metrics as usize + metric as usize] / scale_sq;
            }
            let total_variance = params.epsilon
                + value.epistemic_scale * distances[source]
                + value.aleatoric_scale
                + variance;
            weight = 1.0 / total_variance.max(params.epsilon);
        }
        unsafe { sums.add(thread_index as usize).write(weight) };
        let inverse = 1.0 / unsafe { block_sum(sums) }.max(params.epsilon);

        let weighted = if thread_index < value.used_k {
            weight * outcomes[row as usize * params.metrics as usize + metric as usize]
        } else {
            0.0
        };
        unsafe { sums.add(thread_index as usize).write(weighted) };
        let mean = unsafe { block_sum(sums) } * inverse;

        let noise = if thread_index < value.used_k && params.observation_noise != 0 {
            weight * inverse * (value.aleatoric_scale + variance)
        } else {
            0.0
        };
        unsafe { sums.add(thread_index as usize).write(noise) };
        let aleatoric_var = unsafe { block_sum(sums) };
        let epistemic_var = inverse;
        let total_variance = epistemic_var + aleatoric_var;
        let (error, epi, ale) = if total_variance < params.epsilon {
            let floor = params.epsilon.sqrt() * scale;
            (floor, floor, 0.0)
        } else {
            (
                total_variance.sqrt() * scale,
                epistemic_var.sqrt() * scale,
                aleatoric_var.sqrt() * scale,
            )
        };
        if thread_index == 0 {
            unsafe {
                *means.get_unchecked_mut(block as usize) = mean;
                *errors.get_unchecked_mut(block as usize) = error;
                *epistemic.get_unchecked_mut(block as usize) = epi;
                *aleatoric.get_unchecked_mut(block as usize) = ale;
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_draw(
        weights: &[f32],
        l2: &[f32],
        means: &[f32],
        errors: &[f32],
        indices: &[u32],
        seeds: &[u64],
        mut draws: DisjointSlice<f64>,
        params: DrawParams,
    ) {
        let output = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        let query_values = params.queries * params.metrics;
        let total = params.seed_count * query_values;
        if output >= total {
            return;
        }
        let seed_index = output / query_values;
        let query_metric = output - seed_index * query_values;
        let query = query_metric / params.metrics;
        let metric = query_metric - query * params.metrics;
        let mut weighted = 0.0_f64;
        let mut neighbor = 0_u32;
        while neighbor < params.neighbors {
            let neighbor_offset = query as usize * params.neighbors as usize + neighbor as usize;
            let weight_offset = neighbor_offset * params.metrics as usize + metric as usize;
            weighted += weights[weight_offset] as f64
                * draw_normal(seeds[seed_index as usize], indices[neighbor_offset], metric);
            neighbor += 1;
        }
        let value_offset = query as usize * params.metrics as usize + metric as usize;
        let scale = errors[value_offset] as f64 / (l2[value_offset] as f64).max(1.0e-12);
        unsafe {
            *draws.get_unchecked_mut(output as usize) =
                means[value_offset] as f64 + scale * weighted;
        }
    }

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
        row_stride: u32,
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

        let base_offset = base_slot as usize * row_stride as usize;
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
                                history_slots[h as usize] as usize * row_stride as usize;
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
                                history_slots[h as usize] as usize * row_stride as usize;
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

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn distance_bf16(
        rows: &[u16],
        history_slots: &[u32],
        seeds: &[Seed],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut partials: DisjointSlice<f32>,
        params: Bf16Score,
    ) {
        static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let block_index = thread::blockIdx_x();
        if block_index >= params.candidates * params.tiles {
            return;
        }
        let candidate = block_index / params.tiles;
        let tile_index = block_index % params.tiles;
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
        let warp_status = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_STATUS) };
        if thread_index < params.history {
            unsafe { distances.add(thread_index as usize).write(0.0) };
        }
        if lane == 0 {
            unsafe { warp_status.add(warp_index as usize).write(0) };
        }
        thread::sync_threads();

        let tile = tiles[tile_index as usize];
        tile_distances(
            rows,
            history_slots,
            seeds[candidate as usize],
            leaves[tile.leaf as usize],
            tile,
            values,
            distances,
            warp_status,
            params,
        );
        let invalid = warp_invalid(warp_status);
        if thread_index < params.history {
            let output = block_index as usize * params.history as usize + thread_index as usize;
            unsafe {
                *partials.get_unchecked_mut(output) = if invalid {
                    f32::INFINITY
                } else {
                    distances.add(thread_index as usize).read()
                };
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score_bf16(
        partials: &[f32],
        outcomes: &[f32],
        variances: &[f32],
        draws: &[f32],
        mut scores: DisjointSlice<f32>,
        params: Bf16Score,
    ) {
        static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;

        let candidate = thread::blockIdx_x();
        if candidate >= params.candidates {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
        let nearest_distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_DISTANCES) };
        let nearest_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_INDICES) };
        if thread_index < params.history {
            let mut distance = 0.0f32;
            let mut tile_index = 0u32;
            while tile_index < params.tiles {
                let block_index = candidate * params.tiles + tile_index;
                let input = block_index as usize * params.history as usize + thread_index as usize;
                distance += partials[input];
                tile_index += 1;
            }
            unsafe { distances.add(thread_index as usize).write(distance) };
        }
        thread::sync_threads();

        if thread_index == 0 {
            if unsafe { !distances.read().is_finite() } {
                unsafe {
                    *scores.get_unchecked_mut(candidate as usize) = f32::NEG_INFINITY;
                }
                return;
            }
            let score = acquisition_score(
                distances,
                nearest_distances,
                nearest_indices,
                outcomes,
                variances,
                draws,
                candidate,
                params,
            );
            unsafe {
                *scores.get_unchecked_mut(candidate as usize) = score;
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn draw_bf16(mut draws: DisjointSlice<f32>, seed: u64, count: u32) {
        let mut index = thread::threadIdx_x() + thread::blockIdx_x() * THREADS;
        let stride = thread::gridDim_x() * THREADS;
        while index < count {
            unsafe {
                *draws.get_unchecked_mut(index as usize) = draw_normal(seed, index, 0) as f32;
            }
            index += stride;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn base_distance(
        rows: &[u8],
        history_slots: &[u32],
        leaves: &[Leaf],
        mut distances: DisjointSlice<f32>,
        row_stride: u32,
        history: u32,
        base_slot: u32,
    ) {
        static mut PARTIALS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let observation = thread::blockIdx_x();
        if observation >= history {
            return;
        }
        let thread_index = thread::threadIdx_x();
        let base_offset = base_slot as usize * row_stride as usize;
        let row_offset = history_slots[observation as usize] as usize * row_stride as usize;
        let mut sum = 0.0_f32;
        let mut leaf_index = 0usize;
        while leaf_index < leaves.len() {
            let leaf = leaves[leaf_index];
            let mut element = thread_index;
            while element < leaf.length {
                let delta = decode_code(code_at(rows, base_offset, leaf, element), leaf)
                    - decode_code(code_at(rows, row_offset, leaf, element), leaf);
                sum += delta * delta * leaf.weight;
                element += THREADS;
            }
            leaf_index += 1;
        }
        unsafe { PARTIALS[thread_index as usize] = sum };
        thread::sync_threads();
        let mut width = THREADS / 2;
        while width > 0 {
            if thread_index < width {
                unsafe {
                    PARTIALS[thread_index as usize] += PARTIALS[(thread_index + width) as usize];
                }
            }
            thread::sync_threads();
            width /= 2;
        }
        if thread_index == 0 {
            unsafe {
                *distances.get_unchecked_mut(observation as usize) = PARTIALS[0];
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score_sparse(
        rows: &[u8],
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        draws: &[f32],
        leaves: &[Leaf],
        edits: &[SparseEdit],
        base_distances: &[f32],
        mut scores: DisjointSlice<f32>,
        row_stride: u32,
        history: u32,
        candidates: u32,
        num_pert: u32,
        base_slot: u32,
        neighbors: u32,
        acquisition: u32,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
    ) {
        static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
        static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;
        let candidate = thread::blockIdx_x();
        if candidate >= candidates {
            return;
        }
        let thread_index = thread::threadIdx_x();
        if thread_index < history {
            let observation_offset =
                history_slots[thread_index as usize] as usize * row_stride as usize;
            let base_offset = base_slot as usize * row_stride as usize;
            let seed = seeds[candidate as usize];
            let mut distance = base_distances[thread_index as usize];
            let mut edit_index = 0;
            while edit_index < num_pert {
                let edit = edits[(candidate * num_pert + edit_index) as usize];
                let leaf = leaves[edit.leaf as usize];
                let base_code = code_at(rows, base_offset, leaf, edit.element);
                let observed_code = code_at(rows, observation_offset, leaf, edit.element);
                let candidate_code =
                    sparse_code(base_code, seed, leaf.element_offset + edit.element, leaf);
                let base_delta = decode_code(base_code, leaf) - decode_code(observed_code, leaf);
                let candidate_delta =
                    decode_code(candidate_code, leaf) - decode_code(observed_code, leaf);
                distance +=
                    (candidate_delta * candidate_delta - base_delta * base_delta) * leaf.weight;
                edit_index += 1;
            }
            unsafe { DISTANCES[thread_index as usize] = distance.max(0.0) };
        }
        thread::sync_threads();
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
            unsafe { *scores.get_unchecked_mut(candidate as usize) = score };
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
        row_stride: u32,
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
        let base_offset = base_slot as usize * row_stride as usize;
        let trial_offset = trial_slot as usize * row_stride as usize;
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
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = 0)]
    pub fn write_sparse(
        mut rows: DisjointSlice<u8>,
        seeds: &[Seed],
        selection: &[Selection],
        leaves: &[Leaf],
        edits: &[SparseEdit],
        row_stride: u32,
        trial_slot: u32,
        num_pert: u32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let candidate = selection[0].index;
        let seed = seeds[candidate as usize];
        let trial_offset = trial_slot as usize * row_stride as usize;
        let rows_ptr = rows.as_mut_ptr() as *const u8;
        let mut edit_index = 0;
        while edit_index < num_pert {
            let edit = edits[(candidate * num_pert + edit_index) as usize];
            let leaf = leaves[edit.leaf as usize];
            let byte = trial_offset
                + leaf.byte_offset as usize
                + if leaf.bits == 4 {
                    (edit.element / 2) as usize
                } else {
                    edit.element as usize
                };
            let current = unsafe { rows_ptr.add(byte).read() };
            let shift = if leaf.bits == 4 {
                (edit.element & 1) * 4
            } else {
                0
            };
            let code = if leaf.bits == 4 {
                u32::from((current >> shift) & 0x0f)
            } else {
                u32::from(current)
            };
            let value = sparse_code(code, seed, leaf.element_offset + edit.element, leaf);
            unsafe {
                *rows.get_unchecked_mut(byte) = if leaf.bits == 4 {
                    (current & !(0x0f << shift)) | ((value as u8) << shift)
                } else {
                    value as u8
                };
            }
            edit_index += 1;
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
    pub fn apply_bf16(
        base: &[u16],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
        tiles: &[DenseTile],
        mut output: DisjointSlice<u16>,
        mut status: DisjointSlice<u32>,
        term_count: u32,
    ) {
        static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tile_index = thread::blockIdx_x();
        if tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let mut invalid = false;
        let mut item = thread_index;
        while item < tile.length {
            let coordinate = u64::from(tile.start + item);
            let index = leaf.offset + coordinate;
            unsafe {
                let value = bf16_value(
                    base[index as usize],
                    leaf.key,
                    coordinate,
                    leaf.scale,
                    terms,
                    term_count,
                );
                invalid |= !bf16_finite(value);
                *output.get_unchecked_mut(index as usize) = value;
            }
            item += THREADS;
        }
        let warp_invalid = warp::any(invalid);
        if lane == 0 {
            unsafe {
                WARP_STATUS[warp_index as usize] = u32::from(warp_invalid);
            }
        }
        thread::sync_threads();
        if thread_index == 0 {
            let mut block_status = 0;
            let mut warp_index = 0;
            while warp_index < WARPS {
                block_status |= unsafe { WARP_STATUS[warp_index] };
                warp_index += 1;
            }
            unsafe {
                *status.get_unchecked_mut(tile_index as usize) = block_status;
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn write_bf16(
        mut rows: DisjointSlice<u16>,
        seeds: &[Seed],
        selection: &[Selection],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        trial_slots: &[u32],
        mut status: DisjointSlice<u32>,
        row_stride: u64,
        base_slot: u32,
        tile_count: u32,
        coefficient: f32,
    ) {
        static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let region = block / tile_count;
        let tile_index = block % tile_count;
        if region as usize >= selection.len() || tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let seed = seeds[selection[region as usize].index as usize];
        let base_offset = base_slot as usize * row_stride as usize;
        let row_offset = trial_slots[region as usize] as usize * row_stride as usize;
        let rows_ptr = rows.as_mut_ptr() as *const u16;
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let mut invalid = false;
        let mut item = thread_index;
        while item < tile.length {
            let element = u64::from(tile.start + item);
            let index = leaf.offset + element;
            let value = bf16_seed(
                unsafe { rows_ptr.add(base_offset + index as usize).read() },
                leaf,
                element,
                seed,
                coefficient,
            );
            invalid |= !bf16_finite(value);
            unsafe {
                *rows.get_unchecked_mut(row_offset + index as usize) = value;
            }
            item += THREADS;
        }
        let warp_invalid = warp::any(invalid);
        if lane == 0 {
            unsafe {
                WARP_STATUS[warp_index as usize] = u32::from(warp_invalid);
            }
        }
        thread::sync_threads();
        if thread_index == 0 {
            let mut block_status = 0;
            let mut warp_index = 0;
            while warp_index < WARPS {
                block_status |= unsafe { WARP_STATUS[warp_index] };
                warp_index += 1;
            }
            unsafe {
                *status.get_unchecked_mut(block as usize) = block_status;
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn tell_bf16(
        mut rows: DisjointSlice<u16>,
        mut history_slots: DisjointSlice<u32>,
        mut outcomes: DisjointSlice<f32>,
        mut variances: DisjointSlice<f32>,
        trial_slots: &[u32],
        values: &[f32],
        trial_variances: &[f32],
        mut accepted: DisjointSlice<u32>,
        mut state: DisjointSlice<SearchState>,
        mut summary: DisjointSlice<TellSummary>,
        params: TellParams,
    ) {
        static mut CONTROL: SharedArray<u32, 3> = SharedArray::UNINIT;

        let thread_index = thread::threadIdx_x();
        if thread_index == 0 {
            let state_ptr = state.as_mut_ptr() as *const SearchState;
            let status = tell_validate!(values, trial_variances, accepted, params.trials);
            let mut best_slot = u32::MAX;
            let mut restarted = 0_u32;
            if status == 0 {
                let mut local = unsafe { state_ptr.read() };
                best_slot = tell_history!(
                    local,
                    history_slots,
                    outcomes,
                    variances,
                    trial_slots,
                    values,
                    trial_variances,
                    accepted,
                    params
                );
                restarted = tell_adapt!(local, values, history_slots, outcomes, variances, params);
                unsafe { *state.get_unchecked_mut(0) = local };
                unsafe {
                    *summary.get_unchecked_mut(0) = TellSummary {
                        length: local.length,
                        best: local.best,
                        best_variance: local.best_variance,
                        history: local.history,
                        restarts: local.restarts,
                        restarted,
                        status: 0,
                    };
                }
            } else {
                let local = unsafe { state_ptr.read() };
                unsafe {
                    *summary.get_unchecked_mut(0) = TellSummary {
                        length: local.length,
                        best: local.best,
                        best_variance: local.best_variance,
                        history: local.history,
                        restarts: local.restarts,
                        restarted: 0,
                        status,
                    };
                }
            }
            unsafe {
                CONTROL[0] = best_slot;
                CONTROL[1] = restarted;
                CONTROL[2] = status;
            }
        }
        thread::sync_threads();

        if unsafe { CONTROL[2] } != 0 {
            return;
        }
        let best_slot = unsafe { CONTROL[0] };
        if best_slot != u32::MAX {
            let source = best_slot as usize * params.row_stride as usize;
            let rows_ptr = rows.as_mut_ptr() as *const u16;
            let mut index = thread_index as u64;
            while index < params.row_len {
                let value = unsafe { rows_ptr.add(source + index as usize).read() };
                unsafe { *rows.get_unchecked_mut(index as usize) = value };
                index += u64::from(THREADS);
            }
        }
        thread::sync_threads();

        if unsafe { CONTROL[1] } != 0 {
            let destination = params.row_stride as usize;
            let rows_ptr = rows.as_mut_ptr() as *const u16;
            let mut index = thread_index as u64;
            while index < params.row_len {
                let value = unsafe { rows_ptr.add(index as usize).read() };
                unsafe { *rows.get_unchecked_mut(destination + index as usize) = value };
                index += u64::from(THREADS);
            }
        }
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn check_search(
        rows: &[u16],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut status: DisjointSlice<u32>,
        row_stride: u64,
        slot: u32,
    ) {
        static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tile_index = thread::blockIdx_x();
        if tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let row_offset = slot as usize * row_stride as usize;
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let mut invalid = false;
        let mut item = thread_index;
        while item < tile.length {
            let index = leaf.offset + u64::from(tile.start + item);
            invalid |= !bf16_finite(rows[row_offset + index as usize]);
            item += THREADS;
        }
        let warp_invalid = warp::any(invalid);
        if lane == 0 {
            unsafe {
                WARP_STATUS[warp_index as usize] = u32::from(warp_invalid);
            }
        }
        thread::sync_threads();
        if thread_index == 0 {
            let mut block_status = 0;
            let mut warp_index = 0;
            while warp_index < WARPS {
                block_status |= unsafe { WARP_STATUS[warp_index] };
                warp_index += 1;
            }
            unsafe {
                *status.get_unchecked_mut(tile_index as usize) = block_status;
            }
        }
    }

    #[kernel]
    pub fn check_bf16(
        base: &[u16],
        leaves: &[DenseLeaf],
        tiles: &[DenseTile],
        mut status: DisjointSlice<u32>,
    ) {
        static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let tile_index = thread::blockIdx_x();
        if tile_index as usize >= tiles.len() {
            return;
        }
        let tile = tiles[tile_index as usize];
        let leaf = leaves[tile.leaf as usize];
        let thread_index = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_index = thread_index / 32;
        let mut invalid = false;
        let mut item = thread_index;
        while item < tile.length {
            let coordinate = u64::from(tile.start + item);
            let index = leaf.offset + coordinate;
            invalid |= !bf16_finite(base[index as usize]);
            item += THREADS;
        }
        let warp_invalid = warp::any(invalid);
        if lane == 0 {
            unsafe {
                WARP_STATUS[warp_index as usize] = u32::from(warp_invalid);
            }
        }
        thread::sync_threads();
        if thread_index == 0 {
            let mut block_status = 0;
            let mut warp_index = 0;
            while warp_index < WARPS {
                block_status |= unsafe { WARP_STATUS[warp_index] };
                warp_index += 1;
            }
            unsafe {
                *status.get_unchecked_mut(tile_index as usize) = block_status;
            }
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
