use cuda_device::{thread, warp};

use crate::THREADS;

pub const KNN_MAX_K: usize = 128;
pub const KNN_ROW_TILE: u32 = THREADS;
pub const KNN_WARP_TILE: u32 = THREADS / 32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KnnParams {
    pub rows: u32,
    pub dims: u32,
    pub queries: u32,
    pub lists: u32,
    pub neighbors: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

// SAFETY: KnnParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for KnnParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MergeParams {
    pub queries: u32,
    pub input_lists: u32,
    pub output_lists: u32,
    pub neighbors: u32,
}

// SAFETY: MergeParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MergeParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PosteriorParams {
    pub queries: u32,
    pub input_k: u32,
    pub used_k: u32,
    pub skip: u32,
    pub metrics: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
}

// SAFETY: PosteriorParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for PosteriorParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WeightedParams {
    pub queries: u32,
    pub input_k: u32,
    pub used_k: u32,
    pub skip: u32,
    pub metrics: u32,
    pub has_yvar: u32,
    pub observation_noise: u32,
    pub pad: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
    pub padf: f32,
}

// SAFETY: WeightedParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for WeightedParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchParams {
    pub queries: u32,
    pub input_k: u32,
    pub metrics: u32,
    pub param_count: u32,
    pub has_yvar: u32,
    pub observation_noise: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub epsilon: f32,
    pub padf0: f32,
    pub padf1: f32,
    pub padf2: f32,
}

// SAFETY: BatchParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for BatchParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchValue {
    pub used_k: u32,
    pub skip: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
}

// SAFETY: BatchValue is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for BatchValue {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawParams {
    pub queries: u32,
    pub neighbors: u32,
    pub metrics: u32,
    pub seed_count: u32,
}

// SAFETY: DrawParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DrawParams {}

#[inline(always)]
pub(super) fn pair_before(
    left_distance: f32,
    left_index: u32,
    right_distance: f32,
    right_index: u32,
) -> bool {
    left_distance < right_distance || (left_distance == right_distance && left_index < right_index)
}

#[inline(always)]
pub(super) unsafe fn order_pair(
    distances: *mut f32,
    indices: *mut u32,
    left: usize,
    right: usize,
    ascending: bool,
) {
    let left_distance = unsafe { distances.add(left).read() };
    let left_index = unsafe { indices.add(left).read() };
    let right_distance = unsafe { distances.add(right).read() };
    let right_index = unsafe { indices.add(right).read() };
    let swap = if ascending {
        pair_before(right_distance, right_index, left_distance, left_index)
    } else {
        pair_before(left_distance, left_index, right_distance, right_index)
    };
    if swap {
        unsafe {
            distances.add(left).write(right_distance);
            indices.add(left).write(right_index);
            distances.add(right).write(left_distance);
            indices.add(right).write(left_index);
        }
    }
}

#[inline(always)]
pub(super) unsafe fn sort_pairs(distances: *mut f32, indices: *mut u32, width: u32) {
    let thread_index = thread::threadIdx_x();
    let mut size = 2_u32;
    while size <= width {
        let mut stride = size / 2;
        while stride > 0 {
            let mut index = thread_index;
            while index < width {
                let partner = index ^ stride;
                if partner > index {
                    unsafe {
                        order_pair(
                            distances,
                            indices,
                            index as usize,
                            partner as usize,
                            index & size == 0,
                        );
                    }
                }
                index += THREADS;
            }
            thread::sync_threads();
            stride /= 2;
        }
        size *= 2;
    }
}

#[inline(always)]
pub(super) fn row_distance(rows: &[f32], queries: &[f32], row: u32, query: u32, dims: u32) -> f32 {
    let row_start = row as usize * dims as usize;
    let query_start = query as usize * dims as usize;
    let mut sum = 0.0_f32;
    let mut dimension = 0_u32;
    while dimension < dims {
        let delta =
            rows[row_start + dimension as usize] - queries[query_start + dimension as usize];
        sum = delta.mul_add(delta, sum);
        dimension += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn warp_distance(rows: &[f32], queries: &[f32], row: u32, query: u32, dims: u32) -> f32 {
    let row_start = row as usize * dims as usize;
    let query_start = query as usize * dims as usize;
    let mut sum = 0.0_f32;
    let mut dimension = warp::lane_id();
    while dimension < dims {
        let delta =
            rows[row_start + dimension as usize] - queries[query_start + dimension as usize];
        sum = delta.mul_add(delta, sum);
        dimension += 32;
    }
    warp::reduce_sum_f32(sum)
}

#[inline(always)]
pub(super) unsafe fn init_pairs(distances: *mut f32, indices: *mut u32, width: u32) {
    let mut index = thread::threadIdx_x();
    while index < width {
        unsafe {
            distances.add(index as usize).write(f32::INFINITY);
            indices.add(index as usize).write(u32::MAX);
        }
        index += THREADS;
    }
}

#[inline(always)]
pub(super) unsafe fn write_list(
    distances: *const f32,
    indices: *const u32,
    output_distances: &mut cuda_device::DisjointSlice<f32>,
    output_indices: &mut cuda_device::DisjointSlice<u32>,
    list: u32,
    neighbors: u32,
) {
    let mut neighbor = thread::threadIdx_x();
    let output_start = list as usize * neighbors as usize;
    while neighbor < neighbors {
        unsafe {
            *output_distances.get_unchecked_mut(output_start + neighbor as usize) =
                distances.add(neighbor as usize).read();
            *output_indices.get_unchecked_mut(output_start + neighbor as usize) =
                indices.add(neighbor as usize).read();
        }
        neighbor += THREADS;
    }
}

#[inline(always)]
pub(super) unsafe fn block_sum(values: *mut f32) -> f32 {
    let thread_index = thread::threadIdx_x();
    thread::sync_threads();
    let mut stride = THREADS / 2;
    while stride > 0 {
        if thread_index < stride {
            unsafe {
                let value = values.add(thread_index as usize).read()
                    + values.add((thread_index + stride) as usize).read();
                values.add(thread_index as usize).write(value);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}
