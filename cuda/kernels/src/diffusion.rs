//! Learned mask inputs and block diffusion readout. No autoregressive repair.
use crate::fbt::{half, half_bits, reduce_max, reduce_sum};
use crate::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

#[cuda_module]
pub mod diffusion_model {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn diffusion_embed(
        weights: &[u16],
        tokens: &[u32],
        mask: &[u16],
        confidence: &[f32],
        mut output: DisjointSlice<u16>,
        rows: u32,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let token = tokens[row as usize];
        let alpha = if token == u32::MAX {
            0.0
        } else {
            confidence[row as usize]
        };
        let mut squares = 0.0;
        for i in 0..2 {
            let dim = tid + i * 256;
            let predicted = if token == u32::MAX {
                0.0
            } else {
                half(weights[(token * 512 + dim) as usize])
            };
            let value = alpha * predicted + (1.0 - alpha) * half(mask[dim as usize]);
            squares += value * value;
            unsafe {
                *output.get_unchecked_mut((row * 512 + dim) as usize) = half_bits(value);
            }
        }
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe {
            sums.add(tid as usize).write(squares);
        }
        let scale = 0.02 / (unsafe { reduce_sum(sums) } / 512.0 + 1e-6).sqrt();
        if alpha > 0.0 && alpha < 1.0 {
            for i in 0..2 {
                let offset = (row * 512 + tid + i * 256) as usize;
                unsafe {
                    *output.get_unchecked_mut(offset) =
                        half_bits(half(*output.get_unchecked_mut(offset)) * scale);
                }
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn diffusion_sample(
        logits: &[u16],
        mut tokens: DisjointSlice<u32>,
        mut confidence: DisjointSlice<f32>,
        rows: u32,
        temperature: f32,
        seed: u64,
        first_row: u32,
    ) {
        static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
        let mut maximum = f32::NEG_INFINITY;
        for i in 0..32 {
            maximum = maximum.max(half(logits[(row * 8192 + tid + i * 256) as usize]));
        }
        unsafe {
            values.add(tid as usize).write(maximum);
        }
        let maximum = unsafe { reduce_max(values) };
        thread::sync_threads();
        let mut best = f32::NEG_INFINITY;
        let mut selected = u32::MAX;
        for i in 0..32 {
            let token = tid + i * 256;
            let logit = half(logits[(row * 8192 + token) as usize]);
            if logit.is_finite() {
                let mut score = logit;
                if temperature > 0.0 {
                    let mut hash = seed ^ (u64::from(first_row + row) << 32) ^ u64::from(token);
                    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
                    hash ^= hash >> 31;
                    let uniform = ((hash >> 41) as f32 + 0.5) / 8388608.0;
                    score = (logit - maximum) / temperature - (-uniform.ln()).ln();
                }
                if score > best || (score == best && token < selected) {
                    best = score;
                    selected = token;
                }
            }
        }
        unsafe {
            values.add(tid as usize).write(best);
            ids.add(tid as usize).write(selected);
        }
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let a = values.add(tid as usize).read();
                    let b = values.add((tid + stride) as usize).read();
                    let x = ids.add(tid as usize).read();
                    let y = ids.add((tid + stride) as usize).read();
                    if b > a || (b == a && y < x) {
                        values.add(tid as usize).write(b);
                        ids.add(tid as usize).write(y);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let chosen = unsafe { ids.read() };
        thread::sync_threads();
        let mut sum = 0.0;
        if temperature > 0.0 {
            for i in 0..32 {
                sum += ((half(logits[(row * 8192 + tid + i * 256) as usize]) - maximum)
                    / temperature)
                    .exp();
            }
        }
        unsafe {
            values.add(tid as usize).write(sum);
        }
        let total = unsafe { reduce_sum(values) };
        if tid == 0 {
            let probability = if chosen >= 8192 {
                f32::NAN
            } else if temperature == 0.0 {
                1.0
            } else {
                ((half(logits[(row * 8192 + chosen) as usize]) - maximum) / temperature).exp()
                    / total
            };
            unsafe {
                *tokens.get_unchecked_mut((first_row + row) as usize) = chosen;
                *confidence.get_unchecked_mut((first_row + row) as usize) = probability;
            }
        }
    }

    /// Independent learned index projection. Attention continues to use the
    /// unmodified eight-head QKV tensor and block-causal visibility.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn project_index(qkv: &[u16], weights: &[u16], mut output: DisjointSlice<u16>, rows: u32) {
        static mut POOLED: SharedArray<f32, 64> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let dim = thread::threadIdx_x();
        let pooled = unsafe { SharedArray::as_raw_mut_ptr(&raw mut POOLED) };
        let mut sum = 0.0;
        for head in 0..8 {
            sum += half(qkv[(row * 640 + head * 64 + dim) as usize]);
        }
        unsafe {
            pooled.add(dim as usize).write(sum);
        }
        thread::sync_threads();
        let mut value = 0.0;
        for input in 0..64 {
            value = unsafe { pooled.add(input as usize).read() }
                .mul_add(half(weights[(input * 64 + dim) as usize]), value);
        }
        // pisa_select pools eight heads. Place the projected vector in head
        // zero and zero the remaining heads, without changing its FP16 scale.
        for head in 0..8 {
            unsafe {
                *output.get_unchecked_mut((row * 640 + head * 64 + dim) as usize) =
                    if head == 0 { half_bits(value) } else { 0 };
            }
        }
    }
}
