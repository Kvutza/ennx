#[repr(C)]
#[derive(Default)]
pub(super) struct Matmul {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) transpose_b: u32,
    pub(super) stride_a: u64,
    pub(super) stride_b: u64,
    pub(super) stride_c: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct FlameShape {
    pub(super) rows: u32,
    pub(super) width: u32,
    pub(super) heads: u32,
    pub(super) hidden: u32,
    pub(super) experts: u32,
    pub(super) top_k: u32,
    pub(super) start: u32,
    pub(super) sequence: u32,
    pub(super) epsilon: f32,
    pub(super) rope_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct QwenShape {
    pub(super) rows: u32,
    pub(super) width: u32,
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) hidden: u32,
    pub(super) start: u32,
    pub(super) sequence: u32,
    pub(super) epsilon: f32,
    pub(super) rope_theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct QwenQkvShape {
    pub(super) rows: u32,
    pub(super) hidden: u32,
    pub(super) kv_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct QwenMlpShape {
    pub(super) rows: u32,
    pub(super) hidden: u32,
    pub(super) intermediate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct CacheShape {
    pub(super) rows: u32,
    pub(super) kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) capacity: u32,
    pub(super) position: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct DecodeRopeShape {
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) capacity: u32,
    pub(super) position: u32,
    pub(super) batch: u32,
    pub(super) cache_stride: u32,
    pub(super) rope_theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct DecodeAttentionShape {
    pub(super) sequence: u32,
    pub(super) capacity: u32,
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) batch: u32,
    pub(super) cache_stride: u32,
    pub(super) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct PrefillAttentionShape {
    pub(super) rows: u32,
    pub(super) sequence: u32,
    pub(super) capacity: u32,
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ArgmaxShape {
    pub(super) width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ArgmaxBatchShape {
    pub(super) rows: u32,
    pub(super) width: u32,
}
