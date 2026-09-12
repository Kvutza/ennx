#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// BF16 is stored verbatim. Every arithmetic operand and accumulator is FP32.
inline float widen(ushort x) { return as_type<float>(uint(x) << 16); }
inline float widen(float x) { return x; }

struct Matmul {
    uint m, n, k, transpose_b;
    ulong stride_a, stride_b, stride_c;
};

// A 32x32 output tile, shared 32-wide K tiles, two 8x8 accumulators per SIMD.
template <typename T>
inline void tiled(device const float* a, device const T* b, device float* c,
                  constant Matmul& p, uint3 group, uint tid, uint simd,
                  threadgroup float* at, threadgroup float* bt,
                  threadgroup float* ct) {
    a += ulong(group.z) * p.stride_a;
    b += ulong(group.z) * p.stride_b;
    c += ulong(group.z) * p.stride_c;
    uint row = group.y * 32, col = group.x * 32;
    uint sr = (simd / 4) * 8, sc = (simd % 4) * 8;
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), av, bv;
    for (uint base = 0; base < p.k; base += 32) {
        for (uint i = tid; i < 1024; i += 256) {
            uint r = i / 32, d = i % 32;
            at[i] = row + r < p.m && base + d < p.k
                ? a[ulong(row + r) * p.k + base + d] : 0.0f;
            // Load B in its original row order, then transpose in shared memory.
            float value = col + r < p.n && base + d < p.k
                ? widen(b[p.transpose_b ? ulong(col + r) * p.k + base + d
                                        : ulong(base + d) * p.n + col + r]) : 0.0f;
            bt[d * 32 + r] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < 32; d += 8) {
            simdgroup_load(bv, bt + d * 32 + sc, 32);
            simdgroup_load(av, at + sr * 32 + d, 32);
            simdgroup_multiply_accumulate(acc0, av, bv, acc0);
            simdgroup_load(av, at + (sr + 16) * 32 + d, 32);
            simdgroup_multiply_accumulate(acc1, av, bv, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0, ct + sr * 32 + sc, 32);
    simdgroup_store(acc1, ct + (sr + 16) * 32 + sc, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) {
        uint r = row + i / 32, n = col + i % 32;
        if (r < p.m && n < p.n) c[ulong(r) * p.n + n] = ct[i];
    }
}

kernel void flame_linear(device const float* a [[buffer(0)]],
                         device const ushort* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant Matmul& p [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float at[1024], bt[1024], ct[1024];
    tiled(a, b, c, p, group, tid, simd, at, bt, ct);
}

kernel void flame_matmul(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant Matmul& p [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float at[1024], bt[1024], ct[1024];
    tiled(a, b, c, p, group, tid, simd, at, bt, ct);
}

kernel void flame_linear_small(device const float* a [[buffer(0)]],
                               device const ushort* b [[buffer(1)]],
                               device float* c [[buffer(2)]],
                               constant Matmul& p [[buffer(3)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint col = group.x * 8 + simd, row = group.y;
    float sum = 0.0f;
    if (col < p.n) {
        for (uint d = lane; d < p.k; d += 32)
            sum = fma(a[ulong(row) * p.k + d], widen(b[ulong(col) * p.k + d]), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0 && col < p.n) c[ulong(row) * p.n + col] = sum;
}

// Mirrors the CUDA 256-thread reduction tree and its FP32 rounding.
template <bool Maximum>
inline float reduce(float value, threadgroup float* scratch, uint lane) {
    scratch[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride /= 2) {
        if (lane < stride) scratch[lane] = Maximum
            ? max(scratch[lane], scratch[lane + stride])
            : scratch[lane] + scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

struct Shape { uint rows, width, heads, hidden, experts, top_k, start, sequence; float epsilon, rope_base; };

kernel void flame_embedding(device const ushort* w [[buffer(0)]],
                            device const int* tokens [[buffer(1)]],
                            device float* x [[buffer(2)]],
                            constant Shape& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width)
        x[i] = widen(w[ulong(tokens[i / p.width]) * p.width + i % p.width]);
}

kernel void flame_rms(device const float* x [[buffer(0)]],
                      device const ushort* w [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant Shape& p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    ulong base = ulong(row) * p.width;
    float sum = 0.0f;
    for (uint d = tid; d < p.width; d += 256) sum += x[base + d] * x[base + d];
    float scale = rsqrt(reduce<false>(sum, scratch, tid) / float(p.width) + p.epsilon);
    for (uint d = tid; d < p.width; d += 256)
        out[base + d] = (x[base + d] * scale) * widen(w[d]);
}

kernel void flame_rotary(device const float* qkv [[buffer(0)]],
                         device float* q [[buffer(1)]],
                         device float* k [[buffer(2)]],
                         device float* v [[buffer(3)]],
                         constant Shape& p [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.width) return;
    uint size = p.width / p.heads, half_size = size / 2;
    uint pos = i / p.width, col = i % p.width, head = col / size, d = col % size;
    uint partner = d < half_size ? d + half_size : d - half_size;
    float sign = d < half_size ? -1.0f : 1.0f;
    float angle = float(pos) * pow(p.rope_base, -float(2 * (d % half_size)) / float(size));
    float cs = cos(angle), sn = sin(angle);
    ulong src = ulong(pos) * 3 * p.width + head * 3 * size;
    ulong dst = (ulong(head) * p.rows + pos) * size + d;
    q[dst] = qkv[src + d] * cs + (sign * qkv[src + partner]) * sn;
    k[dst] = qkv[src + size + d] * cs + (sign * qkv[src + size + partner]) * sn;
    v[dst] = qkv[src + 2 * size + d];
}

kernel void flame_softmax(device float* scores [[buffer(0)]],
                          constant Shape& p [[buffer(1)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    uint pos = row % p.rows;
    device float* values = scores + ulong(row) * p.rows;
    float scale = 1.0f / sqrt(float(p.width / p.heads)), peak = -INFINITY;
    for (uint j = tid; j <= pos; j += 256) peak = max(peak, values[j] * scale);
    peak = reduce<true>(peak, scratch, tid);
    float sum = 0.0f;
    for (uint j = tid; j < p.rows; j += 256) {
        float value = j <= pos ? exp(values[j] * scale - peak) : 0.0f;
        values[j] = value;
        sum += value;
    }
    sum = reduce<false>(sum, scratch, tid);
    for (uint j = tid; j < p.rows; j += 256) values[j] /= sum;
}

kernel void flame_unpack(device const float* packed [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant Shape& p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.width) return;
    uint d = p.width / p.heads, col = i % p.width;
    out[i] = packed[(ulong(col / d) * p.rows + i / p.width) * d + col % d];
}

kernel void flame_residual(device float* x [[buffer(0)]],
                           device const float* update [[buffer(1)]],
                           constant Shape& p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width) x[i] += update[i];
}

kernel void flame_silu(device const float* gates [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant Shape& p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.hidden) return;
    ulong src = ulong(i / p.hidden) * 2 * p.hidden + i % p.hidden;
    float gate = gates[src];
    out[i] = (gate * (1.0f / (1.0f + exp(-gate)))) * gates[src + p.hidden];
}

kernel void flame_router(device float* logits [[buffer(0)]],
                         device float* probs [[buffer(1)]],
                         device uint* indices [[buffer(2)]],
                         constant Shape& p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    device float* values = logits + ulong(row) * p.experts;
    float peak = -INFINITY;
    for (uint e = tid; e < p.experts; e += 256) peak = max(peak, values[e]);
    peak = reduce<true>(peak, scratch, tid);
    float sum = 0.0f;
    for (uint e = tid; e < p.experts; e += 256) {
        values[e] = exp(values[e] - peak);
        sum += values[e];
    }
    sum = reduce<false>(sum, scratch, tid);
    for (uint e = tid; e < p.experts; e += 256) values[e] /= sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) for (uint slot = 0; slot < p.top_k; ++slot) {
        float best = -1.0f;
        uint expert = 0;
        for (uint e = 0; e < p.experts; ++e) if (values[e] > best || isnan(values[e])) {
            best = values[e];
            expert = e;
        }
        ulong dst = ulong(row) * p.top_k + slot;
        probs[dst] = best;
        indices[dst] = expert;
        values[expert] = -1.0f;
    }
}

// Stable grouping; an expert can receive each token at most once. Invalid
// router NaNs can repeat a selection, so guard capacity and report a bad count.
kernel void flame_group(device const uint* indices [[buffer(0)]],
                        device uint* slots [[buffer(1)]],
                        device uint* counts [[buffer(2)]],
                        constant Shape& p [[buffer(3)]],
                        uint expert [[thread_position_in_grid]]) {
    if (expert >= p.experts) return;
    uint count = 0;
    for (uint i = 0; i < p.rows * p.top_k; ++i) if (indices[i] == expert) {
        if (count < p.sequence) slots[ulong(expert) * p.sequence + count] = i;
        ++count;
    }
    counts[expert] = count;
}

kernel void flame_gather(device const float* x [[buffer(0)]],
                         device const uint* slots [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant Shape& p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width)
        out[i] = x[ulong(slots[i / p.width] / p.top_k) * p.width + i % p.width];
}

kernel void flame_scatter(device const float* x [[buffer(0)]],
                          device const uint* slots [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant Shape& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width)
        out[ulong(slots[i / p.width]) * p.width + i % p.width] = x[i];
}

kernel void flame_combine(device float* x [[buffer(0)]],
                          device const float* shared [[buffer(1)]],
                          device const float* routed [[buffer(2)]],
                          device const float* probs [[buffer(3)]],
                          constant Shape& p [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.width) return;
    ulong route = ulong(i / p.width) * p.top_k;
    float sum = 0.0f;
    for (uint s = 0; s < p.top_k; ++s)
        sum += routed[(route + s) * p.width + i % p.width] * probs[route + s];
    x[i] += shared[i] + sum;
}

kernel void flame_cross_entropy(device const float* logits [[buffer(0)]],
                                device const int* tokens [[buffer(1)]],
                                device const uchar* masks [[buffer(2)]],
                                device float* losses [[buffer(3)]],
                                device atomic_uint* invalid [[buffer(4)]],
                                constant Shape& p [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    device const float* values = logits + ulong(row) * p.width;
    float peak = -INFINITY;
    for (uint j = tid; j < p.width; j += 256) {
        float v = values[j];
        if (!isfinite(v)) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        peak = max(peak, v);
    }
    peak = reduce<true>(peak, scratch, tid);
    float sum = 0.0f;
    for (uint j = tid; j < p.width; j += 256) sum += exp(values[j] - peak);
    sum = reduce<false>(sum, scratch, tid);
    if (tid == 0) {
        uint target = p.start + row + 1;
        float loss = 0.0f;
        if (target < p.sequence && masks[target]) {
            loss = log(sum) + (peak - values[tokens[target]]);
            if (!isfinite(loss)) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        }
        losses[p.start + row] = loss;
    }
}

kernel void flame_mean(device float* losses [[buffer(0)]],
                       device atomic_uint* invalid [[buffer(1)]],
                       constant Shape& p [[buffer(2)]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    float sum = 0.0f;
    for (uint i = tid; i < p.rows; i += 256) sum += losses[i];
    sum = reduce<false>(sum, scratch, tid);
    if (tid == 0) {
        losses[0] = sum / float(p.hidden);
        if (!isfinite(losses[0])) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
    }
}
