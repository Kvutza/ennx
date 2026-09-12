#include <metal_stdlib>
using namespace metal;

inline float widen(float x) { return x; }

struct Matmul {
    uint m, n, k, transpose_b;
    ulong stride_a, stride_b, stride_c;
};

kernel void flame_matmul(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* c [[buffer(2)]],
                        constant Matmul& p [[buffer(3)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint row = group.y * 32 + tid / 8;
    uint col = group.x * 32 + (tid % 8) * 4;
    if (row >= p.m) return;
    ulong batch_a = ulong(group.z) * p.stride_a;
    ulong batch_b = ulong(group.z) * p.stride_b;
    ulong batch_c = ulong(group.z) * p.stride_c;
    for (uint j = 0; j < 4 && col + j < p.n; ++j) {
        float sum = 0.0f;
        for (uint d = 0; d < p.k; ++d) {
            ulong b_index = p.transpose_b
                ? batch_b + ulong(col + j) * p.k + d
                : batch_b + ulong(d) * p.n + col + j;
            sum = fma(a[batch_a + ulong(row) * p.k + d], b[b_index], sum);
        }
        c[batch_c + ulong(row) * p.n + col + j] = sum;
    }
}

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

kernel void flame_softmax(device float* scores [[buffer(0)]],
                         constant Shape& p [[buffer(1)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    uint pos = row % p.rows;
    device float* values = scores + ulong(row) * p.rows;
    float scale = 1.0f / sqrt(float(p.width / p.heads));
    float peak = -INFINITY;
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

kernel void flame_xent(device const float* logits [[buffer(0)]],
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
        float value = values[j];
        if (!isfinite(value)) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        peak = max(peak, value);
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
