#include <metal_stdlib>
using namespace metal;

struct MoeShape {
    uint rows;
    uint width;
    uint experts;
    uint rows_per_expert;
    uint expert_width;
};

kernel void fbt_moe_balanced_gate(
    device const half* input [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device half* gates [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = gid >> 5u;
    if (token >= p.rows) return;
    const uint expert = token % p.experts;
    float value = 0.0f;
    for (uint column = lane; column < p.width; column += 32u) {
        value += float(input[ulong(token) * p.width + column]) *
            float(router[ulong(column) * p.experts + expert]);
    }
    value = simd_sum(value);
    if (lane == 0u) {
        const float e = exp(-abs(value));
        gates[token] = half(value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
    }
}

kernel void fbt_moe_group_balanced(
    device const half* input [[buffer(0)]],
    device half* grouped [[buffer(1)]],
    constant MoeShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    const uint expert = token % p.experts;
    const uint row = token / p.experts;
    grouped[(ulong(expert) * p.rows_per_expert + row) * p.width + column] = input[gid];
}

kernel void fbt_moe_swiglu(
    device const half* gate_up [[buffer(0)]],
    device half* activation [[buffer(1)]],
    constant MoeShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.rows) * p.expert_width;
    if (gid >= total) return;
    const uint row = gid / p.expert_width;
    const uint column = gid % p.expert_width;
    const ulong base = ulong(row) * 2u * p.expert_width;
    const float gate = float(gate_up[base + column]);
    const float e = exp(-abs(gate));
    const float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    activation[gid] = half(gate * sigmoid * float(gate_up[base + p.expert_width + column]));
}

kernel void fbt_moe_ungroup_residual(
    device const half* input [[buffer(0)]],
    device const half* grouped [[buffer(1)]],
    device const half* gates [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MoeShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    const uint expert = token % p.experts;
    const uint row = token / p.experts;
    const ulong source = (ulong(expert) * p.rows_per_expert + row) * p.width + column;
    output[gid] = half(float(input[gid]) + float(gates[token]) * float(grouped[source]));
}

kernel void fbt_moe_feedback_fuse(
    device const half* state [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const float value = float(gate[gid]);
    const float e = exp(-abs(value));
    const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    output[gid] = half(float(state[gid]) * sigmoid);
}

kernel void fbt_moe_cross_entropy(
    device const half* logits [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device float* losses [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup float shared;
    if (row >= 8192) return;
    const ulong base = ulong(row) * 8192;
    float local_max = -INFINITY;
    for (uint column = tid; column < 8192; column += 256)
        local_max = max(local_max, float(logits[base + column]));
    const float group_max = simd_max(local_max);
    if (lane == 0) partial[simdgroup] = group_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : -INFINITY;
        const float maximum = simd_max(value);
        if (lane == 0) shared = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = shared;
    float local_sum = 0.0f;
    for (uint column = tid; column < 8192; column += 256)
        local_sum += exp(float(logits[base + column]) - maximum);
    const float group_sum = simd_sum(local_sum);
    if (lane == 0) partial[simdgroup] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) {
            const uint label = labels[row];
            losses[row] = log(total) + maximum - float(logits[base + label]);
        }
    }
}

kernel void fbt_moe_rms(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float squared = 0.0f;
    for (uint column = tid; column < p.width; column += 128) {
        const float value = float(input[base + column]);
        squared = fma(value, value, squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < p.width; column += 128)
        output[base + column] = half(
            float(input[base + column]) * scale * float(weight[column]));
}

kernel void fbt_moe_embed(
    device const half* weights [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint rows = 8192;
    constexpr uint width = 512;
    constexpr uint vocab = 8192;
    if (gid >= rows * width) return;
    const uint row = gid / width;
    const uint column = gid % width;
    output[gid] = weights[ulong(column) * vocab + tokens[row]];
}

kernel void fbt_moe_sequence_loss(
    device const float* losses [[buffer(0)]],
    device const uchar* score_mask [[buffer(1)]],
    device float* scores [[buffer(2)]],
    uint sequence [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint context = 4096;
    threadgroup float partial[8];
    threadgroup uint counts[8];
    float sum = 0.0f;
    uint count = 0;
    const uint start = sequence * context;
    for (uint token = tid; token < context; token += 256) {
        const bool scored = score_mask[start + token] != 0;
        sum += scored ? losses[start + token] : 0.0f;
        count += uint(scored);
    }
    sum = simd_sum(sum);
    count = simd_sum(count);
    if (lane == 0) {
        partial[simdgroup] = sum;
        counts[simdgroup] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : 0.0f;
        const uint count_value = lane < 8 ? counts[lane] : 0;
        const float total = simd_sum(value);
        const uint total_count = simd_sum(count_value);
        if (lane == 0) scores[sequence] = total / float(total_count);
    }
}

kernel void fbt_moe_residual(
    device const half* input [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    output[gid] = half(float(input[gid]) + float(branch[gid]));
}

kernel void fbt_moe_residual_rms(
    device const half* input [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* residual [[buffer(3)]],
    device half* normalized [[buffer(4)]],
    constant MoeShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float values[4];
    float squared = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const half value = half(float(input[base + column]) + float(branch[base + column]));
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        residual[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}

kernel void fbt_moe_ungroup_residual_rms(
    device const half* input [[buffer(0)]],
    device const half* grouped [[buffer(1)]],
    device const half* gates [[buffer(2)]],
    device const half* weight [[buffer(3)]],
    device half* output [[buffer(4)]],
    device half* normalized [[buffer(5)]],
    constant MoeShape& p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const uint expert = row % p.experts;
    const uint expert_row = row / p.experts;
    const ulong base = ulong(row) * p.width;
    const ulong source =
        (ulong(expert) * p.rows_per_expert + expert_row) * p.width;
    const float gate = float(gates[row]);
    float values[4];
    float squared = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const half value = half(float(input[base + column])
            + gate * float(grouped[source + column]));
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        output[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}

kernel void fbt_moe_feedback_fuse_rms(
    device const half* state [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    device half* normalized [[buffer(4)]],
    constant MoeShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float values[4];
    float squared = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const float gate_value = float(gate[base + column]);
        const float e = exp(-abs(gate_value));
        const float sigmoid = gate_value >= 0.0f
            ? 1.0f / (1.0f + e)
            : e / (1.0f + e);
        const half value = half(float(state[base + column]) * sigmoid);
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        output[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}
