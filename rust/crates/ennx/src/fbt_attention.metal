#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

struct NormParams { uint width; float epsilon; };

kernel void fbt_rms_affine(
    device const float *input [[buffer(0)]],
    device const ushort *gamma [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    float sum = 0.0f;
    for (uint i = lane; i < p.width; i += 32) sum += input[base+i] * input[base+i];
    float scale = rsqrt(simd_sum(sum) / float(p.width) + p.epsilon);
    for (uint i = lane; i < p.width; i += 32)
        output[base+i] = input[base+i] * scale * as_type<float>(uint(gamma[i]) << 16);
}

struct AttentionParams {
    uint heads;
    uint kv_heads;
    uint dim;
    uint start;
    uint rows;
    uint window;
    float epsilon;
    float rope_base;
    float score_scale;
};

inline ushort to_bf16(float x) {
    uint bits = as_type<uint>(x);
    return ushort((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

// Parallel head normalization/rotation; K/V writes are append-only, never rings.
kernel void fbt_prepare_qkv(
    device const float *queries [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device float *q_rotated [[buffer(3)]],
    device ushort *key_cache [[buffer(4)]],
    device ushort *value_cache [[buffer(5)]],
    constant AttentionParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y;
    bool is_q = head < p.heads;
    uint h = is_q ? head : head - p.heads;
    uint count = is_q ? p.heads : p.kv_heads;
    ulong offset = (ulong(row) * count + h) * p.dim;
    device const float *source = is_q ? queries : keys;
    float sum = 0.0f;
    if (p.epsilon > 0.0f)
        for (uint i = lane; i < p.dim; i += 32) sum += source[offset+i] * source[offset+i];
    sum = simd_sum(sum);
    float scale = p.epsilon > 0.0f ? rsqrt(sum / float(p.dim) + p.epsilon) : 1.0f;
    uint midpoint = p.dim / 2;
    ulong cache = (ulong(p.start + row) * p.kv_heads + h) * p.dim;
    for (uint i = lane; i < p.dim; i += 32) {
        uint pair = i % midpoint;
        float angle = float(p.start + row) * pow(p.rope_base, -2.0f * float(pair) / float(p.dim));
        float a = source[offset+pair] * scale;
        float b = source[offset+midpoint+pair] * scale;
        float x = i < midpoint ? a*cos(angle) - b*sin(angle) : b*cos(angle) + a*sin(angle);
        if (is_q) q_rotated[offset+i] = x;
        else {
            key_cache[cache+i] = to_bf16(x);
            value_cache[cache+i] = to_bf16(values[offset+i]);
        }
    }
}

// Online stable softmax: bounded per-lane accumulators, no sequence-squared buffer.
kernel void fbt_cached_attention(
    device const float *queries [[buffer(0)]],
    device const ushort *keys [[buffer(1)]],
    device const ushort *values [[buffer(2)]],
    device const float *gates [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant AttentionParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y;
    uint kv = head / (p.heads / p.kv_heads);
    uint position = p.start + row;
    uint begin = p.window == 0 || position < p.window ? 0 : position + 1 - p.window;
    ulong qbase = (ulong(row) * p.heads + head) * p.dim;
    float q[8], acc[8];
    for (uint j = 0; j < 8; ++j) {
        uint i = lane + 32*j;
        q[j] = i < p.dim ? queries[qbase+i] : 0.0f;
        acc[j] = 0.0f;
    }
    float maximum = -INFINITY, total = 0.0f;
    for (uint t = begin; t <= position; ++t) {
        ulong base = (ulong(t) * p.kv_heads + kv) * p.dim;
        float score = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            uint i = lane + 32*j;
            if (i < p.dim) score += q[j] * as_type<float>(uint(keys[base+i]) << 16);
        }
        score = simd_sum(score) * p.score_scale;
        float next_max = max(maximum, score);
        float alpha = exp(maximum - next_max), beta = exp(score - next_max);
        total = alpha * total + beta;
        for (uint j = 0; j < 8; ++j) {
            uint i = lane + 32*j;
            if (i < p.dim)
                acc[j] = alpha * acc[j] + beta * as_type<float>(uint(values[base+i]) << 16);
        }
        maximum = next_max;
    }
    float gate = gates[ulong(row) * p.heads + head];
    for (uint j = 0; j < 8; ++j) {
        uint i = lane + 32*j;
        if (i < p.dim) output[qbase+i] = (acc[j] / total) * gate;
    }
}

// Eight queries reuse each 16-key tile. QK and probability-times-V use
// FP32 SIMD-group matrix operations; online softmax avoids a context-square buffer.
// Specialized for the model's 96-wide heads. The scalar SIMD path remains the oracle.
kernel void fbt_tiled_attention96(
    device const float *queries [[buffer(0)]],
    device const ushort *keys [[buffer(1)]],
    device const ushort *values [[buffer(2)]],
    device const float *gates [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant AttentionParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float q[8 * 96], kv[16 * 96], scores[8 * 16];
    threadgroup float result[8 * 96], product[8 * 96];
    threadgroup float maxima[8], totals[8], alphas[8];
    uint head = group.x, kv_head = head / (p.heads / p.kv_heads);
    uint row_base = group.y * 8;
    uint first_pos = p.start + row_base;
    uint last_pos = p.start + min(row_base + 7, p.rows - 1);
    uint begin = p.window == 0 || first_pos < p.window ? 0 : first_pos + 1 - p.window;
    for (uint i = tid; i < 8 * 96; i += 128) {
        uint row = row_base + i / 96;
        q[i] = row < p.rows ? queries[(ulong(row) * p.heads + head) * 96 + i % 96] : 0.0f;
        result[i] = 0.0f;
    }
    if (tid < 8) { maxima[tid] = -INFINITY; totals[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong base = begin; base <= last_pos; base += 16) {
        // K is staged transposed for Q*K^T. The same storage later holds V.
        for (uint i = tid; i < 16 * 96; i += 128) {
            ulong token = base + i % 16;
            kv[i] = token <= last_pos
                ? as_type<float>(uint(keys[(token * p.kv_heads + kv_head) * 96 + i / 16]) << 16) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2) {
            simdgroup_float8x8 score(0);
            for (uint d = 0; d < 96; d += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, &q[d], 96);
                simdgroup_load(b, &kv[d * 16 + sg * 8], 16);
                simdgroup_multiply_accumulate(score, a, b, score);
            }
            simdgroup_store(score, &scores[sg * 8], 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = sg; row < 8; row += 4) {
            uint position = p.start + row_base + row;
            uint lower = p.window == 0 || position < p.window ? 0 : position + 1 - p.window;
            ulong token = base + lane;
            bool valid = lane < 16 && row_base + row < p.rows && token >= lower && token <= position;
            float value = valid ? scores[row * 16 + lane] * p.score_scale : -INFINITY;
            float next_max = max(maxima[row], simd_max(value));
            if (row_base + row >= p.rows) next_max = 0.0f;
            float alpha = exp(maxima[row] - next_max);
            float probability = valid ? exp(value - next_max) : 0.0f;
            float total = totals[row] * alpha + simd_sum(probability);
            if (lane < 16) scores[row * 16 + lane] = probability;
            if (lane == 0) { maxima[row] = next_max; totals[row] = total; alphas[row] = alpha; }
        }
        for (uint i = tid; i < 16 * 96; i += 128) {
            ulong token = base + i / 96;
            kv[i] = token <= last_pos
                ? as_type<float>(uint(values[(token * p.kv_heads + kv_head) * 96 + i % 96]) << 16) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint part = 0; part < 3; ++part) {
            uint col = sg * 8 + part * 32;
            simdgroup_float8x8 acc(0);
            for (uint k = 0; k < 16; k += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, &scores[k], 16);
                simdgroup_load(b, &kv[k * 96 + col], 96);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, &product[col], 96);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 96; i += 128)
            result[i] = alphas[i / 96] * result[i] + product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 8 * 96; i += 128) {
        uint row = row_base + i / 96;
        if (row < p.rows) output[(ulong(row) * p.heads + head) * 96 + i % 96] =
            result[i] / totals[i / 96] * gates[ulong(row) * p.heads + head];
    }
}
