// GEMVT scheduling adapted from MLX GEMVTKernel (Apple, MIT).
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.h
// BM=4, BN=1, SM=4, SN=8, TM=4, TN=4, FP16 storage/FP32 accumulation.
// Copyright (c) 2023-2024 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include <metal_stdlib>
using namespace metal;

#ifndef PISA_CONTEXT
#define PISA_CONTEXT 4096
#endif

kernel void decode_gemv(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const uint* experts [[buffer(3)]],
    constant uint3& shape [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint k = shape.x, n = shape.y;
    const uint expert = group.y == 0 ? 0 : experts[group.y - 1] + 1;
    if (shape.z != 0) weight += ulong(expert) * k * n;
    if (shape.z == 2) input += group.y * k;
    output += group.y * n;
    const uint row = (sg * 4 + lane / 8) * 4;
    const uint column = group.x * 32 + (lane % 8) * 4;
    float result[4] = {0.0f};
    for (uint block = 0; block < k; block += 64) {
        for (uint m = 0; m < 4; ++m) {
            const uint source = block + row + m;
            if (source < k) {
                const float x = float(input[source]);
                for (uint j = 0; j < 4; ++j)
                    if (column + j < n)
                        result[j] = fma(x, float(weight[ulong(source) * n + column + j]), result[j]);
            }
        }
    }
    for (uint j = 0; j < 4; ++j) {
        result[j] += simd_shuffle_down(result[j], 16);
        result[j] += simd_shuffle_down(result[j], 8);
    }
    threadgroup float partial[4][32];
    if (lane < 8)
        for (uint j = 0; j < 4; ++j) partial[sg][lane * 4 + j] = result[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 8) {
        for (uint j = 0; j < 4; ++j) {
            float value = result[j];
            for (uint other = 1; other < 4; ++other) value += partial[other][lane * 4 + j];
            if (column + j < n) output[column + j] = half(value);
        }
    }
}

kernel void decode_gemv_vector(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const uint* experts [[buffer(3)]],
    constant uint3& shape [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint k = shape.x, n = shape.y;
    const uint expert = group.y == 0 ? 0 : experts[group.y - 1] + 1;
    if (shape.z != 0) weight += ulong(expert) * k * n;
    if (shape.z == 2) input += group.y * k;
    output += group.y * n;
    const uint row = (sg * 4 + lane / 8) * 4;
    const uint column = group.x * 32 + (lane % 8) * 4;
    const bool packed = (n & 3u) == 0u;
    float4 result = 0.0f;
    for (uint block = 0; block < k; block += 64) {
        for (uint m = 0; m < 4; ++m) {
            const uint source = block + row + m;
            if (source < k) {
                const float x = float(input[source]);
                if (packed && column + 3 < n) {
                    const half4 values = *reinterpret_cast<device const half4*>(
                        weight + ulong(source) * n + column);
                    result = fma(float4(x), float4(values), result);
                } else if (!packed) {
                    for (uint j = 0; j < 4; ++j)
                        if (column + j < n)
                            result[j] = fma(
                                x,
                                float(weight[ulong(source) * n + column + j]),
                                result[j]);
                }
            }
        }
    }
    result += simd_shuffle_down(result, 16);
    result += simd_shuffle_down(result, 8);
    threadgroup float4 partial[4][8];
    if (lane < 8) partial[sg][lane] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 8) {
        float4 value = result;
        for (uint other = 1; other < 4; ++other) value += partial[other][lane];
        if (packed && column + 3 < n) {
            *reinterpret_cast<device half4*>(output + column) = half4(value);
        } else if (!packed) {
            for (uint j = 0; j < 4; ++j)
                if (column + j < n) output[column + j] = half(value[j]);
        }
    }
}

kernel void decode_pad_router(
    device const half* source [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint layers = 5;
    constexpr uint width = 512;
    constexpr uint experts = 625;
    constexpr uint padded = 628;
    constexpr uint elements = layers * width * padded;
    if (gid >= elements) return;
    const uint column = gid % padded;
    const uint row = gid / padded;
    output[gid] = column < experts
        ? source[ulong(row) * experts + column]
        : half(0.0f);
}

kernel void decode_embed(
    device const half* weight [[buffer(0)]], device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& position [[buffer(3)]],
    uint column [[thread_position_in_grid]]) {
    if (column < 512) output[column] = weight[ulong(column) * 8192 + tokens[position]];
}

// Completed leaves only: routing never scores a node containing future keys.
// A partial current leaf is forced, so it requires no speculative summary.
kernel void decode_leaf(
    device const half* qkv [[buffer(0)]], device half* pyramid [[buffer(1)]],
    constant uint& position [[buffer(2)]], uint dim [[thread_position_in_grid]]) {
    const uint leaf = position / 64;
    float sum = 0.0f;
    for (uint token = 0; token < 64; ++token)
        sum += float(qkv[ulong(leaf * 64 + token) * 640 + 512 + dim]);
    pyramid[leaf * 64 + dim] = half(sum / 64.0f);
    uint child = leaf, child_offset = 0, parent_offset = PISA_CONTEXT / 64, count = PISA_CONTEXT / 64;
    while (count > 1) {
        // An incomplete right subtree will never be scored; only publish
        // ancestors whose last leaf has now completed.
        if ((child & 1u) == 0) break;
        const uint parent = child / 2;
        pyramid[(parent_offset + parent) * 64 + dim] = half(0.5f * (
            float(pyramid[(child_offset + 2 * parent) * 64 + dim])
            + float(pyramid[(child_offset + 2 * parent + 1) * 64 + dim])));
        child = parent;
        child_offset = parent_offset;
        count /= 2;
        parent_offset += count;
    }
}

kernel void decode_combine_residual_rms(
    device const half* experts [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device const half* norm_weight [[buffer(3)]],
    device half* residual [[buffer(4)]],
    device half* normalized [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    float values[4];
    float squared = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        float branch = float(experts[column]);
        for (uint route = 0; route < 3; ++route)
            branch = fma(
                float(route_weights[route]),
                float(experts[(route + 1) * 512 + column]),
                branch);
        const half rounded_branch = half(branch);
        const half value = half(float(input[column]) + float(rounded_branch));
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / 512.0f + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        residual[column] = half(values[item]);
        normalized[column] = half(values[item] * scale * float(norm_weight[column]));
    }
}

kernel void decode_combine_branch(
    device const half* experts [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 512) return;
    float value = float(experts[gid]);
    for (uint route = 0; route < 3; ++route)
        value = fma(
            float(route_weights[route]),
            float(experts[(route + 1) * 512 + gid]),
            value);
    output[gid] = half(value);
}

struct Sample {
    ulong seed;
    uint position;
    uint eos;
    uint first;
    float temperature;
};

METAL_FUNC ulong decode_mix(ulong value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

// Gumbel-max categorical sampling stays on GPU. No CPU token synchronization.
// Stop tokens are absorbing; the host reports only through the first EOS.
kernel void decode_sample(
    device const half* logits [[buffer(0)]], device uint* tokens [[buffer(1)]],
    device uint* invalid [[buffer(2)]], constant Sample& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float values[8];
    threadgroup uint indices[8];
    float best = -INFINITY;
    uint index = UINT_MAX;
    bool bad = false;
    for (uint token = tid; token < 8192; token += 256) {
        float value = float(logits[token]);
        bad |= !isfinite(value);
        if (p.temperature > 0.0f) {
            const ulong bits = decode_mix(p.seed ^ (ulong(p.position) << 32) ^ token);
            const float uniform = (float(uint(bits >> 41)) + 0.5f) / 8388608.0f;
            value -= p.temperature * log(-log(uniform));
            bad |= !isfinite(value);
        }
        if (value > best || (value == best && token < index)) { best = value; index = token; }
    }
    const float maximum = simd_max(best);
    const uint winner = simd_min(best == maximum ? index : UINT_MAX);
    const bool any_bad = simd_any(bad);
    if (lane == 0) { values[sg] = maximum; indices[sg] = winner; }
    if (any_bad && lane == 0) atomic_store_explicit(
        reinterpret_cast<device atomic_uint*>(invalid), 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float value = lane < 8 ? values[lane] : -INFINITY;
        const float total = simd_max(value);
        const uint chosen = simd_min(lane < 8 && value == total ? indices[lane] : UINT_MAX);
        if (lane == 0) tokens[p.position + 1] = p.position >= p.first && tokens[p.position] == p.eos
            ? p.eos : chosen;
    }
}
