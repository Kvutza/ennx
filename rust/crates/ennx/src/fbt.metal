#include <metal_stdlib>
using namespace metal;

inline float stable_sigmoid(float x) {
    float e = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

struct FeedbackParams {
    uint width;
    uint rows;
    float token_epsilon;
    float fused_epsilon;
};

struct LinearParams {
    uint input;
    uint output;
    uint sigmoid;
    uint rows;
};

#include <metal_simdgroup_matrix>

struct GemmParams {
    uint input, output, sigmoid, rows;
    uint mode, start;
    float scale;
    uint padding;
};

union GemmScratch {
    struct { float tail[64 * 8]; float b[32 * 33]; } operands;
    float result[64 * 32];
};

// Full tiles read FP32 activations directly from device memory. BF16 weights
// are widened once per 32-wide reduction slab, shared by four SIMD groups.
// Modes: projection, gated up projection, residual, vocabulary-loss partials.
kernel void fbt_gemm(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    device const float *gate [[buffer(4)]],
    device const uint *targets [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup GemmScratch scratch;
    threadgroup float *tail = scratch.operands.tail;
    threadgroup float *b = scratch.operands.b;
    threadgroup float *c = scratch.result;
    const ulong rb = ulong(group.y) * 64, cb = ulong(group.x) * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (ulong base = 0; base < p.input; base += 32) {
        ulong col = cb + tid / 4;
        ulong k_base = base + (tid % 4) * 8;
        if ((p.input & 7) == 0 && col < p.output && k_base + 8 <= p.input) {
            device const ushort4 *w4 = reinterpret_cast<device const ushort4*>(weights + col * p.input + k_base);
            ushort4 r0 = w4[0];
            ushort4 r1 = w4[1];
            uint k_off = (tid % 4) * 8;
            uint col_off = tid / 4;
            b[k_off * 33 + col_off] = as_type<float>(uint(r0.x) << 16);
            b[(k_off + 1) * 33 + col_off] = as_type<float>(uint(r0.y) << 16);
            b[(k_off + 2) * 33 + col_off] = as_type<float>(uint(r0.z) << 16);
            b[(k_off + 3) * 33 + col_off] = as_type<float>(uint(r0.w) << 16);
            b[(k_off + 4) * 33 + col_off] = as_type<float>(uint(r1.x) << 16);
            b[(k_off + 5) * 33 + col_off] = as_type<float>(uint(r1.y) << 16);
            b[(k_off + 6) * 33 + col_off] = as_type<float>(uint(r1.z) << 16);
            b[(k_off + 7) * 33 + col_off] = as_type<float>(uint(r1.w) << 16);
        } else {
            uint k_off = (tid % 4) * 8;
            uint col_off = tid / 4;
            for (uint e = 0; e < 8; ++e) {
                ulong k = k_base + e;
                b[(k_off + e) * 33 + col_off] = k < p.input && col < p.output
                    ? as_type<float>(uint(weights[col * p.input + k]) << 16) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            bool full = rb + 64 <= p.rows && base + k + 8 <= p.input;
            if (!full) {
                for (uint ix = tid; ix < 512; ix += 128) {
                    ulong row = rb + ix / 8, col = base + k + ix % 8;
                    tail[ix] = row < p.rows && col < p.input ? input[row * p.input + col] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_float8x8 av[2], bv[4];
            for (uint i = 0; i < 2; ++i) {
                if (full) simdgroup_load(av[i], input + (rb + sg * 16 + i * 8) * p.input + base + k, p.input);
                else simdgroup_load(av[i], tail + (sg * 16 + i * 8) * 8, 8);
            }
            for (uint j = 0; j < 4; ++j) simdgroup_load(bv[j], b + k * 33 + j * 8, 33);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
            if (!full) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], c + (sg * 16 + i * 8) * 32 + j * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.mode == 3) {
        ulong tiles = (ulong(p.output) + 31) / 32;
        for (uint r = sg; r < 64; r += 4) {
            if (rb + r >= p.rows) break;
            float v = cb + lane < p.output ? c[r * 32 + lane] : -INFINITY;
            float maximum = simd_max(v);
            float sum = simd_sum(exp(v - maximum));
            float target = simd_sum(cb + lane == targets[p.start + rb + r] ? v : 0.0f);
            if (lane == 0) {
                ulong ix = ((rb + r) * tiles + group.x) * 4;
                output[ix] = maximum; output[ix + 1] = sum;
                output[ix + 2] = target; output[ix + 3] = 0.0f;
            }
        }
    } else {
        for (uint ix = tid; ix < 2048; ix += 128) {
            ulong row = rb + ix / 32, col = cb + ix % 32;
            if (row >= p.rows || col >= p.output) continue;
            ulong dst = row * p.output + col;
            float v = c[ix];
            if (p.mode == 1) { float g = gate[dst]; v *= g * stable_sigmoid(g); }
            else if (p.mode == 2) v = output[dst] + p.scale * v;
            else if (p.sigmoid) v = stable_sigmoid(v);
            output[dst] = v;
        }
    }
}

// 64 prompt rows share each 32-column BF16 weight tile. Four SIMD groups
// accumulate in FP32; shared output staging makes every remainder shape safe.
kernel void fbt_linear_tiled(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float a[64 * 8];
    threadgroup float b[8 * 32];
    threadgroup float c[64 * 32];
    const ulong row_base = ulong(group.y) * 64;
    const ulong col_base = ulong(group.x) * 32;
    const uint row_slice = sg * 16;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (ulong base = 0; base < p.input; base += 8) {
        for (uint index = tid; index < 64 * 8; index += 128) {
            ulong row = row_base + index / 8;
            ulong k = base + index % 8;
            a[index] = row < p.rows && k < p.input ? input[row * p.input + k] : 0.0f;
        }
        for (uint index = tid; index < 8 * 32; index += 128) {
            ulong col = col_base + index % 32;
            ulong k = base + index / 32;
            b[index] = col < p.output && k < p.input
                ? as_type<float>(uint(weights[col * p.input + k]) << 16) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 av[2];
        simdgroup_float8x8 bv[4];
        for (uint i = 0; i < 2; ++i) simdgroup_load(av[i], &a[(row_slice + i * 8) * 8], 8);
        for (uint j = 0; j < 4; ++j) simdgroup_load(bv[j], &b[j * 8], 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], &c[(row_slice + i * 8) * 32 + j * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < 64 * 32; index += 128) {
        ulong row = row_base + index / 32;
        ulong col = col_base + index % 32;
        if (row < p.rows && col < p.output) {
            float value = c[index];
            output[row * p.output + col] = p.sigmoid ? stable_sigmoid(value) : value;
        }
    }
}

struct GroupedParams {
    uint input;
    uint output[3];
    uint sigmoid[3];
};

kernel void fbt_linear_grouped(
    device const ushort *w0 [[buffer(0)]],
    device const ushort *w1 [[buffer(1)]],
    device const ushort *w2 [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *y0 [[buffer(4)]],
    device float *y1 [[buffer(5)]],
    device float *y2 [[buffer(6)]],
    constant GroupedParams &p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint index = group.x;
    uint part = 0;
    if (index >= p.output[0]) { index -= p.output[0]; part = 1; }
    if (part == 1 && index >= p.output[1]) { index -= p.output[1]; part = 2; }
    device const ushort *weights = part == 0 ? w0 : (part == 1 ? w1 : w2);
    device float *output = part == 0 ? y0 : (part == 1 ? y1 : y2);
    float sum = 0.0f;
    for (ulong i = lane; i < p.input; i += 32) {
        float w = as_type<float>(uint(weights[ulong(index) * p.input + i]) << 16);
        sum += w * input[ulong(group.y) * p.input + i];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[ulong(group.y) * p.output[part] + index] =
        p.sigmoid[part] ? stable_sigmoid(sum) : sum;
}

// Correctness-first SIMD projection; tiled prefill GEMM is a separate path.
kernel void fbt_linear(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (ulong i = lane; i < p.input; i += 32) {
        float w = as_type<float>(uint(weights[ulong(group.x) * p.input + i]) << 16);
        sum += w * input[ulong(group.y) * p.input + i];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[ulong(group.y) * p.output + group.x] =
        p.sigmoid ? stable_sigmoid(sum) : sum;
}

// One SIMD group per input row. A plain row bypasses both feedback norms.
kernel void fbt_token_scale(
    device const float *tokens [[buffer(0)]],
    device float *scales [[buffer(1)]],
    constant FeedbackParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    if (p.token_epsilon > 0.0f) {
        if ((p.width & 3) == 0) {
            device const float4 *tok4 = reinterpret_cast<device const float4*>(tokens + ulong(row) * p.width);
            uint num_vec4 = p.width / 4;
            for (uint v = lane; v < num_vec4; v += 32) {
                float4 x = tok4[v];
                sum += dot(x, x);
            }
        } else {
            for (uint i = lane; i < p.width; i += 32) {
                float x = tokens[ulong(row) * p.width + i];
                sum += x * x;
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) scales[row] = p.token_epsilon > 0.0f
        ? rsqrt(sum / float(p.width) + p.token_epsilon) : 1.0f;
}

// One SIMD group per output coordinate, coalesced across each matrix row.
// Vectorized 128-bit loads for BF16 weights and FP32 activations.
kernel void fbt_project(
    device const ushort *weights [[buffer(0)]],
    device const float *previous [[buffer(1)]],
    device const float *tokens [[buffer(2)]],
    device const uint *fused [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device float *scratch [[buffer(5)]],
    constant FeedbackParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint out = group.x, row = group.y;
    ulong base = ulong(row) * p.width;
    if (fused[row] == 0) {
        if (lane == 0) scratch[base + out] = tokens[base + out];
        return;
    }
    ulong w = ulong(out) * p.width;
    ulong gate = ulong(p.width) * p.width + w;
    float row_scale = scales[row];
    float value = 0.0f, logit = 0.0f;
    if ((p.width & 3) == 0) {
        device const ushort4 *w4 = reinterpret_cast<device const ushort4*>(weights + w);
        device const ushort4 *g4 = reinterpret_cast<device const ushort4*>(weights + gate);
        device const float4 *prev4 = reinterpret_cast<device const float4*>(previous + base);
        device const float4 *tok4 = reinterpret_cast<device const float4*>(tokens + base);
        uint num_vec4 = p.width / 4;
        for (uint v = lane; v < num_vec4; v += 32) {
            ushort4 rw = w4[v];
            ushort4 rg = g4[v];
            float4 pv = prev4[v];
            float4 tv = tok4[v] * row_scale;
            float4 uw = float4(as_type<float>(uint(rw.x) << 16), as_type<float>(uint(rw.y) << 16),
                               as_type<float>(uint(rw.z) << 16), as_type<float>(uint(rw.w) << 16));
            float4 gw = float4(as_type<float>(uint(rg.x) << 16), as_type<float>(uint(rg.y) << 16),
                               as_type<float>(uint(rg.z) << 16), as_type<float>(uint(rg.w) << 16));
            value += dot(uw, pv);
            logit += dot(gw, tv);
        }
    } else {
        for (uint i = lane; i < p.width; i += 32) {
            float u = as_type<float>(uint(weights[w + i]) << 16);
            float g = as_type<float>(uint(weights[gate + i]) << 16);
            value += u * previous[base + i];
            logit += g * (tokens[base + i] * row_scale);
        }
    }
    value = simd_sum(value);
    logit = simd_sum(logit);
    if (lane == 0) {
        scratch[base + out] = value * stable_sigmoid(logit);
    }
}

struct FfnParams {
    uint width;
    uint intermediate;
    float residual_scale;
};

kernel void fbt_ffn_up(
    device const ushort *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *activation [[buffer(2)]],
    constant FfnParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint out = group.x, row = group.y;
    ulong matrix = ulong(p.width) * p.intermediate;
    ulong w = ulong(out) * p.width;
    float gate = 0.0f, up = 0.0f;
    for (uint i = lane; i < p.width; i += 32) {
        float x = input[ulong(row) * p.width + i];
        gate += as_type<float>(uint(weights[w + i]) << 16) * x;
        up += as_type<float>(uint(weights[matrix + w + i]) << 16) * x;
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (lane == 0) activation[ulong(row) * p.intermediate + out] =
        (gate * stable_sigmoid(gate)) * up;
}

kernel void fbt_ffn_down_residual(
    device const ushort *weights [[buffer(0)]],
    device const float *activation [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant FfnParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint out = group.x, row = group.y;
    ulong w = 2 * ulong(p.width) * p.intermediate + ulong(out) * p.intermediate;
    float sum = 0.0f;
    for (uint i = lane; i < p.intermediate; i += 32) {
        float weight = as_type<float>(uint(weights[w + i]) << 16);
        sum += weight * activation[ulong(row) * p.intermediate + i];
    }
    sum = simd_sum(sum);
    ulong index = ulong(row) * p.width + out;
    if (lane == 0) output[index] = residual[index] + p.residual_scale * sum;
}

kernel void fbt_fused_norm(
    device const float *scratch [[buffer(0)]],
    device const uint *fused [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant FeedbackParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    float sum = 0.0f;
    bool normalize = fused[row] != 0 && p.fused_epsilon > 0.0f;
    if (normalize) {
        if ((p.width & 3) == 0) {
            device const float4 *s4 = reinterpret_cast<device const float4*>(scratch + base);
            uint num_vec4 = p.width / 4;
            for (uint v = lane; v < num_vec4; v += 32) {
                float4 x = s4[v];
                sum += dot(x, x);
            }
        } else {
            for (uint i = lane; i < p.width; i += 32) {
                float x = scratch[base + i];
                sum += x * x;
            }
        }
    }
    sum = simd_sum(sum);
    float scale = normalize ? rsqrt(sum / float(p.width) + p.fused_epsilon) : 1.0f;
    if ((p.width & 3) == 0) {
        device const float4 *s4 = reinterpret_cast<device const float4*>(scratch + base);
        device float4 *out4 = reinterpret_cast<device float4*>(output + base);
        uint num_vec4 = p.width / 4;
        for (uint v = lane; v < num_vec4; v += 32) {
            out4[v] = s4[v] * scale;
        }
    } else {
        for (uint i = lane; i < p.width; i += 32) output[base + i] = scratch[base + i] * scale;
    }
}
