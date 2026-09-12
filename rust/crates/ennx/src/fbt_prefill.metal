#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void fbt_widen_weights(device const ushort *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant ulong &count [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    ulong i = ulong(group) * 32 + lane;
    if (i < count) output[i] = as_type<float>(uint(input[i]) << 16);
}

kernel void fbt_transpose_half(
    device const ushort *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup half tile[16][17];
    uint in_row = group.y * 16 + tid.y;
    uint in_col = group.x * 16 + tid.x;
    if (in_row < shape.x && in_col < shape.y) {
        float f = as_type<float>(uint(input[ulong(in_row) * shape.y + in_col]) << 16);
        tile[tid.y][tid.x] = half(f);
    } else {
        tile[tid.y][tid.x] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint out_row = group.x * 16 + tid.y;
    uint out_col = group.y * 16 + tid.x;
    if (out_row < shape.y && out_col < shape.x) {
        output[ulong(out_row) * shape.x + out_col] = tile[tid.x][tid.y];
    }
}

kernel void fbt_pack_qkvg(
    device const ushort *q [[buffer(0)]],
    device const ushort *k [[buffer(1)]],
    device const ushort *v [[buffer(2)]],
    device const ushort *gate [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant uint &k_dim [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint col = tid.x;
    uint row = tid.y;
    if (row >= k_dim || col >= 3088) return;
    float val = 0.0f;
    if (col < 1536) {
        val = as_type<float>(uint(q[ulong(col) * k_dim + row]) << 16);
    } else if (col < 2304) {
        uint c = col - 1536;
        val = as_type<float>(uint(k[ulong(c) * k_dim + row]) << 16);
    } else if (col < 3072) {
        uint c = col - 2304;
        val = as_type<float>(uint(v[ulong(c) * k_dim + row]) << 16);
    } else {
        uint c = col - 3072;
        val = as_type<float>(uint(gate[ulong(c) * k_dim + row]) << 16);
    }
    out[ulong(row) * 3088 + col] = half(val);
}

kernel void fbt_pack_gate_up(
    device const ushort *gate [[buffer(0)]],
    device const ushort *up [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint col = tid.x;
    uint row = tid.y;
    if (row >= k_dim || col >= 13312) return;
    float val = 0.0f;
    if (col < 6656) {
        val = as_type<float>(uint(gate[ulong(col) * k_dim + row]) << 16);
    } else {
        uint c = col - 6656;
        val = as_type<float>(uint(up[ulong(c) * k_dim + row]) << 16);
    }
    out[ulong(row) * 13312 + col] = half(val);
}




struct NormParams { uint width; float epsilon; };

kernel void fbt_prefill_rms_half(
    device const float *input [[buffer(0)]],
    device const ushort *gamma [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    device const float4 *in4 = reinterpret_cast<device const float4*>(input + base);
    device half4 *out4 = reinterpret_cast<device half4*>(output + base);
    uint width4 = p.width / 4;
    float sum = 0.0f;
    for (uint i = lane; i < width4; i += 32) {
        float4 v = in4[i];
        sum += dot(v, v);
    }
    float scale = rsqrt(simd_sum(sum) / float(p.width) + p.epsilon);
    for (uint i = lane; i < width4; i += 32) {
        float4 v = in4[i];
        uint base_g = i * 4;
        float4 g = float4(
            as_type<float>(uint(gamma[base_g]) << 16),
            as_type<float>(uint(gamma[base_g + 1]) << 16),
            as_type<float>(uint(gamma[base_g + 2]) << 16),
            as_type<float>(uint(gamma[base_g + 3]) << 16)
        );
        out4[i] = half4(v * scale * g);
    }
}

kernel void fbt_prefill_unit_rms_half(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant NormParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    device const float4 *in4 = reinterpret_cast<device const float4*>(input + base);
    device half4 *out4 = reinterpret_cast<device half4*>(output + base);
    uint width4 = p.width / 4;
    float sum = 0.0f;
    for (uint i = lane; i < width4; i += 32) {
        float4 v = in4[i];
        sum += dot(v, v);
    }
    float scale = rsqrt(simd_sum(sum) / float(p.width) + p.epsilon);
    for (uint i = lane; i < width4; i += 32) {
        out4[i] = half4(in4[i] * scale);
    }
}

struct FeedbackNormParams { uint width; float fused_epsilon; };

kernel void fbt_feedback_combine_norm(
    device const half *value [[buffer(0)]],
    device const half *logit [[buffer(1)]],
    device const float *tokens [[buffer(2)]],
    device const uint *fused [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant FeedbackNormParams &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    if (fused[row] == 0) {
        for (uint i = lane; i < p.width; i += 32) output[base + i] = tokens[base + i];
        return;
    }
    float sum_sq = 0.0f;
    for (uint i = lane; i < p.width; i += 32) {
        float v = float(value[base + i]);
        float g = float(logit[base + i]);
        float e = exp(-abs(g));
        float sig = g >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
        float s = v * sig;
        sum_sq += s * s;
    }
    float scale = rsqrt(simd_sum(sum_sq) / float(p.width) + p.fused_epsilon);
    for (uint i = lane; i < p.width; i += 32) {
        float v = float(value[base + i]);
        float g = float(logit[base + i]);
        float e = exp(-abs(g));
        float sig = g >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
        output[base + i] = (v * sig) * scale;
    }
}

struct GraphParams { uint width, rows, start, vocab; float scale; };

kernel void fbt_prefill_residual_half(
    device const half *embed [[buffer(0)]],
    device float *x [[buffer(1)]],
    constant GraphParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.width;
    device const half4 *embed4 = reinterpret_cast<device const half4*>(embed + base);
    device float4 *x4 = reinterpret_cast<device float4*>(x + base);
    uint width4 = p.width / 4;
    for (uint i = lane; i < width4; i += 32) {
        x4[i] += float4(embed4[i]) * p.scale;
    }
}

kernel void fbt_prefill_glu_half(
    device const half *gate [[buffer(0)]],
    device half *up [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    for (uint i = lane; i < width; i += 32) {
        ulong ix = ulong(row)*width+i;
        float g = float(gate[ix]), e = exp(-abs(g));
        float val = g * (g >= 0.0f ? 1.0f/(1.0f+e) : e/(1.0f+e));
        up[ix] = half(float(up[ix]) * val);
    }
}

kernel void fbt_prefill_glu_half_fused(
    device const half *gate_up [[buffer(0)]],
    device half *out_up [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong base_in = ulong(row) * width * 2;
    ulong base_out = ulong(row) * width;
    device const half4 *gate4 = reinterpret_cast<device const half4*>(gate_up + base_in);
    device const half4 *up4 = reinterpret_cast<device const half4*>(gate_up + base_in + width);
    device half4 *out4 = reinterpret_cast<device half4*>(out_up + base_out);
    uint width4 = width / 4;
    for (uint i = lane; i < width4; i += 32) {
        float4 g = float4(gate4[i]);
        float4 u = float4(up4[i]);
        float4 e = exp(-abs(g));
        float4 sig = float4(
            g.x >= 0.0f ? 1.0f / (1.0f + e.x) : e.x / (1.0f + e.x),
            g.y >= 0.0f ? 1.0f / (1.0f + e.y) : e.y / (1.0f + e.y),
            g.z >= 0.0f ? 1.0f / (1.0f + e.z) : e.z / (1.0f + e.z),
            g.w >= 0.0f ? 1.0f / (1.0f + e.w) : e.w / (1.0f + e.w)
        );
        out4[i] = half4(g * sig * u);
    }
}


struct PrefillParams {
    uint length, heads, kv_heads, dim, start, block, window, key_start, key_rows;
    float epsilon, rope_base;
};

inline float bf16_round(float x) {
    uint bits = as_type<uint>(x);
    return as_type<float>((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u);
}

// Head-major FP32 matrices for MPS. K/V retain the reference's BF16 rounding.
// GQA heads share values logically; materializing copies enables batched GEMM.
kernel void fbt_prefill_prepare(device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
    device half *qr [[buffer(3)]], device half *kr [[buffer(4)]],
    device half *vr [[buffer(5)]], constant PrefillParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y, pos = row % p.length, sample = row / p.length;
    uint kv = head / (p.heads / p.kv_heads);
    ulong qi = (ulong(row) * p.heads + head) * p.dim;
    ulong ki = (ulong(row) * p.kv_heads + kv) * p.dim;
    ulong dst = ((ulong(sample) * p.heads + head) * p.length + pos) * p.dim;
    float qs = 0, ks = 0;
    for (uint i = lane; i < p.dim; i += 32) { qs += q[qi+i]*q[qi+i]; ks += k[ki+i]*k[ki+i]; }
    qs = rsqrt(simd_sum(qs) / float(p.dim) + p.epsilon);
    ks = rsqrt(simd_sum(ks) / float(p.dim) + p.epsilon);
    uint midpoint = p.dim / 2;
    for (uint i = lane; i < p.dim; i += 32) {
        uint pair = i % midpoint;
        float angle = float(pos) * pow(p.rope_base, -2.0f * float(pair) / float(p.dim));
        float co = cos(angle), si = sin(angle);
        float qa = q[qi+pair]*qs, qb = q[qi+midpoint+pair]*qs;
        float ka = k[ki+pair]*ks, kb = k[ki+midpoint+pair]*ks;
        qr[dst+i] = half(i < midpoint ? qa*co - qb*si : qb*co + qa*si);
        kr[dst+i] = half(i < midpoint ? ka*co - kb*si : kb*co + ka*si);
        vr[dst+i] = half(v[ki+i]);
    }
}

kernel void fbt_prefill_prepare_half(device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
    device half *qr [[buffer(3)]], device half *kr [[buffer(4)]],
    device half *vr [[buffer(5)]], constant PrefillParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y, pos = row % p.length, sample = row / p.length;
    uint kv = head / (p.heads / p.kv_heads);
    ulong qi = (ulong(row) * p.heads + head) * p.dim;
    ulong ki = (ulong(row) * p.kv_heads + kv) * p.dim;
    ulong dst = ((ulong(sample) * p.heads + head) * p.length + pos) * p.dim;
    float qs = 0, ks = 0;
    for (uint i = lane; i < p.dim; i += 32) {
        float qv = float(q[qi+i]), kv_val = float(k[ki+i]);
        qs += qv*qv; ks += kv_val*kv_val;
    }
    qs = rsqrt(simd_sum(qs) / float(p.dim) + p.epsilon);
    ks = rsqrt(simd_sum(ks) / float(p.dim) + p.epsilon);
    uint midpoint = p.dim / 2;
    float log_base = log2(p.rope_base);
    for (uint i = lane; i < p.dim; i += 32) {
        uint pair = i % midpoint;
        float inv_freq = exp2(-2.0f * float(pair) / float(p.dim) * log_base);
        float angle = float(pos) * inv_freq;
        float co, si;
        si = sincos(angle, co);
        float qa = float(q[qi+pair])*qs, qb = float(q[qi+midpoint+pair])*qs;
        float ka = float(k[ki+pair])*ks, kb = float(k[ki+midpoint+pair])*ks;
        qr[dst+i] = half(i < midpoint ? qa*co - qb*si : qb*co + qa*si);
        kr[dst+i] = half(i < midpoint ? ka*co - kb*si : kb*co + ka*si);
        vr[dst+i] = v[ki+i];
    }
}

kernel void fbt_prefill_prepare_half_fused(
    device const half *qkvg [[buffer(0)]],
    device half *qr [[buffer(1)]],
    device half *kr [[buffer(2)]],
    device half *vr [[buffer(3)]],
    device half *gates [[buffer(4)]],
    constant PrefillParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y, pos = row % p.length, sample = row / p.length;
    uint kv = head / (p.heads / p.kv_heads);
    ulong row_base = ulong(row) * 3088;
    ulong qi = row_base + head * p.dim;
    ulong ki = row_base + 1536 + kv * p.dim;
    ulong dst = ((ulong(sample) * p.heads + head) * p.length + pos) * p.dim;
    if (lane == 0) {
        gates[ulong(row) * p.heads + head] = qkvg[row_base + 3072 + head];
    }
    float qs = 0;
    for (uint i = lane; i < p.dim; i += 32) {
        float qv = float(qkvg[qi+i]);
        qs += qv*qv;
    }
    qs = rsqrt(simd_sum(qs) / float(p.dim) + p.epsilon);
    uint midpoint = p.dim / 2;
    float log_base = log2(p.rope_base);
    for (uint i = lane; i < p.dim; i += 32) {
        uint pair = i % midpoint;
        float inv_freq = exp2(-2.0f * float(pair) / float(p.dim) * log_base);
        float angle = float(pos) * inv_freq;
        float co, si;
        si = sincos(angle, co);
        float qa = float(qkvg[qi+pair])*qs, qb = float(qkvg[qi+midpoint+pair])*qs;
        qr[dst+i] = half(i < midpoint ? qa*co - qb*si : qb*co + qa*si);
    }
    if (head % (p.heads / p.kv_heads) == 0) {
        float ks = 0;
        for (uint i = lane; i < p.dim; i += 32) {
            float kv_val = float(qkvg[ki+i]);
            ks += kv_val*kv_val;
        }
        ks = rsqrt(simd_sum(ks) / float(p.dim) + p.epsilon);
        ulong dst_kv = ((ulong(sample) * p.kv_heads + kv) * p.length + pos) * p.dim;
        for (uint i = lane; i < p.dim; i += 32) {
            uint pair = i % midpoint;
            float inv_freq = exp2(-2.0f * float(pair) / float(p.dim) * log_base);
            float angle = float(pos) * inv_freq;
            float co, si;
            si = sincos(angle, co);
            float ka = float(qkvg[ki+pair])*ks, kb = float(qkvg[ki+midpoint+pair])*ks;
            kr[dst_kv+i] = half(i < midpoint ? ka*co - kb*si : kb*co + ka*si);
            vr[dst_kv+i] = qkvg[row_base + 2304 + kv * p.dim + i];
        }
    }
}

kernel void fbt_prefill_softmax(device half *scores [[buffer(0)]],
    constant PrefillParams &p [[buffer(1)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint pos = p.start + group.y;
    uint lower = p.window == 0 || pos < p.window ? 0 : pos + 1 - p.window;
    lower -= p.key_start;
    pos -= p.key_start;
    ulong base = (ulong(group.x) * p.block + group.y) * p.key_rows;
    float maximum = -INFINITY;
    for (uint i = lower + lane; i <= pos; i += 32) maximum = max(maximum, float(scores[base+i]));
    maximum = simd_max(maximum);
    float sum = 0;
    for (uint i = lane; i < p.key_rows; i += 32) {
        float value = i >= lower && i <= pos ? exp(float(scores[base+i]) - maximum) : 0.0f;
        scores[base+i] = half(value);
        sum += value;
    }
    sum = simd_sum(sum);
    for (uint i = lane; i < p.key_rows; i += 32) scores[base+i] = half(float(scores[base+i]) / sum);
}

kernel void fbt_prefill_unpack(device const half *packed [[buffer(0)]],
    device const float *gate_logits [[buffer(1)]], device float *out [[buffer(2)]],
    constant PrefillParams &p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, row = group.y;
    ulong src = ((ulong(row / p.length) * p.heads + head) * p.length + row % p.length) * p.dim;
    ulong dst = (ulong(row) * p.heads + head) * p.dim;
    float g = gate_logits[ulong(row) * p.heads + head];
    float e = exp(-abs(g)), sigmoid = g >= 0 ? 1.0f/(1.0f+e) : e/(1.0f+e);
    for (uint i = lane; i < p.dim; i += 32) out[dst+i] = float(packed[src+i]) * sigmoid;
}

struct ShiftParams { uint width, length; };
kernel void fbt_prefill_shift(device const float *history [[buffer(0)]],
    device float *previous [[buffer(1)]], device uint *mask [[buffer(2)]],
    constant ShiftParams &p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    bool fused = row % p.length != 0;
    if (lane == 0) mask[row] = fused;
    for (uint i = lane; i < p.width; i += 32)
        previous[ulong(row)*p.width+i] = fused ? history[ulong(row-1)*p.width+i] : 0;
}

kernel void fbt_prefill_shift_half(device const half *history [[buffer(0)]],
    device half *previous [[buffer(1)]], device uint *mask [[buffer(2)]],
    constant ShiftParams &p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    bool fused = row % p.length != 0;
    if (lane == 0) mask[row] = fused;
    ulong dst_base = ulong(row) * p.width;
    ulong src_base = fused ? ulong(row - 1) * p.width : 0;
    device const half4 *src4 = reinterpret_cast<device const half4*>(history + src_base);
    device half4 *dst4 = reinterpret_cast<device half4*>(previous + dst_base);
    uint width4 = p.width / 4;
    for (uint i = lane; i < width4; i += 32) {
        dst4[i] = fused ? src4[i] : half4(0.0h);
    }
}

kernel void fbt_prefill_glu(device const float *gate [[buffer(0)]],
    device float *up [[buffer(1)]], constant uint &width [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    for (uint i = lane; i < width; i += 32) {
        ulong ix = ulong(row)*width+i;
        float g = gate[ix], e = exp(-abs(g));
        up[ix] *= g * (g >= 0 ? 1.0f/(1.0f+e) : e/(1.0f+e));
    }
}

// Custom fused Flash Attention kernel for dim==128 (BO tune model heads).
// Implements causal + sliding-window attention with online Softmax.
// FP16 tiles (q, k, v, p) + AMX mixed-precision matrix multiply acceleration.
// Dispatch: grid = (ceil(length/16), batch*heads), threads = 64 (2 SIMD groups).
kernel void fbt_prefill_flash_attention(
    device const half *qr [[buffer(0)]],
    device const half *kr [[buffer(1)]],
    device const half *vr [[buffer(2)]],
    device const float *gate_logits [[buffer(3)]],
    device float *branch [[buffer(4)]],
    constant PrefillParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    // group.x: query block index (0 .. ceil(p.length / 16) - 1)
    // group.y: head index across batch (0 .. batch * heads - 1)
    // 64 threads per threadgroup (2 SIMD groups)
    threadgroup half q_tile[16 * 128];
    threadgroup half k_tile[16 * 128];
    threadgroup half v_tile[16 * 128];
    threadgroup float s_tile[16][16];
    threadgroup half p_tile[16][16];
    threadgroup float m_stat[16];
    threadgroup float l_stat[16];
    threadgroup float alpha_stat[16];

    uint head_idx = group.y;
    uint q_block = group.x;
    uint q_start = q_block * 16;
    if (q_start >= p.length) return;

    ulong head_offset = ulong(head_idx) * ulong(p.length) * ulong(p.dim);
    device const half *head_qr = qr + head_offset;
    device const half *head_kr = kr + head_offset;
    device const half *head_vr = vr + head_offset;

    // Load Q tile into FP16 threadgroup memory: 16 rows * 128 cols = 512 half4s.
    device const half4 *q_src4 = reinterpret_cast<device const half4*>(head_qr + ulong(q_start) * 128);
    threadgroup half4 *q_dst4 = reinterpret_cast<threadgroup half4*>(q_tile);
    for (uint k = 0; k < 8; ++k) {
        uint idx = tid * 8 + k;
        uint r = idx / 32;
        if (q_start + r < p.length) {
            q_dst4[idx] = q_src4[idx];
        } else {
            q_dst4[idx] = half4(0.0h);
        }
    }

    // Initialize stats
    if (tid < 16) {
        m_stat[tid] = -INFINITY;
        l_stat[tid] = 0.0f;
        alpha_stat[tid] = 1.0f;
    }

    // Initialize 8 float4 output registers per thread
    float4 o_reg[8];
    for (uint k = 0; k < 8; ++k) {
        o_reg[k] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = 1.0f / sqrt(float(p.dim));
    uint max_k_block = min((q_start + 15) / 16 + 1, (p.length + 15) / 16);

    for (uint k_block = 0; k_block < max_k_block; ++k_block) {
        uint k_start = k_block * 16;
        if (p.window > 0 && q_start >= p.window && k_start + 15 < q_start + 1 - p.window) {
            continue; // entirely before sliding window
        }

        // Load K and V tiles (2048 half elements per tile = 512 half4s)
        device const half4 *k_src4 = reinterpret_cast<device const half4*>(head_kr + ulong(k_start) * 128);
        device const half4 *v_src4 = reinterpret_cast<device const half4*>(head_vr + ulong(k_start) * 128);
        threadgroup half4 *k_dst4 = reinterpret_cast<threadgroup half4*>(k_tile);
        threadgroup half4 *v_dst4 = reinterpret_cast<threadgroup half4*>(v_tile);

        for (uint k = 0; k < 8; ++k) {
            uint idx = tid * 8 + k;
            uint r = idx / 32;
            if (k_start + r < p.length) {
                k_dst4[idx] = k_src4[idx];
                v_dst4[idx] = v_src4[idx];
            } else {
                k_dst4[idx] = half4(0.0h);
                v_dst4[idx] = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q * K^T via Apple AMX simdgroup_multiply_accumulate.
        // Mixed-precision: Q, K matrices are simdgroup_half8x8, S accumulator is simdgroup_float8x8.
        uint simd_id = tid / 32;
        simdgroup_float8x8 s_amx_left  = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 s_amx_right = simdgroup_float8x8(0.0f);

        for (uint d = 0; d < 16; ++d) {
            simdgroup_half8x8 q_blk, k_blk_left, k_blk_right;
            simdgroup_load(q_blk,        q_tile + simd_id * 8 * 128 + d * 8, 128, ulong2(0, 0));
            simdgroup_load(k_blk_left,  k_tile + 0 * 128 + d * 8, 128, ulong2(0, 0), true);
            simdgroup_load(k_blk_right, k_tile + 8 * 128 + d * 8, 128, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_amx_left,  q_blk, k_blk_left,  s_amx_left);
            simdgroup_multiply_accumulate(s_amx_right, q_blk, k_blk_right, s_amx_right);
        }

        simdgroup_store(s_amx_left,  &s_tile[simd_id * 8][0], 16, ulong2(0, 0));
        simdgroup_store(s_amx_right, &s_tile[simd_id * 8][8], 16, ulong2(0, 0));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply causal / sliding-window mask and scale
        for (uint k = 0; k < 4; ++k) {
            uint s_idx = tid * 4 + k;
            uint r = s_idx / 16;
            uint c = s_idx % 16;
            uint q_pos = q_start + r;
            uint k_pos = k_start + c;
            uint lower = (p.window == 0 || q_pos < p.window) ? 0 : q_pos + 1 - p.window;
            if (q_pos < p.length && k_pos < p.length && k_pos <= q_pos && k_pos >= lower) {
                s_tile[r][c] *= scale;
            } else {
                s_tile[r][c] = -INFINITY;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute online softmax stats for 16 rows (threads 0..15 handle rows 0..15)
        if (tid < 16) {
            uint r = tid;
            float row_max = -INFINITY;
            for (uint c = 0; c < 16; ++c) {
                row_max = max(row_max, s_tile[r][c]);
            }
            if (row_max > -INFINITY) {
                float prev_max = m_stat[r];
                float new_max = max(prev_max, row_max);
                float alpha = (prev_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);
                float row_sum = 0.0f;
                for (uint c = 0; c < 16; ++c) {
                    float p_val = (s_tile[r][c] == -INFINITY) ? 0.0f : exp(s_tile[r][c] - new_max);
                    p_tile[r][c] = half(p_val);
                    row_sum += p_val;
                }
                m_stat[r] = new_max;
                l_stat[r] = l_stat[r] * alpha + row_sum;
                alpha_stat[r] = alpha;
            } else {
                for (uint c = 0; c < 16; ++c) {
                    p_tile[r][c] = half(0.0h);
                }
                alpha_stat[r] = 1.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale running O by alpha (per-row) and accumulate P * V via AMX.
        for (uint k = 0; k < 8; ++k) {
            uint idx = tid * 8 + k;
            uint r = idx / 32;
            o_reg[k] = o_reg[k] * alpha_stat[r];
        }

        threadgroup float o_scratch[16 * 128];
        {
            threadgroup float4 *o_dst4 = reinterpret_cast<threadgroup float4*>(o_scratch);
            for (uint k = 0; k < 8; ++k) {
                uint idx = tid * 8 + k;
                o_dst4[idx] = o_reg[k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint v_col = 0; v_col < 16; ++v_col) {
            simdgroup_float8x8 o_blk;
            simdgroup_load(o_blk, o_scratch + simd_id * 8 * 128 + v_col * 8, 128, ulong2(0, 0));

            simdgroup_half8x8 p_left, p_right, v_blk_top, v_blk_bot;
            simdgroup_load(p_left,    &p_tile[simd_id * 8][0], 16, ulong2(0, 0));
            simdgroup_load(p_right,   &p_tile[simd_id * 8][8], 16, ulong2(0, 0));
            simdgroup_load(v_blk_top, v_tile + 0 * 128 + v_col * 8, 128, ulong2(0, 0));
            simdgroup_load(v_blk_bot, v_tile + 8 * 128 + v_col * 8, 128, ulong2(0, 0));

            simdgroup_multiply_accumulate(o_blk, p_left,  v_blk_top, o_blk);
            simdgroup_multiply_accumulate(o_blk, p_right, v_blk_bot, o_blk);
            simdgroup_store(o_blk, o_scratch + simd_id * 8 * 128 + v_col * 8, 128, ulong2(0, 0));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            threadgroup const float4 *o_src4 = reinterpret_cast<threadgroup const float4*>(o_scratch);
            for (uint k = 0; k < 8; ++k) {
                uint idx = tid * 8 + k;
                o_reg[k] = o_src4[idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization, gating and output write to branch
    uint sample = head_idx / p.heads;
    uint head = head_idx % p.heads;

    for (uint k = 0; k < 8; ++k) {
        uint idx = tid * 8 + k;
        uint r = idx / 32;
        uint col4 = idx % 32;
        uint pos = q_start + r;
        if (pos < p.length) {
            float denom = l_stat[r];
            float4 norm_v = denom > 0.0f ? (o_reg[k] / denom) : float4(0.0f);
            ulong row = ulong(sample) * ulong(p.length) + ulong(pos);
            float g = gate_logits[row * ulong(p.heads) + ulong(head)];
            float e = exp(-abs(g));
            float sigmoid = g >= 0.0f ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
            norm_v *= sigmoid;

            ulong dst = (row * ulong(p.heads) + ulong(head)) * ulong(p.dim) + ulong(col4) * 4;
            device float4 *out4 = reinterpret_cast<device float4*>(branch + dst);
            *out4 = norm_v;
        }
    }
}

kernel void fbt_prefill_flash_attention_half(
    device const half *qr [[buffer(0)]],
    device const half *kr [[buffer(1)]],
    device const half *vr [[buffer(2)]],
    device const half *gate_logits [[buffer(3)]],
    device half *branch [[buffer(4)]],
    constant PrefillParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half q_tile[16 * 128];
    threadgroup half k_tile[16 * 128];
    threadgroup half v_tile[16 * 128];
    threadgroup float s_tile[16][16];
    threadgroup half p_tile[16][16];
    threadgroup float m_stat[16];
    threadgroup float l_stat[16];
    threadgroup float alpha_stat[16];

    uint head_idx = group.y;
    uint q_block = group.x;
    uint q_start = q_block * 16;
    if (q_start >= p.length) return;

    ulong head_offset = ulong(head_idx) * ulong(p.length) * ulong(p.dim);
    device const half *head_qr = qr + head_offset;
    device const half *head_kr = kr + head_offset;
    device const half *head_vr = vr + head_offset;

    device const half4 *q_src4 = reinterpret_cast<device const half4*>(head_qr + ulong(q_start) * 128);
    threadgroup half4 *q_dst4 = reinterpret_cast<threadgroup half4*>(q_tile);
    for (uint k = 0; k < 8; ++k) {
        uint idx = tid * 8 + k;
        uint r = idx / 32;
        if (q_start + r < p.length) {
            q_dst4[idx] = q_src4[idx];
        } else {
            q_dst4[idx] = half4(0.0h);
        }
    }

    if (tid < 16) {
        m_stat[tid] = -INFINITY;
        l_stat[tid] = 0.0f;
        alpha_stat[tid] = 1.0f;
    }

    float4 o_reg[8];
    for (uint k = 0; k < 8; ++k) {
        o_reg[k] = float4(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = 1.0f / sqrt(float(p.dim));
    uint max_k_block = min((q_start + 15) / 16 + 1, (p.length + 15) / 16);

    for (uint k_block = 0; k_block < max_k_block; ++k_block) {
        uint k_start = k_block * 16;
        if (p.window > 0 && q_start >= p.window && k_start + 15 < q_start + 1 - p.window) {
            continue;
        }

        device const half4 *k_src4 = reinterpret_cast<device const half4*>(head_kr + ulong(k_start) * 128);
        device const half4 *v_src4 = reinterpret_cast<device const half4*>(head_vr + ulong(k_start) * 128);
        threadgroup half4 *k_dst4 = reinterpret_cast<threadgroup half4*>(k_tile);
        threadgroup half4 *v_dst4 = reinterpret_cast<threadgroup half4*>(v_tile);

        for (uint k = 0; k < 8; ++k) {
            uint idx = tid * 8 + k;
            uint r = idx / 32;
            if (k_start + r < p.length) {
                k_dst4[idx] = k_src4[idx];
                v_dst4[idx] = v_src4[idx];
            } else {
                k_dst4[idx] = half4(0.0h);
                v_dst4[idx] = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint simd_id = tid / 32;
        simdgroup_float8x8 s_amx_left  = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 s_amx_right = simdgroup_float8x8(0.0f);

        for (uint d = 0; d < 16; ++d) {
            simdgroup_half8x8 q_blk, k_blk_left, k_blk_right;
            simdgroup_load(q_blk,        q_tile + simd_id * 8 * 128 + d * 8, 128, ulong2(0, 0));
            simdgroup_load(k_blk_left,  k_tile + 0 * 128 + d * 8, 128, ulong2(0, 0), true);
            simdgroup_load(k_blk_right, k_tile + 8 * 128 + d * 8, 128, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_amx_left,  q_blk, k_blk_left,  s_amx_left);
            simdgroup_multiply_accumulate(s_amx_right, q_blk, k_blk_right, s_amx_right);
        }

        simdgroup_store(s_amx_left,  &s_tile[simd_id * 8][0], 16, ulong2(0, 0));
        simdgroup_store(s_amx_right, &s_tile[simd_id * 8][8], 16, ulong2(0, 0));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 4; ++k) {
            uint s_idx = tid * 4 + k;
            uint r = s_idx / 16;
            uint c = s_idx % 16;
            uint q_pos = q_start + r;
            uint k_pos = k_start + c;
            uint lower = (p.window == 0 || q_pos < p.window) ? 0 : q_pos + 1 - p.window;
            if (q_pos < p.length && k_pos < p.length && k_pos <= q_pos && k_pos >= lower) {
                s_tile[r][c] *= scale;
            } else {
                s_tile[r][c] = -INFINITY;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16) {
            uint r = tid;
            float row_max = -INFINITY;
            for (uint c = 0; c < 16; ++c) {
                row_max = max(row_max, s_tile[r][c]);
            }
            if (row_max > -INFINITY) {
                float prev_max = m_stat[r];
                float new_max = max(prev_max, row_max);
                float alpha = (prev_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);
                float row_sum = 0.0f;
                for (uint c = 0; c < 16; ++c) {
                    float p_val = (s_tile[r][c] == -INFINITY) ? 0.0f : exp(s_tile[r][c] - new_max);
                    p_tile[r][c] = half(p_val);
                    row_sum += p_val;
                }
                m_stat[r] = new_max;
                l_stat[r] = l_stat[r] * alpha + row_sum;
                alpha_stat[r] = alpha;
            } else {
                for (uint c = 0; c < 16; ++c) {
                    p_tile[r][c] = half(0.0h);
                }
                alpha_stat[r] = 1.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 8; ++k) {
            uint idx = tid * 8 + k;
            uint r = idx / 32;
            o_reg[k] = o_reg[k] * alpha_stat[r];
        }

        threadgroup float o_scratch[16 * 128];
        {
            threadgroup float4 *o_dst4 = reinterpret_cast<threadgroup float4*>(o_scratch);
            for (uint k = 0; k < 8; ++k) {
                uint idx = tid * 8 + k;
                o_dst4[idx] = o_reg[k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint v_col = 0; v_col < 16; ++v_col) {
            simdgroup_float8x8 o_blk;
            simdgroup_load(o_blk, o_scratch + simd_id * 8 * 128 + v_col * 8, 128, ulong2(0, 0));

            simdgroup_half8x8 p_left, p_right, v_blk_top, v_blk_bot;
            simdgroup_load(p_left,    &p_tile[simd_id * 8][0], 16, ulong2(0, 0));
            simdgroup_load(p_right,   &p_tile[simd_id * 8][8], 16, ulong2(0, 0));
            simdgroup_load(v_blk_top, v_tile + 0 * 128 + v_col * 8, 128, ulong2(0, 0));
            simdgroup_load(v_blk_bot, v_tile + 8 * 128 + v_col * 8, 128, ulong2(0, 0));

            simdgroup_multiply_accumulate(o_blk, p_left,  v_blk_top, o_blk);
            simdgroup_multiply_accumulate(o_blk, p_right, v_blk_bot, o_blk);
            simdgroup_store(o_blk, o_scratch + simd_id * 8 * 128 + v_col * 8, 128, ulong2(0, 0));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            threadgroup const float4 *o_src4 = reinterpret_cast<threadgroup const float4*>(o_scratch);
            for (uint k = 0; k < 8; ++k) {
                uint idx = tid * 8 + k;
                o_reg[k] = o_src4[idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sample = head_idx / p.heads;
    uint head = head_idx % p.heads;

    for (uint k = 0; k < 8; ++k) {
        uint idx = tid * 8 + k;
        uint r = idx / 32;
        uint col4 = idx % 32;
        uint pos = q_start + r;
        if (pos < p.length) {
            float denom = l_stat[r];
            float4 norm_v = denom > 0.0f ? (o_reg[k] / denom) : float4(0.0f);
            ulong row = ulong(sample) * ulong(p.length) + ulong(pos);
            float g = float(gate_logits[row * ulong(p.heads) + ulong(head)]);
            float e = exp(-abs(g));
            float sigmoid = g >= 0.0f ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
            norm_v *= sigmoid;

            ulong dst = (row * ulong(p.heads) + ulong(head)) * ulong(p.dim) + ulong(col4) * 4;
            device half4 *out4 = reinterpret_cast<device half4*>(branch + dst);
            *out4 = half4(norm_v);
        }
    }
}

kernel void fbt_prefill_flash_attention_half_96(
    device const half *qr [[buffer(0)]],
    device const half *kr [[buffer(1)]],
    device const half *vr [[buffer(2)]],
    device const half *gate_logits [[buffer(3)]],
    device half *branch [[buffer(4)]],
    constant PrefillParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {

    threadgroup half q_tile[32 * 96];
    threadgroup half k_tile[32 * 96];
    threadgroup half v_tile[32 * 96];
    threadgroup half p_tile[4][8][33];
    threadgroup float s_tile[4][8][33];
    threadgroup float A_shared[4][8][9];
    threadgroup float m_stat[32];
    threadgroup float l_stat[32];

    uint head_idx = group.y;
    uint q_block = group.x;
    uint q_start = q_block * 32;
    if (q_start >= p.length) return;

    uint sample = head_idx / p.heads;
    uint head = head_idx % p.heads;
    uint kv_head = head / (p.heads / p.kv_heads);

    ulong q_offset = ulong(head_idx) * ulong(p.length) * 96;
    ulong kv_offset = (ulong(sample) * ulong(p.kv_heads) + ulong(kv_head)) * ulong(p.length) * 96;
    device const half *head_qr = qr + q_offset;
    device const half *head_kr = kr + kv_offset;
    device const half *head_vr = vr + kv_offset;

    device const half4 *q_src4 = reinterpret_cast<device const half4*>(head_qr + ulong(q_start) * 96);
    threadgroup half4 *q_dst4 = reinterpret_cast<threadgroup half4*>(q_tile);
    if (q_start + 32 <= p.length) {
        #pragma unroll
        for (uint idx = tid; idx < 768; idx += 128) {
            q_dst4[idx] = q_src4[idx];
        }
    } else {
        for (uint idx = tid; idx < 768; idx += 128) {
            uint r = idx / 24;
            if (q_start + r < p.length) {
                q_dst4[idx] = q_src4[idx];
            } else {
                q_dst4[idx] = half4(0.0h);
            }
        }
    }

    if (tid < 32) {
        m_stat[tid] = -INFINITY;
        l_stat[tid] = 0.0f;
    }

    simdgroup_float8x8 o_blks[12];
    #pragma unroll
    for (uint v = 0; v < 12; ++v) {
        o_blks[v] = simdgroup_float8x8(0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = 1.0f / sqrt(96.0f);
    uint min_k_block = 0;
    if (p.window > 0 && q_start >= p.window) {
        min_k_block = (q_start + 1 - p.window) / 32;
    }
    uint max_k_block = min((q_start + 31) / 32 + 1, (p.length + 31) / 32);
    uint simd_id = tid / 32;
    uint lane = tid % 32;

    for (uint k_block = min_k_block; k_block < max_k_block; ++k_block) {
        uint k_start = k_block * 32;

        device const half4 *k_src4 = reinterpret_cast<device const half4*>(head_kr + ulong(k_start) * 96);
        device const half4 *v_src4 = reinterpret_cast<device const half4*>(head_vr + ulong(k_start) * 96);
        threadgroup half4 *k_dst4 = reinterpret_cast<threadgroup half4*>(k_tile);
        threadgroup half4 *v_dst4 = reinterpret_cast<threadgroup half4*>(v_tile);

        if (k_start + 32 <= p.length) {
            #pragma unroll
            for (uint idx = tid; idx < 768; idx += 128) {
                k_dst4[idx] = k_src4[idx];
                v_dst4[idx] = v_src4[idx];
            }
        } else {
            for (uint idx = tid; idx < 768; idx += 128) {
                uint r = idx / 24;
                if (k_start + r < p.length) {
                    k_dst4[idx] = k_src4[idx];
                    v_dst4[idx] = v_src4[idx];
                } else {
                    k_dst4[idx] = half4(0.0h);
                    v_dst4[idx] = half4(0.0h);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s_amx[4];
        #pragma unroll
        for (uint b = 0; b < 4; ++b) s_amx[b] = simdgroup_float8x8(0.0f);

        #pragma unroll
        for (uint d = 0; d < 12; ++d) {
            simdgroup_half8x8 q_blk;
            simdgroup_load(q_blk, q_tile + simd_id * 8 * 96 + d * 8, 96, ulong2(0, 0));
            #pragma unroll
            for (uint b = 0; b < 4; ++b) {
                simdgroup_half8x8 k_blk;
                simdgroup_load(k_blk, k_tile + b * 8 * 96 + d * 8, 96, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_amx[b], q_blk, k_blk, s_amx[b]);
            }
        }

        #pragma unroll
        for (uint b = 0; b < 4; ++b) {
            simdgroup_store(s_amx[b], &s_tile[simd_id][0][b * 8], 33, ulong2(0, 0));
        }

        // Intra-SIMD Softmax: exactly 8 rows per SIMD group, no cross-SIMD barriers needed!
        #pragma unroll
        for (uint r = 0; r < 8; ++r) {
            uint row_idx = simd_id * 8 + r;
            uint q_pos = q_start + row_idx;
            uint k_pos = k_start + lane;
            uint lower = (p.window == 0 || q_pos < p.window) ? 0 : q_pos + 1 - p.window;
            float val = s_tile[simd_id][r][lane];
            if (q_pos < p.length && k_pos < p.length && k_pos <= q_pos && k_pos >= lower) {
                val *= scale;
            } else {
                val = -INFINITY;
            }
            float row_max = simd_max(val);
            if (row_max > -INFINITY) {
                float prev_max = m_stat[row_idx];
                float new_max = max(prev_max, row_max);
                float alpha = (prev_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);
                if (lane == 0) m_stat[row_idx] = new_max;
                float p_val = (val == -INFINITY) ? 0.0f : exp(val - new_max);
                p_tile[simd_id][r][lane] = half(p_val);
                float row_sum = simd_sum(p_val);
                if (lane == 0) l_stat[row_idx] = l_stat[row_idx] * alpha + row_sum;
                if (lane < 8) A_shared[simd_id][r][lane] = (lane == r) ? alpha : 0.0f;
            } else {
                p_tile[simd_id][r][lane] = half(0.0h);
                if (lane < 8) A_shared[simd_id][r][lane] = (lane == r) ? 1.0f : 0.0f;
            }
            if (lane == 0) {
                p_tile[simd_id][r][32] = half(0.0h);
                A_shared[simd_id][r][8] = 0.0f;
            }
        }

        simdgroup_half8x8 p_blks[4];
        #pragma unroll
        for (uint b = 0; b < 4; ++b) {
            simdgroup_load(p_blks[b], &p_tile[simd_id][0][b * 8], 33, ulong2(0, 0));
        }

        #pragma unroll
        for (uint v_col = 0; v_col < 12; ++v_col) {
            simdgroup_store(o_blks[v_col], &s_tile[simd_id][0][0], 33, ulong2(0, 0));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint item = 0; item < 2; ++item) {
                uint ix = lane + item * 32;
                uint r = ix / 8;
                uint c = ix - r * 8;
                s_tile[simd_id][r][c] *= A_shared[simd_id][r][r];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o_blks[v_col], &s_tile[simd_id][0][0], 33, ulong2(0, 0));
            #pragma unroll
            for (uint b = 0; b < 4; ++b) {
                simdgroup_half8x8 v_blk;
                simdgroup_load(v_blk, v_tile + b * 8 * 96 + v_col * 8, 96, ulong2(0, 0));
                simdgroup_multiply_accumulate(o_blks[v_col], p_blks[b], v_blk, o_blks[v_col]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float2 inverse_sum, sigmoid;
    #pragma unroll
    for (uint i = 0; i < 2; ++i) {
        uint r = simd_id * 8 + lane / 8 + i * 4;
        float denom = l_stat[r];
        inverse_sum[i] = denom > 0.0f ? 1.0f / denom : 0.0f;
        sigmoid[i] = 0.0f;
        uint pos = q_start + r;
        if (pos < p.length) {
            ulong row = ulong(sample) * ulong(p.length) + ulong(pos);
            float g = float(gate_logits[row * ulong(p.heads) + ulong(head)]);
            float e = exp(-abs(g));
            sigmoid[i] = g >= 0.0f ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
        }
    }

    // Reuse each SIMD group's score tile in 32-column slices. A full float
    // output tile cannot fit in the half-precision q_tile allocation.
    #pragma unroll
    for (uint slice = 0; slice < 3; ++slice) {
        #pragma unroll
        for (uint b = 0; b < 4; ++b) {
            simdgroup_store(o_blks[slice * 4 + b], &s_tile[simd_id][0][b * 8], 33, ulong2(0, 0));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            uint r = lane / 8 + i * 4;
            uint col = (lane % 8) * 4;
            uint pos = q_start + simd_id * 8 + r;
            if (pos < p.length) {
                float4 value = float4(s_tile[simd_id][r][col], s_tile[simd_id][r][col + 1],
                                     s_tile[simd_id][r][col + 2], s_tile[simd_id][r][col + 3]);
                float4 normalized = (value * inverse_sum[i]) * sigmoid[i];
                ulong row = ulong(sample) * ulong(p.length) + ulong(pos);
                ulong dst = (row * ulong(p.heads) + ulong(head)) * 96 + slice * 32 + col;
                device half4 *out4 = reinterpret_cast<device half4*>(branch + dst);
                *out4 = half4(normalized);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fbt_cross_entropy_blocked_half(
    device const half *logits [[buffer(0)]],
    device const uint *targets [[buffer(1)]],
    device float *loss [[buffer(2)]],
    constant GraphParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_max[8];
    threadgroup float s_sum[8];
    constexpr uint block = 8192;
    float local_max = -INFINITY;
    for (uint v = tid; v < p.vocab; v += 256) {
        uint b = v / block;
        uint within = v - b * block;
        uint width = min(block, p.vocab - b * block);
        ulong index = ulong(b) * ulong(p.rows) * ulong(block)
            + ulong(row) * ulong(width) + ulong(within);
        local_max = max(local_max, float(logits[index]));
    }
    local_max = simd_max(local_max);
    if (lane == 0) s_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = -INFINITY;
    if (tid < 8) row_max = s_max[tid];
    row_max = simd_max(row_max);
    if (tid == 0) s_max[0] = row_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = s_max[0];

    float local_sum = 0.0f;
    for (uint v = tid; v < p.vocab; v += 256) {
        uint b = v / block;
        uint within = v - b * block;
        uint width = min(block, p.vocab - b * block);
        ulong index = ulong(b) * ulong(p.rows) * ulong(block)
            + ulong(row) * ulong(width) + ulong(within);
        local_sum += exp(float(logits[index]) - row_max);
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) s_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total_sum = 0.0f;
        for (uint i = 0; i < 8; ++i) total_sum += s_sum[i];
        uint target = targets[p.start + row];
        uint b = target / block;
        uint within = target - b * block;
        uint width = min(block, p.vocab - b * block);
        ulong target_index = ulong(b) * ulong(p.rows) * ulong(block)
            + ulong(row) * ulong(width) + ulong(within);
        loss[p.start + row] = row_max + log(total_sum) - float(logits[target_index]);
    }
}

struct GemmParams { uint m, n, k; };

kernel void fbt_gemm_half(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half a_tile[64][36];
    threadgroup half b_tile[32][68];
    threadgroup float c_scratch[64][68];

    uint group_m = group.y;
    uint group_n = group.x;
    uint simd_id = tid / 32;

    uint sim_row_base = (simd_id / 2) * 32;
    uint sim_col_base = (simd_id % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    for (uint k_start = 0; k_start < p.k; k_start += 32) {
        // Coalesced load of 64x32 tile from A (512 half4s, 4 per thread)
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint idx4 = tid + step * 128;
            uint r = idx4 / 8;
            uint c4 = idx4 % 8;
            uint global_r = group_m * 64 + r;
            uint global_c = k_start + c4 * 4;
            half4 val = half4(0.0h);
            if (global_r < p.m && global_c + 3 < p.k) {
                val = *reinterpret_cast<device const half4*>(A + ulong(global_r) * p.k + global_c);
            } else if (global_r < p.m) {
                for (uint e = 0; e < 4 && global_c + e < p.k; ++e) {
                    val[e] = A[ulong(global_r) * p.k + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&a_tile[r][c4 * 4]) = val;
        }

        // Coalesced load of 32x64 tile from B (512 half4s, 4 per thread)
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint idx4 = tid + step * 128;
            uint r = idx4 / 16;
            uint c4 = idx4 % 16;
            uint global_r = k_start + r;
            uint global_c = group_n * 64 + c4 * 4;
            half4 val = half4(0.0h);
            if (global_r < p.k && global_c + 3 < p.n) {
                val = *reinterpret_cast<device const half4*>(B + ulong(global_r) * p.n + global_c);
            } else if (global_r < p.k) {
                for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                    val[e] = B[ulong(global_r) * p.n + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&b_tile[r][c4 * 4]) = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // AMX compute loop with stride 36 and 68 (ZERO bank conflicts)
        #pragma unroll
        for (uint d = 0; d < 32; d += 8) {
            simdgroup_half8x8 a_frag[4];
            simdgroup_half8x8 b_frag[4];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i], &a_tile[sim_row_base + i * 8][d], 36, ulong2(0, 0));
            }
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j], &b_tile[d][sim_col_base + j * 8], 68, ulong2(0, 0));
            }
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                #pragma unroll
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // All SIMDs store their 32x32 results into threadgroup memory in parallel
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], &c_scratch[sim_row_base + i * 8][sim_col_base + j * 8], 68, ulong2(0, 0));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced write-out of 64x64 output tile (1024 half4s, 8 per thread)
    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint idx4 = tid + step * 128;
        uint r = idx4 / 16;
        uint c4 = idx4 % 16;
        uint global_r = group_m * 64 + r;
        uint global_c = group_n * 64 + c4 * 4;
        if (global_r < p.m) {
            float4 f = *reinterpret_cast<threadgroup const float4*>(&c_scratch[r][c4 * 4]);
            if (global_c + 3 < p.n) {
                *reinterpret_cast<device half4*>(C + ulong(global_r) * p.n + global_c) = half4(f);
            } else {
                for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                    C[ulong(global_r) * p.n + global_c + e] = half(f[e]);
                }
            }
        }
    }
}

kernel void fbt_gemm_glu_half(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half a_tile[64][36];
    threadgroup half b_gate_tile[32][68];
    threadgroup half b_up_tile[32][68];
    threadgroup float c_scratch[64][68];

    uint group_m = group.y;
    uint group_n = group.x;
    uint simd_id = tid / 32;

    uint sim_row_base = (simd_id / 2) * 32;
    uint sim_col_base = (simd_id % 2) * 32;

    simdgroup_float8x8 acc_g[4][4];
    simdgroup_float8x8 acc_u[4][4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            acc_g[i][j] = simdgroup_float8x8(0.0f);
            acc_u[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    ulong ldb = ulong(p.n) * 2;

    for (uint k_start = 0; k_start < p.k; k_start += 32) {
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint idx4 = tid + step * 128;
            uint r = idx4 / 8;
            uint c4 = idx4 % 8;
            uint global_r = group_m * 64 + r;
            uint global_c = k_start + c4 * 4;
            half4 val = half4(0.0h);
            if (global_r < p.m && global_c + 3 < p.k) {
                val = *reinterpret_cast<device const half4*>(A + ulong(global_r) * p.k + global_c);
            } else if (global_r < p.m) {
                for (uint e = 0; e < 4 && global_c + e < p.k; ++e) {
                    val[e] = A[ulong(global_r) * p.k + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&a_tile[r][c4 * 4]) = val;
        }

        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint idx4 = tid + step * 128;
            uint r = idx4 / 16;
            uint c4 = idx4 % 16;
            uint global_r = k_start + r;
            uint global_c = group_n * 64 + c4 * 4;
            half4 val_g = half4(0.0h);
            half4 val_u = half4(0.0h);
            if (global_r < p.k && global_c + 3 < p.n) {
                val_g = *reinterpret_cast<device const half4*>(B + ulong(global_r) * ldb + global_c);
                val_u = *reinterpret_cast<device const half4*>(B + ulong(global_r) * ldb + ulong(p.n) + global_c);
            } else if (global_r < p.k) {
                for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                    val_g[e] = B[ulong(global_r) * ldb + global_c + e];
                    val_u[e] = B[ulong(global_r) * ldb + ulong(p.n) + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&b_gate_tile[r][c4 * 4]) = val_g;
            *reinterpret_cast<threadgroup half4*>(&b_up_tile[r][c4 * 4]) = val_u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint d = 0; d < 32; d += 8) {
            simdgroup_half8x8 a_frag[4];
            simdgroup_half8x8 bg_frag[4];
            simdgroup_half8x8 bu_frag[4];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i], &a_tile[sim_row_base + i * 8][d], 36, ulong2(0, 0));
            }
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(bg_frag[j], &b_gate_tile[d][sim_col_base + j * 8], 68, ulong2(0, 0));
                simdgroup_load(bu_frag[j], &b_up_tile[d][sim_col_base + j * 8], 68, ulong2(0, 0));
            }
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                #pragma unroll
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc_g[i][j], a_frag[i], bg_frag[j], acc_g[i][j]);
                    simdgroup_multiply_accumulate(acc_u[i][j], a_frag[i], bu_frag[j], acc_u[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store Gate results into threadgroup memory
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc_g[i][j], &c_scratch[sim_row_base + i * 8][sim_col_base + j * 8], 68, ulong2(0, 0));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 g_reg[8];
    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint idx4 = tid + step * 128;
        uint r = idx4 / 16;
        uint c4 = idx4 % 16;
        g_reg[step] = *reinterpret_cast<threadgroup const float4*>(&c_scratch[r][c4 * 4]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store Up results into threadgroup memory
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc_u[i][j], &c_scratch[sim_row_base + i * 8][sim_col_base + j * 8], 68, ulong2(0, 0));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced SwiGLU fusion and write-out of 64x64 tile
    #pragma unroll
    for (uint step = 0; step < 8; ++step) {
        uint idx4 = tid + step * 128;
        uint r = idx4 / 16;
        uint c4 = idx4 % 16;
        uint global_r = group_m * 64 + r;
        uint global_c = group_n * 64 + c4 * 4;
        if (global_r < p.m) {
            // Match the FP16 GEMM output boundary before applying SwiGLU.
            float4 g = float4(half4(g_reg[step]));
            float4 u = float4(half4(*reinterpret_cast<threadgroup const float4*>(&c_scratch[r][c4 * 4])));
            float4 e = exp(-abs(g));
            float4 sig = float4(
                g.x >= 0.0f ? 1.0f / (1.0f + e.x) : e.x / (1.0f + e.x),
                g.y >= 0.0f ? 1.0f / (1.0f + e.y) : e.y / (1.0f + e.y),
                g.z >= 0.0f ? 1.0f / (1.0f + e.z) : e.z / (1.0f + e.z),
                g.w >= 0.0f ? 1.0f / (1.0f + e.w) : e.w / (1.0f + e.w)
            );
            half4 res = half4(g * sig * u);
            if (global_c + 3 < p.n) {
                *reinterpret_cast<device half4*>(C + ulong(global_r) * p.n + global_c) = res;
            } else {
                for (uint e_idx = 0; e_idx < 4 && global_c + e_idx < p.n; ++e_idx) {
                    C[ulong(global_r) * p.n + global_c + e_idx] = res[e_idx];
                }
            }
        }
    }
}

kernel void fbt_gemm_128_half(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half a_tile[128][36];
    threadgroup half b_tile[32][68];
    threadgroup float c_scratch[64][68];

    uint group_m = group.y;
    uint group_n = group.x;
    uint simd_id = tid / 32;

    uint sim_row_base = (simd_id / 2) * 32;
    uint sim_col_base = (simd_id % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    for (uint k_start = 0; k_start < p.k; k_start += 32) {
        #pragma unroll
        for (uint step = 0; step < 4; ++step) {
            uint idx4 = tid + step * 256;
            uint r = idx4 / 8;
            uint c4 = idx4 % 8;
            uint global_r = group_m * 128 + r;
            uint global_c = k_start + c4 * 4;
            half4 val = half4(0.0h);
            if (global_r < p.m && global_c + 3 < p.k) {
                val = *reinterpret_cast<device const half4*>(A + ulong(global_r) * p.k + global_c);
            } else if (global_r < p.m) {
                for (uint e = 0; e < 4 && global_c + e < p.k; ++e) {
                    val[e] = A[ulong(global_r) * p.k + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&a_tile[r][c4 * 4]) = val;
        }

        #pragma unroll
        for (uint step = 0; step < 2; ++step) {
            uint idx4 = tid + step * 256;
            uint r = idx4 / 16;
            uint c4 = idx4 % 16;
            uint global_r = k_start + r;
            uint global_c = group_n * 64 + c4 * 4;
            half4 val = half4(0.0h);
            if (global_r < p.k && global_c + 3 < p.n) {
                val = *reinterpret_cast<device const half4*>(B + ulong(global_r) * p.n + global_c);
            } else if (global_r < p.k) {
                for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                    val[e] = B[ulong(global_r) * p.n + global_c + e];
                }
            }
            *reinterpret_cast<threadgroup half4*>(&b_tile[r][c4 * 4]) = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint d = 0; d < 32; d += 8) {
            simdgroup_half8x8 a_frag[4];
            simdgroup_half8x8 b_frag[4];
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i], &a_tile[sim_row_base + i * 8][d], 36, ulong2(0, 0));
            }
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_load(b_frag[j], &b_tile[d][sim_col_base + j * 8], 68, ulong2(0, 0));
            }
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                #pragma unroll
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Top half (simd_id < 4): rows 0..63
    if (simd_id < 4) {
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_store(acc[i][j], &c_scratch[sim_row_base + i * 8][sim_col_base + j * 8], 68, ulong2(0, 0));
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint step = 0; step < 4; ++step) {
        uint idx4 = tid + step * 256;
        if (idx4 < 1024) {
            uint r = idx4 / 16;
            uint c4 = idx4 % 16;
            uint global_r = group_m * 128 + r;
            uint global_c = group_n * 64 + c4 * 4;
            if (global_r < p.m) {
                float4 f = *reinterpret_cast<threadgroup const float4*>(&c_scratch[r][c4 * 4]);
                if (global_c + 3 < p.n) {
                    *reinterpret_cast<device half4*>(C + ulong(global_r) * p.n + global_c) = half4(f);
                } else {
                    for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                        C[ulong(global_r) * p.n + global_c + e] = half(f[e]);
                    }
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bottom half (simd_id >= 4): rows 64..127
    if (simd_id >= 4) {
        uint sim_row_sub = (simd_id - 4) / 2 * 32;
        #pragma unroll
        for (uint i = 0; i < 4; ++i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_store(acc[i][j], &c_scratch[sim_row_sub + i * 8][sim_col_base + j * 8], 68, ulong2(0, 0));
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint step = 0; step < 4; ++step) {
        uint idx4 = tid + step * 256;
        if (idx4 < 1024) {
            uint r = idx4 / 16;
            uint c4 = idx4 % 16;
            uint global_r = group_m * 128 + 64 + r;
            uint global_c = group_n * 64 + c4 * 4;
            if (global_r < p.m) {
                float4 f = *reinterpret_cast<threadgroup const float4*>(&c_scratch[r][c4 * 4]);
                if (global_c + 3 < p.n) {
                    *reinterpret_cast<device half4*>(C + ulong(global_r) * p.n + global_c) = half4(f);
                } else {
                    for (uint e = 0; e < 4 && global_c + e < p.n; ++e) {
                        C[ulong(global_r) * p.n + global_c + e] = half(f[e]);
                    }
                }
            }
        }
    }
}
