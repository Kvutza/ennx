#include <metal_stdlib>
using namespace metal;

struct GraphParams { uint width; uint rows; uint start; uint vocab; float scale; };

kernel void fbt_cross_entropy_partials(device const float4 *partials [[buffer(0)]],
    device float *loss [[buffer(1)]], constant GraphParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    ulong tiles = (ulong(p.vocab) + 31) / 32;
    float maximum = -INFINITY;
    for (ulong i = lane; i < tiles; i += 32) maximum = max(maximum, partials[row * tiles + i].x);
    maximum = simd_max(maximum);
    float sum = 0.0f, target = 0.0f;
    for (ulong i = lane; i < tiles; i += 32) {
        float4 v = partials[row * tiles + i];
        sum += v.y * exp(v.x - maximum);
        target += v.z;
    }
    sum = simd_sum(sum); target = simd_sum(target);
    if (lane == 0) loss[p.start + row] = maximum + log(sum) - target;
}

kernel void fbt_lookup(device const ushort *weights [[buffer(0)]],
    device const uint *tokens [[buffer(1)]], device float *out [[buffer(2)]],
    constant GraphParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint token = tokens[p.start + row];
    for (uint i = lane; i < p.width; i += 32)
        out[ulong(row) * p.width + i] = as_type<float>(uint(weights[ulong(token) * p.width + i]) << 16);
}

kernel void fbt_residual(device const float *branch [[buffer(0)]],
    device float *residual [[buffer(1)]], constant GraphParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    for (uint i = lane; i < p.width; i += 32) {
        ulong ix = ulong(row) * p.width + i;
        residual[ix] += p.scale * branch[ix];
    }
}

kernel void fbt_glu(device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]], device float *out [[buffer(2)]],
    constant GraphParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    for (uint i = lane; i < p.width; i += 32) {
        ulong ix = ulong(row) * p.width + i;
        float g = gate[ix];
        float e = exp(-abs(g));
        float s = g >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
        out[ix] = g * s * up[ix];
    }
}

kernel void fbt_shift_state(device const float *history [[buffer(0)]],
    device float *previous [[buffer(1)]], device uint *mask [[buffer(2)]],
    constant GraphParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint pos = p.start + row;
    if (lane == 0) mask[row] = pos != 0;
    for (uint i = lane; i < p.width; i += 32)
        previous[ulong(row) * p.width + i] = pos ? history[ulong(pos - 1) * p.width + i] : 0.0f;
}

kernel void fbt_capture_state(device const float *state [[buffer(0)]],
    device float *history [[buffer(1)]], constant GraphParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    for (uint i = lane; i < p.width; i += 32)
        history[ulong(p.start + row) * p.width + i] = state[ulong(row) * p.width + i];
}

kernel void fbt_cross_entropy(device const float *logits [[buffer(0)]],
    device const uint *targets [[buffer(1)]], device float *loss [[buffer(2)]],
    constant GraphParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    ulong base = ulong(row) * p.vocab;
    float maximum = -INFINITY;
    for (uint i = lane; i < p.vocab; i += 32) maximum = max(maximum, logits[base + i]);
    maximum = simd_max(maximum);
    float sum = 0.0f;
    for (uint i = lane; i < p.vocab; i += 32) sum += exp(logits[base + i] - maximum);
    sum = simd_sum(sum);
    if (lane == 0) loss[p.start + row] = maximum + log(sum) - logits[base + targets[p.start + row]];
}
