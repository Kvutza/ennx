#include <metal_stdlib>
using namespace metal;

// Mask is outside the tokenizer vocabulary. Its vector is a trainable tensor
// in the candidate arena and never participates in vocabulary sampling.
kernel void denoise_embed(
    device const half* weights [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const half* mask [[buffer(3)]],
    device const float* confidence [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float squares[4];
    threadgroup float scale;
    const uint token = tokens[row];
    const float alpha = token == UINT_MAX ? 0.0f : confidence[row];
    float norm = 0.0f;
    for (uint dim = tid; dim < 512; dim += 128) {
        const float predicted = token == UINT_MAX ? 0.0f : float(weights[ulong(dim) * 8192 + token]);
        const float value = alpha * predicted + (1.0f - alpha) * float(mask[dim]);
        output[ulong(row) * 512 + dim] = half(value);
        norm = fma(value, value, norm);
    }
    norm = simd_sum(norm);
    if (lane == 0) squares[simd] = norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) scale = rsqrt((squares[0] + squares[1] + squares[2] + squares[3]) / 512.0f + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Normalize only mixtures; discrete embeddings retain their original scale.
    if (alpha > 0.0f && alpha < 1.0f)
        for (uint dim = tid; dim < 512; dim += 128)
            output[ulong(row) * 512 + dim] *= half(scale * 0.02f);
}

// Each tile contains Gumbel winner, token ID, ordinary logsumexp maximum and
// denominator. Sampling and confidence use the same temperature distribution.
kernel void denoise_reduce(
    device const float4* partials [[buffer(0)]],
    device uint* tokens [[buffer(1)]],
    device float* confidence [[buffer(2)]],
    device const float2* moments [[buffer(3)]],
    constant float& temperature [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    float best = -INFINITY, maximum = -INFINITY;
    uint token = UINT_MAX;
    float selected_logit = 0.0f;
    for (uint tile = 0; tile < 128; ++tile) {
        const float4 p = partials[ulong(row) * 128 + tile];
        const uint index = as_type<uint>(p.y);
        if (p.x > best || (p.x == best && index < token)) {
            best = p.x; token = index; selected_logit = p.z;
        }
        maximum = max(maximum, moments[ulong(row) * 128 + tile].x);
    }
    float denominator = 0.0f;
    for (uint tile = 0; tile < 128; ++tile) {
        const float2 p = moments[ulong(row) * 128 + tile];
        denominator += p.y * exp(p.x - maximum);
    }
    tokens[row] = token;
    confidence[row] = temperature == 0.0f ? 1.0f : exp(selected_logit / temperature - maximum) / denominator;
}
