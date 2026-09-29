#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <uint M, uint N, uint K>
METAL_FUNC void grouped_matmul(
    device half* input,
    device half* weights,
    device half* output,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, 64, K);
    matmul2d<descriptor, execution_simdgroups<4>> op;

    const ulong input_offset = ulong(tgid.z) * M * K;
    const ulong weight_offset = ulong(tgid.z) * K * N;
    const ulong output_offset = ulong(tgid.z) * M * N;
    auto a = tensor(input + input_offset, dextents<int32_t, 2>{K, M});
    auto b = tensor(weights + weight_offset, dextents<int32_t, 2>{N, K});
    auto c = tensor(output + output_offset, dextents<int32_t, 2>{N, M});
    auto tile_a = a.slice<K, 128>(0, tgid.y * 128);
    auto tile_b = b.slice<64, K>(tgid.x * 64, 0);
    auto tile_c = c.slice<64, 128>(tgid.x * 64, tgid.y * 128);
    op.run(tile_a, tile_b, tile_c);
}

template <uint M, uint N, uint K>
METAL_FUNC void wide_matmul(
    device half* input,
    device half* weights,
    device half* output,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, 128, K);
    matmul2d<descriptor, execution_simdgroups<8>> op;

    auto tile_a = tensor(
        input + ulong(tgid.y) * 128 * K,
        extents<int32_t, K, 128>{});
    auto tile_b = tensor(
        weights + tgid.x * 128,
        extents<int32_t, 128, K>{},
        array<int32_t, 2>{1, N});
    auto tile_c = tensor(
        output + ulong(tgid.y) * 128 * N + tgid.x * 128,
        extents<int32_t, 128, 128>{},
        array<int32_t, 2>{1, N});
    op.run(tile_a, tile_b, tile_c);
}

template <uint N, uint K, uint KI, uint NI>
METAL_FUNC void materialize_kronecker(
    device half* weights,
    device half* inner,
    device half* outer,
    device half* output,
    uint3 gid) {
    const uint n = gid.x * 4;
    const uint k = gid.y;
    const uint expert = gid.z;
    if (n >= N || k >= K || expert >= 32) return;
    const ulong expert_elements = ulong(K) * N;
    const ulong index = ulong(expert) * expert_elements + ulong(k) * N + n;
    const ulong inner_offset = ulong(expert) * KI * NI;
    const ulong outer_offset = ulong(expert) * (K / KI) * (N / NI);
    const half4 base = *reinterpret_cast<const device half4*>(weights + index);
    const half4 inner_values = *reinterpret_cast<const device half4*>(
        inner + inner_offset + (k % KI) * NI + (n % NI));
    const float outer_value = float(
        outer[outer_offset + (k / KI) * (N / NI) + (n / NI)]);
    *reinterpret_cast<device half4*>(output + index) =
        half4(float4(base) + float4(inner_values) * outer_value);
}

kernel void fbt_moe_tensorops_gate_up(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    grouped_matmul<256, 1728, 512>(input, weights, output, tgid);
}

template <uint N>
METAL_FUNC void activated_matmul(
    device half* input,
    device half* weights,
    device half* output,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, N, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto a = tensor(input + ulong(tgid.z) * 256 * 512, dextents<int, 2>{512, 256});
    auto b = tensor(weights + ulong(tgid.z) * 512 * 1728, dextents<int, 2>{1728, 512});
    auto c = tensor(output + ulong(tgid.z) * 256 * 864, dextents<int, 2>{864, 256});
    auto tile_a = a.slice<512, 128>(0, tgid.y * 128);
    auto gate_b = b.slice<N, 512>(tgid.x * 64, 0);
    auto up_b = b.slice<N, 512>(864 + tgid.x * 64, 0);
    auto gate = op.template get_destination_cooperative_tensor<decltype(tile_a), decltype(gate_b), half>();
    auto up = op.template get_destination_cooperative_tensor<decltype(tile_a), decltype(up_b), half>();
    op.run(tile_a, gate_b, gate);
    op.run(tile_a, up_b, up);
    // Keep the reference's half rounding before the float activation.
    #pragma unroll
    for (uint i = 0; i < gate.get_capacity(); ++i) {
        if (gate.is_valid_element(i)) {
            const float value = float(gate[i]);
            const float e = exp(-abs(value));
            const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
            gate[i] = half(value * sigmoid * float(up[i]));
        }
    }
    gate.store(c.slice<N, 128>(tgid.x * 64, tgid.y * 128));
}

kernel void gate_activation(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 13) activated_matmul<32>(input, weights, output, tgid);
    else activated_matmul<64>(input, weights, output, tgid);
}

kernel void fbt_moe_tensorops_down(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    grouped_matmul<256, 512, 864>(input, weights, output, tgid);
}

kernel void fbt_model_tensorops_qkv(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    grouped_matmul<8192, 640, 512>(input, weights, output, tgid);
}

kernel void fbt_model_tensorops_output_projection(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    grouped_matmul<8192, 512, 512>(input, weights, output, tgid);
}

kernel void fbt_model_tensorops_qkv_wide(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    wide_matmul<8192, 640, 512>(input, weights, output, tgid);
}

kernel void fbt_model_tensorops_output_projection_wide(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    wide_matmul<8192, 512, 512>(input, weights, output, tgid);
}

kernel void fbt_model_tensorops_readout(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    grouped_matmul<8192, 8192, 512>(input, weights, output, tgid);
}

kernel void fbt_moe_materialize_kronecker_gate_up(
    device half* weights [[buffer(0)]],
    device half* inner [[buffer(1)]],
    device half* outer [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    materialize_kronecker<1728, 512, 32, 32>(weights, inner, outer, output, gid);
}

kernel void fbt_moe_materialize_kronecker_down(
    device half* weights [[buffer(0)]],
    device half* inner [[buffer(1)]],
    device half* outer [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    materialize_kronecker<512, 864, 27, 16>(weights, inner, outer, output, gid);
}
