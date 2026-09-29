#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct Top3RouteShape {
    uint rows;
    uint width;
    uint experts;
    uint top_k;
    uint block_tokens;
    uint blocks;
};

// Project all router logits with the neural-accelerator path. Dynamic tensor
// extents make the last token tile exact instead of imposing a padded routing
// capacity. The production dimensions are width 512 and 128 routed experts.
kernel void fbt_moe_router_logits(
    device half* input [[buffer(0)]],
    device half* router [[buffer(1)]],
    device half* scores [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;

    const uint first_row = tgid.y * 128;
    const uint first_expert = tgid.x * 64;
    const uint valid_rows = min(128u, p.rows - first_row);
    const uint valid_experts = min(64u, p.experts - first_expert);
    auto a = tensor(
        input + ulong(first_row) * p.width,
        dextents<int32_t, 2>{int32_t(p.width), int32_t(valid_rows)});
    auto b = tensor(
        router + first_expert,
        dextents<int32_t, 2>{int32_t(valid_experts), int32_t(p.width)},
        array<int32_t, 2>{1, int32_t(p.experts)});
    auto c = tensor(
        scores + ulong(first_row) * p.experts + first_expert,
        dextents<int32_t, 2>{int32_t(valid_experts), int32_t(valid_rows)},
        array<int32_t, 2>{1, int32_t(p.experts)});
    op.run(a, b, c);
}

METAL_FUNC bool routed_tile(
    device const uint* expert_offsets,
    device const uint* expert_loads,
    uint tile,
    thread uint& expert,
    thread uint& first_row,
    thread uint& valid_rows) {
    uint first_tile = 0;
    for (uint candidate = 0; candidate < 128; ++candidate) {
        const uint load = expert_loads[candidate];
        const uint tiles = (load + 127) / 128;
        if (tile < first_tile + tiles) {
            const uint local = tile - first_tile;
            expert = candidate;
            first_row = expert_offsets[candidate] + local * 128;
            valid_rows = min(128u, load - local * 128);
            return true;
        }
        first_tile += tiles;
    }
    return false;
}

template <uint N>
METAL_FUNC void routed_gate(
    device half* packed_input,
    device half* weights,
    device const uint* expert_offsets,
    device const uint* expert_loads,
    device half* activation,
    uint3 tgid) {
    uint expert;
    uint first_row;
    uint valid_rows;
    if (!routed_tile(expert_offsets, expert_loads, tgid.y,
                     expert, first_row, valid_rows)) return;

    constexpr auto descriptor = matmul2d_descriptor(128, N, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_column = tgid.x * 64;
    const ulong expert_base = ulong(expert + 1) * 512 * 432;
    auto a = tensor(
        packed_input + ulong(first_row) * 512,
        dextents<int32_t, 2>{512, int32_t(valid_rows)});
    auto gate_b = tensor(
        weights + expert_base + first_column,
        dextents<int32_t, 2>{N, 512},
        array<int32_t, 2>{1, 432});
    auto up_b = tensor(
        weights + expert_base + 216 + first_column,
        dextents<int32_t, 2>{N, 512},
        array<int32_t, 2>{1, 432});
    auto c = tensor(
        activation + ulong(first_row) * 216 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 216});
    auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
    auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
    op.run(a, gate_b, gate);
    op.run(a, up_b, up);
    for (uint index = 0; index < gate.get_capacity(); ++index) {
        if (gate.is_valid_element(index)) {
            const float value = float(gate[index]);
            const float e = exp(-abs(value));
            const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
            gate[index] = half(value * sigmoid * float(up[index]));
        }
    }
    gate.store(c);
}

kernel void fbt_moe_routed_gate(
    device half* packed_input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const uint* expert_offsets [[buffer(2)]],
    device const uint* expert_loads [[buffer(3)]],
    device half* activation [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x >= (216 + 63) / 64 || tgid.y >= (p.rows * p.top_k + 127) / 128 + 127)
        return;
    if (tgid.x == 3) routed_gate<24>(packed_input, weights, expert_offsets,
                                               expert_loads, activation, tgid);
    else routed_gate<64>(packed_input, weights, expert_offsets,
                                    expert_loads, activation, tgid);
}

template <uint N>
METAL_FUNC void routed_down(
    device half* activation,
    device half* weights,
    device const uint* expert_offsets,
    device const uint* expert_loads,
    device half* output,
    uint3 tgid) {
    uint expert;
    uint first_row;
    uint valid_rows;
    if (!routed_tile(expert_offsets, expert_loads, tgid.y,
                     expert, first_row, valid_rows)) return;

    constexpr auto descriptor = matmul2d_descriptor(128, N);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_column = tgid.x * 64;
    const ulong expert_base = ulong(expert + 1) * 216 * 512;
    auto a = tensor(
        activation + ulong(first_row) * 216,
        dextents<int32_t, 2>{216, int32_t(valid_rows)});
    auto b = tensor(
        weights + expert_base + first_column,
        dextents<int32_t, 2>{N, 216},
        array<int32_t, 2>{1, 512});
    auto c = tensor(
        output + ulong(first_row) * 512 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 512});
    op.run(a, b, c);
}

kernel void fbt_moe_routed_down(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const uint* expert_offsets [[buffer(2)]],
    device const uint* expert_loads [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x >= p.width / 64 || tgid.y >= (p.rows * p.top_k + 127) / 128 + 127)
        return;
    routed_down<64>(activation, weights, expert_offsets, expert_loads, output, tgid);
}

template <uint N>
METAL_FUNC void shared_gate(
    device half* input,
    device half* weights,
    device half* activation,
    constant Top3RouteShape& p,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, N, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = tgid.y * 128;
    const uint valid_rows = min(128u, p.rows - first_row);
    const uint first_column = tgid.x * 64;
    auto a = tensor(
        input + ulong(first_row) * 512,
        dextents<int32_t, 2>{512, int32_t(valid_rows)});
    auto gate_b = tensor(
        weights + first_column,
        dextents<int32_t, 2>{N, 512},
        array<int32_t, 2>{1, 432});
    auto up_b = tensor(
        weights + 216 + first_column,
        dextents<int32_t, 2>{N, 512},
        array<int32_t, 2>{1, 432});
    auto c = tensor(
        activation + ulong(first_row) * 216 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 216});
    auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
    auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
    op.run(a, gate_b, gate);
    op.run(a, up_b, up);
    for (uint index = 0; index < gate.get_capacity(); ++index) {
        if (gate.is_valid_element(index)) {
            const float value = float(gate[index]);
            const float e = exp(-abs(value));
            const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
            gate[index] = half(value * sigmoid * float(up[index]));
        }
    }
    gate.store(c);
}

kernel void fbt_moe_shared_gate(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* activation [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 3) shared_gate<24>(input, weights, activation, p, tgid);
    else shared_gate<64>(input, weights, activation, p, tgid);
}

kernel void fbt_moe_shared_down(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(128, 64);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = tgid.y * 128;
    const uint valid_rows = min(128u, p.rows - first_row);
    const uint first_column = tgid.x * 64;
    auto a = tensor(
        activation + ulong(first_row) * 216,
        dextents<int32_t, 2>{216, int32_t(valid_rows)});
    auto b = tensor(
        weights + first_column,
        dextents<int32_t, 2>{64, 216},
        array<int32_t, 2>{1, 512});
    auto c = tensor(
        output + ulong(first_row) * 512 + first_column,
        dextents<int32_t, 2>{64, int32_t(valid_rows)},
        array<int32_t, 2>{1, 512});
    op.run(a, b, c);
}
