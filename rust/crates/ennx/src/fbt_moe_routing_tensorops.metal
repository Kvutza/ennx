#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef UNROLL_GATE
#define UNROLL_GATE
#endif

#ifdef INT8_MOE_GATE
typedef int8_t moe_gate_weight;
#else
typedef half moe_gate_weight;
#endif

#ifndef RELAXED_MOE_TENSOROPS
#define RELAXED_MOE_TENSOROPS
#endif

#ifdef RELAXED_MOE_TENSOROPS
#define ENNX_RELAXED_MOE true
#else
#define ENNX_RELAXED_MOE false
#endif

#ifndef FAST_MOE_ACTIVATION
#define FAST_MOE_ACTIVATION
#endif

METAL_FUNC float moe_activation_exp(float value) {
#ifdef FAST_MOE_ACTIVATION
    return fast::exp(value);
#else
    return exp(value);
#endif
}

#ifndef HALF_MOE_ACTIVATION
#define HALF_MOE_ACTIVATION
#endif

METAL_FUNC half moe_swiglu(half gate_value, half up_value) {
#ifdef HALF_MOE_ACTIVATION
    const half e = exp(-abs(gate_value));
    const half sigmoid = gate_value >= half(0.0h)
        ? half(1.0h) / (half(1.0h) + e)
        : e / (half(1.0h) + e);
    return gate_value * sigmoid * up_value;
#else
    const float value = float(gate_value);
    const float e = moe_activation_exp(-abs(value));
    const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return half(value * sigmoid * float(up_value));
#endif
}

struct Top3RouteShape {
    uint rows;
    uint width;
    uint experts;
    uint top_k;
    uint block_tokens;
    uint blocks;
};

struct RoutedTile {
    uint expert;
    uint first_row;
    uint valid_rows;
    uint reserved;
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
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512, false, false, ENNX_RELAXED_MOE);
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

METAL_FUNC void routed_tile(
    device const RoutedTile* routed_tiles,
    uint tile_index,
    thread uint& expert,
    thread uint& first_row,
    thread uint& valid_rows) {
    const RoutedTile tile = routed_tiles[tile_index];
    expert = tile.expert;
    first_row = tile.first_row;
    valid_rows = tile.valid_rows;
}

template <uint M, uint N, uint Simdgroups, uint ColumnStep>
METAL_FUNC void routed_gate(
    device half* packed_input,
    device moe_gate_weight* weights,
    device const RoutedTile* routed_tiles,
    device half* activation,
    uint3 tgid) {
    uint expert;
    uint first_row;
    uint valid_rows;
    routed_tile(routed_tiles, tgid.y, expert, first_row, valid_rows);

    constexpr auto descriptor = matmul2d_descriptor(M, N, 512, false, false, ENNX_RELAXED_MOE);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> op;
    const uint first_column = tgid.x * ColumnStep;
    const ulong expert_base = ulong(expert + 1) * 512 * 432;
    if (valid_rows == M) {
        auto a = tensor(
            packed_input + ulong(first_row) * 512,
            extents<int32_t, 512, M>{});
        auto gate_b = tensor(
            weights + expert_base + first_column,
            extents<int32_t, N, 512>{},
            array<int32_t, 2>{1, 432});
        auto up_b = tensor(
            weights + expert_base + 216 + first_column,
            extents<int32_t, N, 512>{},
            array<int32_t, 2>{1, 432});
        auto c = tensor(
            activation + ulong(first_row) * 224 + first_column,
            extents<int32_t, N, M>{},
            array<int32_t, 2>{1, 224});
        auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
        auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
        op.run(a, gate_b, gate);
        op.run(a, up_b, up);
#ifdef UNROLL_GATE
        #pragma unroll
#endif
        for (uint index = 0; index < gate.get_capacity(); ++index) {
            if (gate.is_valid_element(index)) {
#ifdef INT8_MOE_GATE
                gate[index] = moe_swiglu(half(gate[index] * 0x1.0p-13f), half(up[index] * 0x1.0p-13f));
#else
                gate[index] = moe_swiglu(gate[index], up[index]);
#endif
            }
        }
        gate.store(c);
        return;
    }
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
        activation + ulong(first_row) * 224 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 224});
    auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
    auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
    op.run(a, gate_b, gate);
    op.run(a, up_b, up);
#ifdef UNROLL_GATE
    #pragma unroll
#endif
    for (uint index = 0; index < gate.get_capacity(); ++index) {
        if (gate.is_valid_element(index)) {
#ifdef INT8_MOE_GATE
            gate[index] = moe_swiglu(half(gate[index] * 0x1.0p-13f), half(up[index] * 0x1.0p-13f));
#else
            gate[index] = moe_swiglu(gate[index], up[index]);
#endif
        }
    }
    gate.store(c);
}

kernel void fbt_moe_routed_gate(
    device half* packed_input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 3) routed_gate<64, 24, 4, 64>(packed_input, weights, routed_tiles, activation, tgid);
    else routed_gate<64, 64, 4, 64>(packed_input, weights, routed_tiles, activation, tgid);
}

kernel void fbt_moe_routed_gate_tall(
    device half* packed_input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 3) routed_gate<128, 24, 4, 64>(packed_input, weights, routed_tiles, activation, tgid);
    else routed_gate<128, 64, 4, 64>(packed_input, weights, routed_tiles, activation, tgid);
}

kernel void fbt_moe_routed_gate_wide(
    device half* packed_input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 1) routed_gate<64, 88, 8, 128>(packed_input, weights, routed_tiles, activation, tgid);
    else routed_gate<64, 128, 8, 128>(packed_input, weights, routed_tiles, activation, tgid);
}

kernel void fbt_moe_routed_gate_fused_columns(
    device half* packed_input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    const uint first_tile = tgid.x * 2;
    const uint3 first = uint3(first_tile, tgid.y, tgid.z);
    routed_gate<64, 64, 4, 64>(packed_input, weights, routed_tiles, activation, first);
    const uint3 second = uint3(first_tile + 1, tgid.y, tgid.z);
    if (first_tile == 0)
        routed_gate<64, 64, 4, 64>(packed_input, weights, routed_tiles, activation, second);
    else
        routed_gate<64, 24, 4, 64>(packed_input, weights, routed_tiles, activation, second);
}

kernel void fbt_moe_routed_gate_all_columns(
    device half* packed_input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    for (uint column_tile = 0; column_tile < 3; ++column_tile) {
        const uint3 tile = uint3(column_tile, tgid.y, tgid.z);
        routed_gate<64, 64, 4, 64>(packed_input, weights, routed_tiles, activation, tile);
    }
    routed_gate<64, 24, 4, 64>(
        packed_input, weights, routed_tiles, activation, uint3(3, tgid.y, tgid.z));
}

kernel void fbt_moe_routed_gate_reuse(
    device half* packed_input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const RoutedTile tile = routed_tiles[tgid.y];
    const uint start = simdgroup * 16;
    if (start >= tile.valid_rows) return;
    const uint rows = min(16u, tile.valid_rows - start);
    constexpr auto descriptor = matmul2d_descriptor(16, 216, 512, false, false, false);
    matmul2d<descriptor, execution_simdgroups<1>> op;
    auto a = tensor(packed_input + ulong(tile.first_row + start) * 512,
        dextents<int32_t, 2>{512, int32_t(rows)});
    const ulong expert_base = ulong(tile.expert + 1) * 512 * 432;
    auto gate_b = tensor(weights + expert_base, extents<int32_t, 216, 512>{},
        array<int32_t, 2>{1, 432});
    auto up_b = tensor(weights + expert_base + 216, extents<int32_t, 216, 512>{},
        array<int32_t, 2>{1, 432});
    auto c = tensor(activation + ulong(tile.first_row + start) * 224,
        dextents<int32_t, 2>{216, int32_t(rows)}, array<int32_t, 2>{1, 224});
    auto cached_a = op.get_left_input_cooperative_tensor<half, half, half>();
    cached_a.load(a);
    auto gate = op.get_destination_cooperative_tensor<decltype(cached_a), decltype(gate_b), half>();
    auto up = op.get_destination_cooperative_tensor<decltype(cached_a), decltype(up_b), half>();
    op.run(cached_a, gate_b, gate);
    op.run(cached_a, up_b, up);
    #pragma unroll
    for (uint index = 0; index < gate.get_capacity(); ++index) {
        if (gate.is_valid_element(index)) gate[index] = moe_swiglu(gate[index], up[index]);
    }
    gate.store(c);
}

// Paired-column SwiGLU dataflow, inspired by the interleaved cuDNN/TIRx
// epilogue. Metal owns the matrix schedule; no CUDA TMA/TMEM emulation.
// A 64-row tile preserves weight reuse. Only a 64x128 FP16 projection is
// live in threadgroup memory, reused for the next pair after its epilogue.
template <uint N>
METAL_FUNC void interleaved_gate_tile(
    device half* input,
    device half* weights,
    device half* activation,
    threadgroup half* projected,
    RoutedTile tile,
    uint column,
    uint tid) {
    constexpr auto descriptor = matmul2d_descriptor(64, 2 * N, 512, false, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto a = tensor(input + ulong(tile.first_row) * 512,
        dextents<int32_t, 2>{512, int32_t(tile.valid_rows)});
    auto b = tensor(weights + ulong(tile.expert + 1) * 512 * 432 + 2 * column,
        extents<int32_t, 2 * N, 512>{}, array<int32_t, 2>{1, 432});
#ifdef INTERLEAVED_GATE_REGISTERS
    auto pair = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    op.run(a, b, pair);
    #pragma unroll
    for (uint i = 0; i < pair.get_capacity(); ++i) {
        if (!pair.is_valid_element(i)) continue;
        const auto index = pair.get_multidimensional_index(i);
        if (index[1] < tile.valid_rows && index[0] >= N && index[0] < 2 * N) {
            projected[index[1] * N + index[0] - N] = pair[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint i = 0; i < pair.get_capacity(); ++i) {
        if (!pair.is_valid_element(i)) continue;
        const auto index = pair.get_multidimensional_index(i);
        if (index[1] < tile.valid_rows && index[0] < N) {
            // Preserve the FP16 projection boundary when the following
            // activation promotes to float under the pipeline's fast math.
            volatile half gate = pair[i];
            activation[ulong(tile.first_row + index[1]) * 224 + column + index[0]] =
                moe_swiglu(gate, projected[index[1] * N + index[0]]);
        }
    }
#else
    auto c = tensor(projected,
        dextents<int32_t, 2>{2 * N, int32_t(tile.valid_rows)});
    op.run(a, b, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint item = tid; item < tile.valid_rows * N; item += 128) {
        const uint row = item / N;
        const uint channel = item % N;
        activation[ulong(tile.first_row + row) * 224 + column + channel] =
            moe_swiglu(projected[row * 2 * N + channel],
                projected[row * 2 * N + N + channel]);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void fbt_moe_routed_gate_interleaved(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
#ifdef INTERLEAVED_GATE_REGISTERS
    threadgroup half projected[64 * 64];
#else
    threadgroup half projected[64 * 128];
#endif
    const RoutedTile tile = tiles[tgid.y];
    const uint column = tgid.x * 128;
    interleaved_gate_tile<64>(input, weights, activation, projected, tile, column, tid);
    if (tgid.x == 0) {
        interleaved_gate_tile<64>(input, weights, activation, projected, tile, column + 64, tid);
    } else {
        interleaved_gate_tile<24>(input, weights, activation, projected, tile, column + 64, tid);
    }
}

// One contiguous projection produces both branches; SRAM holds FP16 results
// only until the paired SiLU product is written to the existing activation.
kernel void fbt_moe_routed_gate_joint(
    device half* packed_input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half projected[32 * 432];
    const RoutedTile tile = routed_tiles[tgid.y];
    constexpr auto descriptor = matmul2d_descriptor(32, 432, 512, false, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    for (uint start = 0; start < tile.valid_rows; start += 32) {
        const uint rows = min(32u, tile.valid_rows - start);
        auto a = tensor(packed_input + ulong(tile.first_row + start) * 512,
            dextents<int32_t, 2>{512, int32_t(rows)});
        auto b = tensor(weights + ulong(tile.expert + 1) * 512 * 432,
            extents<int32_t, 432, 512>{});
        auto c = tensor(projected, dextents<int32_t, 2>{432, int32_t(rows)});
        op.run(a, b, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = tid; item < rows * 216; item += 128) {
            const uint row = item / 216;
            const uint column = item % 216;
            activation[ulong(tile.first_row + start + row) * 224 + column] =
                moe_swiglu(projected[row * 432 + column], projected[row * 432 + 216 + column]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Preserve 64-row weight reuse while projecting both branches together.
// The caller reuses routed_output as temporary storage before the down pass.
kernel void fbt_moe_routed_gate_joint64(
    device half* packed_input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* projection [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    const RoutedTile tile = routed_tiles[tgid.y];
    constexpr auto descriptor = matmul2d_descriptor(64, 432, 512, false, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto a = tensor(packed_input + ulong(tile.first_row) * 512,
        dextents<int32_t, 2>{512, int32_t(tile.valid_rows)});
    auto b = tensor(weights + ulong(tile.expert + 1) * 512 * 432,
        extents<int32_t, 432, 512>{});
    auto c = tensor(projection + ulong(tile.first_row) * 432,
        dextents<int32_t, 2>{432, int32_t(tile.valid_rows)});
    op.run(a, b, c);
}

kernel void fbt_moe_joint_activation(
    device const half* projection [[buffer(0)]],
    device half* activation [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint row = gid / 54;
    const uint column = gid % 54 * 4;
    const half4 gate = *reinterpret_cast<device const half4*>(projection + ulong(row) * 432 + column);
    const half4 up = *reinterpret_cast<device const half4*>(projection + ulong(row) * 432 + 216 + column);
    half4 result;
    for (uint i = 0; i < 4; ++i) result[i] = moe_swiglu(gate[i], up[i]);
    *reinterpret_cast<device half4*>(activation + ulong(row) * 224 + column) = result;
}

#ifndef INT8_MOE_GATE
// Full tiles load straight into SIMD matrices. Both branches reuse each A
// fragment, and the rounded SiLU product stays in registers until its store.
kernel void fbt_moe_routed_gate_manual(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const RoutedTile tile = tiles[tgid.y];
    if (tgid.x == 3) {
        routed_gate<64, 24, 4, 64>(input, weights, tiles, activation, tgid);
        return;
    }
    if (tile.valid_rows != 64) {
        routed_gate<64, 64, 4, 64>(input, weights, tiles, activation, tgid);
        return;
    }
    const uint row = tile.first_row + sg / 2 * 32;
    const uint column = tgid.x * 64 + sg % 2 * 32;
    const ulong base = ulong(tile.expert + 1) * 512 * 432;
    simdgroup_float8x8 gate[4][4];
    simdgroup_float8x8 up[4][4];
    #pragma unroll
    for (uint m = 0; m < 4; ++m) {
        #pragma unroll
        for (uint n = 0; n < 4; ++n) {
            gate[m][n] = simdgroup_float8x8(0.0f);
            up[m][n] = simdgroup_float8x8(0.0f);
        }
    }
    for (uint k = 0; k < 512; k += 8) {
        simdgroup_half8x8 a[4];
        #pragma unroll
        for (uint m = 0; m < 4; ++m)
            simdgroup_load(a[m], input + ulong(row + m * 8) * 512 + k, 512);
        #pragma unroll
        for (uint n = 0; n < 4; ++n) {
            simdgroup_half8x8 b_gate;
            simdgroup_half8x8 b_up;
            simdgroup_load(b_gate, weights + base + ulong(k) * 432 + column + n * 8, 432);
            simdgroup_load(b_up, weights + base + ulong(k) * 432 + 216 + column + n * 8, 432);
            #pragma unroll
            for (uint m = 0; m < 4; ++m) {
                simdgroup_multiply_accumulate(gate[m][n], a[m], b_gate, gate[m][n]);
                simdgroup_multiply_accumulate(up[m][n], a[m], b_up, up[m][n]);
            }
        }
    }
    #pragma unroll
    for (uint m = 0; m < 4; ++m) {
        #pragma unroll
        for (uint n = 0; n < 4; ++n) {
            simdgroup_half8x8 result;
            #pragma unroll
            for (uint element = 0; element < 2; ++element)
                result.thread_elements()[element] = moe_swiglu(
                    half(gate[m][n].thread_elements()[element]),
                    half(up[m][n].thread_elements()[element]));
            simdgroup_store(result, activation + ulong(row + m * 8) * 224 + column + n * 8, 224);
        }
    }
}
#endif

// Manual SIMD-matrix tile adapted from the prefill engine. It stages each
// packed-input K tile once and reuses it for both gate and up projections.
kernel void fbt_moe_routed_gate_staged(
    device half* packed_input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* activation [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half a_tile[64][36];
    threadgroup half b_gate_tile[32][68];
    threadgroup half b_up_tile[32][68];
    threadgroup float c_scratch[64][68];

    const RoutedTile routed = routed_tiles[tgid.y];
    const uint first_column = tgid.x * 64;
    const uint valid_columns = min(64u, 216u - first_column);
    const uint simdgroup = tid / 32;
    const uint simd_row = (simdgroup / 2) * 32;
    const uint simd_column = (simdgroup % 2) * 32;
    const ulong expert_base = ulong(routed.expert + 1) * 512 * 432;

    simdgroup_float8x8 gate[4][4];
    simdgroup_float8x8 up[4][4];
    for (uint row_tile = 0; row_tile < 4; ++row_tile) {
        for (uint column_tile = 0; column_tile < 4; ++column_tile) {
            gate[row_tile][column_tile] = simdgroup_float8x8(0.0f);
            up[row_tile][column_tile] = simdgroup_float8x8(0.0f);
        }
    }

    for (uint k_start = 0; k_start < 512; k_start += 32) {
        for (uint step = 0; step < 4; ++step) {
            const uint vector = tid + step * 128;
            const uint row = vector / 8;
            const uint k_vector = vector % 8;
            half4 value = half4(0.0h);
            if (row < routed.valid_rows) {
                value = *reinterpret_cast<device const half4*>(
                    packed_input + ulong(routed.first_row + row) * 512
                    + k_start + k_vector * 4);
            }
            *reinterpret_cast<threadgroup half4*>(&a_tile[row][k_vector * 4]) = value;
        }
        for (uint step = 0; step < 4; ++step) {
            const uint vector = tid + step * 128;
            const uint k_row = vector / 16;
            const uint column_vector = vector % 16;
            const uint column = column_vector * 4;
            half4 gate_value = half4(0.0h);
            half4 up_value = half4(0.0h);
            if (column < valid_columns) {
                const uint valid = min(4u, valid_columns - column);
                for (uint item = 0; item < valid; ++item) {
                    const ulong source = expert_base
                        + ulong(k_start + k_row) * 432
                        + first_column + column + item;
                    gate_value[item] = weights[source];
                    up_value[item] = weights[source + 216];
                }
            }
            *reinterpret_cast<threadgroup half4*>(
                &b_gate_tile[k_row][column]) = gate_value;
            *reinterpret_cast<threadgroup half4*>(
                &b_up_tile[k_row][column]) = up_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint depth = 0; depth < 32; depth += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 gate_b[4];
            simdgroup_half8x8 up_b[4];
            for (uint row_tile = 0; row_tile < 4; ++row_tile)
                simdgroup_load(
                    a[row_tile],
                    &a_tile[simd_row + row_tile * 8][depth],
                    36,
                    ulong2(0, 0));
            for (uint column_tile = 0; column_tile < 4; ++column_tile) {
                simdgroup_load(
                    gate_b[column_tile],
                    &b_gate_tile[depth][simd_column + column_tile * 8],
                    68,
                    ulong2(0, 0));
                simdgroup_load(
                    up_b[column_tile],
                    &b_up_tile[depth][simd_column + column_tile * 8],
                    68,
                    ulong2(0, 0));
            }
            for (uint row_tile = 0; row_tile < 4; ++row_tile) {
                for (uint column_tile = 0; column_tile < 4; ++column_tile) {
                    simdgroup_multiply_accumulate(
                        gate[row_tile][column_tile],
                        a[row_tile],
                        gate_b[column_tile],
                        gate[row_tile][column_tile]);
                    simdgroup_multiply_accumulate(
                        up[row_tile][column_tile],
                        a[row_tile],
                        up_b[column_tile],
                        up[row_tile][column_tile]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row_tile = 0; row_tile < 4; ++row_tile) {
        for (uint column_tile = 0; column_tile < 4; ++column_tile)
            simdgroup_store(
                gate[row_tile][column_tile],
                &c_scratch[simd_row + row_tile * 8][simd_column + column_tile * 8],
                68,
                ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 gate_values[8];
    for (uint step = 0; step < 8; ++step) {
        const uint vector = tid + step * 128;
        gate_values[step] = *reinterpret_cast<threadgroup const float4*>(
            &c_scratch[vector / 16][vector % 16 * 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row_tile = 0; row_tile < 4; ++row_tile) {
        for (uint column_tile = 0; column_tile < 4; ++column_tile)
            simdgroup_store(
                up[row_tile][column_tile],
                &c_scratch[simd_row + row_tile * 8][simd_column + column_tile * 8],
                68,
                ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < 8; ++step) {
        const uint vector = tid + step * 128;
        const uint row = vector / 16;
        const uint column = vector % 16 * 4;
        if (row < routed.valid_rows && column < valid_columns) {
            const float4 gate_value = float4(half4(gate_values[step]));
            const float4 up_value = float4(half4(
                *reinterpret_cast<threadgroup const float4*>(&c_scratch[row][column])));
            const float4 e = exp(-abs(gate_value));
            const float4 sigmoid = select(
                e / (1.0f + e),
                1.0f / (1.0f + e),
                gate_value >= 0.0f);
            const half4 result = half4(gate_value * sigmoid * up_value);
            const uint valid = min(4u, valid_columns - column);
            for (uint item = 0; item < valid; ++item) {
                activation[ulong(routed.first_row + row) * 224
                    + first_column + column + item] = result[item];
            }
        }
    }
}

template <uint M, uint N, uint Simdgroups>
METAL_FUNC void routed_down(
    device half* activation,
    device half* weights,
    device const RoutedTile* routed_tiles,
    device half* output,
    constant Top3RouteShape& p,
    uint3 tgid) {
    uint expert;
    uint first_row;
    uint valid_rows;
    routed_tile(routed_tiles, tgid.y, expert, first_row, valid_rows);

    constexpr auto dynamic_descriptor = matmul2d_descriptor(
        M, N, static_cast<int>(dynamic_extent), false, false, ENNX_RELAXED_MOE);
    constexpr auto padded_descriptor = matmul2d_descriptor(M, N, 224, false, false, ENNX_RELAXED_MOE);
    matmul2d<dynamic_descriptor, execution_simdgroups<Simdgroups>> dynamic_op;
    matmul2d<padded_descriptor, execution_simdgroups<Simdgroups>> padded_op;
    const uint first_column = tgid.x * N;
    const ulong expert_base = ulong(expert + 1) * 216 * 512;
    if (expert + 1 < p.experts && valid_rows == M) {
        auto a = tensor(
            activation + ulong(first_row) * 224,
            extents<int32_t, 224, M>{});
        auto b = tensor(
            weights + expert_base + first_column,
            extents<int32_t, N, 224>{},
            array<int32_t, 2>{1, 512});
        auto c = tensor(
            output + ulong(first_row) * 512 + first_column,
            extents<int32_t, N, M>{},
            array<int32_t, 2>{1, 512});
        padded_op.run(a, b, c);
        return;
    }
    if (expert + 1 < p.experts) {
        auto a = tensor(
            activation + ulong(first_row) * 224,
            dextents<int32_t, 2>{224, int32_t(valid_rows)},
            array<int32_t, 2>{1, 224});
        auto b = tensor(
            weights + expert_base + first_column,
            dextents<int32_t, 2>{N, 224},
            array<int32_t, 2>{1, 512});
        auto c = tensor(
            output + ulong(first_row) * 512 + first_column,
            dextents<int32_t, 2>{N, int32_t(valid_rows)},
            array<int32_t, 2>{1, 512});
        padded_op.run(a, b, c);
        return;
    }
    auto a = tensor(
        activation + ulong(first_row) * 224,
        dextents<int32_t, 2>{216, int32_t(valid_rows)},
        array<int32_t, 2>{1, 224});
    auto b = tensor(
        weights + expert_base + first_column,
        dextents<int32_t, 2>{N, 216},
        array<int32_t, 2>{1, 512});
    auto c = tensor(
        output + ulong(first_row) * 512 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 512});
    dynamic_op.run(a, b, c);
}

kernel void fbt_moe_routed_down(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    routed_down<64, 64, 4>(activation, weights, routed_tiles, output, p, tgid);
}

kernel void fbt_moe_routed_down_wide(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    routed_down<64, 128, 8>(activation, weights, routed_tiles, output, p, tgid);
}

kernel void fbt_moe_routed_down_tall(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    routed_down<128, 64, 4>(activation, weights, routed_tiles, output, p, tgid);
}

kernel void fbt_moe_routed_down_fused_columns(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const RoutedTile* routed_tiles [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    const uint first_tile = tgid.x * 2;
    routed_down<64, 64, 4>(
        activation,
        weights,
        routed_tiles,
        output,
        p,
        uint3(first_tile, tgid.y, tgid.z));
    routed_down<64, 64, 4>(
        activation,
        weights,
        routed_tiles,
        output,
        p,
        uint3(first_tile + 1, tgid.y, tgid.z));
}

template <uint N>
METAL_FUNC void shared_gate(
    device half* input,
    device moe_gate_weight* weights,
    device half* activation,
    constant Top3RouteShape& p,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, N, 512, false, false, ENNX_RELAXED_MOE);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = tgid.y * 128;
    const uint valid_rows = min(128u, p.rows - first_row);
    const uint first_column = tgid.x * 64;
    if (valid_rows == 128) {
        auto a = tensor(
            input + ulong(first_row) * 512,
            extents<int32_t, 512, 128>{});
        auto gate_b = tensor(
            weights + first_column,
            extents<int32_t, N, 512>{},
            array<int32_t, 2>{1, 432});
        auto up_b = tensor(
            weights + 216 + first_column,
            extents<int32_t, N, 512>{},
            array<int32_t, 2>{1, 432});
        auto c = tensor(
            activation + ulong(first_row) * 224 + first_column,
            extents<int32_t, N, 128>{},
            array<int32_t, 2>{1, 224});
        auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
        auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
        op.run(a, gate_b, gate);
        op.run(a, up_b, up);
        for (uint index = 0; index < gate.get_capacity(); ++index) {
            if (gate.is_valid_element(index)) {
#ifdef INT8_MOE_GATE
                gate[index] = moe_swiglu(half(gate[index] * 0x1.0p-13f), half(up[index] * 0x1.0p-13f));
#else
                gate[index] = moe_swiglu(gate[index], up[index]);
#endif
            }
        }
        gate.store(c);
        return;
    }
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
        activation + ulong(first_row) * 224 + first_column,
        dextents<int32_t, 2>{N, int32_t(valid_rows)},
        array<int32_t, 2>{1, 224});
    auto gate = op.template get_destination_cooperative_tensor<decltype(a), decltype(gate_b), half>();
    auto up = op.template get_destination_cooperative_tensor<decltype(a), decltype(up_b), half>();
    op.run(a, gate_b, gate);
    op.run(a, up_b, up);
    for (uint index = 0; index < gate.get_capacity(); ++index) {
        if (gate.is_valid_element(index)) {
#ifdef INT8_MOE_GATE
            gate[index] = moe_swiglu(half(gate[index] * 0x1.0p-13f), half(up[index] * 0x1.0p-13f));
#else
            gate[index] = moe_swiglu(gate[index], up[index]);
#endif
        }
    }
    gate.store(c);
}

kernel void fbt_moe_shared_gate(
    device half* input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device half* activation [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x == 3) shared_gate<24>(input, weights, activation, p, tgid);
    else shared_gate<64>(input, weights, activation, p, tgid);
}

kernel void fbt_moe_shared_gate_fused_columns(
    device half* input [[buffer(0)]],
    device moe_gate_weight* weights [[buffer(1)]],
    device half* activation [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    const uint first_tile = tgid.x * 2;
    shared_gate<64>(input, weights, activation, p, uint3(first_tile, tgid.y, tgid.z));
    const uint3 second = uint3(first_tile + 1, tgid.y, tgid.z);
    if (first_tile == 0)
        shared_gate<64>(input, weights, activation, p, second);
    else
        shared_gate<24>(input, weights, activation, p, second);
}

METAL_FUNC void shared_down(
    device half* activation,
    device half* weights,
    device half* output,
    constant Top3RouteShape& p,
    uint3 tgid) {
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 224, false, false, ENNX_RELAXED_MOE);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = tgid.y * 128;
    const uint valid_rows = min(128u, p.rows - first_row);
    const uint first_column = tgid.x * 64;
    if (valid_rows == 128) {
        auto a = tensor(
            activation + ulong(first_row) * 224,
            extents<int32_t, 224, 128>{});
        auto b = tensor(
            weights + first_column,
            extents<int32_t, 64, 224>{},
            array<int32_t, 2>{1, 512});
        auto c = tensor(
            output + ulong(first_row) * 512 + first_column,
            extents<int32_t, 64, 128>{},
            array<int32_t, 2>{1, 512});
        op.run(a, b, c);
        return;
    }
    auto a = tensor(
        activation + ulong(first_row) * 224,
        dextents<int32_t, 2>{224, int32_t(valid_rows)});
    auto b = tensor(
        weights + first_column,
        dextents<int32_t, 2>{64, 224},
        array<int32_t, 2>{1, 512});
    auto c = tensor(
        output + ulong(first_row) * 512 + first_column,
        dextents<int32_t, 2>{64, int32_t(valid_rows)},
        array<int32_t, 2>{1, 512});
    op.run(a, b, c);
}

kernel void fbt_moe_shared_down(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    shared_down(activation, weights, output, p, tgid);
}

kernel void fbt_moe_shared_down_fused_columns(
    device half* activation [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    const uint first_tile = tgid.x * 2;
    shared_down(activation, weights, output, p, uint3(first_tile, tgid.y, tgid.z));
    shared_down(activation, weights, output, p, uint3(first_tile + 1, tgid.y, tgid.z));
}
