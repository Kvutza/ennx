#include <metal_stdlib>
using namespace metal;

// Routing scratch shape for up to 640 experts. Routing is token-choice:
// every token independently selects three experts, so its choices never depend
// on which other tokens happen to share the batch.
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

METAL_FUNC float route_sigmoid(float value) {
    const float e = exp(-abs(value));
    const float probability = value >= 0.0f
        ? 1.0f / (1.0f + e)
        : e / (1.0f + e);
    return max(probability, 1.0e-20f);
}

// Eight exact zeros permit a fixed K=224 TensorOps projection. Padding is
// storage-only: the independent model parameters remain width 216.
kernel void fbt_moe_select_top3_wide(
    device const half* scores [[buffer(0)]],
    device uint* route_experts [[buffer(1)]],
    device half* route_weights [[buffer(2)]],
    device float* route_margin [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint expert [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint token = group.y;
    if (token >= p.rows || expert >= p.experts) return;

    // Each SIMD group selects its four best experts, then SIMD group zero
    // merges the resulting sixteen candidates. Every reduction resolves equal
    // scores by the lower expert id, matching the former serial insertion sort.
    threadgroup float local_scores[16];
    threadgroup uint local_experts[16];
    float score = float(scores[ulong(token) * p.experts + expert]);
    for (uint position = 0; position < 4; ++position) {
        const float best_score = simd_max(score);
        const uint best_expert = simd_min(
            score == best_score ? expert : UINT_MAX);
        if (lane == 0) {
            const uint slot = simdgroup * 4 + position;
            local_scores[slot] = best_score;
            local_experts[slot] = best_expert;
        }
        if (expert == best_expert) score = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float candidate_score = lane < 16 ? local_scores[lane] : -INFINITY;
        uint candidate_expert = lane < 16 ? local_experts[lane] : UINT_MAX;
        float best_scores[4];
        uint best_experts[4];
        for (uint position = 0; position < 4; ++position) {
            const float best_score = simd_max(candidate_score);
            const uint best_expert = simd_min(
                candidate_score == best_score ? candidate_expert : UINT_MAX);
            best_scores[position] = best_score;
            best_experts[position] = best_expert;
            if (candidate_expert == best_expert) candidate_score = -INFINITY;
        }
        if (lane == 0) {
            const float top3_denominator = route_sigmoid(best_scores[0])
                + route_sigmoid(best_scores[1]) + route_sigmoid(best_scores[2]);
            for (uint position = 0; position < p.top_k; ++position) {
                const ulong route = ulong(token) * p.top_k + position;
                route_experts[route] = best_experts[position];
                route_weights[route] =
                    half(route_sigmoid(best_scores[position]) / top3_denominator);
            }
            route_margin[token] = best_scores[2] - best_scores[3];
        }
    }
}

// One SIMD group owns a token. Each lane keeps twenty experts locally, exposing
// only its current best to the SIMD-wide reduction. This produces the same
// top-four ordering and lower-id tie break as the 128-thread implementation,
// without its cross-SIMD scratch traffic and barrier.
kernel void fbt_moe_select_top3(
    device const half* scores [[buffer(0)]],
    device uint* route_experts [[buffer(1)]],
    device half* route_weights [[buffer(2)]],
    device float* route_margin [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = group.y;
    if (token >= p.rows) return;

    float lane_scores[20];
    uint lane_experts[20];
    for (uint item = 0; item < 20; ++item) {
        const uint expert = lane + item * 32;
        lane_experts[item] = expert;
        lane_scores[item] = expert < p.experts
            ? float(scores[ulong(token) * p.experts + expert])
            : -INFINITY;
    }

    float best_scores[4];
    uint best_experts[4];
    for (uint position = 0; position < 4; ++position) {
        float candidate_score = lane_scores[0];
        uint candidate_expert = lane_experts[0];
        for (uint item = 1; item < 20; ++item) {
            if (lane_scores[item] > candidate_score
                || (lane_scores[item] == candidate_score
                    && lane_experts[item] < candidate_expert)) {
                candidate_score = lane_scores[item];
                candidate_expert = lane_experts[item];
            }
        }
        const float best_score = simd_max(candidate_score);
        const uint best_expert = simd_min(
            candidate_score == best_score ? candidate_expert : UINT_MAX);
        best_scores[position] = best_score;
        best_experts[position] = best_expert;
        for (uint item = 0; item < 20; ++item) {
            if (lane_experts[item] == best_expert) lane_scores[item] = -INFINITY;
        }
    }

    if (lane == 0) {
        const float top3_denominator = route_sigmoid(best_scores[0])
            + route_sigmoid(best_scores[1]) + route_sigmoid(best_scores[2]);
        for (uint position = 0; position < p.top_k; ++position) {
            const ulong route = ulong(token) * p.top_k + position;
            route_experts[route] = best_experts[position];
            route_weights[route] =
                half(route_sigmoid(best_scores[position]) / top3_denominator);
        }
        route_margin[token] = best_scores[2] - best_scores[3];
    }
}

// Count each assignment once and record its stable local rank in the same pass.
kernel void fbt_moe_route_histogram(
    device const uint* route_experts [[buffer(0)]],
    device uint* block_histogram [[buffer(1)]],
    device uint* packed_rows [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint counts[640];
    if (tid < p.experts) counts[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint first = block * p.block_tokens * p.top_k;
        const uint end = min(first + p.block_tokens * p.top_k, p.rows * p.top_k);
        for (uint assignment = first; assignment < end; ++assignment) {
            const uint expert = route_experts[assignment];
            packed_rows[assignment] = counts[expert]++;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.experts)
        block_histogram[ulong(block) * p.experts + tid] = counts[tid];
}

// Stable exclusive prefixes yield exact variable expert segments, with no
// capacity or dropped-token policy hidden in allocation assumptions.
kernel void fbt_moe_route_prefix(
    device const uint* block_histogram [[buffer(0)]],
    device uint* block_offsets [[buffer(1)]],
    device uint* expert_offsets [[buffer(2)]],
    device uint* expert_loads [[buffer(3)]],
    device RoutedTile* routed_tiles [[buffer(4)]],
    device uint* routed_dispatch [[buffer(5)]],
    constant Top3RouteShape& p [[buffer(6)]],
    uint expert [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint2 group_totals[20];
    uint expert_load = 0;
    if (expert < p.experts) {
        for (uint block = 0; block < p.blocks; ++block) {
            const ulong index = ulong(block) * p.experts + expert;
            block_offsets[index] = expert_load;
            expert_load += block_histogram[index];
        }
    }
#ifdef TALL_ROUTED_ROWS
    constexpr uint tile_rows = 128;
#else
    constexpr uint tile_rows = 64;
#endif
    const uint tiles = (expert_load + tile_rows - 1) / tile_rows;
    uint expert_offset = simd_prefix_exclusive_sum(expert_load);
    uint tile_offset = simd_prefix_exclusive_sum(tiles);
    const uint2 total = uint2(simd_sum(expert_load), simd_sum(tiles));
    if (lane == 0) group_totals[simdgroup] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = 0; group < simdgroup; ++group) {
        expert_offset += group_totals[group].x;
        tile_offset += group_totals[group].y;
    }
    if (expert < p.experts) {
        expert_offsets[expert] = expert_offset;
        expert_loads[expert] = expert_load;
        for (uint block = 0; block < p.blocks; ++block)
            block_offsets[ulong(block) * p.experts + expert] += expert_offset;
        for (uint local = 0; local < tiles; ++local) {
            const uint consumed = local * tile_rows;
            routed_tiles[tile_offset + local] = RoutedTile {
                expert, expert_offset + consumed, min(tile_rows, expert_load - consumed), 0u
            };
        }
    }
    if (expert + 1 == p.experts) {
#ifdef MANUAL_ROUTED_GATE
        routed_dispatch[0] = (216 + 63) / 64;
#elif defined(ALL_ROUTED_GATE_COLUMNS)
        routed_dispatch[0] = 1;
#elif defined(FUSED_ROUTED_GATE_COLUMNS)
        routed_dispatch[0] = 2;
#elif defined(WIDE_ROUTED_GATE)
        routed_dispatch[0] = (216 + 127) / 128;
#else
        routed_dispatch[0] = (216 + 63) / 64;
#endif
        routed_dispatch[1] = tile_offset + tiles;
        routed_dispatch[2] = 1;
#ifdef FUSED_ROUTED_DOWN_COLUMNS
        routed_dispatch[3] = (p.width + 127) / 128;
#elif defined(WIDE_ROUTED_DOWN)
        routed_dispatch[3] = (p.width + 127) / 128;
#else
        routed_dispatch[3] = (p.width + 63) / 64;
#endif
        routed_dispatch[4] = tile_offset + tiles;
        routed_dispatch[5] = 1;
    }
}

// Add the global segment prefix to the already computed stable local rank.
kernel void fbt_moe_route_rank(
    device const uint* route_experts [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    device uint* packed_rows [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint assignment [[thread_position_in_grid]]) {
    if (assignment >= p.rows * p.top_k) return;
    const uint block = assignment / (p.block_tokens * p.top_k);
    const uint expert = route_experts[assignment];
    packed_rows[assignment] += block_offsets[ulong(block) * p.experts + expert];
}

// Materialize stable expert-contiguous token rows and retain token/route
// metadata for weighted output recombination after the expert kernels.
kernel void fbt_moe_route_pack(
    device const half* input [[buffer(0)]],
    device const uint* route_experts [[buffer(1)]],
    device const half* route_weights [[buffer(2)]],
    device const uint* packed_rows [[buffer(3)]],
    device half* packed_input [[buffer(4)]],
    device uint* packed_tokens [[buffer(5)]],
    device uint* packed_experts [[buffer(6)]],
    device half* packed_weights [[buffer(7)]],
    constant Top3RouteShape& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.rows * p.top_k;
    const ulong elements = ulong(total) * p.width;
    if (gid >= elements) return;
    const uint assignment = gid / p.width;
    const uint column = gid % p.width;
    const uint token = assignment / p.top_k;
    const uint packed = packed_rows[assignment];
    packed_input[ulong(packed) * p.width + column] = input[ulong(token) * p.width + column];
    if (column == 0) {
        packed_tokens[packed] = token;
        packed_experts[packed] = route_experts[assignment];
        packed_weights[packed] = route_weights[assignment];
    }
}

// Fuse the global expert-segment offset into a row-oriented, vectorized pack.
// Each threadgroup owns one assignment, so its rank and metadata are computed
// once while 128 threads copy the 512-wide token as half4 vectors.
kernel void fbt_moe_route_pack_rows(
    device const half* input [[buffer(0)]],
    device const uint* route_experts [[buffer(1)]],
    device const half* route_weights [[buffer(2)]],
    device uint* packed_rows [[buffer(3)]],
    device const uint* block_offsets [[buffer(4)]],
    device half* packed_input [[buffer(5)]],
    device uint* packed_tokens [[buffer(6)]],
    device uint* packed_experts [[buffer(7)]],
    device half* packed_weights [[buffer(8)]],
    constant Top3RouteShape& p [[buffer(9)]],
    uint assignment [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (assignment >= p.rows * p.top_k) return;
    threadgroup uint packed_row;
    const uint expert = route_experts[assignment];
    if (tid == 0) {
        const uint block = assignment / (p.block_tokens * p.top_k);
        packed_row = packed_rows[assignment]
            + block_offsets[ulong(block) * p.experts + expert];
        // The combine kernel consumes this assignment-to-global-row map.
        // Persist the fused rank result instead of leaving the block-local rank.
        packed_rows[assignment] = packed_row;
        packed_tokens[packed_row] = assignment / p.top_k;
        packed_experts[packed_row] = expert;
        packed_weights[packed_row] = route_weights[assignment];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint token = assignment / p.top_k;
    const uint vectors = p.width / 4;
    if (tid < vectors) {
        *reinterpret_cast<device half4*>(packed_input + ulong(packed_row) * p.width + tid * 4) =
            *reinterpret_cast<device const half4*>(input + ulong(token) * p.width + tid * 4);
    }
}

// Gather instead of atomically scattering: every token-column pair owns one
// output and reads its three routed assignments in their original route order.
kernel void fbt_moe_combine(
    device const half* route_weights [[buffer(0)]],
    device const uint* packed_rows [[buffer(1)]],
    device const half* routed_output [[buffer(2)]],
    device const half* shared_output [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    float value = float(shared_output[gid]);
    for (uint route = 0; route < p.top_k; ++route) {
        const ulong assignment = ulong(token) * p.top_k + route;
        const uint packed = packed_rows[assignment];
        value = fma(float(route_weights[assignment]),
                    float(routed_output[ulong(packed) * p.width + column]), value);
    }
    output[gid] = half(value);
}

// Row-oriented equivalent of fbt_moe_combine. Route metadata is invariant
// across all 512 columns, so load it once per token and let each lane combine
// four adjacent columns in the same route/FMA order as the scalar kernel.
kernel void fbt_moe_combine_rows(
    device const half* route_weights [[buffer(0)]],
    device const uint* packed_rows [[buffer(1)]],
    device const half* routed_output [[buffer(2)]],
    device const half* shared_output [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (token >= p.rows) return;
    threadgroup uint rows[3];
    threadgroup float weights[3];
    if (tid < p.top_k) {
        const ulong assignment = ulong(token) * p.top_k + tid;
        rows[tid] = packed_rows[assignment];
        weights[tid] = float(route_weights[assignment]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint column = tid * 4;
    if (column + 3 < p.width) {
        const ulong destination = ulong(token) * p.width + column;
        float4 value = float4(
            *reinterpret_cast<device const half4*>(shared_output + destination));
        for (uint route = 0; route < p.top_k; ++route) {
            const ulong source = ulong(rows[route]) * p.width + column;
            const float4 routed = float4(
                *reinterpret_cast<device const half4*>(routed_output + source));
            value = fma(float4(weights[route]), routed, value);
        }
        *reinterpret_cast<device half4*>(output + destination) = half4(value);
    }
}

// Preserve the existing half-rounding boundaries while avoiding the
// materialized combined branch and a second row-wide dispatch.
kernel void fbt_moe_combine_residual_rms(
    device const half* route_weights [[buffer(0)]],
    device const uint* packed_rows [[buffer(1)]],
    device const half* routed_output [[buffer(2)]],
    device const half* shared_output [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device const half* norm [[buffer(5)]],
    device half* output [[buffer(6)]],
    device half* normalized [[buffer(7)]],
    constant Top3RouteShape& p [[buffer(8)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (token >= p.rows) return;
    threadgroup uint rows[3];
    threadgroup float weights[3];
    threadgroup float partial[4];
    threadgroup float scale;
    if (tid < p.top_k) {
        const ulong assignment = ulong(token) * p.top_k + tid;
        rows[tid] = packed_rows[assignment];
        weights[tid] = float(route_weights[assignment]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong base = ulong(token) * p.width;
    float state[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        float branch = float(shared_output[base + column]);
        #pragma unroll
        for (uint route = 0; route < p.top_k; ++route) {
            const ulong source = ulong(rows[route]) * p.width + column;
            branch = fma(weights[route], float(routed_output[source]), branch);
        }
        const half rounded_branch = half(branch);
        const half rounded_state = half(
            float(residual[base + column]) + float(rounded_branch));
        state[item] = float(rounded_state);
        squared = fma(state[item], state[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        output[base + column] = half(state[item]);
        normalized[base + column] =
            half(state[item] * scale * float(norm[column]));
    }
}
