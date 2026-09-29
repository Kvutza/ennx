#include <metal_stdlib>
using namespace metal;

// Routing scratch shape for the 128-expert, top-3 path. Routing is token-choice:
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

METAL_FUNC float route_sigmoid(float value) {
    const float e = exp(-abs(value));
    const float probability = value >= 0.0f
        ? 1.0f / (1.0f + e)
        : e / (1.0f + e);
    return max(probability, 1.0e-20f);
}

kernel void fbt_moe_select_top3(
    device const half* scores [[buffer(0)]],
    device uint* route_experts [[buffer(1)]],
    device half* route_weights [[buffer(2)]],
    device float* route_margin [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint expert [[thread_index_in_threadgroup]]) {
    const uint token = group.y;
    if (token >= p.rows || expert >= p.experts) return;

    threadgroup float logits[128];
    logits[expert] = float(scores[ulong(token) * p.experts + expert]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (expert == 0) {
        float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint best_id[4] = {p.experts, p.experts, p.experts, p.experts};
        for (uint candidate = 0; candidate < p.experts; ++candidate) {
            const float value = logits[candidate];
            for (uint position = 0; position < 4; ++position) {
                if (value > best[position] ||
                    (value == best[position] && candidate < best_id[position])) {
                    for (uint shift = 3; shift > position; --shift) {
                        best[shift] = best[shift - 1];
                        best_id[shift] = best_id[shift - 1];
                    }
                    best[position] = value;
                    best_id[position] = candidate;
                    break;
                }
            }
        }
        const float top3_denominator = route_sigmoid(best[0]) + route_sigmoid(best[1])
            + route_sigmoid(best[2]);
        for (uint position = 0; position < p.top_k; ++position) {
            const ulong route = ulong(token) * p.top_k + position;
            route_experts[route] = best_id[position];
            route_weights[route] = half(route_sigmoid(best[position]) / top3_denominator);
        }
        route_margin[token] = best[2] - best[3];
    }
}

// One deterministic histogram per token block and expert. The fixed block is
// only an indexing aid; no expert has a fixed row capacity.
kernel void fbt_moe_route_histogram(
    device const uint* route_experts [[buffer(0)]],
    device uint* block_histogram [[buffer(1)]],
    constant Top3RouteShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint block = gid / p.experts;
    const uint expert = gid % p.experts;
    if (block >= p.blocks) return;
    const uint first_token = block * p.block_tokens;
    const uint end_token = min(first_token + p.block_tokens, p.rows);
    uint count = 0;
    for (uint token = first_token; token < end_token; ++token)
        for (uint route = 0; route < p.top_k; ++route)
            count += uint(route_experts[ulong(token) * p.top_k + route] == expert);
    block_histogram[ulong(block) * p.experts + expert] = count;
}

// Stable exclusive prefixes yield exact variable expert segments, with no
// capacity or dropped-token policy hidden in allocation assumptions.
kernel void fbt_moe_route_prefix(
    device const uint* block_histogram [[buffer(0)]],
    device uint* block_offsets [[buffer(1)]],
    device uint* expert_offsets [[buffer(2)]],
    device uint* expert_loads [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= p.experts) return;
    uint expert_offset = 0;
    for (uint previous = 0; previous < expert; ++previous)
        for (uint block = 0; block < p.blocks; ++block)
            expert_offset += block_histogram[ulong(block) * p.experts + previous];
    expert_offsets[expert] = expert_offset;
    uint expert_load = 0;
    for (uint block = 0; block < p.blocks; ++block) {
        const ulong index = ulong(block) * p.experts + expert;
        block_offsets[index] = expert_offset;
        const uint count = block_histogram[index];
        expert_offset += count;
        expert_load += count;
    }
    expert_loads[expert] = expert_load;
}

// Compute each assignment's stable slot within its token block and expert.
kernel void fbt_moe_route_rank(
    device const uint* route_experts [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    device uint* packed_rows [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint assignment [[thread_position_in_grid]]) {
    const uint total = p.rows * p.top_k;
    if (assignment >= total) return;
    const uint token = assignment / p.top_k;
    const uint route = assignment % p.top_k;
    const uint expert = route_experts[assignment];
    const uint block = token / p.block_tokens;
    const uint first_token = block * p.block_tokens;
    uint local_rank = 0;
    for (uint previous_token = first_token; previous_token < token; ++previous_token)
        for (uint previous_route = 0; previous_route < p.top_k; ++previous_route)
            local_rank += uint(route_experts[ulong(previous_token) * p.top_k + previous_route] == expert);
    for (uint previous_route = 0; previous_route < route; ++previous_route)
        local_rank += uint(route_experts[ulong(token) * p.top_k + previous_route] == expert);
    packed_rows[assignment] = block_offsets[ulong(block) * p.experts + expert] + local_rank;
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
