#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint kContext = 4096;
constant uint kRows = 8192;
constant uint kQueryHeads = 8;
constant uint kHeadDim = 64;
constant uint kQueryWidth = kQueryHeads * kHeadDim;
constant uint kQkvWidth = kQueryWidth + 2 * kHeadDim;
constant uint kBlock = 64;
constant uint kLeaves = kContext / kBlock;
constant uint kNodes = 2 * kLeaves - 1;
constant uint kSelected = 8;

kernel void fbt_pisa1_leaf_means(
    device const half* qkv [[buffer(0)]],
    device half* pyramid [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    const uint leaf = group.x;
    const uint sequence = group.y;
    if (leaf >= kLeaves || sequence >= 2 || dim >= kHeadDim) return;
    float sum = 0.0f;
    const uint first = sequence * kContext + leaf * kBlock;
    for (uint token = 0; token < kBlock; ++token)
        sum += float(qkv[ulong(first + token) * kQkvWidth + kQueryWidth + dim]);
    pyramid[(ulong(sequence) * kNodes + leaf) * kHeadDim + dim] = half(sum / float(kBlock));
}

kernel void fbt_pisa1_upper_means(
    device half* pyramid [[buffer(0)]],
    uint sequence [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    if (sequence >= 2 || dim >= kHeadDim) return;
    uint child_offset = 0;
    uint parent_offset = kLeaves;
    uint child_count = kLeaves;
    const ulong base = ulong(sequence) * kNodes * kHeadDim;
    while (child_count > 1) {
        const uint parent_count = child_count / 2;
        for (uint parent = 0; parent < parent_count; ++parent) {
            const ulong left = base + ulong(child_offset + 2 * parent) * kHeadDim + dim;
            const ulong right = left + kHeadDim;
            pyramid[base + ulong(parent_offset + parent) * kHeadDim + dim] =
                half(0.5f * (float(pyramid[left]) + float(pyramid[right])));
        }
        threadgroup_barrier(mem_flags::mem_device);
        child_offset = parent_offset;
        parent_offset += parent_count;
        child_count = parent_count;
    }
}

inline bool contains_leaf(uint node, uint level, uint leaf) {
    const uint begin = node << level;
    return leaf >= begin && leaf < begin + (1u << level);
}

kernel void fbt_pisa1_select(
    device const half* qkv [[buffer(0)]],
    device const half* pyramid [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float query[kHeadDim];
    threadgroup float scores[16];
    threadgroup uint candidates[16];
    threadgroup uint chosen[kSelected];
    threadgroup uint expanded[16];
    if (row >= kRows) return;

    for (uint dim = lane; dim < kHeadDim; dim += 32) {
        float sum = 0.0f;
        for (uint head = 0; head < kQueryHeads; ++head)
            sum += float(qkv[ulong(row) * kQkvWidth + head * kHeadDim + dim]);
        query[dim] = sum;
    }
    if (lane < 16) candidates[lane] = lane;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint position = row % kContext;
    const uint current = position / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint sequence = row / kContext;
    for (int level = 2; level >= 0; --level) {
        const uint offset = level == 2 ? 96 : (level == 1 ? 64 : 0);
        if (lane < 16) {
            const uint node = candidates[lane];
            float score = -INFINITY;
            if (node != UINT_MAX) {
                const uint span = 1u << uint(level);
                const uint last = (node + 1) * span - 1;
                const bool forced = contains_leaf(node, uint(level), 0)
                    || contains_leaf(node, uint(level), previous)
                    || contains_leaf(node, uint(level), current);
                if (forced) {
                    score = INFINITY;
                } else if (last < current) {
                    score = 0.0f;
                    const ulong summary =
                        (ulong(sequence) * kNodes + offset + node) * kHeadDim;
                    for (uint dim = 0; dim < kHeadDim; ++dim)
                        score = fma(query[dim], float(pyramid[summary + dim]), score);
                }
            }
            scores[lane] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (uint slot = 0; slot < kSelected; ++slot) {
                uint best = UINT_MAX;
                float best_score = -INFINITY;
                for (uint candidate = 0; candidate < 16; ++candidate) {
                    const float score = scores[candidate];
                    if (score > best_score
                        || (score == best_score && score > -INFINITY
                            && candidates[candidate] < candidates[best])) {
                        best = candidate;
                        best_score = score;
                    }
                }
                chosen[slot] = best == UINT_MAX ? UINT_MAX : candidates[best];
                if (best != UINT_MAX) scores[best] = -INFINITY;
            }
            if (level > 0) {
                for (uint slot = 0; slot < kSelected; ++slot) {
                    const uint node = chosen[slot];
                    expanded[2 * slot] = node == UINT_MAX ? UINT_MAX : 2 * node;
                    expanded[2 * slot + 1] = node == UINT_MAX ? UINT_MAX : 2 * node + 1;
                }
            } else {
                for (uint slot = 0; slot < kSelected; ++slot)
                    blocks[ulong(row) * kSelected + slot] = chosen[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (level > 0 && lane < 16) candidates[lane] = expanded[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void fbt_pisa1_attention(
    device const half* qkv [[buffer(0)]],
    device const uint* blocks [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[kQueryHeads][kBlock + 1];
    threadgroup half probabilities[kQueryHeads][kBlock + 1];
    threadgroup float scratch[kQueryHeads][kHeadDim];
    threadgroup float maxima[kQueryHeads];
    threadgroup float totals[kQueryHeads];
    threadgroup float alphas[kQueryHeads];
    if (row >= kRows) return;

    if (tid < kQueryHeads) {
        maxima[tid] = -INFINITY;
        totals[tid] = 0.0f;
    }
    simdgroup_float8x8 result[2] = {
        simdgroup_float8x8(0.0f),
        simdgroup_float8x8(0.0f),
    };
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sequence = row / kContext;
    const uint position = row % kContext;
    for (uint selected = 0; selected < kSelected; ++selected) {
        const uint block = blocks[ulong(row) * kSelected + selected];
        if (block == UINT_MAX) continue;
        for (uint tile = simdgroup; tile < 8; tile += 4) {
            simdgroup_float8x8 value(0.0f);
            for (uint dim = 0; dim < kHeadDim; dim += 8) {
                simdgroup_half8x8 q;
                simdgroup_half8x8 k;
                simdgroup_load(
                    q,
                    qkv + ulong(row) * kQkvWidth + dim,
                    kHeadDim,
                    ulong2(0, 0));
                const uint source = sequence * kContext + block * kBlock + tile * 8;
                simdgroup_load(
                    k,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + dim,
                    kQkvWidth,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(value, q, k, value);
            }
            simdgroup_store(value, &scores[0][tile * 8], kBlock + 1, ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint head = simdgroup; head < kQueryHeads; head += 4) {
            const uint token0 = lane;
            const uint token1 = lane + 32;
            const bool valid0 = block * kBlock + token0 <= position;
            const bool valid1 = block * kBlock + token1 <= position;
            const float value0 = valid0 ? scores[head][token0] * 0.125f : -INFINITY;
            const float value1 = valid1 ? scores[head][token1] * 0.125f : -INFINITY;
            const float next_max = max(maxima[head], simd_max(max(value0, value1)));
            const float alpha = exp(maxima[head] - next_max);
            const float probability0 = valid0 ? exp(value0 - next_max) : 0.0f;
            const float probability1 = valid1 ? exp(value1 - next_max) : 0.0f;
            const float total = totals[head] * alpha + simd_sum(probability0 + probability1);
            probabilities[head][token0] = half(probability0);
            probabilities[head][token1] = half(probability1);
            if (lane == 0) {
                maxima[head] = next_max;
                totals[head] = total;
                alphas[head] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local = 0; local < 2; ++local) {
            const uint tile = simdgroup + 4 * local;
            simdgroup_store(
                result[local],
                &scratch[0][tile * 8],
                kHeadDim,
                ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < kQueryWidth; index += 128)
            scratch[index / kHeadDim][index % kHeadDim] *= alphas[index / kHeadDim];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local = 0; local < 2; ++local) {
            const uint tile = simdgroup + 4 * local;
            simdgroup_load(
                result[local],
                &scratch[0][tile * 8],
                kHeadDim,
                ulong2(0, 0));
            for (uint token = 0; token < kBlock; token += 8) {
                simdgroup_half8x8 probability;
                simdgroup_half8x8 v;
                simdgroup_load(
                    probability,
                    &probabilities[0][token],
                    kBlock + 1,
                    ulong2(0, 0));
                const uint source = sequence * kContext + block * kBlock + token;
                simdgroup_load(
                    v,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + kHeadDim + tile * 8,
                    kQkvWidth,
                    ulong2(0, 0));
                simdgroup_multiply_accumulate(
                    result[local],
                    probability,
                    v,
                    result[local]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint local = 0; local < 2; ++local) {
        const uint tile = simdgroup + 4 * local;
        simdgroup_store(result[local], &scratch[0][tile * 8], kHeadDim, ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < kQueryWidth; index += 128)
        output[ulong(row) * kQueryWidth + index] =
            half(scratch[index / kHeadDim][index % kHeadDim] / totals[index / kHeadDim]);
}

kernel void fbt_pisa1_select_attention(
    device const half* qkv [[buffer(0)]],
    device const half* pyramid [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float route_query[kHeadDim];
    threadgroup float route_scores[16];
    threadgroup uint route_candidates[16];
    threadgroup uint selected_blocks[kSelected];
    threadgroup uint route_expanded[16];
    threadgroup float scores[kQueryHeads][kBlock + 1];
    threadgroup half probabilities[kQueryHeads][kBlock + 1];
    threadgroup float scratch[kQueryHeads][kHeadDim];
    threadgroup float maxima[kQueryHeads];
    threadgroup float totals[kQueryHeads];
    threadgroup float alphas[kQueryHeads];
    if (row >= kRows) return;

    if (tid < kHeadDim) {
        float sum = 0.0f;
        for (uint head = 0; head < kQueryHeads; ++head)
            sum += float(qkv[ulong(row) * kQkvWidth + head * kHeadDim + tid]);
        route_query[tid] = sum;
    }
    if (tid < 16) route_candidates[tid] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint position = row % kContext;
    const uint current = position / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint sequence = row / kContext;
    for (int level = 2; level >= 0; --level) {
        const uint offset = level == 2 ? 96 : (level == 1 ? 64 : 0);
        if (tid < 16) {
            const uint node = route_candidates[tid];
            float score = -INFINITY;
            if (node != UINT_MAX) {
                const uint span = 1u << uint(level);
                const uint last = (node + 1) * span - 1;
                const bool forced = contains_leaf(node, uint(level), 0)
                    || contains_leaf(node, uint(level), previous)
                    || contains_leaf(node, uint(level), current);
                if (forced) {
                    score = INFINITY;
                } else if (last < current) {
                    score = 0.0f;
                    const ulong summary =
                        (ulong(sequence) * kNodes + offset + node) * kHeadDim;
                    for (uint dim = 0; dim < kHeadDim; ++dim)
                        score = fma(route_query[dim], float(pyramid[summary + dim]), score);
                }
            }
            route_scores[tid] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint slot = 0; slot < kSelected; ++slot) {
                uint best = UINT_MAX;
                float best_score = -INFINITY;
                for (uint candidate = 0; candidate < 16; ++candidate) {
                    const float score = route_scores[candidate];
                    if (score > best_score
                        || (score == best_score && score > -INFINITY
                            && route_candidates[candidate] < route_candidates[best])) {
                        best = candidate;
                        best_score = score;
                    }
                }
                selected_blocks[slot] =
                    best == UINT_MAX ? UINT_MAX : route_candidates[best];
                if (best != UINT_MAX) route_scores[best] = -INFINITY;
            }
            if (level > 0) {
                for (uint slot = 0; slot < kSelected; ++slot) {
                    const uint node = selected_blocks[slot];
                    route_expanded[2 * slot] =
                        node == UINT_MAX ? UINT_MAX : 2 * node;
                    route_expanded[2 * slot + 1] =
                        node == UINT_MAX ? UINT_MAX : 2 * node + 1;
                }
            } else {
                for (uint slot = 0; slot < kSelected; ++slot)
                    blocks[ulong(row) * kSelected + slot] = selected_blocks[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (level > 0 && tid < 16) route_candidates[tid] = route_expanded[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < kQueryHeads) {
        maxima[tid] = -INFINITY;
        totals[tid] = 0.0f;
    }
    simdgroup_float8x8 result[2] = {
        simdgroup_float8x8(0.0f),
        simdgroup_float8x8(0.0f),
    };
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint selected = 0; selected < kSelected; ++selected) {
        const uint block = selected_blocks[selected];
        if (block == UINT_MAX) continue;
        for (uint tile = simdgroup; tile < 8; tile += 4) {
            simdgroup_float8x8 value(0.0f);
            for (uint dim = 0; dim < kHeadDim; dim += 8) {
                simdgroup_half8x8 q;
                simdgroup_half8x8 k;
                simdgroup_load(
                    q,
                    qkv + ulong(row) * kQkvWidth + dim,
                    kHeadDim,
                    ulong2(0, 0));
                const uint source = sequence * kContext + block * kBlock + tile * 8;
                simdgroup_load(
                    k,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + dim,
                    kQkvWidth,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(value, q, k, value);
            }
            simdgroup_store(value, &scores[0][tile * 8], kBlock + 1, ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint head = simdgroup; head < kQueryHeads; head += 4) {
            const uint token0 = lane;
            const uint token1 = lane + 32;
            const bool valid0 = block * kBlock + token0 <= position;
            const bool valid1 = block * kBlock + token1 <= position;
            const float value0 = valid0 ? scores[head][token0] * 0.125f : -INFINITY;
            const float value1 = valid1 ? scores[head][token1] * 0.125f : -INFINITY;
            const float next_max = max(maxima[head], simd_max(max(value0, value1)));
            const float alpha = exp(maxima[head] - next_max);
            const float probability0 = valid0 ? exp(value0 - next_max) : 0.0f;
            const float probability1 = valid1 ? exp(value1 - next_max) : 0.0f;
            const float total = totals[head] * alpha + simd_sum(probability0 + probability1);
            probabilities[head][token0] = half(probability0);
            probabilities[head][token1] = half(probability1);
            if (lane == 0) {
                maxima[head] = next_max;
                totals[head] = total;
                alphas[head] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local = 0; local < 2; ++local) {
            const uint tile = simdgroup + 4 * local;
            simdgroup_store(
                result[local],
                &scratch[0][tile * 8],
                kHeadDim,
                ulong2(0, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < kQueryWidth; index += 128)
            scratch[index / kHeadDim][index % kHeadDim] *= alphas[index / kHeadDim];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local = 0; local < 2; ++local) {
            const uint tile = simdgroup + 4 * local;
            simdgroup_load(
                result[local],
                &scratch[0][tile * 8],
                kHeadDim,
                ulong2(0, 0));
            for (uint token = 0; token < kBlock; token += 8) {
                simdgroup_half8x8 probability;
                simdgroup_half8x8 v;
                simdgroup_load(
                    probability,
                    &probabilities[0][token],
                    kBlock + 1,
                    ulong2(0, 0));
                const uint source = sequence * kContext + block * kBlock + token;
                simdgroup_load(
                    v,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + kHeadDim + tile * 8,
                    kQkvWidth,
                    ulong2(0, 0));
                simdgroup_multiply_accumulate(
                    result[local],
                    probability,
                    v,
                    result[local]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint local = 0; local < 2; ++local) {
        const uint tile = simdgroup + 4 * local;
        simdgroup_store(result[local], &scratch[0][tile * 8], kHeadDim, ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < kQueryWidth; index += 128)
        output[ulong(row) * kQueryWidth + index] =
            half(scratch[index / kHeadDim][index % kHeadDim] / totals[index / kHeadDim]);
}

kernel void fbt_pisa1_select_attention_q4(
    device const half* qkv [[buffer(0)]],
    device const half* pyramid [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint query_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float route_query[4][kHeadDim];
    threadgroup uint selected_blocks[4][kSelected];
    threadgroup float scores[4][kQueryHeads][kBlock + 1];
    threadgroup half probabilities[4][kQueryHeads][kBlock + 1];
    threadgroup float maxima[4][kQueryHeads];
    threadgroup float totals[4][kQueryHeads];
    threadgroup float alphas[4][kQueryHeads];

    const uint row = query_tile * 4 + simdgroup;
    if (row >= kRows) return;

    for (uint dim = lane; dim < kHeadDim; dim += 32) {
        float sum = 0.0f;
        for (uint head = 0; head < kQueryHeads; ++head)
            sum += float(qkv[ulong(row) * kQkvWidth + head * kHeadDim + dim]);
        route_query[simdgroup][dim] = sum;
    }
    uint node = lane < 16 ? lane : UINT_MAX;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint position = row % kContext;
    const uint current = position / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint sequence = row / kContext;
    for (int level = 2; level >= 0; --level) {
        const uint offset = level == 2 ? 96 : (level == 1 ? 64 : 0);
        float score = -INFINITY;
        if (lane < 16) {
            if (node != UINT_MAX) {
                const uint span = 1u << uint(level);
                const uint last = (node + 1) * span - 1;
                const bool forced = contains_leaf(node, uint(level), 0)
                    || contains_leaf(node, uint(level), previous)
                    || contains_leaf(node, uint(level), current);
                if (forced) {
                    score = INFINITY;
                } else if (last < current) {
                    score = 0.0f;
                    const ulong summary =
                        (ulong(sequence) * kNodes + offset + node) * kHeadDim;
                    for (uint dim = 0; dim < kHeadDim; ++dim)
                        score = fma(
                            route_query[simdgroup][dim],
                            float(pyramid[summary + dim]),
                            score);
                }
            }
        }
        for (uint slot = 0; slot < kSelected; ++slot) {
            const float best_score = simd_max(score);
            const uint best = simd_min(
                score == best_score && score > -INFINITY ? node : UINT_MAX);
            if (lane == slot) {
                selected_blocks[simdgroup][slot] = best;
                if (level == 0) blocks[ulong(row) * kSelected + slot] = best;
            }
            if (node == best) score = -INFINITY;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (level > 0 && lane < 16) {
            const uint parent = selected_blocks[simdgroup][lane / 2];
            node = parent == UINT_MAX ? UINT_MAX : 2 * parent + lane % 2;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < kQueryHeads) {
        maxima[simdgroup][lane] = -INFINITY;
        totals[simdgroup][lane] = 0.0f;
    }
    simdgroup_float8x8 result[8];
    for (uint tile = 0; tile < 8; ++tile) result[tile] = simdgroup_float8x8(0.0f);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint selected = 0; selected < kSelected; ++selected) {
        const uint block = selected_blocks[simdgroup][selected];
        if (block == UINT_MAX) continue;
        for (uint tile = 0; tile < 8; ++tile) {
            simdgroup_float8x8 value(0.0f);
            for (uint dim = 0; dim < kHeadDim; dim += 8) {
                simdgroup_half8x8 q;
                simdgroup_half8x8 k;
                simdgroup_load(
                    q,
                    qkv + ulong(row) * kQkvWidth + dim,
                    kHeadDim,
                    ulong2(0, 0));
                const uint source = sequence * kContext + block * kBlock + tile * 8;
                simdgroup_load(
                    k,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + dim,
                    kQkvWidth,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(value, q, k, value);
            }
            simdgroup_store(
                value,
                &scores[simdgroup][0][tile * 8],
                kBlock + 1,
                ulong2(0, 0));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint head = 0; head < kQueryHeads; ++head) {
            const uint token0 = lane;
            const uint token1 = lane + 32;
            const bool valid0 = block * kBlock + token0 <= position;
            const bool valid1 = block * kBlock + token1 <= position;
            const float value0 =
                valid0 ? scores[simdgroup][head][token0] * 0.125f : -INFINITY;
            const float value1 =
                valid1 ? scores[simdgroup][head][token1] * 0.125f : -INFINITY;
            const float next_max =
                max(maxima[simdgroup][head], simd_max(max(value0, value1)));
            const float alpha = exp(maxima[simdgroup][head] - next_max);
            const float probability0 = valid0 ? exp(value0 - next_max) : 0.0f;
            const float probability1 = valid1 ? exp(value1 - next_max) : 0.0f;
            const float total = totals[simdgroup][head] * alpha
                + simd_sum(probability0 + probability1);
            probabilities[simdgroup][head][token0] = half(probability0);
            probabilities[simdgroup][head][token1] = half(probability1);
            if (lane == 0) {
                maxima[simdgroup][head] = next_max;
                totals[simdgroup][head] = total;
                alphas[simdgroup][head] = alpha;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = lane; index < 64; index += 32) {
            const uint head = index / 8;
            const uint column = index % 8;
            scores[simdgroup][head][column] =
                head == column ? alphas[simdgroup][head] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 alpha;
        simdgroup_load(
            alpha,
            &scores[simdgroup][0][0],
            kBlock + 1,
            ulong2(0, 0));
        for (uint tile = 0; tile < 8; ++tile) {
            simdgroup_float8x8 scaled;
            simdgroup_multiply_accumulate(
                scaled,
                alpha,
                result[tile],
                simdgroup_float8x8(0.0f));
            result[tile] = scaled;
        }
        for (uint token = 0; token < kBlock; token += 8) {
            simdgroup_half8x8 probability;
            simdgroup_load(
                probability,
                &probabilities[simdgroup][0][token],
                kBlock + 1,
                ulong2(0, 0));
            for (uint tile = 0; tile < 8; ++tile) {
                simdgroup_half8x8 v;
                const uint source = sequence * kContext + block * kBlock + token;
                simdgroup_load(
                    v,
                    qkv + ulong(source) * kQkvWidth + kQueryWidth + kHeadDim + tile * 8,
                    kQkvWidth,
                    ulong2(0, 0));
                simdgroup_multiply_accumulate(result[tile], probability, v, result[tile]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tile = 0; tile < 8; ++tile)
        simdgroup_store(
            result[tile],
            &scores[simdgroup][0][tile * 8],
            kBlock + 1,
            ulong2(0, 0));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < kQueryWidth; index += 32)
        output[ulong(row) * kQueryWidth + index] = half(
            scores[simdgroup][index / kHeadDim][index % kHeadDim]
            / totals[simdgroup][index / kHeadDim]);
}
