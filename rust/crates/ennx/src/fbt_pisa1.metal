#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#ifndef PISA_NO_DIRECT_RESCALE
#define PISA_DIRECT_RESCALE
#endif

#ifndef PISA_NO_REUSE_SCORES
#define PISA_REUSE_SCORES
#endif

#ifndef PISA_NO_STAGE_SHARED_KV
#define PISA_STAGE_SHARED_KV
#endif

// Loop unrolling directive in macro avoiding MSL lambda compiler restrictions
#define PISA_FRAGMENTS(index, step) \
    _Pragma("unroll") \
    for (uint index = 0; index < 8 * step; index += step) {
#define PISA_END_FRAGMENTS }

#ifndef PISA_CONTEXT
#define PISA_CONTEXT 4096
#endif
constant uint kContext = PISA_CONTEXT;
#ifndef PISA_ROWS
#define PISA_ROWS 8192
#endif
constant uint kRows = PISA_ROWS;
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

// Rebuild only leaves intersecting a changed token range. Each leaf retains
// the full 64-key reduction and FP16 store used by fbt_pisa1_leaf_means, so
// this is bit-identical to rebuilding the complete leaf level.
kernel void fbt_pisa1_leaf_range(
    device const half* qkv [[buffer(0)]],
    device half* pyramid [[buffer(1)]],
    constant uint4& range [[buffer(2)]],
    uint leaf_index [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    const uint leaf = range.x + leaf_index;
    const uint sequence = range.z;
    if (leaf_index >= range.y || leaf >= kLeaves || sequence >= 2 || dim >= kHeadDim) return;
    float sum = 0.0f;
    const uint first = sequence * kContext + leaf * kBlock;
    for (uint token = 0; token < kBlock; ++token)
        sum += float(qkv[ulong(first + token) * kQkvWidth + kQueryWidth + dim]);
    pyramid[(ulong(sequence) * kNodes + leaf) * kHeadDim + dim] = half(sum / float(kBlock));
}

// Per-leaf mean K and mean V for the hierarchical kernel.
// Layout: coarse_kv[(sequence * kLeaves + leaf) * 2 * kHeadDim + {0 | kHeadDim} + dim].
kernel void fbt_pisa1_leaf_kv_means(
    device const half* qkv [[buffer(0)]],
    device half* coarse_kv [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    const uint leaf = group.x;
    const uint sequence = group.y;
    if (leaf >= kLeaves || sequence >= 2 || dim >= kHeadDim) return;
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    const uint first = sequence * kContext + leaf * kBlock;
    for (uint token = 0; token < kBlock; ++token) {
        const ulong base = ulong(first + token) * kQkvWidth + kQueryWidth;
        key_sum += float(qkv[base + dim]);
        value_sum += float(qkv[base + kHeadDim + dim]);
    }
    const ulong out = (ulong(sequence) * kLeaves + leaf) * 2 * kHeadDim;
    coarse_kv[out + dim] = half(key_sum / float(kBlock));
    coarse_kv[out + kHeadDim + dim] = half(value_sum / float(kBlock));
}

kernel void fbt_pisa1_upper_range(
    device half* pyramid [[buffer(0)]],
    constant uint4& range [[buffer(1)]],
    uint dim [[thread_index_in_threadgroup]]) {
    if (dim >= kHeadDim) return;
    uint first = range.x;
    uint end = first + range.y;
    uint child_offset = 0;
    uint parent_offset = kLeaves;
    uint child_count = kLeaves;
    const ulong base = ulong(range.z) * kNodes * kHeadDim;
    while (child_count > 1) {
        const uint parent_first = first / 2;
        const uint parent_end = (end + 1) / 2;
        for (uint parent = parent_first; parent < parent_end; ++parent) {
            const ulong left = base + ulong(child_offset + 2 * parent) * kHeadDim + dim;
            const ulong right = left + kHeadDim;
            pyramid[base + ulong(parent_offset + parent) * kHeadDim + dim] =
                half(0.5f * (float(pyramid[left]) + float(pyramid[right])));
        }
        threadgroup_barrier(mem_flags::mem_device);
        first = parent_first;
        end = parent_end;
        child_offset = parent_offset;
        child_count /= 2;
        parent_offset += child_count;
    }
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
    for (int level = int(31 - clz(kLeaves)) - 4; level >= 0; --level) {
        const uint offset = 2 * kLeaves - (2 * kLeaves >> uint(level));
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

// Numerically stable, full-context fallback for a future certified adaptive
// path. One threadgroup owns one query/head; no node is omitted or approximated.
kernel void fbt_pisa1_exact_attention(
    device const half* qkv [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint2& row_range [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scores[kBlock];
    threadgroup float probabilities[kBlock];
    threadgroup float block_maximum;
    threadgroup float block_total;
    threadgroup float running_maximum;
    threadgroup float running_total;
    threadgroup float prior_scale;
    threadgroup float block_scale;
    const uint row = row_range.x + group / kQueryHeads;
    const uint head = group % kQueryHeads;
    if (group / kQueryHeads >= row_range.y || row >= kRows) return;
    const uint sequence = row / kContext;
    const uint position = row % kContext;
    if (lane == 0) {
        running_maximum = -INFINITY;
        running_total = 0.0f;
    }
    float accumulator = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint last_block = position / kBlock;
    for (uint block = 0; block <= last_block; ++block) {
        const uint first = block * kBlock;
        const uint valid = min(kBlock, position - first + 1);
        if (lane < valid) {
            const uint source = sequence * kContext + first + lane;
            float score = 0.0f;
            for (uint dim = 0; dim < kHeadDim; ++dim) {
                const uint query_index = row * kQkvWidth + head * kHeadDim + dim;
                const uint key_index = source * kQkvWidth + kQueryWidth + dim;
                score = fma(float(qkv[query_index]), float(qkv[key_index]), score);
            }
            scores[lane] = score * 0.125f;
        } else {
            scores[lane] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            float value = -INFINITY;
            for (uint token = 0; token < valid; ++token) value = max(value, scores[token]);
            block_maximum = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < valid) probabilities[lane] = exp(scores[lane] - block_maximum);
        else probabilities[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            float value = 0.0f;
            for (uint token = 0; token < valid; ++token) value += probabilities[token];
            block_total = value;
            const float next_maximum = max(running_maximum, block_maximum);
            prior_scale = running_total == 0.0f ? 0.0f : exp(running_maximum - next_maximum);
            block_scale = exp(block_maximum - next_maximum);
            running_total = running_total * prior_scale + block_total * block_scale;
            running_maximum = next_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float block_value = 0.0f;
        for (uint token = 0; token < valid; ++token) {
            const uint source = sequence * kContext + first + token;
            const uint index = source * kQkvWidth + kQueryWidth + kHeadDim + lane;
            block_value = fma(probabilities[token], float(qkv[index]), block_value);
        }
        accumulator = accumulator * prior_scale + block_value * block_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[ulong(row) * kQueryWidth + head * kHeadDim + lane] = half(accumulator / running_total);
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
    for (int level = PISA_CONTEXT == 8192 ? 3 : 2; level >= 0; --level) {
        const uint offset = 2 * kLeaves - (2 * kLeaves >> uint(level));
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

#ifndef PISA_QUERY_TILE
#define PISA_QUERY_TILE 4
#endif

#if defined(PISA_REUSE_SCORES) && defined(PISA_QUAD_SOFTMAX)
#error PISA_REUSE_SCORES requires the head-wise softmax lifetime schedule
#endif

kernel void fbt_pisa1_select_attention_q4(
    device const half* qkv [[buffer(0)]],
    device const half* pyramid [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint query_tile [[threadgroup_position_in_grid]],
#ifdef PISA_DECODE
    constant uint& decode_position [[buffer(4)]],
#else
    constant uint2& row_range [[buffer(4)]],
#endif
#ifdef PISA_CACHE
    device const half* cache [[buffer(5)]],
#endif
#ifdef PISA_EXTERNAL_INDEX
    constant uint& block_size [[buffer(6)]],
#endif
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float route_query[PISA_QUERY_TILE][kHeadDim];
    threadgroup uint selected_blocks[PISA_QUERY_TILE][kSelected];
    threadgroup float scores[PISA_QUERY_TILE][kQueryHeads][kBlock + 1];
#ifdef PISA_REUSE_SCORES
    // Each head keeps its original byte stride. Read all 64 FP32 scores
    // before overwriting their first half with the 64 FP16 probabilities.
    constexpr uint probability_stride = 2 * (kBlock + 1);
    threadgroup half (*probabilities)[kQueryHeads][probability_stride] =
        reinterpret_cast<threadgroup half (*)[kQueryHeads][probability_stride]>(scores);
#else
    constexpr uint probability_stride = kBlock + 1;
    threadgroup half probabilities[PISA_QUERY_TILE][kQueryHeads][kBlock + 1];
#endif
    threadgroup float maxima[PISA_QUERY_TILE][kQueryHeads];
    threadgroup float totals[PISA_QUERY_TILE][kQueryHeads];
    threadgroup float alphas[PISA_QUERY_TILE][kQueryHeads];
#ifdef PISA_STAGE_SHARED_KV
    threadgroup half shared_kv[2][kBlock * kHeadDim];
#endif

#ifdef PISA_DECODE
    const uint row = decode_position;
#else
    const uint row = row_range.x + query_tile * PISA_QUERY_TILE + simdgroup;
#endif
    if (row >= kRows
#ifndef PISA_DECODE
        || row >= row_range.x + row_range.y
#endif
    ) return;

#ifdef PISA_CACHE
    const uint query_row = row - row_range.x;
    const uint query_stride = kQueryWidth;
    device const half* kv = cache;
    const uint kv_stride = 2 * kHeadDim;
#else
    const uint query_row = row;
    const uint query_stride = kQkvWidth;
    device const half* kv = qkv + kQueryWidth;
    const uint kv_stride = kQkvWidth;
#endif

    for (uint dim = lane; dim < kHeadDim; dim += 32) {
        float sum = 0.0f;
        for (uint head = 0; head < kQueryHeads; ++head)
            sum += float(qkv[ulong(query_row) * query_stride + head * kHeadDim + dim]);
        route_query[simdgroup][dim] = sum;
    }
    uint node = lane < 16 ? lane : UINT_MAX;
    simdgroup_barrier(mem_flags::mem_threadgroup);

#ifdef PISA_EXTERNAL_INDEX
    const uint position = min(kContext, (row / block_size + 1) * block_size) - 1;
#else
    const uint position = row % kContext;
#endif
    const uint current = position / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint sequence = row / kContext;
#ifdef PISA_EXTERNAL_INDEX
    if (lane < kSelected) selected_blocks[simdgroup][lane] = blocks[ulong(query_row) * kSelected + lane];
    simdgroup_barrier(mem_flags::mem_threadgroup);
#else
    for (int level = int(31 - clz(kLeaves)) - 4; level >= 0; --level) {
        const uint offset = 2 * kLeaves - (2 * kLeaves >> uint(level));
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
                    for (uint dim = 0; dim < kHeadDim; dim += 4) {
                        const float4 q = *reinterpret_cast<threadgroup const float4*>(&route_query[simdgroup][dim]);
                        const half4 p_h = *reinterpret_cast<device const half4*>(&pyramid[summary + dim]);
                        const float4 p = float4(p_h);
                        score = fma(p.w, q.w, fma(p.z, q.z, fma(p.y, q.y, fma(p.x, q.x, score))));
                    }
                }
            }
        }
        for (uint slot = 0; slot < kSelected; ++slot) {
            const float best_score = simd_max(score);
            const uint best = simd_min(
                score == best_score && score > -INFINITY ? node : UINT_MAX);
            if (lane == slot) {
                selected_blocks[simdgroup][slot] = best;
                if (level == 0) blocks[ulong(query_row) * kSelected + slot] = best;
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
#endif

    if (lane < kQueryHeads) {
        maxima[simdgroup][lane] = -INFINITY;
        totals[simdgroup][lane] = 0.0f;
    }
    simdgroup_float8x8 result[8];
    PISA_FRAGMENTS(tile, 1)
        result[tile] = simdgroup_float8x8(0.0f);
    PISA_END_FRAGMENTS
    simdgroup_half8x8 query[8];
    PISA_FRAGMENTS(dim, 8)
        simdgroup_load(
            query[dim / 8],
            qkv + ulong(query_row) * query_stride + dim,
            kHeadDim,
            ulong2(0, 0));
    PISA_END_FRAGMENTS
    simdgroup_barrier(mem_flags::mem_threadgroup);
#ifdef PISA_STAGE_SHARED_KV
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

    for (uint selected = 0; selected < kSelected; ++selected) {
        const uint block = selected_blocks[simdgroup][selected];
        if (block == UINT_MAX) continue;
        const uint valid_tokens = block < current
            ? kBlock
            : (block == current ? position % kBlock + 1 : 0);
        const uint active_tiles = (valid_tokens + 7) / 8;
#ifdef PISA_STAGE_SHARED_KV
        const bool shared = block == selected_blocks[0][selected]
            && block == selected_blocks[1][selected]
            && block == selected_blocks[2][selected]
            && block == selected_blocks[3][selected];
        if (shared) {
            for (uint vector = simdgroup * 32 + lane;
                 vector < kBlock * kHeadDim / 4; vector += 128) {
                const uint token = vector / (kHeadDim / 4);
                const uint dim = vector % (kHeadDim / 4) * 4;
                const uint source = sequence * kContext + block * kBlock + token;
                *reinterpret_cast<threadgroup half4*>(shared_kv[0] + token * kHeadDim + dim) =
                    *reinterpret_cast<device const half4*>(
                        kv + ulong(source) * kv_stride + dim);
                *reinterpret_cast<threadgroup half4*>(shared_kv[1] + token * kHeadDim + dim) =
                    *reinterpret_cast<device const half4*>(
                        kv + ulong(source) * kv_stride + kHeadDim + dim);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        for (uint tile = 0; tile < active_tiles; ++tile) {
            simdgroup_float8x8 value(0.0f);
            PISA_FRAGMENTS(dim, 8)
                simdgroup_half8x8 k;
                const uint source = sequence * kContext + block * kBlock + tile * 8;
#ifdef PISA_STAGE_SHARED_KV
                if (shared)
                    simdgroup_load(k, shared_kv[0] + tile * 8 * kHeadDim + dim,
                                   kHeadDim, ulong2(0, 0), true);
                else
#endif
                    simdgroup_load(
                        k,
                        kv + ulong(source) * kv_stride + dim,
                        kv_stride,
                        ulong2(0, 0),
                        true);
                simdgroup_multiply_accumulate(value, query[dim / 8], k, value);
            PISA_END_FRAGMENTS
            simdgroup_store(
                value,
                &scores[simdgroup][0][tile * 8],
                kBlock + 1,
                ulong2(0, 0));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

#ifdef PISA_QUAD_SOFTMAX
        const uint head = lane / 4;
        const uint head_lane = lane % 4;
        float values[16];
        float local_max = -INFINITY;
        for (uint i = 0; i < 16; ++i) {
            const uint token = i * 4 + head_lane;
            values[i] = token < valid_tokens
                ? scores[simdgroup][head][token] * 0.125f : -INFINITY;
            local_max = max(local_max, values[i]);
        }
        local_max = max(local_max, simd_shuffle_xor(local_max, 1));
        local_max = max(local_max, simd_shuffle_xor(local_max, 2));
        const float next_max = max(maxima[simdgroup][head], local_max);
        const float alpha = exp(maxima[simdgroup][head] - next_max);
        for (uint i = 0; i < 16; ++i) {
            const float probability = exp(values[i] - next_max);
            scores[simdgroup][head][i * 4 + head_lane] = probability;
            probabilities[simdgroup][head][i * 4 + head_lane] = half(probability);
        }
        if (head_lane == 0) {
            maxima[simdgroup][head] = next_max;
            alphas[simdgroup][head] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Keep the original denominator reduction order. QK scratch is dead
        // here and temporarily holds the unrounded probabilities.
        for (uint reduced_head = 0; reduced_head < kQueryHeads; ++reduced_head) {
            const float sum = simd_sum(scores[simdgroup][reduced_head][lane]
                + scores[simdgroup][reduced_head][lane + 32]);
            if (lane == 0)
                totals[simdgroup][reduced_head] = totals[simdgroup][reduced_head]
                    * alphas[simdgroup][reduced_head] + sum;
        }
#else
        for (uint head = 0; head < kQueryHeads; ++head) {
            const uint token0 = lane;
            const uint token1 = lane + 32;
            const bool valid0 = block * kBlock + token0 <= position;
            const bool valid1 = block * kBlock + token1 <= position;
            const float score0 = scores[simdgroup][head][token0] * 0.125f;
            const float score1 = scores[simdgroup][head][token1] * 0.125f;
            const float value0 = select(-INFINITY, score0, valid0);
            const float value1 = select(-INFINITY, score1, valid1);
            const float next_max =
                max(maxima[simdgroup][head], simd_max(max(value0, value1)));
            const float alpha = exp(maxima[simdgroup][head] - next_max);
            const float probability0 = select(0.0f, exp(value0 - next_max), valid0);
            const float probability1 = select(0.0f, exp(value1 - next_max), valid1);
            const float total = totals[simdgroup][head] * alpha
                + simd_sum(probability0 + probability1);
#ifdef PISA_REUSE_SCORES
            // A lane's half store can overlap another lane's float load.
            // All score loads must finish before any probability store.
            simdgroup_barrier(mem_flags::mem_threadgroup);
#endif
            probabilities[simdgroup][head][token0] = half(probability0);
            probabilities[simdgroup][head][token1] = half(probability1);
            if (lane == 0) {
                maxima[simdgroup][head] = next_max;
                totals[simdgroup][head] = total;
                alphas[simdgroup][head] = alpha;
            }
        }
#endif
        simdgroup_barrier(mem_flags::mem_threadgroup);

#ifdef PISA_SKIP_IDENTITY_RESCALE
        const bool head_changed = lane < kQueryHeads
            && alphas[simdgroup][lane] != 1.0f;
        const bool rescale = simd_any(head_changed);
        if (rescale) {
#endif
#ifdef PISA_DIRECT_RESCALE
        const uint quad = lane / 4;
        const uint matrix_row = (quad & 4) + (lane / 2) % 4;
        const float row_alpha = alphas[simdgroup][matrix_row];
        PISA_FRAGMENTS(tile, 1)
            result[tile].thread_elements()[0] *= row_alpha;
            result[tile].thread_elements()[1] *= row_alpha;
        PISA_END_FRAGMENTS
#else
        for (uint index = lane; index < 64; index += 32) {
            const uint head = index / 8;
            const uint column = index % 8;
#ifdef PISA_REUSE_SCORES
            // Routing is finished. Its 64-float query buffer holds the
            // diagonal, leaving the aliased probabilities intact for PV.
            route_query[simdgroup][index] =
#else
            scores[simdgroup][head][column] =
#endif
                head == column ? alphas[simdgroup][head] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 alpha;
        simdgroup_load(
            alpha,
#ifdef PISA_REUSE_SCORES
            &route_query[simdgroup][0],
            kQueryHeads,
#else
            &scores[simdgroup][0][0],
            kBlock + 1,
#endif
            ulong2(0, 0));
        PISA_FRAGMENTS(tile, 1)
            simdgroup_float8x8 scaled;
            simdgroup_multiply_accumulate(
                scaled,
                alpha,
                result[tile],
                simdgroup_float8x8(0.0f));
            result[tile] = scaled;
        PISA_END_FRAGMENTS
#endif
#ifdef PISA_SKIP_IDENTITY_RESCALE
        }
#endif
#ifdef PISA_STAGE_SHARED_KV
        // Both tiles were prefetched before QK; V remains live in threadgroup memory.
#endif
        for (uint token = 0; token < valid_tokens; token += 8) {
            simdgroup_half8x8 probability;
            simdgroup_load(
                probability,
                &probabilities[simdgroup][0][token],
                probability_stride,
                ulong2(0, 0));
            PISA_FRAGMENTS(tile, 1)
                simdgroup_half8x8 v;
                const uint source = sequence * kContext + block * kBlock + token;
#ifdef PISA_STAGE_SHARED_KV
                if (shared)
                    simdgroup_load(v, shared_kv[1] + token * kHeadDim + tile * 8,
                                   kHeadDim, ulong2(0, 0));
                else
#endif
                    simdgroup_load(
                        v,
                        kv + ulong(source) * kv_stride + kHeadDim + tile * 8,
                        kv_stride,
                        ulong2(0, 0));
                simdgroup_multiply_accumulate(result[tile], probability, v, result[tile]);
            PISA_END_FRAGMENTS
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
#ifdef PISA_STAGE_SHARED_KV
        if (shared) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }

    PISA_FRAGMENTS(tile, 1)
        simdgroup_store(
            result[tile],
            &scores[simdgroup][0][tile * 8],
            kBlock + 1,
            ulong2(0, 0));
    PISA_END_FRAGMENTS
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < kQueryWidth; index += 32)
        output[ulong(query_row) * kQueryWidth + index] = half(
            scores[simdgroup][index / kHeadDim][index % kHeadDim]
            / totals[simdgroup][index / kHeadDim]);
}

// ============================================================================
// Mass-Preserving Dual-Resolution Hierarchical Attention Kernel
//
// Unifies fine local tokens in selected blocks with coarse pooled super-tokens
// across unselected historical blocks. Coarse clusters receive a logarithmic
// mass compensation (+ log(kBlock) ≈ 4.158883f) so that the probability mass
// over the full 1M context is strictly preserved (sum(P) == 1.0).
// ============================================================================
kernel void fbt_pisa1_hierarchical_attention(
    device const half* qkv [[buffer(0)]],
    device const half* coarse_kv [[buffer(1)]],
    device const uint* blocks [[buffer(2)]],
    device half* output [[buffer(3)]],
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
    const uint current_leaf = position / kBlock;

    // Phase 1: Fine active window across selected blocks
    for (uint selected = 0; selected < kSelected; ++selected) {
        const uint block = blocks[ulong(row) * kSelected + selected];
        if (block == UINT_MAX || block > current_leaf) continue;

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

    // Flush fine-phase accumulators so phase 2 builds on the real partial output.
    for (uint local = 0; local < 2; ++local) {
        const uint tile = simdgroup + 4 * local;
        simdgroup_store(result[local], &scratch[0][tile * 8], kHeadDim, ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Mass-preserving coarse cluster accumulation
    // log(64) ≈ 4.15888308336f
    constexpr float kLogLeafMass = 4.15888308336f;
    for (uint leaf = 0; leaf <= current_leaf; ++leaf) {
        bool is_selected = false;
        for (uint s = 0; s < kSelected; ++s) {
            if (blocks[ulong(row) * kSelected + s] == leaf) {
                is_selected = true;
                break;
            }
        }
        if (is_selected) continue; // Already evaluated at fine resolution

        for (uint head = simdgroup; head < kQueryHeads; head += 4) {
            // Compute cooperative dot product between Q and coarse K
            float qk_coarse = 0.0f;
            for (uint d = lane; d < kHeadDim; d += 32) {
                const float q_val = float(qkv[ulong(row) * kQkvWidth + head * kHeadDim + d]);
                const float k_val = float(coarse_kv[(ulong(sequence) * kLeaves + leaf) * 2 * kHeadDim + d]);
                qk_coarse = fma(q_val, k_val, qk_coarse);
            }
            qk_coarse = simd_sum(qk_coarse);

            const float coarse_score = fma(qk_coarse, 0.125f, kLogLeafMass);
            const float next_max = max(maxima[head], coarse_score);
            const float alpha = exp(maxima[head] - next_max);
            const float p_coarse = exp(coarse_score - next_max);

            if (lane == 0) {
                totals[head] = totals[head] * alpha + p_coarse;
                maxima[head] = next_max;
                alphas[head] = alpha;
            }

            // Rescale scratch and accumulate coarse V
            for (uint d = lane; d < kHeadDim; d += 32) {
                const float v_val = float(coarse_kv[(ulong(sequence) * kLeaves + leaf) * 2 * kHeadDim + kHeadDim + d]);
                scratch[head][d] = fma(scratch[head][d], alpha, p_coarse * v_val);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Unified output store scaled by total probability mass
    for (uint index = tid; index < kQueryWidth; index += 128) {
        const uint h = index / kHeadDim;
        const uint d = index % kHeadDim;
        output[ulong(row) * kQueryWidth + index] = half(scratch[h][d] / totals[h]);
    }
}

