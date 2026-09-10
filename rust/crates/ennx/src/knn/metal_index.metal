#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kTileRows = 1024;
constant uint kMergeRows = 4096;
constant uint kGramRows = 64;
constant uint kGramQueries = 8;
constant uint kGramDim = 8;

inline uint merge_width(uint k) {
    uint width = 1;
    while (width < 2 * k) {
        width <<= 1;
    }
    return width;
}

inline uint topk_width(uint k) {
    uint width = 1;
    while (width < k) {
        width <<= 1;
    }
    return width;
}

struct Params {
    uint rows;
    uint dim;
    uint queries;
    uint tile_start;
    uint tile_rows;
    uint k;
    uint lanes;
    uint tiles;
    uint lists;
    uint groups;
};

kernel void init_results(
    device float* result_distances [[buffer(0)]],
    device uint* result_indices [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * params.k;
    if (gid >= total) return;
    result_distances[gid] = INFINITY;
    result_indices[gid] = 0xffffffffu;
}

inline bool before(float distance_a, uint index_a, float distance_b, uint index_b) {
    return distance_a < distance_b
        || (distance_a == distance_b && index_a < index_b);
}

inline void order_pair(threadgroup float* distances, threadgroup uint* indices,
                       uint left, uint right, bool ascending) {
    float left_distance = distances[left];
    uint left_index = indices[left];
    float right_distance = distances[right];
    uint right_index = indices[right];
    bool swap_pair = ascending
        ? before(right_distance, right_index, left_distance, left_index)
        : before(left_distance, left_index, right_distance, right_index);
    if (swap_pair) {
        distances[left] = right_distance;
        indices[left] = right_index;
        distances[right] = left_distance;
        indices[right] = left_index;
    }
}

inline void sort4(thread float* values, thread uint* indices) {
    for (uint i = 1; i < 4; ++i) {
        float value = values[i];
        uint index = indices[i];
        uint at = i;
        while (at > 0 && before(value, index, values[at - 1], indices[at - 1])) {
            values[at] = values[at - 1];
            indices[at] = indices[at - 1];
            --at;
        }
        values[at] = value;
        indices[at] = index;
    }
}

inline void reduce_pair(thread float& value, thread uint& index, uint lane) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_value = simd_shuffle_down(value, offset);
        uint other_index = simd_shuffle_down(index, offset);
        if (lane + offset < 32 && before(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

inline float l2(
    device const float* row,
    device const float* query,
    uint dim,
    uint lanes) {
    if (lanes == 4) {
        float4 sum = 0.0f;
        uint d = 0;
        for (; d + 3 < dim; d += 4) {
            float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                         - float4(query[d], query[d + 1], query[d + 2], query[d + 3]);
            sum = fma(delta, delta, sum);
        }
        float result = (sum.x + sum.y) + (sum.z + sum.w);
        for (; d < dim; ++d) {
            float delta = row[d] - query[d];
            result = fma(delta, delta, result);
        }
        return result;
    }
    if (lanes == 2) {
        float2 sum = 0.0f;
        uint d = 0;
        for (; d + 1 < dim; d += 2) {
            float2 delta = float2(row[d], row[d + 1]) - float2(query[d], query[d + 1]);
            sum = fma(delta, delta, sum);
        }
        float result = sum.x + sum.y;
        if (d < dim) {
            float delta = row[d] - query[d];
            result = fma(delta, delta, result);
        }
        return result;
    }
    float sum = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        float delta = row[d] - query[d];
        sum = fma(delta, delta, sum);
    }
    return sum;
}

kernel void distance_rows(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float sum = 0.0f;
    for (uint d = 0; d < params.dim; ++d) {
        float delta = row[d] - query[d];
        sum = fma(delta, delta, sum);
    }
    distances[gid] = sum;
}

kernel void distance_rows_2(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float2 sum = 0.0f;
    uint d = 0;
    for (; d + 1 < params.dim; d += 2) {
        float2 delta = float2(row[d], row[d + 1]) - float2(query[d], query[d + 1]);
        sum = fma(delta, delta, sum);
    }
    float result = sum.x + sum.y;
    if (d < params.dim) {
        float delta = row[d] - query[d];
        result = fma(delta, delta, result);
    }
    distances[gid] = result;
}

kernel void distance_rows_4(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float4 sum = 0.0f;
    uint d = 0;
    for (; d + 3 < params.dim; d += 4) {
        float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                     - float4(query[d], query[d + 1], query[d + 2], query[d + 3]);
        sum = fma(delta, delta, sum);
    }
    float result = (sum.x + sum.y) + (sum.z + sum.w);
    for (; d < params.dim; ++d) {
        float delta = row[d] - query[d];
        result = fma(delta, delta, result);
    }
    distances[gid] = result;
}

kernel void distance_simd(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint pair = group * (kThreads / 32) + simd;
    uint total = params.queries * kTileRows;
    if (pair >= total) return;
    uint query_index = pair / kTileRows;
    uint tile_index = pair - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        if (lane == 0) distances[pair] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float4 sum4 = 0.0f;
    uint vector_dim = params.dim & ~3u;
    for (uint d = lane * 4; d < vector_dim; d += 32 * 4) {
        float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                     - float4(query[d], query[d + 1], query[d + 2], query[d + 3]);
        sum4 = fma(delta, delta, sum4);
    }
    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    uint tail = vector_dim + lane;
    if (tail < params.dim) {
        float delta = row[tail] - query[tail];
        sum = fma(delta, delta, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) distances[pair] = sum;
}

kernel void topk_16(
    device const float* distances [[buffer(0)]],
    device float* output_distances [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    float values[4];
    uint indices[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        values[slot] = local < params.tile_rows
            ? distances[query * kTileRows + local]
            : INFINITY;
        indices[slot] = local < params.tile_rows
            ? params.tile_start + local
            : 0xffffffffu;
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint lane = tid % 32;
    uint group = tid / 32;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[group] = value;
            part_indices[group] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (group == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[query * params.k + rank] = value;
                output_indices[query * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void l2_topk_16(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query_index [[threadgroup_position_in_grid]]) {
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float values[4];
    uint indices[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        if (local < params.tile_rows) {
            uint row_index = params.tile_start + local;
            device const float* row = rows + ulong(row_index) * ulong(params.dim);
            values[slot] = l2(row, query, params.dim, params.lanes);
            indices[slot] = row_index;
        } else {
            values[slot] = INFINITY;
            indices[slot] = 0xffffffffu;
        }
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint lane = tid % 32;
    uint group = tid / 32;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[group] = value;
            part_indices[group] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (group == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[query_index * params.k + rank] = value;
                output_indices[query_index * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void l2_topk_16_batch(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
    uint tile = batch / params.queries;
    uint query_index = batch - tile * params.queries;
    uint output_batch = query_index * params.tiles + tile;
    uint tile_start = tile * kTileRows;
    uint tile_rows = min(kTileRows, params.rows - tile_start);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float values[4];
    uint indices[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        if (local < tile_rows) {
            uint row_index = tile_start + local;
            device const float* row = rows + ulong(row_index) * ulong(params.dim);
            values[slot] = l2(row, query, params.dim, params.lanes);
            indices[slot] = row_index;
        } else {
            values[slot] = INFINITY;
            indices[slot] = 0xffffffffu;
        }
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint lane = tid % 32;
    uint group = tid / 32;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[group] = value;
            part_indices[group] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[output_batch * params.k + rank] = value;
                output_indices[output_batch * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void l2_topk_16_tiled(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
    uint tile = batch / params.queries;
    uint query_index = batch - tile * params.queries;
    uint output_batch = query_index * params.tiles + tile;
    uint tile_start = tile * kTileRows;
    uint tile_rows = min(kTileRows, params.rows - tile_start);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    threadgroup float query_tile[kThreads];
    float values[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint indices[4];
    bool valid[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        valid[slot] = local < tile_rows;
        indices[slot] = valid[slot] ? tile_start + local : 0xffffffffu;
    }
    for (uint base = 0; base < params.dim; base += kThreads) {
        uint width = min(kThreads, params.dim - base);
        query_tile[tid] = tid < width ? query[base + tid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slot = 0; slot < 4; ++slot) {
            if (!valid[slot]) continue;
            device const float* row = rows
                + ulong(indices[slot]) * ulong(params.dim)
                + ulong(base);
            uint d = 0;
            if (params.lanes == 4) {
                for (; d + 3 < width; d += 4) {
                    float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                                 - float4(query_tile[d], query_tile[d + 1],
                                          query_tile[d + 2], query_tile[d + 3]);
                    values[slot] += dot(delta, delta);
                }
            } else if (params.lanes == 2) {
                for (; d + 1 < width; d += 2) {
                    float2 delta = float2(row[d], row[d + 1])
                                 - float2(query_tile[d], query_tile[d + 1]);
                    values[slot] += dot(delta, delta);
                }
            }
            for (; d < width; ++d) {
                float delta = row[d] - query_tile[d];
                values[slot] = fma(delta, delta, values[slot]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint slot = 0; slot < 4; ++slot) {
        if (!valid[slot]) values[slot] = INFINITY;
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint lane = tid % 32;
    uint group = tid / 32;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[group] = value;
            part_indices[group] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[output_batch * params.k + rank] = value;
                output_indices[output_batch * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void l2_topk_16_simd(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tile = batch / params.queries;
    uint query_index = batch - tile * params.queries;
    uint output_batch = query_index * params.tiles + tile;
    uint tile_start = tile * kTileRows;
    uint tile_rows = min(kTileRows, params.rows - tile_start);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    threadgroup float tile_distances[kTileRows];
    for (uint local = simd; local < tile_rows; local += kThreads / 32) {
        uint row_index = tile_start + local;
        device const float* row = rows + ulong(row_index) * ulong(params.dim);
        float4 sum4 = 0.0f;
        uint vector_dim = params.dim & ~3u;
        for (uint d = lane * 4; d < vector_dim; d += 32 * 4) {
            float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                         - float4(query[d], query[d + 1], query[d + 2], query[d + 3]);
            sum4 = fma(delta, delta, sum4);
        }
        float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
        uint tail = vector_dim + lane;
        if (tail < params.dim) {
            float delta = row[tail] - query[tail];
            sum = fma(delta, delta, sum);
        }
        sum = simd_sum(sum);
        if (lane == 0) tile_distances[local] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float values[4];
    uint indices[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        values[slot] = local < tile_rows ? tile_distances[local] : INFINITY;
        indices[slot] = local < tile_rows ? tile_start + local : 0xffffffffu;
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[simd] = value;
            part_indices[simd] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[output_batch * params.k + rank] = value;
                output_indices[output_batch * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void l2_topk_16_gram(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    device const float* row_norms [[buffer(4)]],
    device const float* query_norms [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint query_groups = (params.queries + kGramQueries - 1) / kGramQueries;
    uint row_group = batch / query_groups;
    uint query_group = batch - row_group * query_groups;
    uint row_start = row_group * kGramRows;
    threadgroup float row_tile[(kThreads / 32) * kGramDim * kGramDim];
    threadgroup float query_tile[kGramQueries * kGramDim];
    threadgroup float products[(kThreads / 32) * kGramDim * kGramDim];
    threadgroup float values[kGramQueries * kGramRows];
    threadgroup uint indices[kGramQueries * kGramRows];

    simdgroup_float8x8 query_matrix;
    simdgroup_float8x8 row_matrix;
    simdgroup_float8x8 product_matrix(0.0f);
    for (uint base = 0; base < params.dim; base += kGramDim) {
        uint width = min(kGramDim, params.dim - base);
        for (uint load = tid; load < kGramQueries * kGramDim; load += kThreads) {
            uint query_local = load / kGramDim;
            uint d = load - query_local * kGramDim;
            uint query_index = query_group * kGramQueries + query_local;
            query_tile[load] = query_index < params.queries && d < width
                ? queries[ulong(query_index) * ulong(params.dim) + base + d]
                : 0.0f;
        }
        for (uint load = tid;
             load < (kThreads / 32) * kGramDim * kGramDim;
             load += kThreads) {
            uint block = load / (kGramDim * kGramDim);
            uint local = load - block * kGramDim * kGramDim;
            uint d = local / kGramDim;
            uint row_local = local - d * kGramDim;
            uint row_index = row_start + block * kGramDim + row_local;
            row_tile[load] = row_index < params.rows && d < width
                ? rows[ulong(row_index) * ulong(params.dim) + base + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(query_matrix, query_tile, kGramDim);
        simdgroup_load(
            row_matrix,
            row_tile + simd * kGramDim * kGramDim,
            kGramDim);
        simdgroup_multiply_accumulate(
            product_matrix,
            query_matrix,
            row_matrix,
            product_matrix);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(
        product_matrix,
        products + simd * kGramDim * kGramDim,
        kGramDim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint work = tid; work < kGramQueries * kGramRows; work += kThreads) {
        uint query_local = work / kGramRows;
        uint row_local = work - query_local * kGramRows;
        uint block = row_local / kGramDim;
        uint row_in_block = row_local - block * kGramDim;
        uint query_index = query_group * kGramQueries + query_local;
        uint row_index = row_start + row_local;
        if (query_index < params.queries && row_index < params.rows) {
            float dot = products[
                block * kGramDim * kGramDim + query_local * kGramDim + row_in_block];
            values[work] = max(
                0.0f,
                row_norms[row_index] + query_norms[query_index] - 2.0f * dot);
            indices[work] = row_index;
        } else {
            values[work] = INFINITY;
            indices[work] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint size = 2; size <= kGramRows; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint work = tid; work < kGramQueries * kGramRows; work += kThreads) {
                uint local = work & (kGramRows - 1);
                uint partner = local ^ stride;
                if (partner > local) {
                    uint list = work - local;
                    order_pair(
                        values,
                        indices,
                        list + local,
                        list + partner,
                        (local & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < kGramQueries * 16) {
        uint local = tid / 16;
        uint rank = tid - local * 16;
        uint query = query_group * kGramQueries + local;
        if (query < params.queries && rank < params.k) {
            uint input = local * kGramRows + rank;
            uint output = (query * params.tiles + row_group) * params.k + rank;
            output_distances[output] = values[input];
            output_indices[output] = indices[input];
        }
    }
}

kernel void l2_topk_batch(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
    uint tile = batch / params.queries;
    uint query_index = batch - tile * params.queries;
    uint output_batch = query_index * params.tiles + tile;
    uint tile_start = tile * kTileRows;
    uint tile_rows = min(kTileRows, params.rows - tile_start);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    threadgroup float values[kTileRows];
    threadgroup uint indices[kTileRows];
    for (uint local = tid; local < kTileRows; local += kThreads) {
        if (local < tile_rows) {
            uint row_index = tile_start + local;
            device const float* row = rows + ulong(row_index) * ulong(params.dim);
            values[local] = l2(row, query, params.dim, params.lanes);
            indices[local] = row_index;
        } else {
            values[local] = INFINITY;
            indices[local] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_k = min(params.k, kTileRows);
    uint width = topk_width(local_k);
    for (uint size = 2; size <= width; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < kTileRows; i += kThreads) {
                uint local = i & (width - 1);
                uint partner = local ^ stride;
                if (partner > local) {
                    uint base = i - local;
                    order_pair(values, indices, base + local, base + partner, (local & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint run_stride = width; run_stride < kTileRows; run_stride <<= 1) {
        uint pair_count = kTileRows / (2 * run_stride);
        uint work_count = pair_count * width;
        for (uint work = tid; work < work_count; work += kThreads) {
            uint pair = work / width;
            uint lane = work - pair * width;
            uint left = pair * 2 * run_stride + lane;
            uint right = pair * 2 * run_stride + run_stride + (width - 1 - lane);
            if (before(values[right], indices[right], values[left], indices[left])) {
                values[left] = values[right];
                indices[left] = indices[right];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint work = tid; work < work_count; work += kThreads) {
                uint pair = work / width;
                uint lane = work - pair * width;
                uint partner = lane ^ stride;
                if (partner > lane) {
                    uint base = pair * 2 * run_stride;
                    order_pair(values, indices, base + lane, base + partner, true);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint rank = tid; rank < params.k; rank += kThreads) {
        output_distances[output_batch * params.k + rank]
            = rank < local_k ? values[rank] : INFINITY;
        output_indices[output_batch * params.k + rank]
            = rank < local_k ? indices[rank] : 0xffffffffu;
    }
}

kernel void fold_topk(
    device const float* input_distances [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
    uint query_index = batch / params.groups;
    uint output_group = batch - query_index * params.groups;
    uint left_list = output_group * 2;
    uint right_list = left_list + 1;
    ulong left = (ulong(query_index) * ulong(params.lists) + left_list)
               * ulong(params.k);
    ulong right = (ulong(query_index) * ulong(params.lists) + right_list)
                * ulong(params.k);
    for (uint rank = tid; rank < params.k; rank += kThreads) {
        uint low = 0;
        uint high = rank;
        while (low < high) {
            uint left_rank = (low + high) >> 1;
            uint right_rank = rank - left_rank - 1;
            float left_distance = input_distances[left + left_rank];
            uint left_index = input_indices[left + left_rank];
            float right_distance = right_list < params.lists
                ? input_distances[right + right_rank]
                : INFINITY;
            uint right_index = right_list < params.lists
                ? input_indices[right + right_rank]
                : 0xffffffffu;
            if (before(left_distance, left_index, right_distance, right_index)) {
                low = left_rank + 1;
            } else {
                high = left_rank;
            }
        }
        uint left_rank = low;
        uint right_rank = rank - low;
        float left_distance = input_distances[left + left_rank];
        uint left_index = input_indices[left + left_rank];
        float right_distance = right_list < params.lists
            ? input_distances[right + right_rank]
            : INFINITY;
        uint right_index = right_list < params.lists
            ? input_indices[right + right_rank]
            : 0xffffffffu;
        bool take_right = before(
            right_distance,
            right_index,
            left_distance,
            left_index);
        output_distances[batch * params.k + rank]
            = take_right ? right_distance : left_distance;
        output_indices[batch * params.k + rank]
            = take_right ? right_index : left_index;
    }
}

kernel void reduce_topk_16(
    device const float* input_distances [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]],
    device float* output_distances [[buffer(2)]],
    device uint* output_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
    uint query_index = batch / params.groups;
    uint output_group = batch - query_index * params.groups;
    uint fan = kTileRows / params.k;
    float values[4];
    uint indices[4];
    for (uint slot = 0; slot < 4; ++slot) {
        uint local = tid + slot * kThreads;
        uint list_local = local / params.k;
        uint rank = local - list_local * params.k;
        uint list = output_group * fan + list_local;
        if (list_local < fan && list < params.lists) {
            ulong offset = (ulong(query_index) * ulong(params.lists) + list)
                         * ulong(params.k) + rank;
            values[slot] = input_distances[offset];
            indices[slot] = input_indices[offset];
        } else {
            values[slot] = INFINITY;
            indices[slot] = 0xffffffffu;
        }
    }
    sort4(values, indices);

    threadgroup float part_values[kThreads / 32];
    threadgroup uint part_indices[kThreads / 32];
    threadgroup float best_value;
    threadgroup uint best_index;
    uint lane = tid % 32;
    uint group = tid / 32;
    uint cursor = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
        float value = cursor < 4 ? values[cursor] : INFINITY;
        uint index = cursor < 4 ? indices[cursor] : 0xffffffffu;
        reduce_pair(value, index, lane);
        if (lane == 0) {
            part_values[group] = value;
            part_indices[group] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            value = lane < kThreads / 32 ? part_values[lane] : INFINITY;
            index = lane < kThreads / 32 ? part_indices[lane] : 0xffffffffu;
            reduce_pair(value, index, lane);
            if (lane == 0) {
                best_value = value;
                best_index = index;
                output_distances[batch * params.k + rank] = value;
                output_indices[batch * params.k + rank] = index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cursor < 4 && indices[cursor] == best_index) {
            ++cursor;
        }
    }
}

kernel void local_topk(
    device const float* distances [[buffer(0)]],
    device float* output_distances [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query_index [[threadgroup_position_in_grid]]) {
    threadgroup float values[kTileRows];
    threadgroup uint indices[kTileRows];
    for (uint i = tid; i < kTileRows; i += kThreads) {
        values[i] = i < params.tile_rows
            ? distances[query_index * kTileRows + i]
            : INFINITY;
        indices[i] = i < params.tile_rows ? params.tile_start + i : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sort independent power-of-two runs just wide enough to retain k values.
    // Then merge pairs of runs while discarding their upper halves. This keeps
    // the result exact without sorting all 1024 tile values when k is small.
    uint local_k = min(params.k, kTileRows);
    uint width = topk_width(local_k);
    for (uint size = 2; size <= width; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < kTileRows; i += kThreads) {
                uint local = i & (width - 1);
                uint partner = local ^ stride;
                if (partner > local) {
                    uint base = i - local;
                    order_pair(
                        values,
                        indices,
                        base + local,
                        base + partner,
                        (local & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint run_stride = width; run_stride < kTileRows; run_stride <<= 1) {
        uint pair_count = kTileRows / (2 * run_stride);
        uint work_count = pair_count * width;

        // The left run followed by the reversed right run is bitonic. Keep its
        // lower half, then bitonic-merge that half back into ascending order.
        for (uint work = tid; work < work_count; work += kThreads) {
            uint pair = work / width;
            uint lane = work - pair * width;
            uint left = pair * 2 * run_stride + lane;
            uint right = pair * 2 * run_stride + run_stride + (width - 1 - lane);
            if (before(values[right], indices[right], values[left], indices[left])) {
                values[left] = values[right];
                indices[left] = indices[right];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            for (uint work = tid; work < work_count; work += kThreads) {
                uint pair = work / width;
                uint lane = work - pair * width;
                uint partner = lane ^ stride;
                if (partner > lane) {
                    uint base = pair * 2 * run_stride;
                    order_pair(values, indices, base + lane, base + partner, true);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < params.k; i += kThreads) {
        output_distances[query_index * params.k + i] = i < local_k ? values[i] : INFINITY;
        output_indices[query_index * params.k + i] = i < local_k ? indices[i] : 0xffffffffu;
    }
}

kernel void merge_topk(
    device float* result_distances [[buffer(0)]],
    device uint* result_indices [[buffer(1)]],
    device const float* local_distances [[buffer(2)]],
    device const uint* local_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query_index [[threadgroup_position_in_grid]]) {
    threadgroup float values[kMergeRows];
    threadgroup uint indices[kMergeRows];
    uint width = merge_width(params.k);
    for (uint i = tid; i < width; i += kThreads) {
        if (i < params.k) {
            values[i] = result_distances[query_index * params.k + i];
            indices[i] = result_indices[query_index * params.k + i];
        } else if (i < 2 * params.k) {
            uint local = i - params.k;
            values[i] = local_distances[query_index * params.k + local];
            indices[i] = local_indices[query_index * params.k + local];
        } else {
            values[i] = INFINITY;
            indices[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= width; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < width; i += kThreads) {
                uint partner = i ^ stride;
                if (partner > i) {
                    order_pair(values, indices, i, partner, (i & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < params.k; i += kThreads) {
        result_distances[query_index * params.k + i] = values[i];
        result_indices[query_index * params.k + i] = indices[i];
    }
}
