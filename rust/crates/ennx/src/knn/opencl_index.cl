#define K_THREADS 256
#define K_TILE_ROWS 1024
#define K_MAX_K 1024
#define K_MERGE_ROWS 2048

typedef struct {
    uint rows;
    uint dim;
    uint queries;
    uint tile_start;
    uint tile_rows;
    uint k;
} Params;

inline int before(float distance_a, uint index_a, float distance_b, uint index_b) {
    return distance_a < distance_b
        || (distance_a == distance_b && index_a < index_b);
}

inline void order_pair(__local float* distances, __local uint* indices,
                       uint left, uint right, int ascending) {
    float left_distance = distances[left];
    uint left_index = indices[left];
    float right_distance = distances[right];
    uint right_index = indices[right];
    int swap_pair = ascending
        ? before(right_distance, right_index, left_distance, left_index)
        : before(left_distance, left_index, right_distance, right_index);
    if (swap_pair) {
        distances[left] = right_distance;
        indices[left] = right_index;
        distances[right] = left_distance;
        indices[right] = left_index;
    }
}

__kernel void distance_rows(
    __global const float* rows,
    __global const float* queries,
    __global float* distances,
    Params params) {
    uint gid = get_global_id(0);
    uint total = params.queries * K_TILE_ROWS;
    if (gid >= total) return;
    uint query_index = gid / K_TILE_ROWS;
    uint tile_index = gid - query_index * K_TILE_ROWS;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    __global const float* row = rows + (size_t)row_index * params.dim;
    __global const float* query = queries + (size_t)query_index * params.dim;
    float sum = 0.0f;
    for (uint d = 0; d < params.dim; ++d) {
        float delta = row[d] - query[d];
        sum = fma(delta, delta, sum);
    }
    distances[gid] = sum;
}

__kernel void local_topk(
    __global const float* distances,
    __global float* output_distances,
    __global uint* output_indices,
    Params params) {
    __local float values[K_TILE_ROWS];
    __local uint indices[K_TILE_ROWS];
    uint tid = get_local_id(0);
    uint query_index = get_group_id(0);
    for (uint i = tid; i < K_TILE_ROWS; i += K_THREADS) {
        values[i] = i < params.tile_rows
            ? distances[query_index * K_TILE_ROWS + i]
            : INFINITY;
        indices[i] = i < params.tile_rows ? params.tile_start + i : 0xffffffffu;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint size = 2; size <= K_TILE_ROWS; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < K_TILE_ROWS; i += K_THREADS) {
                uint partner = i ^ stride;
                if (partner > i) {
                    order_pair(values, indices, i, partner, (i & size) == 0);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    for (uint i = tid; i < params.k; i += K_THREADS) {
        output_distances[query_index * params.k + i] = values[i];
        output_indices[query_index * params.k + i] = indices[i];
    }
}

__kernel void merge_topk(
    __global float* result_distances,
    __global uint* result_indices,
    __global const float* local_distances,
    __global const uint* local_indices,
    Params params) {
    __local float values[K_MERGE_ROWS];
    __local uint indices[K_MERGE_ROWS];
    uint tid = get_local_id(0);
    uint query_index = get_group_id(0);
    for (uint i = tid; i < K_MERGE_ROWS; i += K_THREADS) {
        if (i < params.k) {
            values[i] = result_distances[query_index * params.k + i];
            indices[i] = result_indices[query_index * params.k + i];
        } else if (i < 2 * params.k) {
            uint local_index = i - params.k;
            values[i] = local_distances[query_index * params.k + local_index];
            indices[i] = local_indices[query_index * params.k + local_index];
        } else {
            values[i] = INFINITY;
            indices[i] = 0xffffffffu;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint size = 2; size <= K_MERGE_ROWS; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < K_MERGE_ROWS; i += K_THREADS) {
                uint partner = i ^ stride;
                if (partner > i) {
                    order_pair(values, indices, i, partner, (i & size) == 0);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    for (uint i = tid; i < params.k; i += K_THREADS) {
        result_distances[query_index * params.k + i] = values[i];
        result_indices[query_index * params.k + i] = indices[i];
    }
}
