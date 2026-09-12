#include <metal_stdlib>
using namespace metal;

inline float widen(ushort x) { return as_type<float>(uint(x) << 16); }

struct Matmul {
    uint m, n, k, transpose_b;
    ulong stride_a, stride_b, stride_c;
};

kernel void qwen_widen(device const ushort* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant ulong& count [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = widen(input[index]);
}

kernel void qwen_bf16_f16(device const ushort* input [[buffer(0)]],
                             device half* output [[buffer(1)]],
                             constant ulong& count [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = half(widen(input[index]));
}

kernel void qwen_f32_to_f16(device const float* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant ulong& count [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = half(input[index]);
}

kernel void qwen_f16_to_f32(device const half* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant ulong& count [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = float(input[index]);
}

// Prefill GEMM. A 32x32 output tile reuses each activation across 32 output
// columns and each weight across 32 prompt rows before moving to the next K
// tile. The old kernel performed a separate global-memory dot product for
// every output element, which made long-prompt prefill bandwidth-bound.
kernel void flame_linear(device const float* a [[buffer(0)]],
                        device const ushort* b [[buffer(1)]],
                        device float* c [[buffer(2)]],
                        constant Matmul& p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a_tile[32 * 32];
    threadgroup ushort b_tile[32 * 32];
    uint row = group.y * 32 + tid / 8;
    uint col = group.x * 32 + (tid % 8) * 4;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k_base = 0; k_base < p.k; k_base += 32) {
        for (uint load = tid * 4; load < (tid + 1) * 4; ++load) {
            uint tile_row = load / 32;
            uint tile_col = load % 32;
            uint a_row = group.y * 32 + tile_row;
            uint a_col = k_base + tile_col;
            a_tile[load] = a_row < p.m && a_col < p.k
                ? a[ulong(a_row) * p.k + a_col]
                : 0.0f;
            uint b_row = k_base + tile_row;
            uint b_col = group.x * 32 + tile_col;
            ulong b_index = p.transpose_b
                ? ulong(b_col) * p.k + b_row
                : ulong(b_row) * p.n + b_col;
            b_tile[load] = b_row < p.k && b_col < p.n ? b[b_index] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < p.m) {
            for (uint d = 0; d < 32 && k_base + d < p.k; ++d) {
                float value = a_tile[(row % 32) * 32 + d];
                for (uint j = 0; j < 4 && col + j < p.n; ++j)
                    sums[j] = fma(value, widen(b_tile[d * 32 + (col % 32) + j]), sums[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.m) {
        for (uint j = 0; j < 4 && col + j < p.n; ++j)
            c[ulong(row) * p.n + col + j] = sums[j];
    }
}

#include <metal_simdgroup_matrix>

// Four SIMD groups cover a 64x32 output tile. Each group computes a 16x32
// slice using 8x8 matrix instructions while the threadgroup cooperatively
// widens one BF16 K tile into FP32 staging memory.
kernel void qwen_simd_gemm(device const float* a [[buffer(0)]],
                                  device const ushort* b [[buffer(1)]],
                                  device float* c [[buffer(2)]],
                                  constant Matmul& p [[buffer(3)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float a_tile[64 * 8];
    threadgroup float b_tile[32 * 8];
    const uint row_base = group.y * 64;
    const uint col_base = group.x * 32;
    const uint row_slice = simdgroup * 16;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = simdgroup_float8x8(0);

    for (uint k_base = 0; k_base < p.k; k_base += 8) {
        for (uint index = tid; index < 64 * 8; index += 128) {
            uint row = index / 8;
            uint k = index % 8;
            uint source_row = row_base + row;
            uint source_k = k_base + k;
            a_tile[index] = source_row < p.m && source_k < p.k
                ? a[ulong(source_row) * p.k + source_k]
                : 0.0f;
        }
        for (uint index = tid; index < 32 * 8; index += 128) {
            uint k = index / 32;
            uint column = index % 32;
            uint source_column = col_base + column;
            uint source_k = k_base + k;
            b_tile[index] = source_column < p.n && source_k < p.k
                ? widen(b[ulong(source_column) * p.k + source_k])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a_frag[2];
        simdgroup_float8x8 b_frag[4];
        for (uint i = 0; i < 2; ++i)
            simdgroup_load(a_frag[i], &a_tile[(row_slice + i * 8) * 8], 8);
        for (uint j = 0; j < 4; ++j)
            simdgroup_load(b_frag[j], &b_tile[j * 8], 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(
                acc[i][j],
                c + ulong(row_base + row_slice + i * 8) * p.n + col_base + j * 8,
                p.n);
}

// Decode GEMV handles small row counts without staging a full 32-row tile.
// Load each activation once and reuse it across four adjacent output columns.
kernel void qwen_gemv(device const float* a [[buffer(0)]],
                             device const ushort* b [[buffer(1)]],
                             device float* c [[buffer(2)]],
                             constant Matmul& p [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    uint row = group.y * 32 + tid / 8;
    uint col = group.x * 32 + (tid % 8) * 4;
    if (row >= p.m) return;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint d = 0; d < p.k; ++d) {
        float value = a[ulong(row) * p.k + d];
        for (uint j = 0; j < 4 && col + j < p.n; ++j) {
            ulong index = p.transpose_b
                ? ulong(col + j) * p.k + d
                : ulong(d) * p.n + col + j;
            sums[j] = fma(value, widen(b[index]), sums[j]);
        }
    }
    for (uint j = 0; j < 4 && col + j < p.n; ++j) {
        c[ulong(row) * p.n + col + j] = sums[j];
    }
}

// Small decode batches share each vocabulary weight across their active rows.
// One lane owns one output column, keeping the tile wide without idle rows.
kernel void qwen_gemv_rows(device const float* a [[buffer(0)]],
                                  device const ushort* b [[buffer(1)]],
                                  device float* c [[buffer(2)]],
                                  constant Matmul& p [[buffer(3)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    uint col = group.x * 32 + tid;
    if (col >= p.n) return;
    if (p.m == 0 || p.m > 4) return;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint d = 0; d < p.k; ++d) {
        float weight = widen(b[ulong(col) * p.k + d]);
        for (uint row = 0; row < p.m; ++row)
            sums[row] = fma(a[ulong(row) * p.k + d], weight, sums[row]);
    }
    for (uint row = 0; row < p.m; ++row)
        c[ulong(row) * p.n + col] = sums[row];
}

// Decode MLP projection for one to four rows. Gate and up consume the same
// normalized activation, so keep both accumulators in registers and write
// only the post-SiLU result needed by the down projection.
struct QwenMlpShape {
    uint rows, hidden, intermediate;
};

kernel void qwen_mlp_rows(device const float* input [[buffer(0)]],
                               device const ushort* gate_weight [[buffer(1)]],
                               device const ushort* up_weight [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant QwenMlpShape& p [[buffer(4)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    uint column = group * 32 + tid;
    if (column >= p.intermediate || p.rows == 0 || p.rows > 4) return;

    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint d = 0; d < p.hidden; ++d) {
        float gate_value = widen(gate_weight[ulong(column) * p.hidden + d]);
        float up_value = widen(up_weight[ulong(column) * p.hidden + d]);
        for (uint row = 0; row < p.rows; ++row) {
            float value = input[ulong(row) * p.hidden + d];
            gate[row] = fma(value, gate_value, gate[row]);
            up[row] = fma(value, up_value, up[row]);
        }
    }
    for (uint row = 0; row < p.rows; ++row)
        output[ulong(row) * p.intermediate + column] =
            gate[row] * (1.0f / (1.0f + exp(-gate[row]))) * up[row];
}
