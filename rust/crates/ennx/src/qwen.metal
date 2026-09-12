// Qwen2 dense kernels. The tiled BF16 linear and FP32 attention kernels are
// shared from flame.metal; these kernels cover only Qwen's different layout.

using namespace metal;
#include <metal_simdgroup_matrix>

inline float widen(ushort x) { return as_type<float>(uint(x) << 16); }

inline ushort narrow(float x) {
    uint bits = as_type<uint>(x);
    uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    return ushort((bits + rounding) >> 16);
}

template <bool Maximum>
inline float reduce(float value, threadgroup float* scratch, uint lane) {
    scratch[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride /= 2) {
        if (lane < stride) scratch[lane] = Maximum
            ? max(scratch[lane], scratch[lane + stride])
            : scratch[lane] + scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

struct QwenShape {
    uint rows, width, heads, kv_heads, head_dim, hidden, start, sequence;
    float epsilon, rope_theta;
};

struct QwenQkvShape {
    uint rows, hidden, kv_width;
};

struct CacheShape {
    uint rows, kv_heads, head_dim, capacity, position;
};

struct DecodeRopeShape {
    uint heads, kv_heads, head_dim, capacity, position;
    uint batch, cache_stride;
    float rope_theta;
};

struct DecodeAttentionShape {
    uint sequence, capacity, heads, kv_heads, head_dim;
    uint batch, cache_stride;
    float scale;
};

struct PrefillAttentionShape {
    uint rows, sequence, capacity, heads, kv_heads, head_dim;
    float scale;
};

struct ArgmaxShape {
    uint width;
};

struct ArgmaxBatchShape {
    uint rows, width;
};

kernel void qwen_embedding(device const ushort* weights [[buffer(0)]],
                           device const int* tokens [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant QwenShape& p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width) {
        uint row = i / p.width;
        output[i] = widen(weights[ulong(tokens[p.start + row]) * p.width + i % p.width]);
    }
}

// Decode Q, K and V together. The three projections share the same input, so
// one dispatch avoids two redundant activation streams and six small launches
// per transformer layer. This path is restricted to the small decode shapes;
// prefill keeps the tiled GEMM implementation.
kernel void qwen_qkv(device const float* input [[buffer(0)]],
                          device const ushort* q_weight [[buffer(1)]],
                          device const ushort* k_weight [[buffer(2)]],
                          device const ushort* v_weight [[buffer(3)]],
                          device const ushort* q_bias [[buffer(4)]],
                          device const ushort* k_bias [[buffer(5)]],
                          device const ushort* v_bias [[buffer(6)]],
                          device float* q_output [[buffer(7)]],
                          device float* k_output [[buffer(8)]],
                          device float* v_output [[buffer(9)]],
                          constant QwenQkvShape& p [[buffer(10)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    uint column = group * 32 + tid;
    uint total = p.hidden + 2 * p.kv_width;
    if (column >= total || p.rows == 0 || p.rows > 4) return;

    uint width;
    device const ushort* weight;
    device const ushort* bias;
    device float* output;
    uint output_column;
    if (column < p.hidden) {
        width = p.hidden;
        weight = q_weight;
        bias = q_bias;
        output = q_output;
        output_column = column;
    } else if (column < p.hidden + p.kv_width) {
        width = p.kv_width;
        weight = k_weight;
        bias = k_bias;
        output = k_output;
        output_column = column - p.hidden;
    } else {
        width = p.kv_width;
        weight = v_weight;
        bias = v_bias;
        output = v_output;
        output_column = column - p.hidden - p.kv_width;
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint d = 0; d < p.hidden; ++d) {
        float value = widen(weight[ulong(output_column) * p.hidden + d]);
        for (uint row = 0; row < p.rows; ++row)
            sums[row] = fma(input[ulong(row) * p.hidden + d], value, sums[row]);
    }
    for (uint row = 0; row < p.rows; ++row)
        output[ulong(row) * width + output_column] = sums[row] + widen(bias[output_column]);
}

kernel void qwen_rms(device const float* input [[buffer(0)]],
                     device const ushort* weights [[buffer(1)]],
                     device float* output [[buffer(2)]],
                     constant QwenShape& p [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    ulong base = ulong(row) * p.width;
    float sum = 0.0f;
    for (uint d = tid; d < p.width; d += 256)
        sum += input[base + d] * input[base + d];
    float scale = rsqrt(reduce<false>(sum, scratch, tid) / float(p.width) + p.epsilon);
    for (uint d = tid; d < p.width; d += 256)
        output[base + d] = input[base + d] * scale * widen(weights[d]);
}

kernel void qwen_rope(device const float* q_input [[buffer(0)]],
                      device const float* k_input [[buffer(1)]],
                      device const float* v_input [[buffer(2)]],
                      device float* q_output [[buffer(3)]],
                      device float* k_output [[buffer(4)]],
                      device float* v_output [[buffer(5)]],
                      device const float* rope_table [[buffer(6)]],
                      constant QwenShape& p [[buffer(7)]],
                      uint i [[thread_position_in_grid]]) {
    uint q_width = p.heads * p.head_dim;
    uint kv_width = p.kv_heads * p.head_dim;
    if (ulong(i) >= ulong(p.rows) * q_width) return;
    uint local_pos = i / q_width, pos = p.start + local_pos, col = i % q_width;
    uint head = col / p.head_dim, d = col % p.head_dim;
    uint half_dim = p.head_dim / 2;
    uint partner = d < half_dim ? d + half_dim : d - half_dim;
    float sign = d < half_dim ? -1.0f : 1.0f;
    ulong table_index = ulong(pos) * p.head_dim + 2 * (d % half_dim);
    float cs = rope_table[table_index], sn = rope_table[table_index + 1];
    ulong dst = (ulong(head) * p.rows + local_pos) * p.head_dim + d;
    ulong src = ulong(local_pos) * q_width + col;
    q_output[dst] = q_input[src] * cs + sign * q_input[ulong(local_pos) * q_width + head * p.head_dim + partner] * sn;
    if (col < kv_width) {
        uint kv_head = col / p.head_dim, kv_d = col % p.head_dim;
        uint kv_partner = kv_d < half_dim ? kv_d + half_dim : kv_d - half_dim;
        float kv_sign = kv_d < half_dim ? -1.0f : 1.0f;
        ulong kv_table_index = ulong(pos) * p.head_dim + 2 * (kv_d % half_dim);
        float kv_cs = rope_table[kv_table_index], kv_sn = rope_table[kv_table_index + 1];
        ulong kv_dst = (ulong(kv_head) * p.rows + local_pos) * p.head_dim + kv_d;
        ulong kv_src = ulong(local_pos) * kv_width + col;
        k_output[kv_dst] = k_input[kv_src] * kv_cs + kv_sign * k_input[ulong(local_pos) * kv_width + kv_head * p.head_dim + kv_partner] * kv_sn;
        v_output[kv_dst] = v_input[kv_src];
    }
}

// One SIMD-group owns one (query row, query head). It attends to all cached
// positions up to that row, so chunk boundaries preserve exact causal order
// without materializing a score matrix.
kernel void qwen_arow(device const float* query [[buffer(0)]],
                                         device const ushort* key_cache [[buffer(1)]],
                                         device const ushort* value_cache [[buffer(2)]],
                                         device float* output [[buffer(3)]],
                                         constant PrefillAttentionShape& p [[buffer(4)]],
                                         uint group [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    uint total = p.rows * p.heads;
    if (group >= total) return;
    uint row = group / p.heads;
    uint head = group % p.heads;
    uint kv_head = head / (p.heads / p.kv_heads);
    uint absolute_row = p.sequence - p.rows + row;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint position = 0; position <= absolute_row; ++position) {
        float partial = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32) {
            ulong query_index = (ulong(head) * p.rows + row) * p.head_dim + d;
            ulong cache_index = (ulong(kv_head) * p.capacity + position) * p.head_dim + d;
            partial = fma(query[query_index], widen(key_cache[cache_index]), partial);
        }
        float score = simd_sum(partial) * p.scale;
        float next_maximum = max(maximum, score);
        float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_maximum);
        float new_scale = exp(score - next_maximum);
        for (uint d = lane, slot = 0; d < p.head_dim; d += 32, ++slot) {
            ulong cache_index = (ulong(kv_head) * p.capacity + position) * p.head_dim + d;
            accumulated[slot] = accumulated[slot] * old_scale
                + new_scale * widen(value_cache[cache_index]);
        }
        denominator = denominator * old_scale + new_scale;
        maximum = next_maximum;
    }
    for (uint d = lane, slot = 0; d < p.head_dim; d += 32, ++slot)
        output[ulong(row) * p.heads * p.head_dim + ulong(head) * p.head_dim + d] = accumulated[slot] / denominator;
}

// Eight queries share each 16-key tile. The serial kernel above remains the
// oracle while this exact, online-softmax path is qualified end to end.
kernel void qwen_atile(device const float* query [[buffer(0)]],
                           device const ushort* key_cache [[buffer(1)]],
                           device const ushort* value_cache [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           constant PrefillAttentionShape& p [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float q[8 * 128];
    threadgroup float kv[16 * 128];
    threadgroup float scores[8 * 16];
    threadgroup float result[8 * 128];
    threadgroup float product[8 * 128];
    threadgroup float maxima[8], totals[8], alphas[8];

    uint head = group.x;
    uint row_base = group.y * 8;
    if (head >= p.heads || row_base >= p.rows || p.head_dim != 128) return;
    uint kv_head = head / (p.heads / p.kv_heads);
    uint start = p.sequence - p.rows;
    uint last_row = min(row_base + 7, p.rows - 1);
    uint last_pos = start + last_row;

    for (uint i = tid; i < 8 * 128; i += 128) {
        uint row = row_base + i / 128;
        uint d = i % 128;
        q[i] = row < p.rows
            ? query[(ulong(head) * p.rows + row) * 128 + d]
            : 0.0f;
        result[i] = 0.0f;
    }
    if (tid < 8) {
        maxima[tid] = -INFINITY;
        totals[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base <= last_pos; base += 16) {
        // K is staged transposed for Q*K^T. The same storage then holds V.
        for (uint i = tid; i < 16 * 128; i += 128) {
            uint token = base + i % 16;
            uint d = i / 16;
            kv[i] = token <= last_pos
                ? widen(key_cache[(ulong(kv_head) * p.capacity + token) * 128 + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2) {
            simdgroup_float8x8 score(0);
            for (uint d = 0; d < 128; d += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, &q[d], 128);
                simdgroup_load(b, &kv[d * 16 + sg * 8], 16);
                simdgroup_multiply_accumulate(score, a, b, score);
            }
            simdgroup_store(score, &scores[sg * 8], 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = sg; row < 8; row += 4) {
            uint local_row = row_base + row;
            uint position = start + local_row;
            uint token = base + lane;
            bool valid = lane < 16 && local_row < p.rows && token <= position;
            float value = valid ? scores[row * 16 + lane] * p.scale : -INFINITY;
            float next_max = max(maxima[row], simd_max(value));
            if (local_row >= p.rows) next_max = 0.0f;
            float alpha = exp(maxima[row] - next_max);
            float probability = valid ? exp(value - next_max) : 0.0f;
            float total = totals[row] * alpha + simd_sum(probability);
            if (lane < 16) scores[row * 16 + lane] = probability;
            if (lane == 0) {
                maxima[row] = next_max;
                totals[row] = total;
                alphas[row] = alpha;
            }
        }

        for (uint i = tid; i < 16 * 128; i += 128) {
            uint token = base + i / 128;
            uint d = i % 128;
            kv[i] = token <= last_pos
                ? widen(value_cache[(ulong(kv_head) * p.capacity + token) * 128 + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint part = 0; part < 4; ++part) {
            uint col = sg * 8 + part * 32;
            simdgroup_float8x8 acc(0);
            for (uint k = 0; k < 16; k += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, &scores[k], 16);
                simdgroup_load(b, &kv[k * 128 + col], 128);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, &product[col], 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 128; i += 128)
            result[i] = alphas[i / 128] * result[i] + product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < 8 * 128; i += 128) {
        uint row = row_base + i / 128;
        uint d = i % 128;
        if (row < p.rows)
            output[ulong(row) * p.heads * 128 + ulong(head) * 128 + d] =
                result[i] / totals[i / 128];
    }
}

// Sixteen queries share each 16-key tile. The staged V tile is reused for the
// temporary P*V product after every SIMD group has finished reading V.
kernel void qwen_attn16(device const float* query [[buffer(0)]],
                        device const ushort* key_cache [[buffer(1)]],
                        device const ushort* value_cache [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        constant PrefillAttentionShape& p [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float q[16 * 128];
    threadgroup float kv[16 * 128];
    threadgroup float scores[16 * 16];
    threadgroup float result[16 * 128];
    threadgroup float maxima[16], totals[16], alphas[16];

    uint head = group.x;
    uint row_base = group.y * 16;
    if (head >= p.heads || row_base >= p.rows || p.head_dim != 128) return;
    uint kv_head = head / (p.heads / p.kv_heads);
    uint start = p.sequence - p.rows;
    uint last_row = min(row_base + 15, p.rows - 1);
    uint last_pos = start + last_row;

    for (uint i = tid; i < 16 * 128; i += 256) {
        uint row = row_base + i / 128;
        uint d = i % 128;
        q[i] = row < p.rows
            ? query[(ulong(head) * p.rows + row) * 128 + d]
            : 0.0f;
        result[i] = 0.0f;
    }
    if (tid < 16) {
        maxima[tid] = -INFINITY;
        totals[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base <= last_pos; base += 16) {
        for (uint i = tid; i < 16 * 128; i += 256) {
            uint token = base + i % 16;
            uint d = i / 16;
            kv[i] = token <= last_pos
                ? widen(key_cache[(ulong(kv_head) * p.capacity + token) * 128 + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 4) {
            uint row = (sg / 2) * 8;
            uint key = (sg % 2) * 8;
            simdgroup_float8x8 score(0);
            for (uint d = 0; d < 128; d += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, &q[row * 128 + d], 128);
                simdgroup_load(b, &kv[d * 16 + key], 16);
                simdgroup_multiply_accumulate(score, a, b, score);
            }
            simdgroup_store(score, &scores[row * 16 + key], 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = sg; row < 16; row += 8) {
            uint local_row = row_base + row;
            uint position = start + local_row;
            uint token = base + lane;
            bool valid = lane < 16 && local_row < p.rows && token <= position;
            float value = valid ? scores[row * 16 + lane] * p.scale : -INFINITY;
            float next_max = max(maxima[row], simd_max(value));
            if (local_row >= p.rows) next_max = 0.0f;
            float alpha = exp(maxima[row] - next_max);
            float probability = valid ? exp(value - next_max) : 0.0f;
            float total = totals[row] * alpha + simd_sum(probability);
            if (lane < 16) scores[row * 16 + lane] = probability;
            if (lane == 0) {
                maxima[row] = next_max;
                totals[row] = total;
                alphas[row] = alpha;
            }
        }

        for (uint i = tid; i < 16 * 128; i += 256) {
            uint token = base + i / 128;
            uint d = i % 128;
            kv[i] = token <= last_pos
                ? widen(value_cache[(ulong(kv_head) * p.capacity + token) * 128 + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint row = (sg / 4) * 8;
        uint col = (sg % 4) * 8;
        simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);
        for (uint k = 0; k < 16; k += 8) {
            simdgroup_float8x8 a, b0, b1, b2, b3;
            simdgroup_load(a, &scores[row * 16 + k], 16);
            simdgroup_load(b0, &kv[k * 128 + col], 128);
            simdgroup_load(b1, &kv[k * 128 + col + 32], 128);
            simdgroup_load(b2, &kv[k * 128 + col + 64], 128);
            simdgroup_load(b3, &kv[k * 128 + col + 96], 128);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, &kv[row * 128 + col], 128);
        simdgroup_store(acc1, &kv[row * 128 + col + 32], 128);
        simdgroup_store(acc2, &kv[row * 128 + col + 64], 128);
        simdgroup_store(acc3, &kv[row * 128 + col + 96], 128);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 16 * 128; i += 256)
            result[i] = alphas[i / 128] * result[i] + kv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < 16 * 128; i += 256) {
        uint row = row_base + i / 128;
        uint d = i % 128;
        if (row < p.rows)
            output[ulong(row) * p.heads * 128 + ulong(head) * 128 + d] =
                result[i] / totals[i / 128];
    }
}

kernel void qwen_cache(device const float* k_input [[buffer(0)]],
                               device const float* v_input [[buffer(1)]],
                               device ushort* k_cache [[buffer(2)]],
                               device ushort* v_cache [[buffer(3)]],
                               constant CacheShape& p [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
    uint row_width = p.rows * p.head_dim;
    if (ulong(i) >= ulong(p.kv_heads) * row_width) return;
    uint head = i / row_width;
    uint remainder = i % row_width;
    uint position = remainder / p.head_dim;
    uint d = remainder % p.head_dim;
    ulong destination = (ulong(head) * p.capacity + p.position + position) * p.head_dim + d;
    k_cache[destination] = narrow(k_input[ulong(head) * row_width + remainder]);
    v_cache[destination] = narrow(v_input[ulong(head) * row_width + remainder]);
}

kernel void qwen_drope(device const float* q_input [[buffer(0)]],
                                   device const float* k_input [[buffer(1)]],
                                   device const float* v_input [[buffer(2)]],
                                   device float* q_output [[buffer(3)]],
                                   device ushort* k_cache [[buffer(4)]],
                                   device ushort* v_cache [[buffer(5)]],
                                   device const float* rope_table [[buffer(6)]],
                                   device const uint* positions [[buffer(7)]],
                                   constant DecodeRopeShape& p [[buffer(8)]],
                                   uint i [[thread_position_in_grid]]) {
    uint q_width = p.heads * p.head_dim;
    uint kv_width = p.kv_heads * p.head_dim;
    if (i >= p.batch * q_width) return;
    uint row = i / q_width;
    uint col = i % q_width;
    uint head = col / p.head_dim;
    uint d = col % p.head_dim;
    uint position = positions[row];
    uint half_dim = p.head_dim / 2;
    uint partner = d < half_dim ? d + half_dim : d - half_dim;
    float sign = d < half_dim ? -1.0f : 1.0f;
    ulong table_index = ulong(position) * p.head_dim + 2 * (d % half_dim);
    float cs = rope_table[table_index], sn = rope_table[table_index + 1];
    ulong q_base = ulong(row) * q_width;
    ulong kv_base = ulong(row) * kv_width;
    ulong q_output_base = (ulong(head) * p.batch + row) * p.head_dim;
    q_output[q_output_base + d] =
        q_input[q_base + col] * cs + sign * q_input[q_base + head * p.head_dim + partner] * sn;
    if (col < kv_width) {
        uint kv_head = col / p.head_dim;
        uint kv_d = col % p.head_dim;
        uint kv_partner = kv_d < half_dim ? kv_d + half_dim : kv_d - half_dim;
        float kv_sign = kv_d < half_dim ? -1.0f : 1.0f;
        ulong kv_table_index = ulong(position) * p.head_dim + 2 * (kv_d % half_dim);
        float kv_cs = rope_table[kv_table_index], kv_sn = rope_table[kv_table_index + 1];
        ulong destination = ulong(row) * p.cache_stride
            + (ulong(kv_head) * p.capacity + position) * p.head_dim + kv_d;
        k_cache[destination] = narrow(
            k_input[kv_base + col] * kv_cs
                + kv_sign * k_input[kv_base + kv_head * p.head_dim + kv_partner] * kv_sn);
        v_cache[destination] = narrow(v_input[kv_base + col]);
    }
}

// One SIMD-group owns one query head. Each lane carries four head dimensions
// and performs an online softmax over the cached sequence, avoiding a score
// matrix and avoiding a separate softmax/value pass.
kernel void qwen_dattn(device const float* query [[buffer(0)]],
                                  device const ushort* key_cache [[buffer(1)]],
                                  device const ushort* value_cache [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  device const uint* positions [[buffer(4)]],
                                  constant DecodeAttentionShape& p [[buffer(5)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    if (group.x >= p.batch * p.heads) return;
    uint row = group.x / p.heads;
    uint head = group.x % p.heads;
    uint kv_head = head / (p.heads / p.kv_heads);
    uint sequence = positions[row] + 1;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint position = 0; position < sequence; ++position) {
        float partial = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32) {
            ulong index = (ulong(head) * p.batch + row) * p.head_dim + d;
            ulong cache_index = ulong(row) * p.cache_stride
                + (ulong(kv_head) * p.capacity + position) * p.head_dim + d;
            partial = fma(query[index], widen(key_cache[cache_index]), partial);
        }
        float score = simd_sum(partial) * p.scale;
        float next_maximum = max(maximum, score);
        float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next_maximum);
        float new_scale = exp(score - next_maximum);
        for (uint d = lane, slot = 0; d < p.head_dim; d += 32, ++slot) {
            ulong cache_index = ulong(row) * p.cache_stride
                + (ulong(kv_head) * p.capacity + position) * p.head_dim + d;
            accumulated[slot] = accumulated[slot] * old_scale
                + new_scale * widen(value_cache[cache_index]);
        }
        denominator = denominator * old_scale + new_scale;
        maximum = next_maximum;
    }
    for (uint d = lane, slot = 0; d < p.head_dim; d += 32, ++slot)
        output[ulong(row) * p.heads * p.head_dim + ulong(head) * p.head_dim + d] =
            accumulated[slot] / denominator;
}

inline bool argmax_better(float value, uint index, float best, uint best_index) {
    return value > best || (value == best && index < best_index);
}

kernel void qwen_argmax(device const float* values [[buffer(0)]],
                        device uint* result [[buffer(1)]],
                        device atomic_uint* invalid [[buffer(2)]],
                        constant ArgmaxShape& p [[buffer(3)]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];
    float best_value = -INFINITY;
    uint best_index = 0xffffffffu;
    for (uint index = tid; index < p.width; index += 256) {
        float value = values[index];
        if (!isfinite(value)) {
            atomic_store_explicit(invalid, 1u, memory_order_relaxed);
            continue;
        }
        if (argmax_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride /= 2) {
        if (tid < stride && argmax_better(
                best_values[tid + stride], best_indices[tid + stride],
                best_values[tid], best_indices[tid])) {
            best_values[tid] = best_values[tid + stride];
            best_indices[tid] = best_indices[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) result[0] = best_indices[0];
}

kernel void qwen_argn(device const float* values [[buffer(0)]],
                              device uint* result [[buffer(1)]],
                              device atomic_uint* invalid [[buffer(2)]],
                              constant ArgmaxBatchShape& p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];
    float best_value = -INFINITY;
    uint best_index = 0xffffffffu;
    device const float* row_values = values + ulong(row) * p.width;
    for (uint index = tid; index < p.width; index += 256) {
        float value = row_values[index];
        if (!isfinite(value)) {
            atomic_store_explicit(invalid, 1u, memory_order_relaxed);
            continue;
        }
        if (argmax_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride /= 2) {
        if (tid < stride && argmax_better(
                best_values[tid + stride], best_indices[tid + stride],
                best_values[tid], best_indices[tid])) {
            best_values[tid] = best_values[tid + stride];
            best_indices[tid] = best_indices[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) result[row] = best_indices[0];
}

kernel void qwen_repkv(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant QwenShape& p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.width) return;
    uint head = i / (p.rows * p.head_dim);
    uint remainder = i % (p.rows * p.head_dim);
    uint source_head = head / (p.heads / p.kv_heads);
    output[i] = input[ulong(source_head) * p.rows * p.head_dim + remainder];
}

kernel void qwen_aout(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  constant QwenShape& p [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.width) return;
    uint pos = i / p.width, head = (i % p.width) / p.head_dim;
    uint d = i % p.head_dim;
    output[i] = input[(ulong(head) * p.rows + pos) * p.head_dim + d];
}

kernel void qwen_silu(device const float* gate [[buffer(0)]],
                      device const float* up [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant QwenShape& p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(p.rows) * p.hidden) return;
    float value = gate[i];
    output[i] = value * (1.0f / (1.0f + exp(-value))) * up[i];
}

kernel void qwen_silu16(device const half* values [[buffer(0)]],
                        device half* output [[buffer(1)]],
                        constant ulong2& p [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= p.x) return;
    ulong row = ulong(i) / p.y;
    ulong col = ulong(i) - row * p.y;
    ulong base = row * 2 * p.y + col;
    float gate = float(values[base]);
    output[i] = half(gate * (1.0f / (1.0f + exp(-gate))) * float(values[base + p.y]));
}

kernel void qwen_residual(device float* input [[buffer(0)]],
                          device const float* update [[buffer(1)]],
                          constant QwenShape& p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width) input[i] += update[i];
}

kernel void qwen_bias(device float* values [[buffer(0)]],
                      device const ushort* bias [[buffer(1)]],
                      constant QwenShape& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (ulong(i) < ulong(p.rows) * p.width)
        values[i] += widen(bias[i % p.width]);
}
