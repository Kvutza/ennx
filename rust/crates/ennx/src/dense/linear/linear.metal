constant uint kLinearThreads = 256;

struct EnnxLinearParams {
    uint rows;
    uint columns;
    uint has_bias;
    uint term_count;
    ulong weight_key;
    ulong weight_start;
    ulong bias_key;
    ulong bias_start;
    float weight_scale;
    float bias_scale;
    uint pad0;
    uint pad1;
};

kernel void dense_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const EnnxDenseTerm* terms [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant EnnxLinearParams& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (row >= params.rows) {
        return;
    }
    ulong row_start = ulong(row) * ulong(params.columns);
    float sum = 0.0f;
    for (
        uint column = thread_index;
        column < params.columns;
        column += kLinearThreads
    ) {
        ulong index = row_start + ulong(column);
        float value = ennx_dense_value(
            weight[index],
            params.weight_key,
            params.weight_start + index,
            params.weight_scale,
            terms,
            params.term_count
        );
        sum = fma(input[column], value, sum);
    }

    threadgroup float partials[kLinearThreads / 32];
    float lane_sum = simd_sum(sum);
    uint lane = thread_index % 32u;
    uint simd_group = thread_index / 32u;
    if (lane == 0u) {
        partials[simd_group] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
        float total = 0.0f;
        for (uint index = 0u; index < kLinearThreads / 32u; ++index) {
            total += partials[index];
        }
        if (params.has_bias != 0u) {
            total += ennx_dense_value(
                bias[row],
                params.bias_key,
                params.bias_start + ulong(row),
                params.bias_scale,
                terms,
                params.term_count
            );
        }
        out[row] = total;
    }
}
