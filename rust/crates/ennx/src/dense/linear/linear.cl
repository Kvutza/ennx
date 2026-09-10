#define ENNX_LINEAR_THREADS 256

typedef struct {
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
} EnnxLinearParams;

__kernel void dense_linear(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global const EnnxDenseTerm* terms,
    __global float* out,
    EnnxLinearParams params
) {
    uint row = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (row >= params.rows) {
        return;
    }
    ulong row_start = ((ulong)row) * ((ulong)params.columns);
    float sum = 0.0f;
    for (
        uint column = thread_index;
        column < params.columns;
        column += ENNX_LINEAR_THREADS
    ) {
        ulong index = row_start + ((ulong)column);
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

    __local float partials[ENNX_LINEAR_THREADS];
    partials[thread_index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = ENNX_LINEAR_THREADS / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partials[thread_index] += partials[thread_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (thread_index == 0) {
        float total = partials[0];
        if (params.has_bias != 0) {
            total += ennx_dense_value(
                bias[row],
                params.bias_key,
                params.bias_start + ((ulong)row),
                params.bias_scale,
                terms,
                params.term_count
            );
        }
        out[row] = total;
    }
}
