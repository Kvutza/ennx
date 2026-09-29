#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct ReadoutShape {
    uint rows;
    uint vocab;
};
struct ProposalShape {
    uint rows;
    uint vocab;
    uint row_start;
    uint context;
};

// Preserve FP16 logit rounding and the complete vocabulary. Only loss
// sufficient statistics cross the device-memory boundary.
kernel void fbt_readout_loss_tiles(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const uint* labels [[buffer(2)]],
    device float4* partials [[buffer(3)]],
    constant ReadoutShape& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half logits[128 * 64];
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = group.y * 128;
    const uint first_column = group.x * 64;
    const uint rows = min(128u, p.rows - first_row);
    auto a = tensor(input + ulong(first_row) * 512,
                    dextents<int32_t, 2>{512, int32_t(rows)});
    auto b = tensor(weights + first_column,
                    dextents<int32_t, 2>{64, 512},
                    array<int32_t, 2>{1, int32_t(p.vocab)});
    auto c = tensor(logits, dextents<int32_t, 2>{64, int32_t(rows)});
    auto values = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    op.run(a, b, values);
    values.store(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= rows) return;
    const uint row = first_row + tid;
    float maximum = -INFINITY;
    for (uint column = 0; column < 64; ++column)
        maximum = max(maximum, float(logits[tid * 64 + column]));
    float sum = 0.0f;
    for (uint column = 0; column < 64; ++column)
        sum += exp(float(logits[tid * 64 + column]) - maximum);
    const uint label = labels[row];
    const float target = label >= first_column && label < first_column + 64
        ? float(logits[tid * 64 + label - first_column]) : 0.0f;
    partials[ulong(row) * (p.vocab / 64) + group.x] = float4(maximum, sum, target, 0.0f);
}

METAL_FUNC ulong fbt_decode_mix(ulong value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

// One full-vocabulary proposal per row without materializing [rows, vocab].
// The same absolute-position Gumbel draw makes repeated Jacobi verification
// deterministic and therefore capable of reproducing sampled AR decoding.
kernel void fbt_readout_proposal_tiles(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float4* partials [[buffer(2)]],
    constant ProposalShape& p [[buffer(3)]],
    device const ulong* seeds [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    device const uint* labels [[buffer(6)]],
    device float4* loss_partials [[buffer(7)]],
    constant bool& score_targets [[buffer(8)]],
#ifdef DENOISE
    device float2* moments [[buffer(9)]],
#endif
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half logits[128 * 64];
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = group.y * 128;
    const uint first_column = group.x * 64;
    const uint rows = min(128u, p.rows - first_row);
    auto a = tensor(input + ulong(first_row) * 512,
                    dextents<int32_t, 2>{512, int32_t(rows)});
    auto b = tensor(weights + first_column,
                    dextents<int32_t, 2>{64, 512},
                    array<int32_t, 2>{1, int32_t(p.vocab)});
    auto c = tensor(logits, dextents<int32_t, 2>{64, int32_t(rows)});
    auto values = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    op.run(a, b, values);
    values.store(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= rows) return;
    const uint row = first_row + tid;
    float best = -INFINITY;
    uint index = UINT_MAX;
    float maximum = -INFINITY;
#ifdef DENOISE
    float chosen_logit = 0.0f;
    float scaled_max = -INFINITY;
#endif
    for (uint column = 0; column < 64 && first_column + column < p.vocab; ++column) {
        const uint token = first_column + column;
        const float logit = float(logits[tid * 64 + column]);
        maximum = max(maximum, logit);
#ifdef DENOISE
        scaled_max = max(scaled_max, logit / max(temperature, 1e-6f));
#endif
        float value = logit;
        if (temperature > 0.0f) {
            // A draft predicts the token at its row; a causal target predicts
            // that token from the preceding row. Share its Gumbel draw so equal
            // distributions do not incur random rejection.
#ifdef DENOISE
            const uint absolute_row = max(1u, p.row_start + row) - 1;
#else
            const uint absolute_row = p.row_start + row;
#endif
            const uint sequence = absolute_row / p.context;
            const uint position = absolute_row % p.context;
            const ulong bits = fbt_decode_mix(
                seeds[sequence] ^ (ulong(position) << 32) ^ token);
            const float uniform = (float(uint(bits >> 41)) + 0.5f) / 8388608.0f;
            value -= temperature * log(-log(uniform));
        }
        if (value > best || (value == best && token < index)) {
            best = value;
            index = token;
#ifdef DENOISE
            chosen_logit = logit;
#endif
        }
    }
    partials[ulong(row) * (p.vocab / 64) + group.x] =
        float4(best, as_type<float>(index),
#ifdef DENOISE
            chosen_logit,
#else
            0.0f,
#endif
            0.0f);
#ifdef DENOISE
    float scaled_sum = 0.0f;
    for (uint column = 0; column < 64; ++column)
        scaled_sum += exp(float(logits[tid * 64 + column]) / max(temperature, 1e-6f) - scaled_max);
    moments[ulong(row) * (p.vocab / 64) + group.x] = float2(scaled_max, scaled_sum);
#endif
    if (score_targets) {
        float sum = 0.0f;
        for (uint column = 0; column < 64 && first_column + column < p.vocab; ++column)
            sum += exp(float(logits[tid * 64 + column]) - maximum);
        const uint label = labels[p.row_start + row];
        const float target = label >= first_column && label < first_column + 64
            ? float(logits[tid * 64 + label - first_column]) : 0.0f;
        loss_partials[ulong(row) * (p.vocab / 64) + group.x] =
            float4(maximum, sum, target, 0.0f);
    }
}
