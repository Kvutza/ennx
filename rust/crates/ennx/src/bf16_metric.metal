#include <metal_stdlib>
using namespace metal;

// The Rust ABI fixes the history at 128 and ties the metric to four families.
// No raw model tensors, projections, or approximate neighbor searches enter here.
struct MetricParams {
    uint rows, candidates, samples, neighbors, local, pad;
    float epistemic, aleatoric;
};

kernel void metric_distances(
    device const float4* components [[buffer(0)]],
    device const float4* weights [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant MetricParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.candidates * p.rows * p.rows) return;
    uint candidate = index / (p.rows * p.rows);
    uint pair = index % (p.rows * p.rows);
    float4 c = components[(pair / p.rows) * 128 + pair % p.rows];
    float4 w = weights[candidate];
    // Ordered scalar sums; compile with fast math disabled.
    distances[index] = ((c.x * w.x + c.y * w.y) + c.z * w.z) + c.w * w.w;
}

inline void metric_sort(threadgroup float* d, threadgroup uint* ids, uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = 2; span <= 128; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            uint other = lane ^ stride;
            if (other > lane) {
                bool greater = d[lane] > d[other] ||
                    (d[lane] == d[other] && ids[lane] > ids[other]);
                bool less = d[lane] < d[other] ||
                    (d[lane] == d[other] && ids[lane] < ids[other]);
                if (((lane & span) == 0 && greater) || ((lane & span) != 0 && less)) {
                    float value = d[lane]; d[lane] = d[other]; d[other] = value;
                    uint id = ids[lane]; ids[lane] = ids[other]; ids[other] = id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void metric_radii(
    device const float* distances [[buffer(0)]],
    device float* radii [[buffer(1)]],
    constant MetricParams& p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint row = group % p.rows;
    threadgroup float d[128];
    threadgroup uint ids[128];
    d[lane] = lane < p.rows && lane != row ? distances[group * p.rows + lane] : INFINITY;
    ids[lane] = lane;
    metric_sort(d, ids, lane);
    if (lane == 0) radii[group] = sqrt(max(d[min(p.local, p.rows - 1) - 1], 1e-12f));
}

kernel void metric_scores(
    device const float* distances [[buffer(0)]],
    device const float* radii [[buffer(1)]],
    device const float2* outcomes [[buffer(2)]],
    device const uint* selected [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant MetricParams& p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    uint candidate = group / p.samples;
    uint row = selected[group % p.samples];
    uint offset = candidate * p.rows;
    threadgroup float d[128];
    threadgroup uint ids[128];
    float value = lane < p.rows && lane != row ?
        distances[(offset + row) * p.rows + lane] : INFINITY;
    if (p.local && lane < p.rows && lane != row) value /= radii[offset + row] * radii[offset + lane];
    d[lane] = value;
    ids[lane] = lane;
    metric_sort(d, ids, lane);
    float3 moment = float3(0.0f);
    if (lane < min(p.neighbors, p.rows - 1)) {
        float2 y = outcomes[ids[lane]];
        float noise = p.aleatoric + y.y;
        float w = 1.0f / (1e-9f + p.epistemic * d[lane] + noise);
        moment = float3(w, w * y.x, w * noise);
    }
    float3 reduced = simd_sum(moment);
    threadgroup float3 partials[4];
    if ((lane & 31) == 0) partials[lane / 32] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float3 total = ((partials[0] + partials[1]) + partials[2]) + partials[3];
        float mu = total.y / total.x;
        float variance = max((1.0f + total.z) / total.x, 1e-9f);
        float residual = outcomes[row].x - mu;
        scores[group] = -0.9189385332046727f - 0.5f * log(variance)
            - 0.5f * residual * residual / variance;
    }
}
