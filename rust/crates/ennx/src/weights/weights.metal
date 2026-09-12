#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kMaxNeighbors = 2048;

struct Block {
    uint offset;
    uint length;
    uint bits;
    float quantization_scale;
    float metric_scale;
    float weight;
};

struct Params {
    uint row_bytes;
    uint observations;
    uint candidates;
    uint blocks;
    uint neighbors;
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
    uint acquisition;
};

struct Best {
    uint index;
    float score;
};

inline float qvalue(device const uchar* row, uint byte_base, uint element, uint bits) {
    if (bits == 4) {
        uchar byte = row[byte_base + element / 2];
        uint shift = (element & 1u) * 4u;
        return float((byte >> shift) & 0x0fu);
    }
    return float(row[byte_base + element]);
}

kernel void select_best(
    device const float* scores [[buffer(0)]],
    device Best* best_out [[buffer(1)]],
    constant uint& candidates [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    threadgroup float best_scores[kThreads];
    threadgroup uint best_indices[kThreads];

    float best_score = -INFINITY;
    uint best_index = 0;
    for (uint index = thread_index; index < candidates; index += kThreads) {
        float score = scores[index];
        if (score > best_score || (score == best_score && index < best_index)) {
            best_score = score;
            best_index = index;
        }
    }
    best_scores[thread_index] = best_score;
    best_indices[thread_index] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreads >> 1; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            float right_score = best_scores[thread_index + stride];
            uint right_index = best_indices[thread_index + stride];
            if (right_score > best_scores[thread_index]
                || (right_score == best_scores[thread_index] && right_index < best_indices[thread_index])) {
                best_scores[thread_index] = right_score;
                best_indices[thread_index] = right_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        best_out[0].index = best_indices[0];
        best_out[0].score = best_scores[0];
    }
}

kernel void score_weight_neighbors(
    device const uchar* observations [[buffer(0)]],
    device const float* outcomes [[buffer(1)]],
    device const uchar* candidates [[buffer(2)]],
    device const Block* blocks [[buffer(3)]],
    device float* scores [[buffer(4)]],
    device const float* thompson_draws [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint candidate_index = group_index.x;
    if (candidate_index >= params.candidates) {
        return;
    }

    threadgroup float partials[kThreads];
    threadgroup float nearest_distances[kMaxNeighbors];
    threadgroup uint nearest_indices[kMaxNeighbors];

    if (thread_index == 0) {
        for (uint k = 0; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* candidate = candidates + ulong(candidate_index) * ulong(params.row_bytes);
    for (uint observation_index = 0; observation_index < params.observations; ++observation_index) {
        device const uchar* observation = observations + ulong(observation_index) * ulong(params.row_bytes);
        float local = 0.0f;
        uint byte_base = 0;
        for (uint block_index = 0; block_index < params.blocks; ++block_index) {
            Block block = blocks[block_index];
            float scale = block.quantization_scale;
            for (uint element = thread_index; element < block.length; element += kThreads) {
                float a = qvalue(candidate, byte_base, element, block.bits) * scale;
                float b = qvalue(observation, byte_base, element, block.bits) * scale;
                float delta = a - b;
                local = fma(delta, delta * block.weight, local);
            }
            byte_base += (block.bits == 4) ? ((block.length + 1u) / 2u) : block.length;
        }
        partials[thread_index] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = kThreads >> 1; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (thread_index == 0) {
            float distance = partials[0];
            uint insert_at = params.neighbors;
            for (uint k = 0; k < params.neighbors; ++k) {
                if (distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && observation_index < nearest_indices[k])) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at < params.neighbors) {
                for (uint k = params.neighbors - 1; k > insert_at; --k) {
                    nearest_distances[k] = nearest_distances[k - 1];
                    nearest_indices[k] = nearest_indices[k - 1];
                }
                nearest_distances[insert_at] = distance;
                nearest_indices[insert_at] = observation_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        float weight_sum = 0.0f;
        float weighted_outcome = 0.0f;
        float squared_weight_sum = 0.0f;
        float weighted_noise = 0.0f;
        float reference_weight = 1.0f;
        if (params.acquisition == 1u) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[0]
                + params.aleatoric_scale;
            reference_weight = max(1.0f / max(variance, 1.0e-12f), 1.17549435e-38f);
        }
        for (uint k = 0; k < params.neighbors; ++k) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / max(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_outcome += weight * outcomes[nearest_indices[k]];
            if (params.acquisition == 1u) {
                // Ratios to the strongest weight avoid underflow in the squared norm.
                float draw_weight = weight / reference_weight;
                squared_weight_sum += draw_weight * draw_weight;
                weighted_noise += draw_weight * thompson_draws[nearest_indices[k]];
            }
        }
        float mean = weighted_outcome / max(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / max(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            // Shared observation noise gives a consistent posterior across candidates.
            float normalized_noise = weighted_noise / max(sqrt(squared_weight_sum), 1.0e-12f);
            scores[candidate_index] = mean + se * normalized_noise;
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}
