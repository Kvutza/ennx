#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kMaxHistory = 128;
constant uint kHistoryBatch = 8;

constant float kFp4E2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

inline float decode_fp8_e4m3(uint code) {
    uint sign = (code & 0x80u) != 0u ? 1u : 0u;
    uint exp = (code >> 3u) & 0x0fu;
    uint mant = code & 0x07u;
    float s = sign != 0u ? -1.0f : 1.0f;
    if (exp == 0u) {
        return s * (float(mant) / 8.0f) * 0.015625f;
    }
    return s * (1.0f + float(mant) / 8.0f) * pow(2.0f, float(exp) - 7.0f);
}

inline float decode_fp8_e5m2(uint code) {
    uint sign = (code & 0x80u) != 0u ? 1u : 0u;
    uint exp = (code >> 2u) & 0x1fu;
    uint mant = code & 0x03u;
    float s = sign != 0u ? -1.0f : 1.0f;
    if (exp == 0u) {
        return s * (float(mant) / 4.0f) * 0.00006103515625f;
    }
    return s * (1.0f + float(mant) / 4.0f) * pow(2.0f, float(exp) - 15.0f);
}

inline float decode_code(uint code, uint encoding, float scale) {
    if (encoding == 0u || encoding == 1u) {
        return float(code) * scale;
    } else if (encoding == 2u) {
        return kFp4E2M1LUT[code & 0x0fu] * scale;
    } else if (encoding == 3u) {
        return decode_fp8_e4m3(code) * scale;
    } else if (encoding == 4u) {
        return decode_fp8_e5m2(code) * scale;
    }
    return float(code) * scale;
}

struct Leaf {
    uint byte_offset;
    uint element_offset;
    uint length;
    uint bits;
    uint encoding;
    float scale;
    float weight;
    uint whole;
    uint threshold;
};


struct Tile {
    uint leaf;
    uint start;
    uint length;
    uint pad;
};

struct Seed {
    uint low;
    uint high;
};

struct Params {
    uint row_stride;
    uint history;
    uint candidates;
    uint leaves;
    uint tiles;
    uint neighbors;
    uint base_slot;
    uint trial_slot;
    uint center_count;
    uint acquisition;
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
};

struct CenterStep {
    uint parent;
    Seed seed;
};

inline uint hash(Seed seed, uint element) {
    uint value = seed.low ^ element * 0x9e3779b9;
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= seed.high;
    value *= 0x846ca68b;
    return value ^ (value >> 15);
}

inline uint code_at(device const uchar* row, Leaf leaf, uint element) {
    if (leaf.bits == 4) {
        uchar byte = row[leaf.byte_offset + element / 2];
        return (byte >> ((element & 1u) * 4u)) & 0x0fu;
    }
    return row[leaf.byte_offset + element];
}

inline uint perturb(uint code, Seed seed, uint element, Leaf leaf) {
    uint random = hash(seed, element);
    uint amount = leaf.whole + uint((random >> 1u) < (leaf.threshold >> 1u));
    if (amount == 0) {
        return code;
    }
    uint max_code = (1u << leaf.bits) - 1u;
    if ((random & 1u) == 0u) {
        return code >= amount ? code - amount : min(code + amount, max_code);
    }
    return code + amount <= max_code ? code + amount : code >= amount ? code - amount : 0u;
}

inline uint resolve_center(
    uint code,
    device const CenterStep* centers,
    uint center,
    uint element,
    Leaf leaf
) {
    Seed chain[8];
    uint depth = 0u;
    while (center != UINT_MAX && depth < 8u) {
        chain[depth++] = centers[center].seed;
        center = centers[center].parent;
    }
    while (depth > 0u) {
        code = perturb(code, chain[--depth], element, leaf);
    }
    return code;
}

kernel void distance_trials(
    device const uchar* rows [[buffer(0)]],
    device const uint* history_slots [[buffer(1)]],
    device const Seed* seeds [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const Tile* tiles [[buffer(4)]],
    device float* partials_out [[buffer(5)]],
    device const CenterStep* centers [[buffer(6)]],
    device const uint* candidate_centers [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint tile_index = group_index.x % params.tiles;
    uint work_index = group_index.x / params.tiles;
    uint history_groups = (params.history + kHistoryBatch - 1u) / kHistoryBatch;
    uint candidate_group = work_index / history_groups;
    uint history_start = (work_index % history_groups) * kHistoryBatch;
    uint history_count = min(kHistoryBatch, params.history - history_start);
    uint first_candidate = candidate_group * 2u;
    if (first_candidate >= params.candidates) {
        return;
    }
    bool has_second = first_candidate + 1u < params.candidates;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed first_seed = seeds[first_candidate];
    Seed second_seed = has_second ? seeds[first_candidate + 1u] : first_seed;
    uint first_center =
        params.center_count == 0u ? UINT_MAX : candidate_centers[first_candidate];
    uint second_center = params.center_count == 0u || !has_second
        ? UINT_MAX
        : candidate_centers[first_candidate + 1u];
    device const uchar* base =
        rows + ulong(params.base_slot) * ulong(params.row_stride);
    float first_distances[kHistoryBatch];
    float second_distances[kHistoryBatch];
    for (uint h = 0; h < history_count; ++h) {
        first_distances[h] = 0.0f;
        second_distances[h] = 0.0f;
    }

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += kThreads) {
            uint first = tile.start + local_byte * 2u;
            uchar base_byte = base[leaf.byte_offset + first_byte + local_byte];
            uint first_base_low = resolve_center(
                uint(base_byte & 0x0fu),
                centers,
                first_center,
                leaf.element_offset + first,
                leaf
            );
            uint first_low = perturb(
                first_base_low,
                first_seed,
                leaf.element_offset + first,
                leaf
            );
            uint first_high = 0u;
            if (first + 1u < leaf.length) {
                uint first_base_high = resolve_center(
                    uint(base_byte >> 4u),
                    centers,
                    first_center,
                    leaf.element_offset + first + 1u,
                    leaf
                );
                first_high = perturb(
                    first_base_high,
                    first_seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            uint second_low = 0u;
            uint second_high = 0u;
            if (has_second) {
                uint second_base_low = resolve_center(
                    uint(base_byte & 0x0fu),
                    centers,
                    second_center,
                    leaf.element_offset + first,
                    leaf
                );
                second_low = perturb(
                    second_base_low,
                    second_seed,
                    leaf.element_offset + first,
                    leaf
                );
                if (first + 1u < leaf.length) {
                    uint second_base_high = resolve_center(
                        uint(base_byte >> 4u),
                        centers,
                        second_center,
                        leaf.element_offset + first + 1u,
                        leaf
                    );
                    second_high = perturb(
                        second_base_high,
                        second_seed,
                        leaf.element_offset + first + 1u,
                        leaf
                    );
                }
            }
            float first_low_val = decode_code(first_low, leaf.encoding, leaf.scale);
            float first_high_val = decode_code(first_high, leaf.encoding, leaf.scale);
            float second_low_val = decode_code(second_low, leaf.encoding, leaf.scale);
            float second_high_val = decode_code(second_high, leaf.encoding, leaf.scale);
            for (uint h = 0; h < history_count; ++h) {
                device const uchar* observation =
                    rows + ulong(history_slots[history_start + h]) * ulong(params.row_stride);
                uchar observed = observation[leaf.byte_offset + first_byte + local_byte];
                float obs_low_val = decode_code(uint(observed & 0x0fu), leaf.encoding, leaf.scale);
                float first_low_delta = first_low_val - obs_low_val;
                first_distances[h] = fma(
                    first_low_delta,
                    first_low_delta * leaf.weight,
                    first_distances[h]
                );
                if (first + 1u < leaf.length) {
                    float obs_high_val = decode_code(uint(observed >> 4u), leaf.encoding, leaf.scale);
                    float first_high_delta = first_high_val - obs_high_val;
                    first_distances[h] = fma(
                        first_high_delta,
                        first_high_delta * leaf.weight,
                        first_distances[h]
                    );
                }
                if (has_second) {
                    float second_low_delta = second_low_val - obs_low_val;
                    second_distances[h] = fma(
                        second_low_delta,
                        second_low_delta * leaf.weight,
                        second_distances[h]
                    );
                    if (first + 1u < leaf.length) {
                        float obs_high_val = decode_code(uint(observed >> 4u), leaf.encoding, leaf.scale);
                        float second_high_delta = second_high_val - obs_high_val;
                        second_distances[h] = fma(
                            second_high_delta,
                            second_high_delta * leaf.weight,
                            second_distances[h]
                        );
                    }
                }
            }
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += kThreads) {
            uint first_base = resolve_center(
                uint(base[leaf.byte_offset + element]),
                centers,
                first_center,
                leaf.element_offset + element,
                leaf
            );
            uint first_value = perturb(
                first_base,
                first_seed,
                leaf.element_offset + element,
                leaf
            );
            uint second_value = has_second
                ? perturb(
                    resolve_center(
                        uint(base[leaf.byte_offset + element]),
                        centers,
                        second_center,
                        leaf.element_offset + element,
                        leaf
                    ),
                    second_seed,
                    leaf.element_offset + element,
                    leaf
                )
                : 0u;
            float first_val = decode_code(first_value, leaf.encoding, leaf.scale);
            float second_val = decode_code(second_value, leaf.encoding, leaf.scale);
            for (uint h = 0; h < history_count; ++h) {
                device const uchar* observation =
                    rows + ulong(history_slots[history_start + h]) * ulong(params.row_stride);
                float obs_val = decode_code(uint(observation[leaf.byte_offset + element]), leaf.encoding, leaf.scale);
                float first_delta = first_val - obs_val;
                first_distances[h] = fma(
                    first_delta,
                    first_delta * leaf.weight,
                    first_distances[h]
                );
                if (has_second) {
                    float second_delta = second_val - obs_val;
                    second_distances[h] = fma(
                        second_delta,
                        second_delta * leaf.weight,
                        second_distances[h]
                    );
                }
            }
        }
    }

    threadgroup float partials[kThreads / 32];
    for (uint h = 0; h < history_count; ++h) {
        float first_simd_val = simd_sum(first_distances[h]);
        uint simd_lane = thread_index % 32;
        uint simd_id = thread_index / 32;
        if (simd_lane == 0) {
            partials[simd_id] = first_simd_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            float total = 0.0f;
            for (uint i = 0; i < kThreads / 32; ++i) {
                total += partials[i];
            }
            ulong offset =
                (ulong(first_candidate) * ulong(params.history) + ulong(history_start + h))
                * ulong(params.tiles)
                + ulong(tile_index);
            partials_out[offset] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (has_second) {
            float second_simd_val = simd_sum(second_distances[h]);
            if (simd_lane == 0) {
                partials[simd_id] = second_simd_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index == 0) {
                float total = 0.0f;
                for (uint i = 0; i < kThreads / 32; ++i) {
                    total += partials[i];
                }
                ulong offset =
                    (ulong(first_candidate + 1u) * ulong(params.history) + ulong(history_start + h))
                    * ulong(params.tiles)
                    + ulong(tile_index);
                partials_out[offset] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


kernel void score_trials(
    device const float* partials_in [[buffer(0)]],
    device const float* outcomes [[buffer(1)]],
    device const float* draws [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint candidate_index = group_index.x;
    if (candidate_index >= params.candidates) {
        return;
    }
    threadgroup float partials[kThreads];
    threadgroup float nearest_distances[kMaxHistory];
    threadgroup uint nearest_indices[kMaxHistory];
    if (thread_index == 0) {
        for (uint k = 0; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < params.history; ++h) {
        float local = 0.0f;
        ulong base =
            (ulong(candidate_index) * ulong(params.history) + ulong(h))
            * ulong(params.tiles);
        for (uint tile = thread_index; tile < params.tiles; tile += kThreads) {
            local += partials_in[base + ulong(tile)];
        }
        float simd_val = simd_sum(local);
        uint simd_lane = thread_index % 32;
        uint simd_id = thread_index / 32;
        if (simd_lane == 0) {
            partials[simd_id] = simd_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0) {
            float distance = 0.0f;
            for (uint i = 0; i < kThreads / 32; ++i) {
                distance += partials[i];
            }
            uint insert_at = params.neighbors;
            for (uint k = 0; k < params.neighbors; ++k) {
                if (
                    distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && h < nearest_indices[k])
                ) {
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
                nearest_indices[insert_at] = h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }


    if (thread_index == 0) {
        float weight_sum = 0.0f;
        float weighted_value = 0.0f;
        for (uint k = 0; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / max(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
        }
        float mean = weighted_value / max(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / max(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * draws[candidate_index];
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

kernel void pick_trial(
    device const float* scores [[buffer(0)]],
    device uint* choice [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    float best_score = -INFINITY;
    uint best = UINT_MAX;
    for (uint index = thread_index; index < params.candidates; index += threads) {
        float score = scores[index];
        if (score > best_score || (score == best_score && index < best)) {
            best = index;
            best_score = score;
        }
    }

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_score = simd_shuffle_down(best_score, offset);
        uint other = simd_shuffle_down(best, offset);
        if (
            other_score > best_score
            || (other_score == best_score && other < best)
        ) {
            best_score = other_score;
            best = other;
        }
    }

    threadgroup float group_scores[kThreads / 32];
    threadgroup uint group_indices[kThreads / 32];
    uint lane = thread_index % 32u;
    uint simd_index = thread_index / 32u;
    if (lane == 0u) {
        group_scores[simd_index] = best_score;
        group_indices[simd_index] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0u) {
        uint simd_count = (threads + 31u) / 32u;
        best_score = lane < simd_count ? group_scores[lane] : -INFINITY;
        best = lane < simd_count ? group_indices[lane] : UINT_MAX;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_score = simd_shuffle_down(best_score, offset);
            uint other = simd_shuffle_down(best, offset);
            if (
                other_score > best_score
                || (other_score == best_score && other < best)
            ) {
                best_score = other_score;
                best = other;
            }
        }
        if (lane == 0u) {
            choice[0] = best;
        }
    }
}

struct MultiTRParams {
    uint num_regions;
    uint candidates_per_region;
};

kernel void multi_tr_pick_trials(
    device const float* scores [[buffer(0)]],
    device uint* choices [[buffer(1)]],
    device float* selected_scores [[buffer(2)]],
    constant MultiTRParams& params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint region_index [[threadgroup_position_in_grid]]
) {
    if (region_index >= params.num_regions) {
        return;
    }
    device const float* region_scores =
        scores + region_index * params.candidates_per_region;
    float best_score = -INFINITY;
    uint best_index = UINT_MAX;
    for (
        uint index = thread_index;
        index < params.candidates_per_region;
        index += threads
    ) {
        float score = region_scores[index];
        if (
            score > best_score
            || (score == best_score && index < best_index)
        ) {
            best_score = score;
            best_index = index;
        }
    }

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_score = simd_shuffle_down(best_score, offset);
        uint other_index = simd_shuffle_down(best_index, offset);
        if (
            other_score > best_score
            || (other_score == best_score && other_index < best_index)
        ) {
            best_score = other_score;
            best_index = other_index;
        }
    }

    threadgroup float group_scores[kThreads / 32];
    threadgroup uint group_indices[kThreads / 32];
    uint lane = thread_index % 32u;
    uint simd_index = thread_index / 32u;
    if (lane == 0u) {
        group_scores[simd_index] = best_score;
        group_indices[simd_index] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0u) {
        uint simd_count = (threads + 31u) / 32u;
        best_score = lane < simd_count ? group_scores[lane] : -INFINITY;
        best_index = lane < simd_count ? group_indices[lane] : UINT_MAX;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_score = simd_shuffle_down(best_score, offset);
            uint other_index = simd_shuffle_down(best_index, offset);
            if (
                other_score > best_score
                || (other_score == best_score && other_index < best_index)
            ) {
                best_score = other_score;
                best_index = other_index;
            }
        }
        if (lane == 0u) {
            choices[region_index] =
                region_index * params.candidates_per_region + best_index;
            selected_scores[region_index] = best_score;
        }
    }
}

kernel void write_trial(
    device uchar* rows [[buffer(0)]],
    device const Seed* seeds [[buffer(1)]],
    device const uint* choice [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const Tile* tiles [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint tile_index = group_index.x;
    if (tile_index >= params.tiles) {
        return;
    }
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed seed = seeds[choice[0]];
    device const uchar* base =
        rows + ulong(params.base_slot) * ulong(params.row_stride);
    device uchar* trial =
        rows + ulong(params.trial_slot) * ulong(params.row_stride);

    if (leaf.bits == 4) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += kThreads) {
            uint first = tile.start + local_byte * 2u;
            uint low = perturb(
                code_at(base, leaf, first),
                seed,
                leaf.element_offset + first,
                leaf
            );
            uint high = 0;
            if (first + 1u < leaf.length) {
                high = perturb(
                    code_at(base, leaf, first + 1u),
                    seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            trial[leaf.byte_offset + first_byte + local_byte] = uchar(low | (high << 4u));
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += kThreads) {
            trial[leaf.byte_offset + element] = uchar(perturb(
                code_at(base, leaf, element),
                seed,
                leaf.element_offset + element,
                leaf
            ));
        }
    }
}
