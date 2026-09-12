#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

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

struct SparseEdit {
    uint leaf;
    uint element;
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
    uint num_pert;
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

inline Seed seed_at(Seed base, uint index) {
    Seed seed;
    seed.low = hash(base, index);
    seed.high = hash(base, index ^ 0xa511e9b3);
    return seed;
}

kernel void fill_seeds(
    device Seed* seeds [[buffer(0)]],
    constant Seed& base [[buffer(1)]],
    constant uint& candidates [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < candidates) {
        seeds[index] = seed_at(base, index);
    }
}

inline uint edit_leaf(uint global, device const Leaf* leaves, uint leaf_count) {
    uint start = 0;
    for (uint index = 0; index < leaf_count; ++index) {
        uint end = start + leaves[index].length;
        if (global < end) {
            return index;
        }
        start = end;
    }
    return leaf_count - 1;
}

inline uint edit_element(uint global, device const Leaf* leaves, uint leaf) {
    uint start = 0;
    for (uint index = 0; index < leaf; ++index) {
        start += leaves[index].length;
    }
    return global - start;
}

kernel void fill_edits(
    device SparseEdit* edits [[buffer(0)]],
    device const Seed* seeds [[buffer(1)]],
    device const Leaf* leaves [[buffer(2)]],
    device const Params& params [[buffer(3)]],
    uint candidate [[thread_position_in_grid]]
) {
    if (params.candidates == 0) return;
    if (candidate >= params.candidates) {
        return;
    }
    Seed seed = seeds[candidate];
    uint start = candidate * params.num_pert;
    for (uint draw = 0; draw < params.num_pert; ++draw) {
        uint key = draw * 0x85ebca6b;
        Seed edit_seed;
        edit_seed.low = seed.low ^ 0xd192ed03;
        edit_seed.high = seed.high ^ 0xd1b54a32;
        uint global = hash(edit_seed, key) % params.center_count;
        bool duplicate = true;
        while (duplicate) {
            duplicate = false;
            for (uint prior = 0; prior < draw; ++prior) {
                SparseEdit edit = edits[start + prior];
                uint prior_global = edit.element;
                for (uint leaf = 0; leaf < edit.leaf; ++leaf) {
                    prior_global += leaves[leaf].length;
                }
                if (prior_global == global) {
                    duplicate = true;
                    global = (global + 1) % params.center_count;
                    break;
                }
            }
        }
        uint leaf = edit_leaf(global, leaves, params.leaves);
        edits[start + draw].leaf = leaf;
        edits[start + draw].element = edit_element(global, leaves, leaf);
    }
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

inline uint sparse_code(uint code, Seed seed, uint element, Leaf leaf) {
    if (leaf.whole == 0u && leaf.threshold == 0u) {
        return code;
    }
    leaf.whole = max(leaf.whole, 1u);
    leaf.threshold = 0u;
    return perturb(code, seed, element, leaf);
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
    device const Params& params [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    if (params.candidates == 0) return;
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

kernel void base_distance(
    device const uchar* rows [[buffer(0)]],
    device const uint* history_slots [[buffer(1)]],
    device const Leaf* leaves [[buffer(2)]],
    device float* distances [[buffer(3)]],
    device const Params& params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    if (params.candidates == 0) return;
    uint observation = group_index.x;
    if (observation >= params.history) {
        return;
    }
    device const uchar* base =
        rows + ulong(params.base_slot) * ulong(params.row_stride);
    device const uchar* row =
        rows + ulong(history_slots[observation]) * ulong(params.row_stride);
    float sum = 0.0f;
    for (uint leaf_index = 0; leaf_index < params.leaves; ++leaf_index) {
        Leaf leaf = leaves[leaf_index];
        for (uint element = thread_index; element < leaf.length; element += kThreads) {
            float delta = decode_code(code_at(base, leaf, element), leaf.encoding, leaf.scale)
                - decode_code(code_at(row, leaf, element), leaf.encoding, leaf.scale);
            sum = fma(delta, delta * leaf.weight, sum);
        }
    }

    threadgroup float partials[kThreads / 32];
    float simd_val = simd_sum(sum);
    uint lane = thread_index % 32;
    uint simd_index = thread_index / 32;
    if (lane == 0) {
        partials[simd_index] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f;
        for (uint i = 0; i < kThreads / 32; ++i) {
            total += partials[i];
        }
        distances[observation] = total;
    }
}

kernel void score_trials(
    device const float* partials_in [[buffer(0)]],
    device const float* outcomes [[buffer(1)]],
    device const float* draws [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device const Params& params [[buffer(4)]],
    device const uint* history_slots [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    if (params.candidates == 0) return;
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
        float weighted_noise = 0.0f;
        float weight_squared_sum = 0.0f;
        float reference_weight = 1.0f;
        if (params.acquisition == 1u) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[0]
                + params.aleatoric_scale;
            reference_weight = max(1.0f / max(variance, 1.0e-12f), 1.17549435e-38f);
        }
        for (uint k = 0; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / max(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
            if (params.acquisition == 1u) {
                float draw_weight = weight / reference_weight;
                weighted_noise += draw_weight * draws[history_slots[nearest_indices[k]]];
                weight_squared_sum += draw_weight * draw_weight;
            }
        }
        float mean = weighted_value / max(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / max(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * (weighted_noise / max(sqrt(weight_squared_sum), 1.0e-12f));
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

kernel void score_sparse(
    device const uchar* rows [[buffer(0)]],
    device const uint* history_slots [[buffer(1)]],
    device const float* outcomes [[buffer(2)]],
    device const Seed* seeds [[buffer(3)]],
    device const float* draws [[buffer(4)]],
    device const Leaf* leaves [[buffer(5)]],
    device const SparseEdit* edits [[buffer(6)]],
    device const float* base_distances [[buffer(7)]],
    device float* scores [[buffer(8)]],
    device const Params& params [[buffer(9)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    if (params.candidates == 0) return;
    uint candidate = group_index.x;
    if (candidate >= params.candidates) {
        return;
    }
    threadgroup float distances[kMaxHistory];
    threadgroup float nearest_distances[kMaxHistory];
    threadgroup uint nearest_indices[kMaxHistory];
    if (thread_index < params.history) {
        device const uchar* base =
            rows + ulong(params.base_slot) * ulong(params.row_stride);
        device const uchar* row =
            rows + ulong(history_slots[thread_index]) * ulong(params.row_stride);
        Seed seed = seeds[candidate];
        float distance = base_distances[thread_index];
        for (uint edit_index = 0; edit_index < params.num_pert; ++edit_index) {
            SparseEdit edit = edits[candidate * params.num_pert + edit_index];
            Leaf leaf = leaves[edit.leaf];
            uint base_code = code_at(base, leaf, edit.element);
            uint observed_code = code_at(row, leaf, edit.element);
            uint candidate_code = sparse_code(
                base_code,
                seed,
                leaf.element_offset + edit.element,
                leaf
            );
            float base_delta = decode_code(base_code, leaf.encoding, leaf.scale)
                - decode_code(observed_code, leaf.encoding, leaf.scale);
            float candidate_delta = decode_code(candidate_code, leaf.encoding, leaf.scale)
                - decode_code(observed_code, leaf.encoding, leaf.scale);
            distance += (candidate_delta * candidate_delta - base_delta * base_delta)
                * leaf.weight;
        }
        distances[thread_index] = max(distance, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        for (uint k = 0; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0;
        }
        for (uint h = 0; h < params.history; ++h) {
            float distance = distances[h];
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
        float weight_sum = 0.0f;
        float weighted_value = 0.0f;
        float weighted_noise = 0.0f;
        float weight_squared_sum = 0.0f;
        float reference_weight = 1.0f;
        if (params.acquisition == 1u) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[0]
                + params.aleatoric_scale;
            reference_weight = max(1.0f / max(variance, 1.0e-12f), 1.17549435e-38f);
        }
        for (uint k = 0; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / max(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
            if (params.acquisition == 1u) {
                float draw_weight = weight / reference_weight;
                weighted_noise += draw_weight * draws[history_slots[nearest_indices[k]]];
                weight_squared_sum += draw_weight * draw_weight;
            }
        }
        float mean = weighted_value / max(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / max(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate] = mean + se * (weighted_noise / max(sqrt(weight_squared_sum), 1.0e-12f));
        } else if (params.acquisition == 2u) {
            scores[candidate] = mean + se;
        } else {
            scores[candidate] = mean + params.beta * se;
        }
    }
}

kernel void pick_trial(
    device const float* scores [[buffer(0)]],
    device uint* choice [[buffer(1)]],
    device float* selected_scores [[buffer(2)]],
    device const Params& params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (params.candidates == 0) return;
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
            selected_scores[0] = best_score;
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

kernel void write_sparse(
    device uchar* rows [[buffer(0)]],
    device const Seed* seeds [[buffer(1)]],
    device const uint* choice [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const SparseEdit* edits [[buffer(4)]],
    device const Params& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    if (params.candidates == 0) return;
    if (thread_index != 0u) {
        return;
    }
    uint candidate = choice[0];
    Seed seed = seeds[candidate];
    device uchar* trial =
        rows + ulong(params.trial_slot) * ulong(params.row_stride);
    for (uint edit_index = 0; edit_index < params.num_pert; ++edit_index) {
        SparseEdit edit = edits[candidate * params.num_pert + edit_index];
        Leaf leaf = leaves[edit.leaf];
        uint byte = leaf.byte_offset
            + (leaf.bits == 4u ? edit.element / 2u : edit.element);
        uchar current = trial[byte];
        uint shift = leaf.bits == 4u ? (edit.element & 1u) * 4u : 0u;
        uint code = leaf.bits == 4u
            ? uint((current >> shift) & 0x0fu)
            : uint(current);
        uint value = sparse_code(code, seed, leaf.element_offset + edit.element, leaf);
        if (leaf.bits == 4u) {
            trial[byte] = uchar((current & ~(0x0fu << shift)) | ((value & 0x0fu) << shift));
        } else {
            trial[byte] = uchar(value);
        }
    }
}

kernel void write_trial(
    device uchar* rows [[buffer(0)]],
    device const Seed* seeds [[buffer(1)]],
    device const uint* choice [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const Tile* tiles [[buffer(4)]],
    device const Params& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    if (params.candidates == 0) return;
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

struct RowSumParams {
    uint row_stride;
    uint row_bytes;
    uint slot;
    uint pad;
};

kernel void row_sum(
    device const uchar* rows [[buffer(0)]],
    device ulong* sums [[buffer(1)]],
    constant RowSumParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    threadgroup ulong partials[256];
    device const uchar* row =
        rows + ulong(params.slot) * ulong(params.row_stride);
    ulong sum = 0;
    for (uint byte = thread_index; byte < params.row_bytes; byte += kThreads) {
        sum += ulong(row[byte]);
    }
    partials[thread_index] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            partials[thread_index] += partials[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        sums[0] = partials[0];
    }
}
