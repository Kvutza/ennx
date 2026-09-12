#pragma OPENCL FP_CONTRACT OFF

#define THREADS 256u
#define MAX_HISTORY 128u

__constant float kFp4E2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

inline float decode_fp8_e4m3(uint code) {
    uint sign = (code & 0x80u) != 0u ? 1u : 0u;
    uint exp = (code >> 3u) & 0x0fu;
    uint mant = code & 0x07u;
    float s = sign != 0u ? -1.0f : 1.0f;
    if (exp == 0u) {
        return s * (float)(mant) / 8.0f * 0.015625f;
    }
    return s * (1.0f + (float)(mant) / 8.0f) * pown(2.0f, (int)(exp) - 7);
}

inline float decode_fp8_e5m2(uint code) {
    uint sign = (code & 0x80u) != 0u ? 1u : 0u;
    uint exp = (code >> 2u) & 0x1fu;
    uint mant = code & 0x03u;
    float s = sign != 0u ? -1.0f : 1.0f;
    if (exp == 0u) {
        return s * (float)(mant) / 4.0f * 0.00006103515625f;
    }
    return s * (1.0f + (float)(mant) / 4.0f) * pown(2.0f, (int)(exp) - 15);
}

inline float decode_code(uint code, uint encoding, float scale) {
    if (encoding == 0u || encoding == 1u) {
        return (float)(code) * scale;
    } else if (encoding == 2u) {
        return kFp4E2M1LUT[code & 0x0fu] * scale;
    } else if (encoding == 3u) {
        return decode_fp8_e4m3(code) * scale;
    } else if (encoding == 4u) {
        return decode_fp8_e5m2(code) * scale;
    }
    return (float)(code) * scale;
}

typedef struct {
    uint byte_offset;
    uint element_offset;
    uint length;
    uint bits;
    uint encoding;
    float scale;
    float weight;
    uint whole;
    uint threshold;
} Leaf;

typedef struct {
    uint leaf;
    uint start;
    uint length;
    uint pad;
} Tile;

typedef struct {
    uint leaf;
    uint element;
} SparseEdit;

typedef struct {
    uint low;
    uint high;
} Seed;

typedef struct {
    uint row_bytes;
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
} Params;

typedef struct {
    uint parent;
    Seed seed;
} CenterStep;

inline uint trial_hash(Seed seed, uint element) {
    uint value = seed.low ^ element * 0x9e3779b9u;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= seed.high;
    value *= 0x846ca68bu;
    return value ^ (value >> 15u);
}

inline Seed seed_at(Seed base, uint index) {
    Seed seed = {
        trial_hash(base, index),
        trial_hash(base, index ^ 0xa511e9b3u)
    };
    return seed;
}

__kernel void fill_seeds(
    __global Seed* seeds,
    Seed base,
    uint candidates
) {
    uint index = get_global_id(0);
    if (index < candidates) {
        seeds[index] = seed_at(base, index);
    }
}

inline uint edit_leaf(uint coord, __global const Leaf* leaves, uint leaf_count) {
    uint start = 0u;
    for (uint index = 0u; index < leaf_count; ++index) {
        uint end = start + leaves[index].length;
        if (coord < end) {
            return index;
        }
        start = end;
    }
    return leaf_count - 1u;
}

inline uint edit_element(uint coord, __global const Leaf* leaves, uint leaf) {
    uint start = 0u;
    for (uint index = 0u; index < leaf; ++index) {
        start += leaves[index].length;
    }
    return coord - start;
}

__kernel void fill_edits(
    __global SparseEdit* edits,
    __global const Seed* seeds,
    __global const Leaf* leaves,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint candidate = get_global_id(0);
    if (candidate >= params.candidates) {
        return;
    }
    Seed seed = seeds[candidate];
    uint start = candidate * params.num_pert;
    for (uint draw = 0u; draw < params.num_pert; ++draw) {
        uint key = draw * 0x85ebca6bu;
        Seed edit_seed;
        edit_seed.low = seed.low ^ 0xd192ed03u;
        edit_seed.high = seed.high ^ 0xd1b54a32u;
        uint coord = trial_hash(edit_seed, key) % params.center_count;
        bool duplicate = true;
        while (duplicate) {
            duplicate = false;
            for (uint prior = 0u; prior < draw; ++prior) {
                SparseEdit edit = edits[start + prior];
                uint prior_global = edit.element;
                for (uint leaf = 0u; leaf < edit.leaf; ++leaf) {
                    prior_global += leaves[leaf].length;
                }
                if (prior_global == coord) {
                    duplicate = true;
                    coord = (coord + 1u) % params.center_count;
                    break;
                }
            }
        }
        uint leaf = edit_leaf(coord, leaves, params.leaves);
        edits[start + draw].leaf = leaf;
        edits[start + draw].element = edit_element(coord, leaves, leaf);
    }
}

inline uint code_at(__global const uchar* row, Leaf leaf, uint element) {
    if (leaf.bits == 4u) {
        uchar byte = row[leaf.byte_offset + element / 2u];
        return (byte >> ((element & 1u) * 4u)) & 0x0fu;
    }
    return row[leaf.byte_offset + element];
}

inline uint perturb_code(uint code, Seed seed, uint element, Leaf leaf) {
    uint random = trial_hash(seed, element);
    uint amount =
        leaf.whole + (uint)((random >> 1u) < (leaf.threshold >> 1u));
    if (amount == 0u) {
        return code;
    }
    uint max_code = (1u << leaf.bits) - 1u;
    if ((random & 1u) == 0u) {
        return code >= amount ? code - amount : min(code + amount, max_code);
    }
    return code + amount <= max_code
        ? code + amount
        : code >= amount ? code - amount : 0u;
}

inline uint sparse_code(uint code, Seed seed, uint element, Leaf leaf) {
    if (leaf.whole == 0u && leaf.threshold == 0u) {
        return code;
    }
    leaf.whole = max(leaf.whole, 1u);
    leaf.threshold = 0u;
    return perturb_code(code, seed, element, leaf);
}

inline uint resolve_center(
    uint code,
    __global const CenterStep* centers,
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
        code = perturb_code(code, chain[--depth], element, leaf);
    }
    return code;
}

__kernel void distance_trials(
    __global const uchar* rows,
    __global const uint* history_slots,
    __global const Seed* seeds,
    __global const Leaf* leaves,
    __global const Tile* tiles,
    __global float* partials_out,
    __global const CenterStep* centers,
    __global const uint* candidate_centers,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint thread_index = get_local_id(0);
    uint group_index = get_group_id(0);
    uint candidate_group = group_index / params.tiles;
    uint tile_index = group_index - candidate_group * params.tiles;
    uint first_candidate = candidate_group * 2u;
    if (first_candidate >= params.candidates) {
        return;
    }
    int has_second = first_candidate + 1u < params.candidates;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed first_seed = seeds[first_candidate];
    Seed second_seed = has_second ? seeds[first_candidate + 1u] : first_seed;
    uint first_center =
        params.center_count == 0u ? UINT_MAX : candidate_centers[first_candidate];
    uint second_center = params.center_count == 0u || !has_second
        ? UINT_MAX
        : candidate_centers[first_candidate + 1u];
    __global const uchar* base =
        rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
    float first_distances[MAX_HISTORY];
    float second_distances[MAX_HISTORY];
    for (uint h = 0u; h < params.history; ++h) {
        first_distances[h] = 0.0f;
        second_distances[h] = 0.0f;
    }

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += THREADS) {
            uint first = tile.start + local_byte * 2u;
            uchar base_byte = base[leaf.byte_offset + first_byte + local_byte];
            uint first_base_low = resolve_center(
                (uint)(base_byte & 0x0fu),
                centers,
                first_center,
                leaf.element_offset + first,
                leaf
            );
            uint first_low = perturb_code(
                first_base_low,
                first_seed,
                leaf.element_offset + first,
                leaf
            );
            uint first_high = 0u;
            if (first + 1u < leaf.length) {
                uint first_base_high = resolve_center(
                    (uint)(base_byte >> 4u),
                    centers,
                    first_center,
                    leaf.element_offset + first + 1u,
                    leaf
                );
                first_high = perturb_code(
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
                    (uint)(base_byte & 0x0fu),
                    centers,
                    second_center,
                    leaf.element_offset + first,
                    leaf
                );
                second_low = perturb_code(
                    second_base_low,
                    second_seed,
                    leaf.element_offset + first,
                    leaf
                );
                if (first + 1u < leaf.length) {
                    uint second_base_high = resolve_center(
                        (uint)(base_byte >> 4u),
                        centers,
                        second_center,
                        leaf.element_offset + first + 1u,
                        leaf
                    );
                    second_high = perturb_code(
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
                for (uint h = 0u; h < params.history; ++h) {
                    __global const uchar* observation =
                        rows + ((ulong)history_slots[h]) * ((ulong)params.row_bytes);
                    uchar observed = observation[leaf.byte_offset + first_byte + local_byte];
                    float obs_low_val = decode_code((uint)(observed & 0x0fu), leaf.encoding, leaf.scale);
                    float first_low_delta = first_low_val - obs_low_val;
                    first_distances[h] = fma(
                        first_low_delta,
                        first_low_delta * leaf.weight,
                        first_distances[h]
                    );
                    if (first + 1u < leaf.length) {
                        float obs_high_val = decode_code((uint)(observed >> 4u), leaf.encoding, leaf.scale);
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
                            float obs_high_val = decode_code((uint)(observed >> 4u), leaf.encoding, leaf.scale);
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
            for (uint element = tile.start + thread_index; element < end; element += THREADS) {
                uint first_base = resolve_center(
                    (uint)base[leaf.byte_offset + element],
                    centers,
                    first_center,
                    leaf.element_offset + element,
                    leaf
                );
                uint first_value = perturb_code(
                    first_base,
                    first_seed,
                    leaf.element_offset + element,
                    leaf
                );
                uint second_value = has_second
                    ? perturb_code(
                        resolve_center(
                            (uint)base[leaf.byte_offset + element],
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
                for (uint h = 0u; h < params.history; ++h) {
                    __global const uchar* observation =
                        rows + ((ulong)history_slots[h]) * ((ulong)params.row_bytes);
                    float obs_val = decode_code((uint)observation[leaf.byte_offset + element], leaf.encoding, leaf.scale);
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


    __local float partials[THREADS];
    for (uint h = 0u; h < params.history; ++h) {
        partials[thread_index] = first_distances[h];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (thread_index == 0u) {
            ulong offset =
                (((ulong)first_candidate) * ((ulong)params.history) + ((ulong)h))
                * ((ulong)params.tiles)
                + ((ulong)tile_index);
            partials_out[offset] = partials[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (has_second) {
            partials[thread_index] = second_distances[h];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
                if (thread_index < stride) {
                    partials[thread_index] += partials[thread_index + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (thread_index == 0u) {
                ulong offset =
                    (((ulong)(first_candidate + 1u)) * ((ulong)params.history) + ((ulong)h))
                    * ((ulong)params.tiles)
                    + ((ulong)tile_index);
                partials_out[offset] = partials[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

__kernel void base_distance(
    __global const uchar* rows,
    __global const uint* history_slots,
    __global const Leaf* leaves,
    __global float* distances,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint observation = get_group_id(0);
    if (observation >= params.history) {
        return;
    }
    uint thread_index = get_local_id(0);
    __global const uchar* base =
        rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
    __global const uchar* row =
        rows + ((ulong)history_slots[observation]) * ((ulong)params.row_bytes);
    float sum = 0.0f;
    for (uint leaf_index = 0u; leaf_index < params.leaves; ++leaf_index) {
        Leaf leaf = leaves[leaf_index];
        for (uint element = thread_index; element < leaf.length; element += THREADS) {
            float delta =
                decode_code(code_at(base, leaf, element), leaf.encoding, leaf.scale)
                - decode_code(code_at(row, leaf, element), leaf.encoding, leaf.scale);
            sum = fma(delta, delta * leaf.weight, sum);
        }
    }

    __local float partials[THREADS];
    partials[thread_index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            partials[thread_index] += partials[thread_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (thread_index == 0u) {
        distances[observation] = partials[0];
    }
}

__kernel void score_trials(
    __global const float* partials_in,
    __global const float* outcomes,
    __global const float* draws,
    __global float* scores,
    __global const Params *input,
    __global const uint* history_slots
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint candidate_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (candidate_index >= params.candidates) {
        return;
    }
    __local float partials[THREADS];
    __local float nearest_distances[MAX_HISTORY];
    __local uint nearest_indices[MAX_HISTORY];
    if (thread_index == 0u) {
        for (uint k = 0u; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint h = 0u; h < params.history; ++h) {
        float accum = 0.0f;
        ulong base =
            (((ulong)candidate_index) * ((ulong)params.history) + ((ulong)h))
            * ((ulong)params.tiles);
        for (uint tile = thread_index; tile < params.tiles; tile += THREADS) {
            accum += partials_in[base + (ulong)tile];
        }
        partials[thread_index] = accum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (thread_index == 0u) {
            float distance = partials[0];
            uint insert_at = params.neighbors;
            for (uint k = 0u; k < params.neighbors; ++k) {
                if (
                    distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && h < nearest_indices[k])
                ) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at < params.neighbors) {
                for (uint k = params.neighbors - 1u; k > insert_at; --k) {
                    nearest_distances[k] = nearest_distances[k - 1u];
                    nearest_indices[k] = nearest_indices[k - 1u];
                }
                nearest_distances[insert_at] = distance;
                nearest_indices[insert_at] = h;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_index == 0u) {
        float weight_sum = 0.0f;
        float weighted_value = 0.0f;
        float weighted_noise = 0.0f;
        float weight_squared_sum = 0.0f;
        float reference_weight = 1.0f;
        if (params.acquisition == 1u) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[0]
                + params.aleatoric_scale;
            reference_weight = fmax(1.0f / fmax(variance, 1.0e-12f), 1.17549435e-38f);
        }
        for (uint k = 0u; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / fmax(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
            if (params.acquisition == 1u) {
                float draw_weight = weight / reference_weight;
                weighted_noise += draw_weight * draws[history_slots[nearest_indices[k]]];
                weight_squared_sum += draw_weight * draw_weight;
            }
        }
        float mean = weighted_value / fmax(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / fmax(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * (weighted_noise / fmax(sqrt(weight_squared_sum), 1.0e-12f));
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

__kernel void score_sparse(
    __global const uchar* rows,
    __global const uint* history_slots,
    __global const float* outcomes,
    __global const Seed* seeds,
    __global const float* draws,
    __global const Leaf* leaves,
    __global const SparseEdit* edits,
    __global const float* base_distances,
    __global float* scores,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint candidate_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (candidate_index >= params.candidates) {
        return;
    }
    __local float distances[MAX_HISTORY];
    __local float nearest_distances[MAX_HISTORY];
    __local uint nearest_indices[MAX_HISTORY];
    if (thread_index < params.history) {
        __global const uchar* base =
            rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
        __global const uchar* row =
            rows + ((ulong)history_slots[thread_index]) * ((ulong)params.row_bytes);
        Seed seed = seeds[candidate_index];
        float distance = base_distances[thread_index];
        for (uint edit_index = 0u; edit_index < params.num_pert; ++edit_index) {
            SparseEdit edit = edits[(candidate_index * params.num_pert) + edit_index];
            Leaf leaf = leaves[edit.leaf];
            uint base_code = code_at(base, leaf, edit.element);
            uint observed_code = code_at(row, leaf, edit.element);
            uint candidate_code = sparse_code(
                base_code,
                seed,
                leaf.element_offset + edit.element,
                leaf
            );
            float base_delta =
                decode_code(base_code, leaf.encoding, leaf.scale)
                - decode_code(observed_code, leaf.encoding, leaf.scale);
            float candidate_delta =
                decode_code(candidate_code, leaf.encoding, leaf.scale)
                - decode_code(observed_code, leaf.encoding, leaf.scale);
            distance += (candidate_delta * candidate_delta - base_delta * base_delta)
                * leaf.weight;
        }
        distances[thread_index] = fmax(distance, 0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (thread_index == 0u) {
        for (uint k = 0u; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0u;
        }
        for (uint h = 0u; h < params.history; ++h) {
            float distance = distances[h];
            uint insert_at = params.neighbors;
            for (uint k = 0u; k < params.neighbors; ++k) {
                if (
                    distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && h < nearest_indices[k])
                ) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at < params.neighbors) {
                for (uint k = params.neighbors - 1u; k > insert_at; --k) {
                    nearest_distances[k] = nearest_distances[k - 1u];
                    nearest_indices[k] = nearest_indices[k - 1u];
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
            reference_weight = fmax(1.0f / fmax(variance, 1.0e-12f), 1.17549435e-38f);
        }
        for (uint k = 0u; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / fmax(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
            if (params.acquisition == 1u) {
                float draw_weight = weight / reference_weight;
                weighted_noise += draw_weight * draws[history_slots[nearest_indices[k]]];
                weight_squared_sum += draw_weight * draw_weight;
            }
        }
        float mean = weighted_value / fmax(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / fmax(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * (weighted_noise / fmax(sqrt(weight_squared_sum), 1.0e-12f));
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

__kernel void pick_trial(
    __global const float* scores,
    __global uint* choice,
    __global float* selected_scores,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint thread_index = get_local_id(0);
    float best_score = -INFINITY;
    uint best = UINT_MAX;
    for (uint index = thread_index; index < params.candidates; index += THREADS) {
        float score = scores[index];
        if (score > best_score || (score == best_score && index < best)) {
            best = index;
            best_score = score;
        }
    }

    __local float local_scores[THREADS];
    __local uint local_indices[THREADS];
    local_scores[thread_index] = best_score;
    local_indices[thread_index] = best;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            float other_score = local_scores[thread_index + stride];
            uint other_index = local_indices[thread_index + stride];
            if (
                other_score > local_scores[thread_index]
                || (
                    other_score == local_scores[thread_index]
                    && other_index < local_indices[thread_index]
                )
            ) {
                local_scores[thread_index] = other_score;
                local_indices[thread_index] = other_index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_index == 0u) {
        choice[0] = local_indices[0];
        selected_scores[0] = local_scores[0];
    }
}

typedef struct {
    uint num_regions;
    uint candidates_per_region;
} MultiTrParams;

__kernel void multi_tr_pick_trials(
    __global const float* scores,
    __global uint* choices,
    __global float* selected_scores,
    MultiTrParams params
) {
    uint region = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (region >= params.num_regions) {
        return;
    }

    __global const float* region_scores =
        scores + region * params.candidates_per_region;
    float best_score = -INFINITY;
    uint best_index = UINT_MAX;
    for (
        uint index = thread_index;
        index < params.candidates_per_region;
        index += THREADS
    ) {
        float score = region_scores[index];
        if (score > best_score || (score == best_score && index < best_index)) {
            best_score = score;
            best_index = index;
        }
    }

    __local float local_scores[THREADS];
    __local uint local_indices[THREADS];
    local_scores[thread_index] = best_score;
    local_indices[thread_index] = best_index;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            float other_score = local_scores[thread_index + stride];
            uint other_index = local_indices[thread_index + stride];
            if (
                other_score > local_scores[thread_index]
                || (
                    other_score == local_scores[thread_index]
                    && other_index < local_indices[thread_index]
                )
            ) {
                local_scores[thread_index] = other_score;
                local_indices[thread_index] = other_index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_index == 0u) {
        choices[region] = region * params.candidates_per_region + local_indices[0];
        selected_scores[region] = local_scores[0];
    }
}

__kernel void write_sparse(
    __global uchar* rows,
    __global const Seed* seeds,
    __global const uint* choice,
    __global const Leaf* leaves,
    __global const SparseEdit* edits,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    if (get_global_id(0) != 0u) {
        return;
    }
    uint candidate = choice[0];
    Seed seed = seeds[candidate];
    __global uchar* trial =
        rows + ((ulong)params.trial_slot) * ((ulong)params.row_bytes);
    for (uint edit_index = 0u; edit_index < params.num_pert; ++edit_index) {
        SparseEdit edit = edits[candidate * params.num_pert + edit_index];
        Leaf leaf = leaves[edit.leaf];
        uint byte = leaf.byte_offset
            + (leaf.bits == 4u ? edit.element / 2u : edit.element);
        uchar current = trial[byte];
        uint shift = leaf.bits == 4u ? (edit.element & 1u) * 4u : 0u;
        uint code = leaf.bits == 4u
            ? (uint)((current >> shift) & 0x0fu)
            : (uint)current;
        uint value = sparse_code(code, seed, leaf.element_offset + edit.element, leaf);
        if (leaf.bits == 4u) {
            trial[byte] = (uchar)((current & ~(0x0fu << shift)) | ((value & 0x0fu) << shift));
        } else {
            trial[byte] = (uchar)value;
        }
    }
}

__kernel void write_trial(
    __global uchar* rows,
    __global const Seed* seeds,
    __global const uint* choice,
    __global const Leaf* leaves,
    __global const Tile* tiles,
    __global const Params *input
) {
    Params params = *input;
    if (params.candidates == 0) return;
    uint tile_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (tile_index >= params.tiles) {
        return;
    }
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed seed = seeds[choice[0]];
    __global const uchar* base =
        rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
    __global uchar* trial =
        rows + ((ulong)params.trial_slot) * ((ulong)params.row_bytes);

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += THREADS) {
            uint first = tile.start + local_byte * 2u;
            uint low = perturb_code(
                code_at(base, leaf, first),
                seed,
                leaf.element_offset + first,
                leaf
            );
            uint high = 0u;
            if (first + 1u < leaf.length) {
                high = perturb_code(
                    code_at(base, leaf, first + 1u),
                    seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            trial[leaf.byte_offset + first_byte + local_byte] =
                (uchar)(low | (high << 4u));
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += THREADS) {
            trial[leaf.byte_offset + element] = (uchar)perturb_code(
                code_at(base, leaf, element),
                seed,
                leaf.element_offset + element,
                leaf
            );
        }
    }
}

typedef struct {
    uint row_bytes;
    uint slot;
    uint pad0;
    uint pad1;
} RowSumParams;

__kernel void row_sum(
    __global const uchar* rows,
    __global ulong* sums,
    RowSumParams params
) {
    __local ulong partials[THREADS];
    uint thread_index = get_local_id(0);
    __global const uchar* row =
        rows + ((ulong)params.slot) * ((ulong)params.row_bytes);
    ulong sum = 0ul;
    for (uint byte = thread_index; byte < params.row_bytes; byte += THREADS) {
        sum += (ulong)row[byte];
    }
    partials[thread_index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (thread_index < stride) {
            partials[thread_index] += partials[thread_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (thread_index == 0u) {
        sums[0] = partials[0];
    }
}
