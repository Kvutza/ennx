#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Leaf {
    ulong key;
    ulong offset;
    ulong length;
    float scale;
    float weight;
};
struct Tile { uint leaf; uint start; uint length; uint pad; };
struct Params {
    ulong seed;
    ulong stream_seed;
    float radius;
    float alternate_radius;
    uint candidate;
    uint tiles;
    uint history;
    uint initialize;
    uint mode;
};
struct SelectionParams {
    ulong root_seed;
    float outcomes[128];
    float variances[128];
    float draws[128];
    float base_distances[128];
    float local_scales[128];
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
    float radius;
    float alternate_radius;
    uint neighbors;
    uint history;
    uint acquisition;
    uint tiles;
    uint mode;
    uint resident_history;
    uint resident_indices[2];
    uint implicit_history;
    uint forced_candidate;
    uint distance_scaling;
    uint local_scale_neighbors;
    uint incumbent_index;
    uint candidate_floor;
};
struct Decision {
    uint index;
    uint valid;
    ulong root_seed;
    ulong seed;
    float radius;
    float score;
    uint mode;
    uint pad;
    float predicted_mean;
    float predicted_standard_error;
    float incumbent_mean;
    float incumbent_standard_error;
};
struct Partial { float anchor; float rejected; float squared; uint changed; uint invalid; };

inline ulong mix64(ulong x) {
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    return x ^ (x >> 31);
}
#ifdef FP16_INDEPENDENT
inline float decode(ushort x) { return float(as_type<half>(x)); }
inline ushort encode(float x) { return as_type<ushort>(half(x)); }
inline bool invalid_value(ushort x) { return (x & 0x7c00u) == 0x7c00u; }
#else
inline float decode(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort encode(float x) {
    uint bits = as_type<uint>(x);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline bool invalid_value(ushort x) { return (x & 0x7f80u) == 0x7f80u; }
#endif
inline float2 noise_pair(ulong seed, ulong key, ulong pair, uint mode) {
    ulong first = mix64(seed ^ mix64(key ^ 0xd6e8feb86659fd93ul)
        ^ mix64(pair ^ 0xa0761d6478bd642ful));
#ifdef RADEMACHER_ONLY
    return float2(
        (first & 1ul) == 0ul ? -1.0f : 1.0f,
        (first & 2ul) == 0ul ? -1.0f : 1.0f
    );
#else
    if (mode == 1u) {
        return float2(
            (first & 1ul) == 0ul ? -1.0f : 1.0f,
            (first & 2ul) == 0ul ? -1.0f : 1.0f
        );
    }
    ulong second = mix64(first ^ 0xd2b74407b1ce6e93ul);
    // Integer addition before FP32 conversion preserves CUDA's FP64-to-FP32 uniform.
    float u1 = float((first >> 11) + 1ul) * 0x1.0p-53f;
    float u2 = float(second >> 11) * 0x1.0p-53f;
    float radius = sqrt(-2.0f * log(clamp(u1, 1e-12f, 0.99999994f)));
    float angle = 6.283185307179586f * u2;
    float cosine;
    float sine = sincos(angle, cosine);
    return radius * float2(cosine, sine);
#endif
}
inline float sample(ulong seed, ulong key, ulong element, uint mode) {
    float2 pair = noise_pair(seed, key, element / 2, mode);
    return (element & 1ul) == 0ul ? pair.x : pair.y;
}
inline float direction(float reference, float inverse_rms, float noise, uint candidate) {
#ifdef FP16_INDEPENDENT
    return noise;
#else
    return candidate < 2 ? 0.75f * (reference * inverse_rms) + sqrt(0.4375f) * noise : noise;
#endif
}
inline uint trial_hash(uint low, uint high, uint element) {
    uint value = low ^ (element * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= high;
    value *= 0x846ca68bu;
    return value ^ (value >> 15);
}
inline ulong candidate_seed(ulong root, uint candidate) {
    uint stream = candidate / 2;
    uint low = uint(root);
    uint high = uint(root >> 32);
    return ulong(trial_hash(low, high, stream))
        | (ulong(trial_hash(high, low, stream ^ 0x9e3779b9u)) << 32);
}

inline ushort proposed_value(
    device const ushort *base,
    ulong index,
    Leaf leaf,
    device const ushort *reference,
    float inverse_rms,
    float noise,
    uint candidate,
    float radius) {
    float d = direction(decode(reference[index]), inverse_rms, noise, candidate);
    return encode(decode(base[index]) + (leaf.scale * radius) * d);
}
inline ushort proposed_value_bits(
    ushort base_bits,
    ushort reference_bits,
    Leaf leaf,
    float inverse_rms,
    float noise,
    uint candidate,
    float radius) {
    float d = direction(decode(reference_bits), inverse_rms, noise, candidate);
    return encode(decode(base_bits) + (leaf.scale * radius) * d);
}

kernel void bf16_propose(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const ushort *reference [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device const Leaf *leaves [[buffer(5)]],
    device const Tile *tiles [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    constant Params &p [[buffer(9)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float a = 0.0f, b = 0.0f, sq = 0.0f;
    uint changed = 0, invalid = 0;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
        float d = direction(decode(reference[index]), scales[tile.leaf], noise, p.candidate);
        ushort value = encode(decode(base[index]) + (leaf.scale * p.radius) * d);
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base[index]);
        float delta = decode(value) - decode(base[index]);
        sq += delta * delta;
        delta = decode(value) - decode(anchor[index]);
        a += (delta * delta) * leaf.weight;
        if (p.history == 2) {
            delta = decode(value) - decode(rejected[index]);
            b += (delta * delta) * leaf.weight;
        }
    }
    threadgroup float sums_a[256], sums_b[256], sums_sq[256];
    threadgroup uint counts[256], flags[256];
    sums_a[tid] = a; sums_b[tid] = b; sums_sq[tid] = sq;
    counts[tid] = changed; flags[tid] = invalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            sums_a[tid] += sums_a[tid + stride];
            sums_b[tid] += sums_b[tid + stride];
            sums_sq[tid] += sums_sq[tid + stride];
            counts[tid] += counts[tid + stride];
            flags[tid] |= flags[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[p.candidate * p.tiles + tile_index] = {
            sums_a[0], sums_b[0], sums_sq[0], counts[0], flags[0]
        };
    }
}

kernel void bf16_propose_pool(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const ushort *reference [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device const Leaf *leaves [[buffer(5)]],
    device const Tile *tiles [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    device float *geometry [[buffer(9)]],
    constant Params &p [[buffer(10)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float inverse_rms = scales[tile.leaf];
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float reference_norm = 0.0f;
    float reference_dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        float2 persistent_pair = noise_pair(persistent_seed, leaf.key, pair, p.mode);
        float2 fresh_pair = noise_pair(fresh_seed, leaf.key, pair, p.mode);
        for (uint pair_offset = 0; pair_offset < 2; pair_offset++) {
            uint item_offset = item + pair_offset;
            if (item_offset >= tile.length) break;
            ulong element = ulong(tile.start) + item_offset;
            ulong index = leaf.offset + element;
            float persistent = pair_offset == 0 ? persistent_pair.x : persistent_pair.y;
            float fresh = pair_offset == 0 ? fresh_pair.x : fresh_pair.y;
            ushort base_bits = base[index];
#ifdef FP16_INDEPENDENT
            ushort reference_bits = 0;
#else
            ushort reference_bits = reference[index];
#endif
            ushort rejected_bits = p.history == 2 ? rejected[index] : 0;
            float base_value = decode(base_bits);
            float anchor_value = decode(anchor[index]);
            float reference_value = decode(reference_bits);
            float rejected_value = p.history == 2 ? decode(rejected_bits) : 0.0f;
            float persistent_direction = direction(reference_value, inverse_rms, persistent, 0);
            float reference_delta = leaf.scale * reference_value * inverse_rms;
            reference_norm += reference_delta * reference_delta * leaf.weight;
            float radius = leaf.scale * p.radius;
            float alternate_radius = leaf.scale * p.alternate_radius;
            ushort values[4] = {
                encode(base_value + radius * persistent_direction),
                encode(base_value + alternate_radius * persistent_direction),
                encode(base_value + radius * fresh),
                encode(base_value + alternate_radius * fresh),
            };
            float deltas[4];
            for (uint candidate = 0; candidate < 4; candidate++) {
                ushort value = values[candidate];
                invalid[candidate] |= uint(invalid_value(value));
                changed[candidate] += uint(value != base_bits);
                float value_float = decode(value);
                float delta = value_float - base_value;
                deltas[candidate] = delta;
                sq[candidate] += delta * delta;
                norms[candidate] += (delta * delta) * leaf.weight;
                reference_dots[candidate] += delta * reference_delta * leaf.weight;
                delta = value_float - anchor_value;
                a[candidate] += (delta * delta) * leaf.weight;
                if (p.history == 2) {
                    delta = value_float - rejected_value;
                    b[candidate] += (delta * delta) * leaf.weight;
                }
            }
            dots[0] += deltas[0] * deltas[1] * leaf.weight;
            dots[1] += deltas[0] * deltas[2] * leaf.weight;
            dots[2] += deltas[0] * deltas[3] * leaf.weight;
            dots[3] += deltas[1] * deltas[2] * leaf.weight;
            dots[4] += deltas[1] * deltas[3] * leaf.weight;
            dots[5] += deltas[2] * deltas[3] * leaf.weight;
        }
    }
    threadgroup float sums_a[1024], sums_b[1024], sums_sq[1024];
    threadgroup float sums_ref[768];
    threadgroup uint counts[1024], flags[1024];
    for (uint candidate = 0; candidate < 4; candidate++) {
        uint slot = candidate * 256 + tid;
        sums_a[slot] = a[candidate];
        sums_b[slot] = b[candidate];
        sums_sq[slot] = sq[candidate];
        counts[slot] = changed[candidate];
        flags[slot] = invalid[candidate];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint candidate = 0; candidate < 4; candidate++) {
                uint slot = candidate * 256 + tid;
                uint other = slot + stride;
                sums_a[slot] += sums_a[other];
                sums_b[slot] += sums_b[other];
                sums_sq[slot] += sums_sq[other];
                counts[slot] += counts[other];
                flags[slot] |= flags[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            uint slot = candidate * 256;
            partials[(p.initialize * 4 + candidate) * p.tiles + tile_index] = {
                sums_a[slot], sums_b[slot], sums_sq[slot], counts[slot], flags[slot]
            };
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint candidate = 0; candidate < 4; candidate++)
        sums_a[candidate * 256 + tid] = norms[candidate];
    for (uint pair = 0; pair < 4; pair++)
        sums_b[pair * 256 + tid] = dots[pair];
    for (uint pair = 0; pair < 2; pair++)
        sums_sq[pair * 256 + tid] = dots[pair + 4];
    sums_sq[2 * 256 + tid] = reference_norm;
    sums_sq[3 * 256 + tid] = reference_dots[0];
    for (uint candidate = 1; candidate < 4; candidate++)
        sums_ref[(candidate - 1) * 256 + tid] = reference_dots[candidate];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint candidate = 0; candidate < 4; candidate++) {
                uint slot = candidate * 256 + tid;
                sums_a[slot] += sums_a[slot + stride];
            }
            for (uint pair = 0; pair < 4; pair++) {
                uint slot = pair * 256 + tid;
                sums_b[slot] += sums_b[slot + stride];
            }
            for (uint pair = 0; pair < 4; pair++) {
                uint slot = pair * 256 + tid;
                sums_sq[slot] += sums_sq[slot + stride];
            }
            for (uint candidate = 0; candidate < 3; candidate++) {
                uint slot = candidate * 256 + tid;
                sums_ref[slot] += sums_ref[slot + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint candidate = 0; candidate < 4; candidate++)
            geometry[candidate * p.tiles + tile_index] = sums_a[candidate * 256];
        for (uint pair = 0; pair < 4; pair++)
            geometry[(pair + 4) * p.tiles + tile_index] = sums_b[pair * 256];
        for (uint pair = 0; pair < 2; pair++)
            geometry[(pair + 8) * p.tiles + tile_index] = sums_sq[pair * 256];
        geometry[10 * p.tiles + tile_index] = sums_sq[2 * 256];
        geometry[11 * p.tiles + tile_index] = sums_sq[3 * 256];
        for (uint candidate = 1; candidate < 4; candidate++)
            geometry[(candidate + 11) * p.tiles + tile_index]
                = sums_ref[(candidate - 1) * 256];
    }
}

inline float acquisition_score(
    thread const float *distances,
    constant SelectionParams &q,
    uint excluded_index,
    thread float &mean_out,
    thread float &standard_error_out) {
    float scaled_distances[128];
    if (q.distance_scaling != 0) {
        float query_neighbors[128];
        uint query_count = 0;
        for (uint i = 0; i < q.history; i++) {
            if (i == excluded_index) continue;
            uint position = query_count;
            while (position > 0 && distances[i] < query_neighbors[position - 1]) {
                query_neighbors[position] = query_neighbors[position - 1];
                position--;
            }
            query_neighbors[position] = distances[i];
            query_count++;
        }
        uint local_rank = min(q.local_scale_neighbors, query_count);
        float query_scale = local_rank == 0
            ? 1.0f
            : max(query_neighbors[local_rank - 1], 1.0e-12f);
        for (uint i = 0; i < q.history; i++) {
            float history_scale = max(q.local_scales[i], 1.0e-12f);
            scaled_distances[i] = distances[i] / (sqrt(query_scale) * sqrt(history_scale));
        }
    } else {
        for (uint i = 0; i < q.history; i++) scaled_distances[i] = distances[i];
    }
    uint nearest[128];
    for (uint i = 0; i < q.history; i++) {
        uint position = i;
        while (position > 0
            && scaled_distances[i] < scaled_distances[nearest[position - 1]]) {
            nearest[position] = nearest[position - 1];
            position--;
        }
        nearest[position] = i;
    }
    uint count = min(q.neighbors, q.history);
    float sum = 0.0f;
    float value = 0.0f;
    float reference = 1.17549435e-38f;
    float y_scale_sq = max(q.y_scale * q.y_scale, 1.0e-12f);
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + q.variances[index] / y_scale_sq;
        float weight = 1.0f / max(variance, 1.0e-12f);
        sum += weight;
        value += weight * q.outcomes[index];
        reference = max(reference, weight);
    }
    float mean = value / max(sum, 1.0e-12f);
    float aleatoric = 0.0f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + q.variances[index] / y_scale_sq;
        float weight = (1.0f / max(variance, 1.0e-12f)) / max(sum, 1.0e-12f);
        aleatoric += weight * (q.aleatoric_scale + q.variances[index] / y_scale_sq);
    }
    float standard_error = sqrt(1.0f / max(sum, 1.0e-12f) + aleatoric) * q.y_scale;
    mean_out = mean;
    standard_error_out = standard_error;
    if (q.acquisition == 1) {
        float noise = 0.0f;
        float squared = 0.0f;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float variance = 1.0e-9f
                + q.epistemic_scale * scaled_distances[index]
                + q.aleatoric_scale
                + q.variances[index] / y_scale_sq;
            float weight = (1.0f / max(variance, 1.0e-12f)) / reference;
            noise += weight * q.draws[index];
            squared += weight * weight;
        }
        return mean + standard_error * (noise / max(sqrt(squared), 1.0e-12f));
    }
    if (q.acquisition == 2) return mean + standard_error;
    return mean + q.beta * standard_error;
}

kernel void bf16_select(
    device const Partial *partials [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device float *pool_distances [[buffer(2)]],
    device const float *geometry [[buffer(3)]],
    constant SelectionParams &q [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    float distances[4][128];
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool changed[4] = {false, false, false, false};
    bool invalid[4] = {false, false, false, false};
    for (uint candidate = 0; candidate < 4; candidate++) {
        for (uint tile = 0; tile < q.tiles; tile++)
            proposal_norms[candidate] += geometry[candidate * q.tiles + tile];
        for (uint row = 0; row < q.history; row++) {
            distances[candidate][row] = q.implicit_history != 0
                ? q.base_distances[row] + proposal_norms[candidate]
                : 0.0f;
        }
    }
    uint tile_count = q.tiles;
    for (uint candidate = 0; candidate < 4; candidate++) {
        for (uint row = 0; row < q.resident_history; row += 2) {
            float first = 0.0f;
            float second = 0.0f;
            for (uint tile = 0; tile < tile_count; tile++) {
                Partial p = partials[((row / 2) * 4 + candidate) * tile_count + tile];
                first += p.anchor;
                if (row + 1 < q.resident_history) second += p.rejected;
                changed[candidate] = changed[candidate] || p.changed != 0;
                invalid[candidate] = invalid[candidate] || p.invalid != 0 || !isfinite(p.squared);
            }
            uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
            distances[candidate][first_index] = first;
            if (row + 1 < q.resident_history) {
                uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
                distances[candidate][second_index] = second;
            }
            invalid[candidate] = invalid[candidate] || !isfinite(first)
                || (row + 1 < q.resident_history && !isfinite(second));
        }
    }
    bool any_changed = false;
    for (uint candidate = 0; candidate < 4; candidate++) any_changed = any_changed || changed[candidate];
    uint selected = 4;
    float best = -INFINITY;
    float candidate_means[4];
    float candidate_standard_errors[4];
    float incumbent_distances[128];
    for (uint row = 0; row < q.history; row++)
        incumbent_distances[row] = q.base_distances[row];
    float incumbent_mean;
    float incumbent_standard_error;
    acquisition_score(
        incumbent_distances,
        q,
        q.incumbent_index,
        incumbent_mean,
        incumbent_standard_error);
    for (uint candidate = 0; candidate < 4; candidate++) {
        if (candidate < q.candidate_floor
            || invalid[candidate]
            || (any_changed && !changed[candidate])) continue;
        float mean;
        float standard_error;
        float score = acquisition_score(distances[candidate], q, 128u, mean, standard_error);
        candidate_means[candidate] = mean;
        candidate_standard_errors[candidate] = standard_error;
        if (q.forced_candidate < 4) {
            if (candidate == q.forced_candidate) {
                selected = candidate;
                best = 0.0f;
            }
            continue;
        }
        if (isfinite(score) && (selected == 4 || score > best)) {
            selected = candidate;
            best = score;
        }
    }
    decision->index = selected;
    decision->valid = selected < 4 ? 1u : 0u;
    decision->root_seed = q.root_seed;
    decision->seed = selected < 4 ? candidate_seed(q.root_seed, selected) : 0ul;
    decision->radius = selected < 4
        ? ((selected & 1u) == 0 ? q.radius : q.alternate_radius)
        : 0.0f;
    decision->score = selected < 4 ? best : 0.0f;
    decision->mode = q.mode;
    decision->pad = 0;
    decision->predicted_mean = selected < 4 ? candidate_means[selected] : 0.0f;
    decision->predicted_standard_error = selected < 4
        ? candidate_standard_errors[selected]
        : 0.0f;
    decision->incumbent_mean = incumbent_mean;
    decision->incumbent_standard_error = incumbent_standard_error;
    for (uint candidate = 0; candidate < 4; candidate++)
        for (uint row = 0; row < q.history; row++)
            pool_distances[candidate * q.history + row] = distances[candidate][row];
}

kernel void bf16_materialize(
    device const ushort *base [[buffer(0)]],
    device const ushort *reference [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device const Tile *tiles [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    device const Decision *decision [[buffer(6)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (decision->valid == 0) return;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    ulong seed = decision->seed ^ 0x8ebc6af09c88c6e3ul;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        float2 samples = noise_pair(seed, leaf.key, pair, decision->mode);
        for (uint pair_offset = 0; pair_offset < 2; pair_offset++) {
            uint item_offset = item + pair_offset;
            if (item_offset >= tile.length) break;
            ulong element = ulong(tile.start) + item_offset;
            ulong index = leaf.offset + element;
            float noise = pair_offset == 0 ? samples.x : samples.y;
            ushort base_bits = base[index];
            float d;
            if (decision->index < 2) {
                float inverse_rms = scales[tile.leaf];
                d = direction(decode(reference[index]), inverse_rms, noise, decision->index);
            } else {
                d = noise;
            }
            float base_value = decode(base_bits);
            output[index] = encode(base_value + (leaf.scale * decision->radius) * d);
        }
    }
}

kernel void bf16_reference(
    device ushort *reference [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const Leaf *leaves [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sum = 0.0f;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        float value;
        if (p.initialize != 0) {
            value = sample(p.seed ^ 0xe7037ed1a0b428dbul, leaf.key, element, p.mode);
        } else {
            float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
            value = direction(decode(reference[index]), scales[tile.leaf], noise, p.candidate);
        }
        ushort bits = encode(value);
        reference[index] = bits;
        float stored = decode(bits);
        sum += stored * stored;
    }
    threadgroup float sums[256];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[tile_index] = sums[0];
}

kernel void bf16_reference_rms(
    device const float *partials [[buffer(0)]],
    device const Leaf *leaves [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device float *scales [[buffer(3)]],
    uint leaf [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint tile = offsets[leaf] + tid; tile < offsets[leaf + 1]; tile += 256)
        sum += partials[tile];
    threadgroup float sums[256];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scales[leaf] = sqrt(float(leaves[leaf].length) / sums[0]);
}
