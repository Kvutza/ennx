#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// ENNX_ZIGGURAT_TABLES

struct Leaf {
    ulong key;
    ulong offset;
    ulong length;
    float scale;
    float weight;
    uint address;
    uint pad;
};
struct Tile { uint leaf; uint start; uint length; uint pad; };
struct Params {
    ulong seed;
    ulong stream_seed;
    ulong basis_seed;
    float radius;
    float alternate_radius;
    uint candidate;
    uint tiles;
    uint history;
    uint initialize;
    uint mode;
    uint base_slot;
    uint blocks;
    uint program;
};
struct ReplayStep {
    ulong seed;
    float radius;
    uint accepted;
    uint candidate;
    uint mode;
};
struct SelectionParams {
    ulong root_seed;
    ulong basis_seed;
    float outcomes[128];
    float variances[128];
    float draws[128];
    float base_distances[128];
    float local_scales[128];
    float latent_norms[4];
    float axis_history[128];
    float axis_candidates[4];
    float axis_base;
    float axis_weight;
    uint axis_enabled;
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
    uint program;
    uint resident_history;
    uint resident_indices[2];
    uint implicit_history;
    uint exact_history;
    uint latent_history;
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
    ulong basis_seed;
    float radius;
    float score;
    uint mode;
    uint program;
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
inline uint mix32(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
#ifndef RADEMACHER_ONLY
inline uint ziggurat_domain(ulong seed, ulong key) {
    return uint(seed) ^ uint(seed >> 32) * 0xc2b2ae35u
        ^ uint(key) * 0x27d4eb2fu ^ uint(key >> 32) * 0x165667b1u;
}
inline uint ziggurat_bits_domain(
    uint domain,
    ulong element,
    uint attempt,
    uint stream) {
    uint counter = uint(element) * 0x9e3779b9u
        ^ uint(element >> 32) * 0x85ebca6bu;
    return mix32(
        domain ^ counter ^ attempt * 0xd3a2646cu ^ stream * 0xfd7046c5u);
}
inline float ziggurat_uniform(uint bits) {
    return min(float((bits >> 8) + 1u) * 0x1.0p-24f, 0.99999994f);
}
inline float ziggurat_sample_domain(uint domain, ulong element) {
    for (uint attempt = 0;; attempt++) {
        uint bits = ziggurat_bits_domain(domain, element, attempt, 0u);
        int signed_sample = as_type<int>(bits);
        uint layer = uint(signed_sample) & 255u;
        uint magnitude = signed_sample < 0
            ? (~uint(signed_sample) + 1u)
            : uint(signed_sample);
        float sample = float(signed_sample) * ziggurat_widths[layer];
        if (magnitude < ziggurat_thresholds[layer]) return sample;
        if (layer == 0u) {
            for (uint tail = 0;; tail++) {
                float x = -log(ziggurat_uniform(
                    ziggurat_bits_domain(domain, element, attempt + tail, 1u)))
                    / 3.654152885361009f;
                float y = -log(ziggurat_uniform(
                    ziggurat_bits_domain(domain, element, attempt + tail, 2u)));
                if (2.0f * y >= x * x) {
                    float value = 3.654152885361009f + x;
                    return signed_sample < 0 ? -value : value;
                }
            }
        }
        float uniform = ziggurat_uniform(
            ziggurat_bits_domain(domain, element, attempt, 1u));
        float density = ziggurat_densities[layer]
            + uniform * (ziggurat_densities[layer - 1u] - ziggurat_densities[layer]);
        if (density < exp(-0.5f * sample * sample)) return sample;
    }
}
inline float ziggurat_sample(ulong seed, ulong key, ulong element) {
    return ziggurat_sample_domain(ziggurat_domain(seed, key), element);
}
inline float2 ziggurat_pair_domain(uint domain, ulong pair) {
    return float2(
        ziggurat_sample_domain(domain, pair * 2ul),
        ziggurat_sample_domain(domain, pair * 2ul + 1ul));
}
#endif
#ifdef FP16_INDEPENDENT
inline float decode(ushort x) { return float(as_type<half>(x)); }
inline ushort encode(float x) { return as_type<ushort>(half(x)); }
inline float2 decode_pair(ushort2 x) { return float2(as_type<half2>(x)); }
inline ushort2 encode_pair(float2 x) { return as_type<ushort2>(half2(x)); }
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
#ifdef RADEMACHER_ONLY
    uint first = mix32(
        uint(seed) ^ uint(seed >> 32)
        ^ mix32(uint(key) ^ uint(key >> 32))
        ^ mix32(uint(pair)));
    return float2(
        (first & 1u) == 0u ? -1.0f : 1.0f,
        (first & 2u) == 0u ? -1.0f : 1.0f
    );
#else
    if (mode == 1u) {
        ulong first = mix64(seed ^ mix64(key ^ 0xd6e8feb86659fd93ul)
            ^ mix64(pair ^ 0xa0761d6478bd642ful));
        return float2(
            (first & 1ul) == 0ul ? -1.0f : 1.0f,
            (first & 2ul) == 0ul ? -1.0f : 1.0f
        );
    }
    return ziggurat_pair_domain(ziggurat_domain(seed, key), pair);
#endif
}
inline float sample(ulong seed, ulong key, ulong element, uint mode) {
    float2 pair = noise_pair(seed, key, element / 2, mode);
    return (element & 1ul) == 0ul ? pair.x : pair.y;
}
inline float direction(float reference, float inverse_rms, float noise, uint candidate) {
#if defined(FP16_INDEPENDENT) || (defined(PROCEDURAL_SLOTS) && PROCEDURAL_SLOTS == 1)
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
#ifdef PROCEDURAL_SLOTS
    uint arm = candidate / PROCEDURAL_SLOTS;
    candidate %= PROCEDURAL_SLOTS;
    if (arm != 0u) {
        ulong z = (root ^ 0x70726f6361726d31ul ^ ulong(arm)) + 0x9e3779b97f4a7c15ul;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
        root = z ^ (z >> 31);
    }
#endif
    uint stream = candidate / 2;
    uint low = uint(root);
    uint high = uint(root >> 32);
    return ulong(trial_hash(low, high, stream))
        | (ulong(trial_hash(high, low, stream ^ 0x9e3779b9u)) << 32);
}

inline float threshold_value(device const ulong *tables, uint candidate, uint index) {
    ulong word = tables[ulong(candidate) * 1024ul + ulong(index >> 6)];
    return ((word >> (index & 63u)) & 1ul) == 0ul ? -1.0f : 1.0f;
}

inline float2 threshold_pair(
    device const ulong *tables,
    uint candidate,
    ulong basis_seed,
    uint address,
    ulong pair) {
    ulong bits = mix64(basis_seed ^ mix64(pair ^ 0xa0761d6478bd642ful));
    uint prefix = address & 0xfffu;
    return float2(
        threshold_value(tables, candidate, prefix | ((uint(bits) & 15u) << 12)),
        threshold_value(tables, candidate, prefix | ((uint(bits >> 15) & 15u) << 12)));
}

#if defined(PROCEDURAL_SLOTS) && defined(FP16_INDEPENDENT)
inline ushort2 procedural_pair(float2 base_value, Leaf leaf, constant Params &p,
    device const ulong *tables, ulong pair, uint candidate) {
    float2 noise = p.program == 1u
        ? threshold_pair(tables, candidate, p.basis_seed, leaf.address, pair)
        : noise_pair(candidate_seed(p.stream_seed, candidate)
            ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, p.mode);
    float radius = (candidate % PROCEDURAL_SLOTS) & 1u ? p.alternate_radius : p.radius;
    return encode_pair(base_value + (leaf.scale * radius) * noise);
}
#endif

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
    device float *geometry [[buffer(9)]],
    constant Params &p [[buffer(10)]],
    device const ulong *threshold_tables [[buffer(11)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float a = 0.0f, b = 0.0f, sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        float2 threshold = threshold_pair(
            threshold_tables, p.candidate, p.basis_seed, leaf.address, element / 2ul);
        float noise = p.program == 1u
            ? threshold[element & 1ul]
            : sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
        float base_value = decode(base[index]);
        float d = direction(decode(reference[index]), scales[tile.leaf], noise, p.candidate);
        ushort value = encode(base_value + (leaf.scale * p.radius) * d);
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base[index]);
        float value_float = decode(value);
        float delta = value_float - base_value;
        sq += delta * delta;
        norm += (delta * delta) * leaf.weight;
        if (p.base_slot != 0) {
            delta = value_float - decode(anchor[index]);
            a += (delta * delta) * leaf.weight;
        }
        if (p.history == 2 && p.base_slot != 1) {
            delta = value_float - decode(rejected[index]);
            b += (delta * delta) * leaf.weight;
        }
    }
    if (p.base_slot == 0) a = norm;
    if (p.history == 2 && p.base_slot == 1) b = norm;
    threadgroup float sums_a[8], sums_b[8], sums_sq[8], sums_norm[8];
    threadgroup uint counts[8], flags[8];
    const uint lane = tid & 31u;
    const uint simdgroup = tid >> 5;
    const float a_sum = simd_sum(a);
    const float b_sum = simd_sum(b);
    const float sq_sum = simd_sum(sq);
    const float norm_sum = simd_sum(norm);
    const uint changed_sum = simd_sum(changed);
    const uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        sums_a[simdgroup] = a_sum;
        sums_b[simdgroup] = b_sum;
        sums_sq[simdgroup] = sq_sum;
        sums_norm[simdgroup] = norm_sum;
        counts[simdgroup] = changed_sum;
        flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const bool active = lane < 8;
        const uint slot = min(lane, 7u);
        const float total_a = simd_sum(active ? sums_a[slot] : 0.0f);
        const float total_b = simd_sum(active ? sums_b[slot] : 0.0f);
        const float total_sq = simd_sum(active ? sums_sq[slot] : 0.0f);
        const float total_norm = simd_sum(active ? sums_norm[slot] : 0.0f);
        const uint total_changed = simd_sum(active ? counts[slot] : 0u);
        const uint total_invalid = simd_sum(active ? flags[slot] : 0u);
        if (lane != 0) return;
        partials[p.candidate * p.tiles + tile_index] = {
            total_a, total_b, total_sq, total_changed, uint(total_invalid != 0)
        };
        geometry[p.candidate * p.tiles + tile_index] = total_norm;
    }
}

// First-round FP16 search has exactly one history row and the base is that row.
// Keeping a separate entry point removes two distance accumulators and all
// history/reference traffic from the billion-coordinate hot loop.
kernel void bf16_propose_initial(
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
    device const ulong *threshold_tables [[buffer(11)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
#ifdef FP16_INDEPENDENT
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + index)
            : ushort2(base[index], complete_pair ? base[index + 1] : 0);
        float2 base_values = decode_pair(base_pair);
        float2 noise = p.program == 1u
            ? threshold_pair(
                threshold_tables, p.candidate, p.basis_seed, leaf.address, element / 2ul)
            : noise_pair(
                p.seed ^ 0x8ebc6af09c88c6e3ul,
                leaf.key,
                element / 2,
                p.mode);
        ushort2 values = encode_pair(base_values + (leaf.scale * p.radius) * noise);
        if (aligned_pair) {
            *reinterpret_cast<device ushort2*>(output + index) = values;
        } else {
            output[index] = values.x;
            if (complete_pair) output[index + 1] = values.y;
        }
        uint pair_length = complete_pair ? 2u : 1u;
        for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
            ushort value = values[pair_offset];
            invalid |= uint(invalid_value(value));
            changed += uint(value != base_pair[pair_offset]);
            float delta = decode(value) - base_values[pair_offset];
            sq += delta * delta;
            norm += (delta * delta) * leaf.weight;
        }
    }
#else
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        ushort base_bits = base[index];
        float base_value = decode(base_bits);
        float2 threshold = threshold_pair(
            threshold_tables, p.candidate, p.basis_seed, leaf.address, element / 2ul);
        float noise = p.program == 1u
            ? threshold[element & 1ul]
            : sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
        ushort value = encode(base_value + (leaf.scale * p.radius) * noise);
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base_bits);
        float delta = decode(value) - base_value;
        sq += delta * delta;
        norm += (delta * delta) * leaf.weight;
    }
#endif
    threadgroup float sums_sq[8], sums_norm[8];
    threadgroup uint counts[8], flags[8];
    const uint lane = tid & 31u;
    const uint simdgroup = tid >> 5;
    const float sq_sum = simd_sum(sq);
    const float norm_sum = simd_sum(norm);
    const uint changed_sum = simd_sum(changed);
    const uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        sums_sq[simdgroup] = sq_sum;
        sums_norm[simdgroup] = norm_sum;
        counts[simdgroup] = changed_sum;
        flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const bool active = lane < 8;
        const uint slot = min(lane, 7u);
        const float total_sq = simd_sum(active ? sums_sq[slot] : 0.0f);
        const float total_norm = simd_sum(active ? sums_norm[slot] : 0.0f);
        const uint total_changed = simd_sum(active ? counts[slot] : 0u);
        const uint total_invalid = simd_sum(active ? flags[slot] : 0u);
        if (lane != 0) return;
        partials[p.candidate * p.tiles + tile_index] = {
            total_norm, 0.0f, total_sq, total_changed, uint(total_invalid != 0)
        };
        geometry[p.candidate * p.tiles + tile_index] = total_norm;
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
    device const ulong *threshold_tables [[buffer(11)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
#ifndef FP16_INDEPENDENT
    float inverse_rms = scales[tile.leaf];
#endif
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    float dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float reference_norm = 0.0f;
    float reference_dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        float2 persistent_pair = p.program == 1u
            ? threshold_pair(threshold_tables, 0u, p.basis_seed, leaf.address, pair)
            : noise_pair(persistent_seed, leaf.key, pair, p.mode);
        float2 fresh_pair = p.program == 1u
            ? threshold_pair(threshold_tables, 2u, p.basis_seed, leaf.address, pair)
            : noise_pair(fresh_seed, leaf.key, pair, p.mode);
#ifdef FP16_INDEPENDENT
        ulong first_index = leaf.offset + pair_element;
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (first_index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + first_index)
            : ushort2(base[first_index], complete_pair ? base[first_index + 1] : 0);
        float2 base_values = decode_pair(base_pair);
        float2 anchor_values = base_values;
        if (p.base_slot != 0) {
            ushort2 anchor_pair = aligned_pair
                ? *reinterpret_cast<device const ushort2*>(anchor + first_index)
                : ushort2(anchor[first_index], complete_pair ? anchor[first_index + 1] : 0);
            anchor_values = decode_pair(anchor_pair);
        }
        float2 rejected_values = float2(0.0f);
        if (p.history == 2 && p.base_slot != 1) {
            ushort2 rejected_pair = aligned_pair
                ? *reinterpret_cast<device const ushort2*>(rejected + first_index)
                : ushort2(rejected[first_index], complete_pair ? rejected[first_index + 1] : 0);
            rejected_values = decode_pair(rejected_pair);
        } else if (p.history == 2) {
            rejected_values = base_values;
        }
        float radius = leaf.scale * p.radius;
        float alternate_radius = leaf.scale * p.alternate_radius;
        ushort2 values[4] = {
            encode_pair(base_values + radius * persistent_pair),
            encode_pair(base_values + alternate_radius * persistent_pair),
            encode_pair(base_values + radius * fresh_pair),
            encode_pair(base_values + alternate_radius * fresh_pair),
        };
#ifdef PROCEDURAL_SLOTS
        for (uint candidate = 0; candidate < 4; candidate++)
            values[candidate] = procedural_pair(
                base_values, leaf, p, threshold_tables, pair, candidate);
#endif
        uint pair_length = complete_pair ? 2u : 1u;
        for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
            ushort base_bits = base_pair[pair_offset];
            float base_value = base_values[pair_offset];
            float anchor_value = anchor_values[pair_offset];
            float rejected_value = rejected_values[pair_offset];
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float deltas[4];
#endif
            for (uint candidate = 0; candidate < 4; candidate++) {
                ushort value = values[candidate][pair_offset];
                invalid[candidate] |= uint(invalid_value(value));
                changed[candidate] += uint(value != base_bits);
                float value_float = decode(value);
                float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                deltas[candidate] = delta;
#endif
                sq[candidate] += delta * delta;
                norms[candidate] += (delta * delta) * leaf.weight;
                if (p.base_slot != 0) {
                    delta = value_float - anchor_value;
                    a[candidate] += (delta * delta) * leaf.weight;
                }
                if (p.history == 2 && p.base_slot != 1) {
                    delta = value_float - rejected_value;
                    b[candidate] += (delta * delta) * leaf.weight;
                }
            }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            dots[0] += deltas[0] * deltas[1] * leaf.weight;
            dots[1] += deltas[0] * deltas[2] * leaf.weight;
            dots[2] += deltas[0] * deltas[3] * leaf.weight;
            dots[3] += deltas[1] * deltas[2] * leaf.weight;
            dots[4] += deltas[1] * deltas[3] * leaf.weight;
            dots[5] += deltas[2] * deltas[3] * leaf.weight;
#endif
        }
#else
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
            float base_value = decode(base_bits);
            float anchor_value = base_value;
            if (p.base_slot != 0) anchor_value = decode(anchor[index]);
            float reference_value = decode(reference_bits);
            float rejected_value = 0.0f;
            if (p.history == 2 && p.base_slot != 1)
                rejected_value = decode(rejected[index]);
            else if (p.history == 2)
                rejected_value = base_value;
            float persistent_direction = direction(reference_value, inverse_rms, persistent, 0);
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float reference_delta = leaf.scale * reference_value * inverse_rms;
            reference_norm += reference_delta * reference_delta * leaf.weight;
#endif
            float radius = leaf.scale * p.radius;
            float alternate_radius = leaf.scale * p.alternate_radius;
            ushort values[4] = {
                encode(base_value + radius * persistent_direction),
                encode(base_value + alternate_radius * persistent_direction),
                encode(base_value + radius * fresh),
                encode(base_value + alternate_radius * fresh),
            };
#ifdef PROCEDURAL_SLOTS
            for (uint candidate = 0; candidate < 4; candidate++) {
                float2 samples = p.program == 1u
                    ? threshold_pair(
                        threshold_tables, candidate, p.basis_seed, leaf.address, pair)
                    : noise_pair(candidate_seed(p.stream_seed, candidate)
                        ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, p.mode);
                float noise = samples[pair_offset];
                float candidate_radius = (candidate % PROCEDURAL_SLOTS) & 1u ? p.alternate_radius : p.radius;
                values[candidate] = proposed_value_bits(base_bits, reference_bits, leaf,
                    inverse_rms, noise, candidate, candidate_radius);
            }
#endif
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float deltas[4];
#endif
            for (uint candidate = 0; candidate < 4; candidate++) {
                ushort value = values[candidate];
                invalid[candidate] |= uint(invalid_value(value));
                changed[candidate] += uint(value != base_bits);
                float value_float = decode(value);
                float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                deltas[candidate] = delta;
#endif
                sq[candidate] += delta * delta;
                norms[candidate] += (delta * delta) * leaf.weight;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                reference_dots[candidate] += delta * reference_delta * leaf.weight;
#endif
                if (p.base_slot != 0) {
                    delta = value_float - anchor_value;
                    a[candidate] += (delta * delta) * leaf.weight;
                }
                if (p.history == 2 && p.base_slot != 1) {
                    delta = value_float - rejected_value;
                    b[candidate] += (delta * delta) * leaf.weight;
                }
            }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            dots[0] += deltas[0] * deltas[1] * leaf.weight;
            dots[1] += deltas[0] * deltas[2] * leaf.weight;
            dots[2] += deltas[0] * deltas[3] * leaf.weight;
            dots[3] += deltas[1] * deltas[2] * leaf.weight;
            dots[4] += deltas[1] * deltas[3] * leaf.weight;
            dots[5] += deltas[2] * deltas[3] * leaf.weight;
#endif
        }
#endif
    }
    if (p.base_slot == 0) {
        for (uint candidate = 0; candidate < 4; candidate++) a[candidate] = norms[candidate];
    } else if (p.base_slot == 1) {
        for (uint candidate = 0; candidate < 4; candidate++) b[candidate] = norms[candidate];
    }
    threadgroup float reduced_a[32], reduced_b[32], reduced_sq[32];
    threadgroup uint reduced_counts[32], reduced_flags[32];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    for (uint candidate = 0; candidate < 4; candidate++) {
        float a_sum = simd_sum(a[candidate]);
        float b_sum = simd_sum(b[candidate]);
        float sq_sum = simd_sum(sq[candidate]);
        uint changed_sum = simd_sum(changed[candidate]);
        uint invalid_sum = simd_sum(invalid[candidate]);
        if (lane == 0) {
            uint slot = candidate * 8 + simdgroup;
            reduced_a[slot] = a_sum;
            reduced_b[slot] = b_sum;
            reduced_sq[slot] = sq_sum;
            reduced_counts[slot] = changed_sum;
            reduced_flags[slot] = invalid_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            bool active = lane < 8;
            uint slot = candidate * 8 + min(lane, 7u);
            float a_sum = simd_sum(active ? reduced_a[slot] : 0.0f);
            float b_sum = simd_sum(active ? reduced_b[slot] : 0.0f);
            float sq_sum = simd_sum(active ? reduced_sq[slot] : 0.0f);
            uint changed_sum = simd_sum(active ? reduced_counts[slot] : 0u);
            uint invalid_sum = simd_sum(active ? reduced_flags[slot] : 0u);
            if (lane != 0) continue;
            partials[(p.initialize * 4 + candidate) * p.tiles + tile_index] =
                {a_sum, b_sum, sq_sum, changed_sum, uint(invalid_sum != 0)};
#ifndef POOL_GEOMETRY_DIAGNOSTICS
            if (p.base_slot == 0)
                geometry[candidate * p.tiles + tile_index] = a_sum;
            else if (p.base_slot == 1)
                geometry[candidate * p.tiles + tile_index] = b_sum;
#endif
        }
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float sums_a[1024], sums_b[1024], sums_sq[1024], sums_ref[768];
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
#endif
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

// For the short-history regime, spend every lane on distinct coordinates and
// reconstruct all historical observations in registers.  This avoids leaving
// most lanes idle when the replay lineage contains only a handful of rows.
kernel void bf16_replay_short(
    device ushort *lineage [[buffer(0)]],
    device const ushort *base [[buffer(1)]],
    device const Leaf *leaves [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device const ReplayStep *steps [[buffer(4)]],
    device const float *historical_scales [[buffer(5)]],
    device const float *base_weights [[buffer(6)]],
    device float *partials [[buffer(7)]],
    device float *components [[buffer(8)]],
    device Partial *pool_partials [[buffer(9)]],
    device float *pool_geometry [[buffer(10)]],
    constant Params &p [[buffer(11)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
#ifndef FP16_INDEPENDENT
    return;
#else
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sums[64] = {0.0f};
    float component_sums[64] = {0.0f};
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    float proposal_dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#endif
    float proposal_squared[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint proposal_changed[4] = {0u, 0u, 0u, 0u};
    uint proposal_invalid[4] = {0u, 0u, 0u, 0u};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
#ifndef RADEMACHER_ONLY
    uint persistent_domain = ziggurat_domain(persistent_seed, leaf.key);
    uint fresh_domain = ziggurat_domain(fresh_seed, leaf.key);
    uint history_domains[16] = {0u};
    for (uint row = 1u; row < p.history; row++)
        history_domains[row] = ziggurat_domain(
            steps[row].seed ^ 0x8ebc6af09c88c6e3ul,
            leaf.key);
#endif
    uint pair_count = (tile.length + 1u) / 2u;
    for (uint pair = tid; pair < pair_count; pair += 256u) {
        uint item = pair * 2u;
        bool has_second = item + 1u < tile.length;
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        ushort2 current_lineage = ushort2(
            lineage[index],
            has_second ? lineage[index + 1ul] : 0u);
        ushort2 base_bits = ushort2(base[index], has_second ? base[index + 1ul] : 0u);
        float2 base_value = decode_pair(base_bits);
#ifdef RADEMACHER_ONLY
        float2 persistent = noise_pair(persistent_seed, leaf.key, element / 2ul, p.mode);
        float2 fresh = noise_pair(fresh_seed, leaf.key, element / 2ul, p.mode);
#else
        float2 persistent = ziggurat_pair_domain(persistent_domain, element / 2ul);
        float2 fresh = ziggurat_pair_domain(fresh_domain, element / 2ul);
#endif
        ushort2 candidates[4] = {
            encode_pair(base_value + leaf.scale * p.radius * persistent),
            encode_pair(base_value + leaf.scale * p.alternate_radius * persistent),
            encode_pair(base_value + leaf.scale * p.radius * fresh),
            encode_pair(base_value + leaf.scale * p.alternate_radius * fresh),
        };
#ifdef PROCEDURAL_SLOTS
        for (uint candidate = 0; candidate < 4; candidate++) {
            float2 samples = noise_pair(candidate_seed(p.stream_seed, candidate)
                ^ 0x8ebc6af09c88c6e3ul, leaf.key, element / 2ul, p.mode);
            float radius = (candidate % PROCEDURAL_SLOTS) & 1u
                ? p.alternate_radius : p.radius;
            candidates[candidate] = encode_pair(
                base_value + (leaf.scale * radius) * samples);
        }
#endif
#ifdef POOL_GEOMETRY_DIAGNOSTICS
        float2 candidate_deltas[4];
#endif
        for (uint candidate = 0; candidate < 4; candidate++) {
            float2 delta = decode_pair(candidates[candidate]) - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            candidate_deltas[candidate] = delta;
#endif
            float2 squared = delta * delta;
            float pair_squared = squared.x + (has_second ? squared.y : 0.0f);
            proposal_norms[candidate] += pair_squared * leaf.weight;
            proposal_squared[candidate] += pair_squared;
            proposal_changed[candidate] += uint(candidates[candidate].x != base_bits.x)
                + uint(has_second && candidates[candidate].y != base_bits.y);
            proposal_invalid[candidate] |= uint(invalid_value(candidates[candidate].x))
                | uint(has_second && invalid_value(candidates[candidate].y));
        }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
        const uint2 candidate_pairs[6] = {
            uint2(0u, 1u), uint2(0u, 2u), uint2(0u, 3u),
            uint2(1u, 2u), uint2(1u, 3u), uint2(2u, 3u),
        };
        for (uint dot = 0; dot < 6u; dot++) {
            float2 product = candidate_deltas[candidate_pairs[dot].x]
                * candidate_deltas[candidate_pairs[dot].y];
            proposal_dots[dot] += (product.x + (has_second ? product.y : 0.0f))
                * leaf.weight;
        }
#endif
        for (uint row = 0; row < p.history; row++) {
            ushort2 historical = current_lineage;
            if (row != 0u) {
                ReplayStep step = steps[row];
#ifdef RADEMACHER_ONLY
                float2 noise = noise_pair(
                    step.seed ^ 0x8ebc6af09c88c6e3ul,
                    leaf.key,
                    element / 2ul,
                    step.mode);
#else
                float2 noise = ziggurat_pair_domain(
                    history_domains[row], element / 2ul);
#endif
                float scale = historical_scales[row * p.blocks + tile.leaf];
                historical = encode_pair(
                    decode_pair(current_lineage) + scale * step.radius * noise);
            }
            float2 historical_value = decode_pair(historical);
            for (uint candidate = 0; candidate < 4; candidate++) {
                float2 delta = decode_pair(candidates[candidate]) - historical_value;
                float2 squared = delta * delta;
                float pair_squared = squared.x + (has_second ? squared.y : 0.0f);
                uint metric = row * 4u + candidate;
                sums[metric] += pair_squared * leaf.weight;
                component_sums[metric] += pair_squared * base_weights[tile.leaf];
            }
            if (steps[row].accepted != 0u) current_lineage = historical;
        }
        lineage[index] = current_lineage.x;
        if (has_second) lineage[index + 1ul] = current_lineage.y;
    }
    threadgroup float weighted_groups[8 * 64];
    threadgroup float component_groups[8 * 64];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    uint metrics = p.history * 4u;
    for (uint metric = 0; metric < metrics; metric++) {
        float weighted = simd_sum(sums[metric]);
        float component = simd_sum(component_sums[metric]);
        if (lane == 0u) {
            weighted_groups[simdgroup * 64u + metric] = weighted;
            component_groups[simdgroup * 64u + metric] = component;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < metrics) {
        float weighted = 0.0f;
        float component = 0.0f;
        for (uint simd = 0; simd < 8u; simd++) {
            weighted += weighted_groups[simd * 64u + tid];
            component += component_groups[simd * 64u + tid];
        }
        uint row = tid / 4u;
        uint candidate = tid - row * 4u;
        ulong output = (ulong(candidate) * 128ul + row) * p.tiles + tile_index;
        partials[output] = weighted;
        components[output] = component;
    }
    threadgroup float norm_groups[8 * 4];
    threadgroup float squared_groups[8 * 4];
    threadgroup uint changed_groups[8 * 4];
    threadgroup uint invalid_groups[8 * 4];
    for (uint candidate = 0; candidate < 4; candidate++) {
        float norm = simd_sum(proposal_norms[candidate]);
        float squared = simd_sum(proposal_squared[candidate]);
        uint changed = simd_sum(proposal_changed[candidate]);
        uint invalid = simd_sum(proposal_invalid[candidate]);
        if (lane == 0u) {
            uint slot = simdgroup * 4u + candidate;
            norm_groups[slot] = norm;
            squared_groups[slot] = squared;
            changed_groups[slot] = changed;
            invalid_groups[slot] = invalid;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        float norm = 0.0f;
        float squared = 0.0f;
        uint changed = 0u;
        uint invalid = 0u;
        for (uint simd = 0; simd < 8u; simd++) {
            uint slot = simd * 4u + tid;
            norm += norm_groups[slot];
            squared += squared_groups[slot];
            changed += changed_groups[slot];
            invalid |= invalid_groups[slot];
        }
        pool_partials[tid * p.tiles + tile_index] = {
            0.0f, 0.0f, squared, changed, invalid
        };
        pool_geometry[tid * p.tiles + tile_index] = norm;
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float dot_groups[8 * 6];
    for (uint dot = 0; dot < 6u; dot++) {
        float value = simd_sum(proposal_dots[dot]);
        if (lane == 0u) dot_groups[simdgroup * 6u + dot] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) {
        float value = 0.0f;
        for (uint simd = 0; simd < 8u; simd++)
            value += dot_groups[simd * 6u + tid];
        pool_geometry[(tid + 4u) * p.tiles + tile_index] = value;
    }
    if (tid < 5u)
        pool_geometry[(tid + 10u) * p.tiles + tile_index] = 0.0f;
#endif
#endif
}

// Replay a complete 32-observation slice as a SIMD pipeline.  Each lane owns
// one historical observation while candidate and lineage values stream from
// lane zero to lane 31.  This evaluates every realized FP16 coordinate without
// materializing historical model rows.
kernel void bf16_replay_history(
    device ushort *lineage [[buffer(0)]],
    device const ushort *base [[buffer(1)]],
    device const Leaf *leaves [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device const ReplayStep *steps [[buffer(4)]],
    device const float *historical_scales [[buffer(5)]],
    device const float *base_weights [[buffer(6)]],
    device float *partials [[buffer(7)]],
    device float *components [[buffer(8)]],
    constant Params &p [[buffer(9)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
#ifndef FP16_INDEPENDENT
    return;
#else
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    uint active_rows = min(32u, p.history - p.initialize);
    uint width = 1u;
    while (width < active_rows) width <<= 1u;
    uint pipelines_per_simdgroup = 32u / width;
    uint local_lane = lane & (width - 1u);
    uint local_pipeline = lane / width;
    uint pipeline = simdgroup * pipelines_per_simdgroup + local_pipeline;
    uint pipeline_count = 8u * pipelines_per_simdgroup;
    uint row = p.initialize + local_lane;
    uint pair_count = (tile.length + 1u) / 2u;
    uint coordinates = pair_count > pipeline
        ? (pair_count - pipeline + pipeline_count - 1u) / pipeline_count
        : 0u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float component_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint carried_lineage = 0u;
    uint carried_candidates[4] = {0u, 0u, 0u, 0u};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
#ifndef RADEMACHER_ONLY
    uint persistent_domain = ziggurat_domain(persistent_seed, leaf.key);
    uint fresh_domain = ziggurat_domain(fresh_seed, leaf.key);
    uint history_domain = row < p.history
        ? ziggurat_domain(
            steps[row].seed ^ 0x8ebc6af09c88c6e3ul,
            leaf.key)
        : 0u;
#endif
    for (uint cycle = 0; cycle < coordinates + width - 1u; cycle++) {
        bool valid = local_lane < active_rows
            && cycle >= local_lane
            && cycle - local_lane < coordinates;
        uint source_lane = local_lane == 0u ? lane : lane - 1u;
        uint incoming = simd_shuffle(carried_lineage, source_lane);
        uint current[4];
        for (uint candidate = 0; candidate < 4; candidate++)
            current[candidate] = simd_shuffle(carried_candidates[candidate], source_lane);
        if (local_lane == 0u) {
            uint ordinal = cycle;
            bool inject = ordinal < coordinates;
            uint pair = ordinal * pipeline_count + pipeline;
            uint item = pair * 2u;
            bool has_second = item + 1u < tile.length;
            ulong element = ulong(tile.start) + item;
            ulong index = leaf.offset + element;
            ushort2 lineage_bits = ushort2(
                inject ? lineage[index] : 0u,
                inject && has_second ? lineage[index + 1ul] : 0u);
            ushort2 base_bits = ushort2(
                inject ? base[index] : 0u,
                inject && has_second ? base[index + 1ul] : 0u);
            incoming = as_type<uint>(lineage_bits);
            float2 base_value = decode_pair(base_bits);
#ifdef RADEMACHER_ONLY
            float2 persistent = inject
                ? noise_pair(persistent_seed, leaf.key, element / 2ul, p.mode)
                : float2(0.0f);
            float2 fresh = inject
                ? noise_pair(fresh_seed, leaf.key, element / 2ul, p.mode)
                : float2(0.0f);
#else
            float2 persistent = inject
                ? ziggurat_pair_domain(persistent_domain, element / 2ul)
                : float2(0.0f);
            float2 fresh = inject
                ? ziggurat_pair_domain(fresh_domain, element / 2ul)
                : float2(0.0f);
#endif
            current[0] = as_type<uint>(
                encode_pair(base_value + leaf.scale * p.radius * persistent));
            current[1] = as_type<uint>(
                encode_pair(base_value + leaf.scale * p.alternate_radius * persistent));
            current[2] = as_type<uint>(
                encode_pair(base_value + leaf.scale * p.radius * fresh));
            current[3] = as_type<uint>(
                encode_pair(base_value + leaf.scale * p.alternate_radius * fresh));
#ifdef PROCEDURAL_SLOTS
            for (uint candidate = 0; candidate < 4; candidate++) {
                float2 samples = noise_pair(candidate_seed(p.stream_seed, candidate)
                    ^ 0x8ebc6af09c88c6e3ul, leaf.key, element / 2ul, p.mode);
                float radius = (candidate % PROCEDURAL_SLOTS) & 1u
                    ? p.alternate_radius : p.radius;
                current[candidate] = as_type<uint>(encode_pair(
                    base_value + (leaf.scale * radius) * samples));
            }
#endif
        }
        uint historical = incoming;
        if (valid && row != 0 && row < p.history) {
            ReplayStep step = steps[row];
            uint ordinal = cycle - local_lane;
            uint pair = ordinal * pipeline_count + pipeline;
            ulong element = ulong(tile.start) + ulong(pair * 2u);
#ifdef RADEMACHER_ONLY
            float2 noise = noise_pair(
                step.seed ^ 0x8ebc6af09c88c6e3ul,
                leaf.key,
                element / 2ul,
                step.mode);
#else
            float2 noise = ziggurat_pair_domain(history_domain, element / 2ul);
#endif
            float scale = historical_scales[row * p.blocks + tile.leaf];
            historical = as_type<uint>(encode_pair(
                decode_pair(as_type<ushort2>(incoming)) + scale * step.radius * noise));
        }
        if (valid && row < p.history) {
            uint ordinal = cycle - local_lane;
            uint pair = ordinal * pipeline_count + pipeline;
            bool has_second = pair * 2u + 1u < tile.length;
            float2 historical_value = decode_pair(as_type<ushort2>(historical));
            for (uint candidate = 0; candidate < 4; candidate++) {
                float2 delta = decode_pair(as_type<ushort2>(current[candidate]))
                    - historical_value;
                float2 squared = delta * delta;
                float pair_squared = squared.x + (has_second ? squared.y : 0.0f);
                sums[candidate] += pair_squared * leaf.weight;
                component_sums[candidate] += pair_squared * base_weights[tile.leaf];
            }
        }
        carried_lineage = valid && row < p.history && steps[row].accepted != 0
            ? historical
            : incoming;
        for (uint candidate = 0; candidate < 4; candidate++)
            carried_candidates[candidate] = current[candidate];
        if (local_lane == width - 1u && cycle >= local_lane
                && cycle - local_lane < coordinates) {
            uint ordinal = cycle - local_lane;
            uint pair = ordinal * pipeline_count + pipeline;
            uint item = pair * 2u;
            bool has_second = item + 1u < tile.length;
            ulong index = leaf.offset + ulong(tile.start) + item;
            ushort2 lineage_bits = as_type<ushort2>(carried_lineage);
            lineage[index] = lineage_bits.x;
            if (has_second) lineage[index + 1ul] = lineage_bits.y;
        }
    }
    threadgroup float weighted[4 * 256];
    threadgroup float unweighted[4 * 256];
    for (uint candidate = 0; candidate < 4; candidate++) {
        uint slot = candidate * 256u + tid;
        weighted[slot] = sums[candidate];
        unweighted[slot] = component_sums[candidate];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < active_rows) {
        uint output_row = p.initialize + tid;
        for (uint candidate = 0; candidate < 4; candidate++) {
            float total = 0.0f;
            float component = 0.0f;
            for (uint group = 0; group < 8u; group++) {
                for (uint packed = 0; packed < pipelines_per_simdgroup; packed++) {
                    uint source = group * 32u + packed * width + tid;
                    total += weighted[candidate * 256u + source];
                    component += unweighted[candidate * 256u + source];
                }
            }
            ulong output = (ulong(candidate) * 128ul + output_row)
                * p.tiles + tile_index;
            partials[output] = total;
            components[output] = component;
        }
    }
#endif
}

kernel void bf16_reduce_replay(
    device const float *partials [[buffer(0)]],
    device const float *components [[buffer(1)]],
    device const uint *family_groups [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device float *distances [[buffer(4)]],
    device float *family_distances [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    uint output [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint candidate = output / p.history;
    uint row = output - candidate * p.history;
    float total = 0.0f;
    float families[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ulong source = (ulong(candidate) * 128ul + row) * p.tiles;
    for (uint tile = tid; tile < p.tiles; tile += 256u) {
        total += partials[source + tile];
        uint family = min(family_groups[tiles[tile].leaf], 3u);
        families[family] += components[source + tile];
    }
    threadgroup float reduced[5 * 256];
    reduced[tid] = total;
    for (uint family = 0; family < 4; family++)
        reduced[(family + 1u) * 256u + tid] = families[family];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride)
            for (uint value = 0; value < 5; value++)
                reduced[value * 256u + tid] += reduced[value * 256u + tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        distances[candidate * p.history + row] = reduced[0];
        ulong family_output = (ulong(candidate) * p.history + row) * 4ul;
        for (uint family = 0; family < 4; family++)
            family_distances[family_output + family] = reduced[(family + 1u) * 256u];
    }
}

// Collapse the per-tile records in parallel before the deliberately scalar
// acquisition step.  The old selector walked every tile on one GPU thread.
kernel void bf16_reduce_pool(
    device const Partial *partials [[buffer(0)]],
    device const float *geometry [[buffer(1)]],
    device Partial *aggregates [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    float anchor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rejected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    for (uint tile = tid; tile < q.tiles; tile += 256) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            Partial p = partials[(pair * 4 + candidate) * q.tiles + tile];
            anchor[candidate] += p.anchor;
            rejected[candidate] += p.rejected;
            changed[candidate] |= p.changed;
            invalid[candidate] |= p.invalid | uint(!isfinite(p.squared));
            if (pair == 0)
                norm[candidate] += geometry[candidate * q.tiles + tile];
        }
    }
    threadgroup float anchor_sums[1024], rejected_sums[1024], norm_sums[1024];
    threadgroup uint changed_flags[1024], invalid_flags[1024];
    for (uint candidate = 0; candidate < 4; candidate++) {
        uint slot = candidate * 256 + tid;
        anchor_sums[slot] = anchor[candidate];
        rejected_sums[slot] = rejected[candidate];
        norm_sums[slot] = norm[candidate];
        changed_flags[slot] = changed[candidate];
        invalid_flags[slot] = invalid[candidate];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint candidate = 0; candidate < 4; candidate++) {
                uint slot = candidate * 256 + tid;
                uint other = slot + stride;
                anchor_sums[slot] += anchor_sums[other];
                rejected_sums[slot] += rejected_sums[other];
                norm_sums[slot] += norm_sums[other];
                changed_flags[slot] |= changed_flags[other];
                invalid_flags[slot] |= invalid_flags[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            uint slot = candidate * 256;
            aggregates[pair * 4 + candidate] = {
                anchor_sums[slot],
                rejected_sums[slot],
                norm_sums[slot],
                changed_flags[slot],
                invalid_flags[slot]
            };
        }
    }
}

kernel void bf16_select(
    device const Partial *aggregates [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device float *pool_distances [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    float distances[4][128];
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool changed[4] = {false, false, false, false};
    bool invalid[4] = {false, false, false, false};
    for (uint candidate = 0; candidate < 4; candidate++) {
        proposal_norms[candidate] = q.latent_history != 0
            ? q.latent_norms[candidate]
            : aggregates[candidate].squared;
        for (uint row = 0; row < q.history; row++) {
            distances[candidate][row] = q.exact_history != 0
                ? pool_distances[candidate * q.history + row]
                : q.implicit_history != 0
                ? q.base_distances[row] + proposal_norms[candidate]
                : 0.0f;
            if (q.program != 0 && q.exact_history == 0)
                distances[candidate][row] += pool_distances[candidate * q.history + row];
            if (q.axis_enabled != 0) {
                float delta = q.axis_candidates[candidate] - q.axis_history[row];
                distances[candidate][row] += q.axis_weight * delta * delta;
            }
        }
        if (q.exact_history != 0) {
            proposal_norms[candidate] = distances[candidate][q.incumbent_index];
            changed[candidate] = proposal_norms[candidate] > 0.0f;
            for (uint row = 0; row < q.history; row++)
                invalid[candidate] = invalid[candidate]
                    || !isfinite(distances[candidate][row]);
        }
    }
    if (q.latent_history != 0) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            changed[candidate] = proposal_norms[candidate] > 0.0f;
            invalid[candidate] = !isfinite(proposal_norms[candidate]);
        }
    }
    if (q.exact_history == 0 && q.latent_history == 0)
    for (uint candidate = 0; candidate < 4; candidate++) {
        for (uint row = 0; row < q.resident_history; row += 2) {
            Partial aggregate = aggregates[(row / 2) * 4 + candidate];
            float first = aggregate.anchor;
            float second = aggregate.rejected;
            changed[candidate] = changed[candidate] || aggregate.changed != 0;
            invalid[candidate] = invalid[candidate] || aggregate.invalid != 0;
            uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
            if (q.exact_history == 0 && q.latent_history == 0)
                distances[candidate][first_index] = first;
            if (row + 1 < q.resident_history) {
                uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
                if (q.exact_history == 0 && q.latent_history == 0)
                    distances[candidate][second_index] = second;
            }
            invalid[candidate] = invalid[candidate] || !isfinite(first)
                || (row + 1 < q.resident_history && !isfinite(second));
        }
    }
    bool any_changed = false;
    for (uint candidate = 0; candidate < 4; candidate++) {
        changed[candidate] = changed[candidate] || (q.axis_enabled != 0
            && q.axis_candidates[candidate] != q.axis_base);
        any_changed = any_changed || changed[candidate];
    }
    uint selected = 4;
    float best = -INFINITY;
    float candidate_means[4];
    float candidate_standard_errors[4];
    float incumbent_distances[128];
    for (uint row = 0; row < q.history; row++)
        incumbent_distances[row] = q.base_distances[row]
            + (q.axis_enabled != 0
                ? q.axis_weight * (q.axis_base - q.axis_history[row])
                    * (q.axis_base - q.axis_history[row])
                : 0.0f)
            + (q.program != 0 ? pool_distances[4 * q.history + row] : 0.0f);
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
    decision->basis_seed = q.basis_seed;
    decision->radius = selected < 4
#if defined(PROCEDURAL_SLOTS) && PROCEDURAL_SLOTS == 1
        ? q.radius
#else
        ? ((selected & 1u) == 0 ? q.radius : q.alternate_radius)
#endif
        : 0.0f;
    decision->score = selected < 4 ? best : 0.0f;
    decision->mode = q.mode;
    decision->program = q.program;
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
    device const ulong *threshold_tables [[buffer(7)]],
    constant uint &account [[buffer(8)]],
    device Partial *partials [[buffer(9)]],
    constant Params &p [[buffer(10)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (decision->valid == 0) return;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    ulong seed = decision->seed ^ 0x8ebc6af09c88c6e3ul;
    float squared = 0.0f;
    uint changed = 0u;
    uint invalid = 0u;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        float2 samples = decision->program == 1u
            ? threshold_pair(
                threshold_tables,
                decision->index,
                decision->basis_seed,
                leaf.address,
                pair)
            : noise_pair(seed, leaf.key, pair, decision->mode);
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
#ifdef FP16_INDEPENDENT
        float step = leaf.scale * decision->radius;
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + index)
            : ushort2(base[index], complete_pair ? base[index + 1] : 0);
        ushort2 result = encode_pair(decode_pair(base_pair) + step * samples);
        if (aligned_pair) {
            *reinterpret_cast<device ushort2*>(output + index) = result;
        } else {
            output[index] = result.x;
            if (complete_pair) output[index + 1] = result.y;
        }
        if (account != 0u) {
            uint pair_length = complete_pair ? 2u : 1u;
            for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
                ushort value = result[pair_offset];
                changed += uint(value != base_pair[pair_offset]);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - decode(base_pair[pair_offset]);
                squared += delta * delta;
            }
        }
#else
        float inverse_rms = scales[tile.leaf];
        float step = leaf.scale * decision->radius;

        ushort first_bits = base[index];
        float first_direction = decision->index < 2
            ? direction(decode(reference[index]), inverse_rms, samples.x, decision->index)
            : samples.x;
        output[index] = encode(decode(first_bits) + step * first_direction);
        if (account != 0u) {
            ushort value = output[index];
            changed += uint(value != first_bits);
            invalid |= uint(invalid_value(value));
            float delta = decode(value) - decode(first_bits);
            squared += delta * delta;
        }

        if (item + 1 < tile.length) {
            ushort second_bits = base[index + 1];
            float second_direction = decision->index < 2
                ? direction(
                    decode(reference[index + 1]),
                    inverse_rms,
                    samples.y,
                    decision->index)
                : samples.y;
            output[index + 1] = encode(decode(second_bits) + step * second_direction);
            if (account != 0u) {
                ushort value = output[index + 1];
                changed += uint(value != second_bits);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - decode(second_bits);
                squared += delta * delta;
            }
        }
#endif
    }
    if (account == 0u) return;
    threadgroup float squared_sums[8];
    threadgroup uint changed_sums[8], invalid_flags[8];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    float squared_sum = simd_sum(squared);
    uint changed_sum = simd_sum(changed);
    uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        squared_sums[simdgroup] = squared_sum;
        changed_sums[simdgroup] = changed_sum;
        invalid_flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        bool active = lane < 8;
        uint slot = min(lane, 7u);
        float total_squared = simd_sum(active ? squared_sums[slot] : 0.0f);
        uint total_changed = simd_sum(active ? changed_sums[slot] : 0u);
        uint total_invalid = simd_sum(active ? invalid_flags[slot] : 0u);
        if (lane == 0)
            partials[decision->index * p.tiles + tile_index] = {
                0.0f, 0.0f, total_squared, total_changed,
                uint(total_invalid != 0u)
            };
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
