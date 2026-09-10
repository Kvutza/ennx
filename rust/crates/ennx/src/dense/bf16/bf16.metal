constant uint kBf16Threads = 256;

struct EnnxBf16Leaf {
    ulong key;
    ulong offset;
    ulong len;
    float scale;
    uint pad;
};

struct EnnxBf16Tile {
    uint leaf;
    uint start;
    uint len;
    uint pad;
};

inline float ennx_bf16_decode(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort ennx_bf16_encode(float value) {
    uint bits = as_type<uint>(value);
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

inline ushort ennx_bf16_next(ushort value, bool positive) {
    if ((value & 0x7fffu) == 0u) {
        return positive ? ushort(1u) : ushort(0x8001u);
    }
    bool grows = ((value & 0x8000u) == 0u) == positive;
    ushort candidate = grows ? ushort(value + 1u) : ushort(value - 1u);
    if (isfinite(ennx_bf16_decode(candidate))) {
        return candidate;
    }
    return grows ? ushort(value - 1u) : ushort(value + 1u);
}

kernel void materialize_bf16(
    device const ushort* base [[buffer(0)]],
    device const EnnxBf16Leaf* leaves [[buffer(1)]],
    device const EnnxBf16Tile* tiles [[buffer(2)]],
    device ushort* candidate [[buffer(3)]],
    constant EnnxDenseTerm* terms [[buffer(4)]],
    constant uint& term_count [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]
) {
    EnnxBf16Tile tile = tiles[group_index];
    EnnxBf16Leaf leaf = leaves[tile.leaf];
    for (uint item = thread_index; item < tile.len; item += kBf16Threads) {
        ulong local = ulong(tile.start + item);
        float sum = 0.0f;
        float strongest = 0.0f;
        bool positive = true;
        for (uint term = 0; term < term_count; ++term) {
            float coefficient = terms[term].coefficient;
            if (coefficient == 0.0f) continue;
            float direction = ennx_dense_sign(terms[term].seed, leaf.key, local);
            sum = fma(coefficient, direction, sum);
            if (abs(coefficient) > strongest) {
                strongest = abs(coefficient);
                positive = (coefficient > 0.0f) == (direction > 0.0f);
            }
        }
        ulong index = leaf.offset + local;
        ushort value = ennx_bf16_encode(
            ennx_bf16_decode(base[index]) + leaf.scale * sum
        );
        candidate[index] = (sum == 0.0f || value == base[index])
            ? ennx_bf16_next(base[index], positive)
            : value;
    }
}
