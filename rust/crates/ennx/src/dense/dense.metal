constant uint kThreads = 256;

struct Leaf {
    ulong key;
    ulong offset;
    ulong len;
    float scale;
    uint pad;
};

struct Tile {
    uint leaf;
    uint start;
    uint len;
    uint pad;
};

kernel void apply_dense(
    device const float* base [[buffer(0)]],
    device const Leaf* leaves [[buffer(1)]],
    device const EnnxDenseTerm* terms [[buffer(2)]],
    device const Tile* tiles [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& term_count [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]
) {
    Tile tile = tiles[group_index];
    Leaf leaf = leaves[tile.leaf];
    for (uint item = thread_index; item < tile.len; item += kThreads) {
        ulong local = ulong(tile.start + item);
        ulong index = leaf.offset + local;
        out[index] = ennx_dense_value(
            base[index],
            leaf.key,
            local,
            leaf.scale,
            terms,
            term_count
        );
    }
}
