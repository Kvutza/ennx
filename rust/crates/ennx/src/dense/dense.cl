#define ENNX_THREADS 256

typedef struct {
    ulong key;
    ulong offset;
    ulong len;
    float scale;
    uint pad;
} Leaf;

typedef struct {
    uint leaf;
    uint start;
    uint len;
    uint pad;
} Tile;

__kernel void apply_dense(
    __global const float* base,
    __global const Leaf* leaves,
    __global const EnnxDenseTerm* terms,
    __global const Tile* tiles,
    __global float* out,
    uint term_count
) {
    Tile tile = tiles[get_group_id(0)];
    Leaf leaf = leaves[tile.leaf];
    for (uint item = get_local_id(0); item < tile.len; item += ENNX_THREADS) {
        ulong coordinate = (ulong)(tile.start + item);
        ulong index = leaf.offset + coordinate;
        out[index] = ennx_dense_value(
            base[index],
            leaf.key,
            coordinate,
            leaf.scale,
            terms,
            term_count
        );
    }
}
