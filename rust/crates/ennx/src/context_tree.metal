#include <metal_stdlib>
using namespace metal;

// Projected, rotated chunk rows are packed Q(512), K(64), V(64).
// Only KV persists across chunks; queries remain in bounded scratch.
kernel void context_pack(
    device const half* qkv [[buffer(0)]],
    device half* kv [[buffer(1)]],
    device half* queries [[buffer(2)]],
    constant uint2& range [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= range.y * 640) return;
    const uint row = gid / 640, dim = gid % 640;
    if (dim < 512) queries[ulong(row) * 512 + dim] = qkv[gid];
    else kv[ulong(range.x + row) * 128 + dim - 512] = qkv[gid];
}

// KV rows contain one 64-dimensional key followed by one value.
kernel void context_leaves(
    device const half* kv [[buffer(0)]],
    device half* tree [[buffer(1)]],
    constant uint2& range [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    const uint leaf = range.x + group;
    if (group >= range.y || dim >= 64) return;
    float sum = 0.0f;
    for (uint token = 0; token < 64; ++token)
        sum += float(kv[ulong(leaf * 64 + token) * 128 + dim]);
    tree[ulong(leaf) * 64 + dim] = half(sum / 64.0f);
}

// A dispatch owns one complete tree level; the host inserts a device barrier
// before dispatching its parents. Dirty ranges can rebuild only their closure.
kernel void context_parents(
    device half* tree [[buffer(0)]],
    constant uint4& level [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    if (group >= level.w || dim >= 64) return;
    const uint parent = level.z + group;
    const ulong left = ulong(level.x + 2 * parent) * 64 + dim;
    const ulong output = ulong(level.y + parent) * 64 + dim;
    tree[output] = half(0.5f * (float(tree[left]) + float(tree[left + 64])));
}
