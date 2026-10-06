#include <metal_stdlib>
using namespace metal;

struct IndexShape {
    uint start, rows, block, mode;
    uint layer, fresh;
    float reuse;
    uint tokens;
};

// A four-query proxy walks the bounded tree frontier once. Refined mode
// rescoring is restricted to its sixteen final leaves. Reused supports are
// approximate: query and selected-summary drift are measured, not certified.
kernel void pisa_index(
    device const half* queries [[buffer(0)]],
    device const half* tree [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    device const half* weights [[buffer(3)]],
    device float* history [[buffer(4)]],
    device uint* saved [[buffer(5)]],
    device uint* stamps [[buffer(6)]],
    device atomic_uint* counters [[buffer(7)]],
    constant IndexShape& p [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float q[4][64];
    threadgroup float pooled[4][64];
    threadgroup float proxy[64];
    threadgroup uint candidates[16];
    threadgroup uint selected[4][8];
    const uint leaves = p.tokens / 64;
    const uint row = p.start + tile * 4;
    const uint current = min(p.tokens, (row / p.block + 1) * p.block) / 64 - 1;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint bank = p.layer * 1024 + tile;
    const uint stamp = row;
    const ulong state = ulong(bank) * 2304;
    for (uint input = lane; input < 64; input += 32) {
        for (uint query = 0; query < 4; ++query) {
            float value = 0.0f;
            for (uint head = 0; head < 8; ++head)
                value += float(queries[ulong(tile * 4 + query) * 512 + head * 64 + input]);
            pooled[query][input] = value;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dim = lane; dim < 64; dim += 32) {
        float mean = 0.0f;
        for (uint query = 0; query < 4; ++query) {
            float value = 0.0f;
            for (uint input = 0; input < 64; ++input) {
                value = fma(pooled[query][input], float(weights[input * 64 + dim]), value);
            }
            q[query][dim] = value;
            mean += value;
        }
        proxy[dim] = mean * 0.25f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float diff = 0.0f, norm = 0.0f;
    for (uint i = lane; i < 256; i += 32) {
        float a = q[i / 64][i % 64], b = history[state + i];
        diff = fma(a - b, a - b, diff); norm = fma(b, b, norm);
    }
    float drift = simd_sum(diff) / max(simd_sum(norm), 1e-12f);
    for (uint choice = 0; choice < 32; ++choice) {
        const uint block = saved[ulong(bank) * 32 + choice];
        float delta = 0.0f, magnitude = 0.0f;
        if (block != UINT_MAX && block < leaves) {
            for (uint dim = lane; dim < 64; dim += 32) {
                float a = float(tree[ulong(block) * 64 + dim]);
                float b = history[state + 256 + choice * 64 + dim];
                delta = fma(a - b, a - b, delta); magnitude = fma(b, b, magnitude);
            }
        }
        drift = max(drift, simd_sum(delta) / max(simd_sum(magnitude), 1e-12f));
    }
    const bool reuse = !p.fresh && p.reuse > 0.0f && stamps[bank] == stamp
        && isfinite(drift) && drift <= p.reuse * p.reuse;
    if (reuse) {
        if (lane < 32) blocks[ulong(tile) * 32 + lane] = saved[ulong(bank) * 32 + lane];
        if (lane == 0) atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
        return;
    }
    if (lane == 0) atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
    // Independent mode traverses for each query. Shared/refined traverse once.
    const uint searches = p.mode == 0 ? 4 : 1;
    for (uint search = 0; search < searches; ++search) {
        uint node = lane < 16 ? lane : UINT_MAX;
        for (int level = int(31 - clz(leaves)) - 4; level >= 0; --level) {
            const uint offset = 2 * leaves - (2 * leaves >> uint(level));
            float score = -INFINITY;
            if (lane < 16 && node != UINT_MAX) {
                const uint first = node << uint(level);
                const uint last = ((node + 1) << uint(level)) - 1;
                const bool forced = first == 0 || (first <= previous && previous <= last)
                    || (first <= current && current <= last);
                if (forced) score = INFINITY;
                else if (last < current) {
                    score = 0.0f;
                    for (uint dim = 0; dim < 64; ++dim)
                        score = fma(p.mode == 0 ? q[search][dim] : proxy[dim],
                            float(tree[(ulong(offset) + node) * 64 + dim]), score);
                }
            }
            if (level == 0 && p.mode == 2) {
                if (lane < 16) candidates[lane] = score > -INFINITY ? node : UINT_MAX;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                break;
            }
            float max_finite_score = -INFINITY;
            for (uint slot = 0; slot < 8; ++slot) {
                const float maximum = simd_max(score);
                if (maximum < INFINITY && max_finite_score == -INFINITY) {
                    max_finite_score = maximum;
                }
                const bool drop = (level == 0) && (max_finite_score > -INFINITY)
                    && (maximum < max_finite_score - 12.0f);
                const uint best = simd_min(score == maximum && score > -INFINITY && !drop ? node : UINT_MAX);
                if (lane == slot) selected[search][slot] = best;
                if (node == best) score = -INFINITY;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (level > 0 && lane < 16) {
                const uint parent = selected[search][lane / 2];
                node = parent == UINT_MAX ? UINT_MAX : parent * 2 + lane % 2;
            }
        }
    }
    if (p.mode == 1) {
        for (uint query = 1; query < 4; ++query)
            if (lane < 8) selected[query][lane] = selected[0][lane];
    } else if (p.mode == 2) {
        for (uint query = 0; query < 4; ++query) {
            const uint node = lane < 16 ? candidates[lane] : UINT_MAX;
            float score = -INFINITY;
            if (node != UINT_MAX) {
                if (node == 0 || node == previous || node == current) score = INFINITY;
                else {
                    score = 0.0f;
                    for (uint dim = 0; dim < 64; ++dim)
                        score = fma(q[query][dim], float(tree[ulong(node) * 64 + dim]), score);
                }
            }
            float max_finite_score = -INFINITY;
            for (uint slot = 0; slot < 8; ++slot) {
                const float maximum = simd_max(score);
                if (maximum < INFINITY && max_finite_score == -INFINITY) {
                    max_finite_score = maximum;
                }
                const bool drop = (max_finite_score > -INFINITY)
                    && (maximum < max_finite_score - 12.0f);
                const uint best = simd_min(score == maximum && score > -INFINITY && !drop ? node : UINT_MAX);
                if (lane == slot) selected[query][slot] = best;
                if (node == best) score = -INFINITY;
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        const uint value = selected[lane / 8][lane % 8];
        blocks[ulong(tile) * 32 + lane] = value;
        saved[ulong(bank) * 32 + lane] = value;
    }
    for (uint i = lane; i < 256; i += 32) history[state + i] = q[i / 64][i % 64];
    for (uint choice = 0; choice < 32; ++choice) {
        const uint block = selected[choice / 8][choice % 8];
        for (uint dim = lane; dim < 64; dim += 32)
            history[state + 256 + choice * 64 + dim] = block == UINT_MAX
                ? 0.0f : float(tree[ulong(block) * 64 + dim]);
    }
    if (lane == 0) stamps[bank] = stamp;
}
