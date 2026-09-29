#include <metal_stdlib>
using namespace metal;

struct ProposalWire {
    ulong seed;
    uint round;
    float radius;
    ulong coordinates;
};

inline uint4 philox(uint4 value, uint2 key) {
    for (uint iteration = 0; iteration < 10; ++iteration) {
        const ulong first = ulong(value.x) * 0xD2511F53ul;
        const ulong second = ulong(value.z) * 0xCD9E8D57ul;
        value = uint4(
            uint(second >> 32) ^ value.y ^ key.x,
            uint(second),
            uint(first >> 32) ^ value.w ^ key.y,
            uint(first)
        );
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return value;
}

kernel void ennx_proposal(
    device const half* base [[buffer(0)]],
    device half* candidate [[buffer(1)]],
    constant ProposalWire& wire [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong first = ulong(gid) * 4;
    if (first >= wire.coordinates) return;
    const uint4 random = philox(
        uint4(uint(first), uint(first >> 32), wire.round, 0x12345678u),
        uint2(uint(wire.seed), uint(wire.seed >> 32))
    );
    for (uint lane = 0; lane < 4 && first + lane < wire.coordinates; ++lane) {
        const float sign = (random[lane] & 1u) == 0u ? -1.0f : 1.0f;
        candidate[first + lane] = half(float(base[first + lane]) + wire.radius * sign);
    }
}
