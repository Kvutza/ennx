#include <metal_stdlib>
using namespace metal;

struct DraftCaptureShape {
    uint vectors;
    uint slot;
    uint row_start;
    uint context;
};

// Capture target hidden states without scalar conversion. The target scorer
// overwrites its activation buffer at every layer, so selected visits must be
// copied before the next layer starts.
kernel void fbt_draft_capture(
    device const half4* input [[buffer(0)]],
    device half4* features [[buffer(1)]],
    constant DraftCaptureShape& shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < shape.vectors) {
        const ulong slot_vectors = ulong(shape.context) * 512ul / 4ul;
        const ulong row_vectors = ulong(shape.row_start) * 512ul / 4ul;
        features[ulong(shape.slot) * slot_vectors + row_vectors + gid] = input[gid];
    }
}
