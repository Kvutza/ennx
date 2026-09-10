typedef struct {
    uint base;
    uint count;
    uint capacity;
    uint total;
    uint pending[32];
} SearchState;

#ifdef REGION_METAL
kernel void configure(device SearchState *state [[buffer(0)]],
                      const device uint *history [[buffer(1)]],
                      device Params *params [[buffer(2)]],
                      device uint *choice [[buffer(3)]]) {
#else
__kernel void configure(__global SearchState *state, __global const uint *history,
                        __global Params *params, __global uint *choice) {
#endif
    uint handle = params->trial_slot;
    choice[0] = 0xffffffffU;
    if (handle >= 32 || state->pending[handle] != 0xffffffffU ||
        params->neighbors > state->count) {
        params->history = 0;
        params->candidates = 0;
        return;
    }
    for (uint slot = 0; slot < state->total; slot++) {
        bool used = slot == state->base;
        for (uint i = 0; i < state->count; i++) used |= history[i] == slot;
        for (uint i = 0; i < 32; i++) used |= state->pending[i] == slot;
        if (!used) {
            state->pending[handle] = slot;
            params->base_slot = state->base;
            params->trial_slot = slot;
            params->history = state->count;
            return;
        }
    }
    params->history = 0;
        params->candidates = 0;
}

#ifdef REGION_METAL
kernel void advance(device SearchState *state [[buffer(0)]],
                    device uint *history [[buffer(1)]],
                    device float *outcomes [[buffer(2)]],
                    device Region *region [[buffer(3)]],
                    device Region *next [[buffer(4)]],
                    constant uint &handle [[buffer(5)]],
                    constant Word &value [[buffer(6)]]) {
#else
__kernel void advance(__global SearchState *state, __global uint *history,
                      __global float *outcomes, __global Region *region,
                      __global Region *next, uint handle, Word value) {
#endif
    uint slot = state->pending[handle];
    if (slot == 0xffffffffU) return;
    state->pending[handle] = 0xffffffffU;
    Word remaining = 0;
    for (uint i = 0; i < 32; i++) remaining += state->pending[i] != 0xffffffffU;
    adapt_region(region, next, value, remaining);
    *region = *next;
    if (region->accepted) state->base = slot;
    if (state->count == state->capacity) {
        for (uint i = 1; i < state->count; i++) {
            history[i - 1] = history[i];
            outcomes[i - 1] = outcomes[i];
        }
        state->count--;
    }
    history[state->count] = slot;
#ifdef REGION_METAL
    outcomes[state->count++] = as_type<float>(word_float(value));
#else
    outcomes[state->count++] = as_float(word_float(value));
#endif
    if (region->restarted) {
        state->count = 1;
        history[0] = state->base;
#ifdef REGION_METAL
        outcomes[0] = as_type<float>(word_float(region->best));
#else
        outcomes[0] = as_float(word_float(region->best));
#endif
    }
}

#ifdef REGION_METAL
kernel void copy_row(device uchar *rows [[buffer(0)]],
                     const device Params *params [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
    uint stride = params->row_stride;
#else
__kernel void copy_row(__global uchar *rows, __global const Params *params) {
    uint index = get_global_id(0);
    uint stride = params->row_bytes;
#endif
    if (params->candidates == 0 || index >= stride) return;
    rows[(ulong)params->trial_slot * stride + index] =
        rows[(ulong)params->base_slot * stride + index];
}
