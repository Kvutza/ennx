// Only the adaptation buffer supplies the current length. Layout and radii are immutable.
#ifdef REGION_METAL
kernel void region_steps(const device Word *region [[buffer(0)]],
                         device Leaf *leaves [[buffer(1)]],
                         const device float *radii [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    uint radius = as_type<uint>(radii[index]);
    uint scale = as_type<uint>(leaves[index].scale);
#else
__kernel void region_steps(__global const Word *region,
                           __global Leaf *leaves,
                           __global const float *radii) {
    uint index = get_global_id(0);
    uint radius = as_uint(radii[index]);
    uint scale = as_uint(leaves[index].scale);
#endif
    uint length = word_float(region[0]);
    uint product = length == 0x7f800000U ? length :
        word_float(word_mul(float_word(length), float_word(radius)));
    uint amplitude = float_div(product, scale);
#ifdef REGION_METAL
    float amount = as_type<float>(amplitude);
#else
    float amount = as_float(amplitude);
#endif
    uint maximum = (1U << leaves[index].bits) - 1;
    amount = min(amount, (float)maximum);
    uint whole = (uint)amount;
    leaves[index].whole = whole;
    leaves[index].threshold = whole == maximum ? 0 :
        (uint)((amount - (float)whole) * 4294967296.0f);
}
