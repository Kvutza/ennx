#include <ennx.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    if (ennx_abi_version() != 1) {
        return 1;
    }

    const float base[] = {
        0.5f, -1.0f, 2.0f, 0.25f,
        4.0f, -2.0f, 0.75f, -0.125f,
    };
    const ennx_dense_leaf leaves[] = {
        {.key = 11, .offset = 0, .len = 4, .scale = 0.5f},
        {.key = 29, .offset = 4, .len = 4, .scale = 1.25f},
    };
    const ennx_dense_term terms[] = {
        {.seed = UINT64_C(0x123456789abcdef0), .coefficient = 0.01f},
    };
    const ennx_dense_term other[] = {
        {.seed = UINT64_C(0x123456789abcdef0), .coefficient = -0.01f},
    };
    float first[8] = {0};
    float second[8] = {0};
    size_t first_changed = 0;
    size_t second_changed = 0;

    if (ennx_dense_apply_f32(
            base,
            8,
            leaves,
            2,
            terms,
            1,
            first,
            &first_changed) != ENNX_OK) {
        return 2;
    }
    if (ennx_dense_apply_f32(
            base,
            8,
            leaves,
            2,
            terms,
            1,
            second,
            &second_changed) != ENNX_OK) {
        return 3;
    }
    if (first_changed != 8 || second_changed != 8) {
        return 4;
    }
    if (memcmp(first, second, sizeof(first)) != 0) {
        return 5;
    }
    for (size_t index = 0; index < 8; ++index) {
        if (first[index] == base[index]) {
            return 6;
        }
    }
    double distance = 0.0;
    if (ennx_dense_dist2(
            leaves,
            2,
            terms,
            1,
            other,
            1,
            &distance) != ENNX_OK) {
        return 7;
    }
    if (!(distance > 0.0)) {
        return 8;
    }

    const float input[2] = {0.5f, -1.0f};
    const float weight[4] = {0.25f, 0.75f, -0.5f, 1.25f};
    float linear_out[2] = {0.0f, 0.0f};
    const ennx_dense_view weight_view = {
        .key = 9,
        .start = 0,
        .scale = 0.5f,
    };
    const ennx_dense_view bias_view = {
        .key = 0,
        .start = 0,
        .scale = 1.0f,
    };
    if (ennx_dense_linear_f32(
            input,
            2,
            weight,
            2,
            NULL,
            weight_view,
            bias_view,
            terms,
            1,
            linear_out) != ENNX_OK) {
        return 9;
    }
    if (linear_out[0] == 0.0f || linear_out[1] == 0.0f) {
        return 10;
    }
    return 0;
}
