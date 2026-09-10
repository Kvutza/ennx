#ifndef ENNX_H
#define ENNX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ENNX_OK = 0,
    ENNX_INVALID_ARGUMENT = 1,
};

typedef struct {
    uint64_t key;
    size_t offset;
    size_t len;
    float scale;
} ennx_dense_leaf;

typedef struct {
    uint64_t seed;
    float coefficient;
} ennx_dense_term;

typedef struct {
    uint64_t key;
    uint64_t start;
    float scale;
} ennx_dense_view;

uint32_t ennx_abi_version(void);

int32_t ennx_dense_apply_f32(
    const float* base,
    size_t num_values,
    const ennx_dense_leaf* leaves,
    size_t num_leaves,
    const ennx_dense_term* terms,
    size_t num_terms,
    float* out,
    size_t* out_changed);

int32_t ennx_dense_dist2(
    const ennx_dense_leaf* leaves,
    size_t num_leaves,
    const ennx_dense_term* left,
    size_t num_left,
    const ennx_dense_term* right,
    size_t num_right,
    double* out_distance);

int32_t ennx_dense_linear_f32(
    const float* input,
    size_t columns,
    const float* weight,
    size_t rows,
    const float* bias,
    ennx_dense_view weight_view,
    ennx_dense_view bias_view,
    const ennx_dense_term* terms,
    size_t num_terms,
    float* out);

#ifdef __cplusplus
}
#endif

#endif
