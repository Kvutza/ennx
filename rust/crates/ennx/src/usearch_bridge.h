#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* ennx_usearch_new(size_t num_dim, char* error, size_t error_capacity) noexcept;
void ennx_usearch_destroy(void* handle) noexcept;
bool ennx_usearch_reserve(void* handle, size_t capacity, char* error, size_t error_capacity) noexcept;
bool ennx_usearch_add(
    void* handle,
    uint64_t key,
    float const* vector,
    size_t num_dim,
    char* error,
    size_t error_capacity) noexcept;
bool ennx_usearch_search(
    void const* handle,
    float const* query,
    size_t num_dim,
    size_t wanted,
    bool exact,
    uint64_t* out_keys,
    size_t out_capacity,
    size_t* out_count,
    char* error,
    size_t error_capacity) noexcept;
size_t ennx_usearch_expansion_search(void const* handle) noexcept;
size_t ennx_usearch_memory_usage(void const* handle) noexcept;

#ifdef __cplusplus
}
#endif
