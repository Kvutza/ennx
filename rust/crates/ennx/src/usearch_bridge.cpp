#include "usearch_bridge.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <usearch/index_dense.hpp>

namespace {
using index_t = unum::usearch::index_dense_t;
using metric_t = unum::usearch::metric_punned_t;
using metric_kind_t = unum::usearch::metric_kind_t;
using scalar_kind_t = unum::usearch::scalar_kind_t;

struct EnnUsearchIndex {
    index_t index;
};

void write_error(char* error, size_t error_capacity, std::string_view message) noexcept {
    if (!error || error_capacity == 0) {
        return;
    }
    size_t const copy_len = std::min(message.size(), error_capacity - 1);
    std::memcpy(error, message.data(), copy_len);
    error[copy_len] = '\0';
}

template <typename Fn>
bool guarded(char* error, size_t error_capacity, Fn&& fn) noexcept {
    try {
        fn();
        return true;
    } catch (std::exception const& e) {
        write_error(error, error_capacity, e.what());
        return false;
    } catch (...) {
        write_error(error, error_capacity, "unknown USearch error");
        return false;
    }
}

} // namespace

extern "C" {

void* ennx_usearch_new(size_t num_dim, char* error, size_t error_capacity) noexcept {
    void* handle = nullptr;
    guarded(error, error_capacity, [&] {
        metric_t metric(num_dim, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
        unum::usearch::index_dense_config_t config;
        auto result = index_t::make(metric, config);
        if (!result) {
            throw std::runtime_error(result.error.what());
        }
        handle = new EnnUsearchIndex{std::move(result)};
    });
    return handle;
}

void ennx_usearch_destroy(void* handle) noexcept {
    delete static_cast<EnnUsearchIndex*>(handle);
}

bool ennx_usearch_reserve(void* handle, size_t capacity, char* error, size_t error_capacity) noexcept {
    return guarded(error, error_capacity, [&] {
        if (!handle) {
            throw std::invalid_argument("USearch handle is null");
        }
        static_cast<EnnUsearchIndex*>(handle)->index.reserve(capacity);
    });
}

bool ennx_usearch_add(
    void* handle,
    uint64_t key,
    float const* vector,
    size_t num_dim,
    char* error,
    size_t error_capacity) noexcept {
    return guarded(error, error_capacity, [&] {
        if (!handle) {
            throw std::invalid_argument("USearch handle is null");
        }
        if (!vector) {
            throw std::invalid_argument("USearch vector is null");
        }
        auto& index = static_cast<EnnUsearchIndex*>(handle)->index;
        if (index.scalar_words() != num_dim) {
            throw std::invalid_argument("Vector length must match index dimensionality");
        }
        index.add(key, vector).error.raise();
    });
}

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
    size_t error_capacity) noexcept {
    return guarded(error, error_capacity, [&] {
        if (!handle) {
            throw std::invalid_argument("USearch handle is null");
        }
        if (!query || !out_keys || !out_count) {
            throw std::invalid_argument("USearch search buffers are null");
        }
        auto const& index = static_cast<EnnUsearchIndex const*>(handle)->index;
        if (index.scalar_words() != num_dim) {
            throw std::invalid_argument("Vector length must match index dimensionality");
        }
        auto result = index.search(query, wanted, index_t::any_thread(), exact);
        result.error.raise();
        *out_count = result.dump_to(out_keys, out_capacity);
    });
}

size_t ennx_usearch_expansion_search(void const* handle) noexcept {
    if (!handle) {
        return 0;
    }
    return static_cast<EnnUsearchIndex const*>(handle)->index.expansion_search();
}

size_t ennx_usearch_memory_usage(void const* handle) noexcept {
    if (!handle) {
        return 0;
    }
    return static_cast<EnnUsearchIndex const*>(handle)->index.memory_usage();
}

} // extern "C"
