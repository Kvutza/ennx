#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

float next(uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
}

void fill(std::vector<float>& values, uint32_t seed) {
    for (float& value : values) {
        value = next(seed);
    }
}

} // namespace

int main() {
    constexpr int dims = 32;
    constexpr faiss::idx_t rows = 2048;
    constexpr faiss::idx_t queries = 257;
    constexpr faiss::idx_t k = 16;

    std::vector<float> data(static_cast<size_t>(rows) * dims);
    std::vector<float> input(static_cast<size_t>(queries) * dims);
    fill(data, 17U);
    fill(input, 29U);

    faiss::IndexFlatL2 cpu(dims);
    cpu.add(rows, data.data());

    faiss::gpu_metal::StandardMetalResources resources;
    if (!resources.isAvailable()) {
        std::cerr << "FAISS Metal device unavailable\n";
        return 1;
    }
    faiss::gpu_metal::MetalIndexFlat metal(
            resources.getResources(), dims, faiss::METRIC_L2, 0.0F);
    metal.add(rows, data.data());

    const size_t count = static_cast<size_t>(queries * k);
    std::vector<float> cpu_dist(count);
    std::vector<float> metal_dist(count);
    std::vector<faiss::idx_t> cpu_ids(count);
    std::vector<faiss::idx_t> metal_ids(count);
    cpu.search(queries, input.data(), k, cpu_dist.data(), cpu_ids.data());
    metal.search(
            queries,
            input.data(),
            k,
            metal_dist.data(),
            metal_ids.data());

    for (size_t i = 0; i < count; ++i) {
        const float tolerance = 2.0e-4F * (1.0F + std::abs(cpu_dist[i]));
        if (cpu_ids[i] != metal_ids[i] ||
            std::abs(cpu_dist[i] - metal_dist[i]) > tolerance) {
            std::cerr << "mismatch at " << i << ": cpu=(" << cpu_ids[i]
                      << ", " << cpu_dist[i] << ") metal=(" << metal_ids[i]
                      << ", " << metal_dist[i] << ")\n";
            return 1;
        }
    }
    return 0;
}
