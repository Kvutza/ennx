#include <faiss/IndexFlat.h>
#include <faiss/gpu_metal/MetalIndexFlat.h>
#include <faiss/gpu_metal/StandardMetalResources.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Point {
    std::string axis;
    int rows;
    int queries;
    int dims;
    int k;
};

constexpr size_t max_temp = 256ULL * 1024ULL * 1024ULL;

Point parse(const std::string& spec) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t end = spec.find(':', start);
        fields.push_back(spec.substr(start, end - start));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    if (fields.size() != 5) {
        throw std::invalid_argument("point must be AXIS:ROWS:QUERIES:DIMS:K");
    }
    Point point{
            fields[0],
            std::stoi(fields[1]),
            std::stoi(fields[2]),
            std::stoi(fields[3]),
            std::stoi(fields[4]),
    };
    if (point.axis.empty() || point.rows < 1 || point.queries < 1 ||
        point.dims < 1 || point.k < 1 || point.k > point.rows ||
        point.k > 2048) {
        throw std::invalid_argument("invalid point " + spec);
    }
    const size_t temp = static_cast<size_t>(point.rows) * point.queries *
            sizeof(float);
    if (temp > max_temp) {
        throw std::invalid_argument("distance matrix exceeds 256 MiB: " + spec);
    }
    return point;
}

float next(uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
}

void fill(std::vector<float>& values, uint32_t seed) {
    for (float& value : values) {
        value = next(seed);
    }
}

double median(int rounds, const std::function<void()>& search) {
    search();
    std::vector<double> samples;
    samples.reserve(rounds);
    for (int round = 0; round < rounds; ++round) {
        const auto start = std::chrono::steady_clock::now();
        search();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        samples.push_back(
                std::chrono::duration<double, std::milli>(elapsed).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

double recall(
        const std::vector<faiss::idx_t>& expected,
        const std::vector<faiss::idx_t>& actual,
        int queries,
        int k) {
    size_t matches = 0;
    std::vector<faiss::idx_t> left(k);
    std::vector<faiss::idx_t> right(k);
    for (int query = 0; query < queries; ++query) {
        const size_t start = static_cast<size_t>(query) * k;
        std::copy_n(expected.begin() + start, k, left.begin());
        std::copy_n(actual.begin() + start, k, right.begin());
        std::sort(left.begin(), left.end());
        std::sort(right.begin(), right.end());
        size_t i = 0;
        size_t j = 0;
        while (i < left.size() && j < right.size()) {
            if (left[i] == right[j]) {
                ++matches;
                ++i;
                ++j;
            } else if (left[i] < right[j]) {
                ++i;
            } else {
                ++j;
            }
        }
    }
    return static_cast<double>(matches) / expected.size();
}

void bench(
        std::ostream& out,
        faiss::gpu_metal::StandardMetalResources& resources,
        const Point& point,
        int rounds) {
    std::vector<float> data(static_cast<size_t>(point.rows) * point.dims);
    std::vector<float> input(static_cast<size_t>(point.queries) * point.dims);
    fill(data, static_cast<uint32_t>(point.rows));
    fill(input, static_cast<uint32_t>(point.queries));

    faiss::IndexFlatL2 cpu(point.dims);
    faiss::gpu_metal::MetalIndexFlat metal(
            resources.getResources(), point.dims, faiss::METRIC_L2, 0.0F);
    cpu.add(point.rows, data.data());
    metal.add(point.rows, data.data());

    const size_t count = static_cast<size_t>(point.queries) * point.k;
    std::vector<float> cpu_dist(count);
    std::vector<float> metal_dist(count);
    std::vector<faiss::idx_t> cpu_ids(count);
    std::vector<faiss::idx_t> metal_ids(count);
    const auto cpu_search = [&] {
        cpu.search(
                point.queries,
                input.data(),
                point.k,
                cpu_dist.data(),
                cpu_ids.data());
    };
    const auto metal_search = [&] {
        metal.search(
                point.queries,
                input.data(),
                point.k,
                metal_dist.data(),
                metal_ids.data());
    };
    const double cpu_ms = median(rounds, cpu_search);
    const double metal_ms = median(rounds, metal_search);

    size_t matches = 0;
    float max_error = 0.0F;
    for (size_t i = 0; i < count; ++i) {
        matches += cpu_ids[i] == metal_ids[i];
        max_error = std::max(
                max_error, std::abs(cpu_dist[i] - metal_dist[i]));
    }
    const double rank_match = static_cast<double>(matches) / count;
    const double set_recall = recall(
            cpu_ids, metal_ids, point.queries, point.k);
    const double distance_mib = static_cast<double>(point.rows) * point.queries *
            sizeof(float) / (1024.0 * 1024.0);
    out << point.axis << ",cpu," << point.rows << ',' << point.queries << ','
        << point.dims << ',' << point.k << ',' << cpu_ms << ",1,1,0,0\n";
    out << point.axis << ",metal," << point.rows << ',' << point.queries << ','
        << point.dims << ',' << point.k << ',' << metal_ms << ',' << rank_match
        << ',' << set_recall << ',' << max_error << ',' << distance_mib << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: faiss-metal-bench OUT ROUNDS POINT...\n";
        return 2;
    }
    const int rounds = std::atoi(argv[2]);
    if (rounds < 1) {
        std::cerr << "rounds must be positive\n";
        return 2;
    }
    std::ofstream out(argv[1]);
    if (!out) {
        std::cerr << "cannot open output " << argv[1] << '\n';
        return 2;
    }
    faiss::gpu_metal::StandardMetalResources resources;
    if (!resources.isAvailable()) {
        std::cerr << "FAISS Metal device unavailable\n";
        return 1;
    }
    out << "axis,backend,rows,queries,dims,k,search_ms,rank_match,recall_at_k,max_abs_error,distance_mib\n";
    try {
        for (int i = 3; i < argc; ++i) {
            bench(out, resources, parse(argv[i]), rounds);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return 0;
}
