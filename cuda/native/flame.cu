#include "flame.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kLogitRows = 128;
constexpr size_t kBlasWorkspace = 4 * 1024 * 1024;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(static_cast<int>(status)));
}

size_t multiply(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b,
            "FLAME allocation or shape overflow");
    return a * b;
}

size_t add(size_t a, size_t b) {
    require(a <= std::numeric_limits<size_t>::max() - b, "FLAME offset overflow");
    return a + b;
}

int blocks(size_t elements) {
    return static_cast<int>(std::min<size_t>(1 + (elements - 1) / kThreads, 65535));
}

int row_blocks(size_t rows) {
    return static_cast<int>(std::min<size_t>(rows, 65535));
}

__device__ float bf16(uint16_t value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

// All callers use exactly kThreads threads and participate in each reduction.
template <bool Maximum>
__device__ float reduce(float value, float* scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2) {
        if (lane < stride) {
            if (Maximum) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
            else scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__global__ void decode(const uint16_t* input, float* output, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        output[i] = bf16(input[i]);
}

__global__ void embedding(const uint16_t* weights, const int32_t* tokens,
                          float* output, size_t count, int width) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        output[i] = bf16(weights[static_cast<size_t>(tokens[i / width]) * width + i % width]);
}

__global__ void rms_norm(const float* x, const uint16_t* weight, float* out,
                         int rows, int width, float epsilon) {
    __shared__ float scratch[kThreads];
    for (size_t row = blockIdx.x; row < static_cast<size_t>(rows); row += gridDim.x) {
        const size_t base = row * width;
        float sum = 0.0f;
        for (size_t i = threadIdx.x; i < static_cast<size_t>(width); i += blockDim.x) {
            const float value = x[base + i];
            sum += value * value;
        }
        const float scale = rsqrtf(reduce<false>(sum, scratch) / width + epsilon);
        for (size_t i = threadIdx.x; i < static_cast<size_t>(width); i += blockDim.x)
            out[base + i] = (x[base + i] * scale) * bf16(weight[i]);
    }
}

__global__ void pack_rotary(const float* input, float* q, float* k, float* v,
                            int rows, int width, int head_size, float rope_base) {
    const size_t count = static_cast<size_t>(rows) * width;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int position = static_cast<int>(i / width);
        const int column = static_cast<int>(i % width);
        const int head = column / head_size, d = column % head_size;
        const int half = head_size / 2;
        const int partner = d < half ? d + half : d - half;
        const float sign = d < half ? -1.0f : 1.0f;
        const float inverse = powf(rope_base, -static_cast<float>(2 * (d % half)) / head_size);
        const float angle = position * inverse;
        const float cosine = cosf(angle), sine = sinf(angle);
        const size_t src = static_cast<size_t>(position) * 3 * width + head * 3 * head_size;
        const size_t dst = (static_cast<size_t>(head) * rows + position) * head_size + d;
        q[dst] = input[src + d] * cosine + (sign * input[src + partner]) * sine;
        k[dst] = input[src + head_size + d] * cosine +
                 (sign * input[src + head_size + partner]) * sine;
        v[dst] = input[src + 2 * head_size + d];
    }
}

__global__ void causal_softmax(float* scores, int rows, int heads, float scale) {
    __shared__ float scratch[kThreads];
    const size_t total = static_cast<size_t>(rows) * heads;
    for (size_t row = blockIdx.x; row < total; row += gridDim.x) {
        const int position = static_cast<int>(row % rows);
        float* values = scores + row * rows;
        float maximum = -CUDART_INF_F;
        for (size_t j = threadIdx.x; j <= static_cast<size_t>(position); j += blockDim.x)
            maximum = fmaxf(maximum, values[j] * scale);
        maximum = reduce<true>(maximum, scratch);
        float sum = 0.0f;
        for (size_t j = threadIdx.x; j < static_cast<size_t>(rows); j += blockDim.x) {
            const float value = j <= static_cast<size_t>(position)
                ? expf(values[j] * scale - maximum) : 0.0f;
            values[j] = value;
            sum += value;
        }
        sum = reduce<false>(sum, scratch);
        for (size_t j = threadIdx.x; j < static_cast<size_t>(rows); j += blockDim.x)
            values[j] /= sum;
        __syncthreads();
    }
}

__global__ void unpack_heads(const float* packed, float* out, int rows,
                             int width, int head_size) {
    const size_t count = static_cast<size_t>(rows) * width;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int column = static_cast<int>(i % width);
        out[i] = packed[(static_cast<size_t>(column / head_size) * rows + i / width)
                        * head_size + column % head_size];
    }
}

__global__ void residual(float* x, const float* update, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        x[i] += update[i];
}

__global__ void silu_gate(const float* input, float* out, size_t count, int hidden) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t src = (i / hidden) * 2 * hidden + i % hidden;
        const float gate = input[src];
        out[i] = (gate * (1.0f / (1.0f + expf(-gate)))) * input[src + hidden];
    }
}

__global__ void router_topk(float* logits, float* probabilities, int* indices,
                            int* counts, int rows, int experts, int top_k) {
    __shared__ float scratch[kThreads];
    for (size_t row = blockIdx.x; row < static_cast<size_t>(rows); row += gridDim.x) {
        float* values = logits + row * experts;
        float maximum = -CUDART_INF_F;
        for (size_t e = threadIdx.x; e < static_cast<size_t>(experts); e += blockDim.x)
            maximum = fmaxf(maximum, values[e]);
        maximum = reduce<true>(maximum, scratch);
        float sum = 0.0f;
        for (size_t e = threadIdx.x; e < static_cast<size_t>(experts); e += blockDim.x) {
            values[e] = expf(values[e] - maximum);
            sum += values[e];
        }
        sum = reduce<false>(sum, scratch);
        for (size_t e = threadIdx.x; e < static_cast<size_t>(experts); e += blockDim.x)
            values[e] /= sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int slot = 0; slot < top_k; ++slot) {
                float best = -1.0f;
                int expert = 0;
                // Compare rounded softmax probabilities, as lax.top_k does.
                for (int e = 0; e < experts; ++e) {
                    if (values[e] > best || isnan(values[e])) {
                        best = values[e];
                        expert = e;
                    }
                }
                const size_t dst = row * top_k + slot;
                probabilities[dst] = best;
                indices[dst] = expert;
                atomicAdd(counts + expert, 1);
                values[expert] = -1.0f;
            }
        }
        __syncthreads();
    }
}

__global__ void group_routes(const int* indices, const int* offsets, int* cursors,
                             int* slots, int count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < static_cast<size_t>(count); i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int expert = indices[i];
        slots[offsets[expert] + atomicAdd(cursors + expert, 1)] = static_cast<int>(i);
    }
}

__global__ void gather_expert(const float* x, const int* slots, float* gathered,
                              size_t count, int width, int top_k) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        gathered[i] = x[static_cast<size_t>(slots[i / width] / top_k) * width + i % width];
}

__global__ void scatter_expert(const float* output, const int* slots, float* routed,
                               size_t count, int width) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        routed[static_cast<size_t>(slots[i / width]) * width + i % width] = output[i];
}

__global__ void combine(float* x, const float* shared, const float* routed,
                        const float* probabilities, size_t count, int width, int top_k) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t route = (i / width) * top_k;
        float sum = 0.0f;
        for (int slot = 0; slot < top_k; ++slot)
            sum += routed[(route + slot) * width + i % width] * probabilities[route + slot];
        x[i] += shared[i] + sum;
    }
}

// Also checks unscored rows, so a masked nonfinite output cannot pass silently.
__global__ void cross_entropy(const float* logits, const int32_t* tokens,
                              const uint8_t* masks, float* losses, int* invalid,
                              int rows, int vocab, int start, int sequence) {
    __shared__ float scratch[kThreads];
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* values = logits + static_cast<size_t>(row) * vocab;
        float maximum = -CUDART_INF_F;
        for (size_t j = threadIdx.x; j < static_cast<size_t>(vocab); j += blockDim.x) {
            const float value = values[j];
            if (!isfinite(value)) atomicExch(invalid, 1);
            maximum = fmaxf(maximum, value);
        }
        maximum = reduce<true>(maximum, scratch);
        if (losses) {
            float sum = 0.0f;
            for (size_t j = threadIdx.x; j < static_cast<size_t>(vocab); j += blockDim.x)
                sum += expf(values[j] - maximum);
            sum = reduce<false>(sum, scratch);
            if (threadIdx.x == 0) {
                const int target = start + row + 1;
                float loss = 0.0f;
                if (target < sequence && masks[target]) {
                    loss = logf(sum) + (maximum - values[tokens[target]]);
                    if (!isfinite(loss)) atomicExch(invalid, 1);
                }
                losses[start + row] = loss;
            }
        }
        __syncthreads();
    }
}

__global__ void mean_loss(float* losses, int rows, int scored, int* invalid) {
    __shared__ float scratch[kThreads];
    float sum = 0.0f;
    for (size_t row = threadIdx.x; row < static_cast<size_t>(rows); row += blockDim.x)
        sum += losses[row];
    sum = reduce<false>(sum, scratch);
    if (threadIdx.x == 0) {
        losses[0] = sum / scored;
        if (!isfinite(losses[0])) atomicExch(invalid, 1);
    }
}

struct Layer {
    size_t attention_norm = 0, qkv = 0, projection = 0, mlp_norm = 0;
    size_t first = 0, second = 0, router = 0, expert_first = 0, expert_second = 0;
};

struct Engine {
    EnnxFlameConfig c{};
    int capacity = 0, logit_rows = 0;
    size_t weight_count = 0, matrix_count = 0;
    uint64_t workspace_bytes = 0;
    std::vector<Layer> layers;
    std::vector<void*> allocations;
    std::vector<int> counts, offsets;
    size_t embedding_offset = 0, final_norm_offset = 0, output_offset = 0;
    cublasHandle_t blas = nullptr;
    float *matrix = nullptr, *x = nullptr, *norm = nullptr, *qkv = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *scores = nullptr;
    float *attended = nullptr, *update = nullptr, *gates = nullptr, *activation = nullptr;
    float *route_logits = nullptr, *probabilities = nullptr, *routed = nullptr;
    float *gathered = nullptr, *expert_output = nullptr, *logits = nullptr, *losses = nullptr;
    int32_t* tokens = nullptr;
    uint8_t* masks = nullptr;
    int *indices = nullptr, *device_counts = nullptr, *device_offsets = nullptr;
    int *slots = nullptr, *invalid = nullptr;

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    ~Engine() noexcept {
        (void)cudaSetDevice(0);
        (void)cudaDeviceSynchronize();
        if (blas) (void)cublasDestroy(blas);
        for (void* pointer : allocations) if (pointer) (void)cudaFree(pointer);
    }

    template <typename T>
    T* allocate(size_t count) {
        const size_t bytes = multiply(count, sizeof(T));
        const size_t total = add(static_cast<size_t>(workspace_bytes), bytes);
        allocations.push_back(nullptr);
        cuda_check(cudaMalloc(&allocations.back(), bytes));
        workspace_bytes = total;
        return static_cast<T*>(allocations.back());
    }

    void initialize(const EnnxFlameConfig& config, uint32_t max_tokens);
    void linear(const float* input, const uint16_t* weight, float* output,
                int rows, int in, int out);
    void normalized(const uint16_t* weights, size_t offset, int rows);
    void mlp(const float* input, const uint16_t* first, const uint16_t* second,
             float* output, int rows, int hidden);
    void forward(const uint16_t* weights, int rows);
    void output(const uint16_t* weights, int rows, int scored, float* host_logits,
                float* host_loss);
    void validate_weights(const uint16_t* weights, size_t length) const;
    int validate_sequence(const int32_t* input, const uint8_t* mask,
                           uint32_t rows, bool loss) const;
};

void Engine::initialize(const EnnxFlameConfig& config, uint32_t max_tokens) {
    c = config;
    const uint32_t dimensions[] = {c.layers, c.width, c.heads, c.vocab, c.dense_width,
        c.expert_width, c.shared_width, c.experts, c.top_k, c.context};
    for (uint32_t value : dimensions)
        require(value > 0 && value <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
                "FLAME dimensions must be positive and fit signed 32-bit integers");
    require(c.width % c.heads == 0 && (c.width / c.heads) % 2 == 0,
            "FLAME heads must divide width and have even dimension");
    require(c.top_k <= c.experts, "FLAME top_k exceeds experts");
    require(std::isfinite(c.epsilon) && c.epsilon > 0.0f && c.epsilon < 1.0f,
            "FLAME epsilon must be finite and between zero and one");
    require(std::isfinite(c.rope_base) && c.rope_base > 1.0f,
            "FLAME rope_base must be finite and greater than one");
    require(max_tokens > 0 && max_tokens <= c.context,
            "FLAME max_tokens must be positive and within context");
    const size_t int_max = std::numeric_limits<int>::max();
    require(c.width <= int_max / 3 && c.dense_width <= int_max / 2 &&
            c.expert_width <= int_max / 2 && c.shared_width <= int_max / 2,
            "FLAME projected dimensions exceed cuBLAS limits");
    require(multiply(max_tokens, c.top_k) <= int_max,
            "FLAME token routing capacity exceeds signed 32-bit limits");
    capacity = static_cast<int>(max_tokens);
    logit_rows = std::min(capacity, kLogitRows);
    const size_t expert_count = c.experts;
    const size_t expert_offsets = add(expert_count, 1);
    require(expert_count <= counts.max_size() && expert_offsets <= offsets.max_size(),
            "FLAME expert count/offset allocation exceeds host vector limits");
    (void)multiply(expert_offsets, sizeof(int));
    layers.resize(c.layers);
    counts.resize(expert_count);
    offsets.resize(expert_offsets);

    // Sort complete tensor names (including decimal layer numbers), exactly as
    // Layout.from_params does. Expert tensors remain single contiguous blocks.
    std::map<std::string, std::pair<size_t, size_t*>> tensors;
    auto tensor = [&](const std::string& name, size_t length, size_t& offset) {
        tensors.emplace(name, std::make_pair(length, &offset));
    };
    const size_t h = c.width;
    tensor("embedding.word_embeddings.weight", multiply(c.vocab, h), embedding_offset);
    tensor("output_layer.weight", multiply(c.vocab, h), output_offset);
    tensor("decoder.final_layernorm.weight", h, final_norm_offset);
    for (uint32_t i = 0; i < c.layers; ++i) {
        Layer& layer = layers[i];
        const std::string p = "decoder.layers." + std::to_string(i) + ".";
        tensor(p + "self_attention.linear_qkv.weight", multiply(3 * h, h), layer.qkv);
        tensor(p + "self_attention.linear_qkv.layer_norm_weight", h, layer.attention_norm);
        tensor(p + "self_attention.linear_proj.weight", multiply(h, h), layer.projection);
        if (i == 0) {
            tensor(p + "mlp.linear_fc1.layer_norm_weight", h, layer.mlp_norm);
            tensor(p + "mlp.linear_fc1.weight", multiply(2 * c.dense_width, h), layer.first);
            tensor(p + "mlp.linear_fc2.weight", multiply(h, c.dense_width), layer.second);
        } else {
            tensor(p + "pre_mlp_layernorm.weight", h, layer.mlp_norm);
            tensor(p + "mlp.router.weight", multiply(c.experts, h), layer.router);
            tensor(p + "mlp.shared_experts.linear_fc1.weight",
                   multiply(2 * c.shared_width, h), layer.first);
            tensor(p + "mlp.shared_experts.linear_fc2.weight",
                   multiply(h, c.shared_width), layer.second);
            tensor(p + "mlp.experts.experts.linear_fc1.weight",
                   multiply(c.experts, multiply(2 * c.expert_width, h)), layer.expert_first);
            tensor(p + "mlp.experts.experts.linear_fc2.weight",
                   multiply(c.experts, multiply(h, c.expert_width)), layer.expert_second);
        }
    }
    for (const auto& item : tensors) {
        *item.second.second = weight_count;
        weight_count = add(weight_count, item.second.first);
    }
    (void)multiply(weight_count, sizeof(uint16_t));
    const size_t hidden = std::max({c.dense_width, c.expert_width, c.shared_width});
    matrix_count = std::max({multiply(c.vocab, h), multiply(3 * h, h),
                             multiply(2 * hidden, h), multiply(c.experts, h)});
    const size_t nh = multiply(max_tokens, h);
    const size_t routes = multiply(max_tokens, c.top_k);
    const size_t nhidden = multiply(max_tokens, hidden);
    allocations.reserve(32);
    matrix = allocate<float>(matrix_count);
    x = allocate<float>(nh);
    norm = allocate<float>(nh);
    qkv = allocate<float>(multiply(nh, 3));
    q = allocate<float>(nh);
    k = allocate<float>(nh);
    v = allocate<float>(nh);
    scores = allocate<float>(multiply(multiply(max_tokens, max_tokens), c.heads));
    attended = allocate<float>(nh);
    update = allocate<float>(nh);
    gates = allocate<float>(multiply(nhidden, 2));
    activation = allocate<float>(nhidden);
    route_logits = allocate<float>(multiply(max_tokens, c.experts));
    probabilities = allocate<float>(routes);
    routed = allocate<float>(multiply(routes, h));
    gathered = allocate<float>(nh);
    expert_output = allocate<float>(nh);
    logits = allocate<float>(multiply(logit_rows, c.vocab));
    losses = allocate<float>(max_tokens);
    tokens = allocate<int32_t>(max_tokens);
    masks = allocate<uint8_t>(max_tokens);
    indices = allocate<int>(routes);
    device_counts = allocate<int>(c.experts);
    device_offsets = allocate<int>(static_cast<size_t>(c.experts) + 1);
    slots = allocate<int>(routes);
    invalid = allocate<int>(1);
    void* blas_workspace = allocate<uint8_t>(kBlasWorkspace);
    blas_check(cublasCreate(&blas));
    blas_check(cublasSetStream(blas, cudaStreamLegacy));
    blas_check(cublasSetWorkspace(blas, blas_workspace, kBlasWorkspace));
    blas_check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
    blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    blas_check(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
}

void Engine::linear(const float* input, const uint16_t* weight, float* output,
                    int rows, int in, int out) {
    const size_t count = static_cast<size_t>(in) * out;
    require(count <= matrix_count, "FLAME matrix scratch capacity exceeded");
    decode<<<blocks(count), kThreads, 0, cudaStreamLegacy>>>(weight, matrix, count);
    cuda_check(cudaGetLastError());
    const float one = 1.0f, zero = 0.0f;
    // Row-major Y[N,O] = X[N,I] * W[O,I]^T is column-major Y^T = W * X^T.
    blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in,
                          &one, matrix, in, input, in, &zero, output, out));
}

void Engine::normalized(const uint16_t* weights, size_t offset, int rows) {
    rms_norm<<<row_blocks(rows), kThreads, 0, cudaStreamLegacy>>>(
        x, weights + offset, norm, rows, c.width, c.epsilon);
    cuda_check(cudaGetLastError());
}

void Engine::mlp(const float* input, const uint16_t* first, const uint16_t* second,
                 float* output, int rows, int hidden) {
    linear(input, first, gates, rows, c.width, 2 * hidden);
    const size_t count = static_cast<size_t>(rows) * hidden;
    silu_gate<<<blocks(count), kThreads, 0, cudaStreamLegacy>>>(gates, activation, count, hidden);
    cuda_check(cudaGetLastError());
    linear(activation, second, output, rows, hidden, c.width);
}

void Engine::forward(const uint16_t* weights, int rows) {
    const int h = static_cast<int>(c.width), d = h / c.heads;
    const size_t nh = static_cast<size_t>(rows) * h;
    embedding<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(
        weights + embedding_offset, tokens, x, nh, h);
    cuda_check(cudaGetLastError());
    const float one = 1.0f, zero = 0.0f;
    const long long head_stride = static_cast<long long>(rows) * d;
    const long long score_stride = static_cast<long long>(rows) * rows;
    for (uint32_t i = 0; i < c.layers; ++i) {
        const Layer& layer = layers[i];
        normalized(weights, layer.attention_norm, rows);
        linear(norm, weights + layer.qkv, qkv, rows, h, 3 * h);
        pack_rotary<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(
            qkv, q, k, v, rows, h, d, c.rope_base);
        cuda_check(cudaGetLastError());
        // Per-head row-major Q K^T; the custom softmax applies scaling/causality.
        blas_check(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, rows, d, &one, k, d, head_stride, q, d, head_stride,
            &zero, scores, rows, score_stride, c.heads));
        causal_softmax<<<row_blocks(static_cast<size_t>(rows) * c.heads),
                           kThreads, 0, cudaStreamLegacy>>>(
            scores, rows, c.heads, 1.0f / std::sqrt(static_cast<float>(d)));
        cuda_check(cudaGetLastError());
        // Reuse the expired interleaved QKV buffer for packed attention output.
        blas_check(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            d, rows, rows, &one, v, d, head_stride, scores, rows, score_stride,
            &zero, qkv, d, head_stride, c.heads));
        unpack_heads<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(qkv, attended, rows, h, d);
        cuda_check(cudaGetLastError());
        linear(attended, weights + layer.projection, update, rows, h, h);
        residual<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(x, update, nh);
        cuda_check(cudaGetLastError());
        normalized(weights, layer.mlp_norm, rows);
        if (i == 0) {
            mlp(norm, weights + layer.first, weights + layer.second, update, rows, c.dense_width);
            residual<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(x, update, nh);
            cuda_check(cudaGetLastError());
            continue;
        }
        linear(norm, weights + layer.router, route_logits, rows, h, c.experts);
        cuda_check(cudaMemsetAsync(device_counts, 0, c.experts * sizeof(int), cudaStreamLegacy));
        router_topk<<<row_blocks(rows), kThreads, 0, cudaStreamLegacy>>>(
            route_logits, probabilities, indices, device_counts, rows, c.experts, c.top_k);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(counts.data(), device_counts, c.experts * sizeof(int),
                                   cudaMemcpyDeviceToHost, cudaStreamLegacy));
        cuda_check(cudaStreamSynchronize(cudaStreamLegacy));
        offsets[0] = 0;
        for (uint32_t e = 0; e < c.experts; ++e) {
            require(counts[e] >= 0 && counts[e] <= rows, "FLAME invalid expert route count");
            require(offsets[e] <= rows * static_cast<int>(c.top_k) - counts[e],
                    "FLAME expert route count overflow");
            offsets[e + 1] = offsets[e] + counts[e];
        }
        require(offsets[c.experts] == rows * static_cast<int>(c.top_k),
                "FLAME expert dispatch lost token assignments");
        cuda_check(cudaMemcpyAsync(device_offsets, offsets.data(), offsets.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, cudaStreamLegacy));
        cuda_check(cudaMemsetAsync(device_counts, 0, c.experts * sizeof(int), cudaStreamLegacy));
        const int assignments = rows * static_cast<int>(c.top_k);
        group_routes<<<blocks(assignments), kThreads, 0, cudaStreamLegacy>>>(
            indices, device_offsets, device_counts, slots, assignments);
        cuda_check(cudaGetLastError());
        mlp(norm, weights + layer.first, weights + layer.second, update, rows, c.shared_width);
        const size_t first_stride = static_cast<size_t>(2) * c.expert_width * h;
        const size_t second_stride = static_cast<size_t>(h) * c.expert_width;
        for (uint32_t e = 0; e < c.experts; ++e) {
            const int count = counts[e];
            if (!count) continue;
            const int* expert_slots = slots + offsets[e];
            const size_t elements = static_cast<size_t>(count) * h;
            gather_expert<<<blocks(elements), kThreads, 0, cudaStreamLegacy>>>(
                norm, expert_slots, gathered, elements, h, c.top_k);
            cuda_check(cudaGetLastError());
            mlp(gathered, weights + layer.expert_first + e * first_stride,
                weights + layer.expert_second + e * second_stride,
                expert_output, count, c.expert_width);
            scatter_expert<<<blocks(elements), kThreads, 0, cudaStreamLegacy>>>(
                expert_output, expert_slots, routed, elements, h);
            cuda_check(cudaGetLastError());
        }
        combine<<<blocks(nh), kThreads, 0, cudaStreamLegacy>>>(
            x, update, routed, probabilities, nh, h, c.top_k);
        cuda_check(cudaGetLastError());
    }
    normalized(weights, final_norm_offset, rows);
}

void Engine::output(const uint16_t* weights, int rows, int scored, float* host_logits,
                    float* host_loss) {
    const size_t count = static_cast<size_t>(c.vocab) * c.width;
    decode<<<blocks(count), kThreads, 0, cudaStreamLegacy>>>(weights + output_offset, matrix, count);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemsetAsync(invalid, 0, sizeof(int), cudaStreamLegacy));
    const float one = 1.0f, zero = 0.0f;
    for (int start = 0; start < rows;) {
        const int chunk = std::min(logit_rows, rows - start);
        blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, c.vocab, chunk, c.width,
            &one, matrix, c.width, norm + static_cast<size_t>(start) * c.width, c.width,
            &zero, logits, c.vocab));
        cross_entropy<<<row_blocks(chunk), kThreads, 0, cudaStreamLegacy>>>(
            logits, tokens, masks, host_loss ? losses : nullptr, invalid,
            chunk, c.vocab, start, rows);
        cuda_check(cudaGetLastError());
        if (host_logits)
            cuda_check(cudaMemcpyAsync(host_logits + static_cast<size_t>(start) * c.vocab,
                logits, static_cast<size_t>(chunk) * c.vocab * sizeof(float),
                cudaMemcpyDeviceToHost, cudaStreamLegacy));
        start += chunk;
    }
    if (host_loss) {
        mean_loss<<<1, kThreads, 0, cudaStreamLegacy>>>(losses, rows, scored, invalid);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(host_loss, losses, sizeof(float),
                                   cudaMemcpyDeviceToHost, cudaStreamLegacy));
    }
    int bad = 0;
    // Synchronize before the stack-backed status or caller-owned output expires.
    const cudaError_t copied = cudaMemcpyAsync(&bad, invalid, sizeof(int),
                                               cudaMemcpyDeviceToHost, cudaStreamLegacy);
    const cudaError_t synced = cudaStreamSynchronize(cudaStreamLegacy);
    cuda_check(copied);
    cuda_check(synced);
    require(!bad, "FLAME produced nonfinite logits or loss");
}

void Engine::validate_weights(const uint16_t* weights, size_t length) const {
    require(weights != nullptr, "FLAME weights pointer is null");
    require(length == weight_count, "FLAME weights_len does not match canonical layout");
    require(reinterpret_cast<uintptr_t>(weights) % alignof(uint16_t) == 0,
            "FLAME weights pointer is misaligned");
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, weights));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == 0,
            "FLAME weights must be a device-0 CUDA allocation");
}

int Engine::validate_sequence(const int32_t* input, const uint8_t* mask,
                               uint32_t rows, bool loss) const {
    require(input != nullptr, "FLAME tokens pointer is null");
    require(rows >= (loss ? 2u : 1u) && rows <= static_cast<uint32_t>(capacity) && rows <= c.context,
            "FLAME sequence length is invalid or exceeds max_tokens/context");
    require(!loss || mask != nullptr, "FLAME masks pointer is null");
    int scored = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        require(input[i] >= 0 && static_cast<uint32_t>(input[i]) < c.vocab,
                "FLAME token ID is outside the vocabulary");
        if (loss && i > 0 && mask[i] != 0) ++scored;
    }
    require(!loss || scored > 0, "FLAME loss requires a nonzero shifted target mask");
    return scored;
}

void error_text(char* error, size_t capacity, const char* message) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", message);
}

void synchronize_failure() noexcept {
    (void)cudaSetDevice(0);
    (void)cudaDeviceSynchronize();
}

template <typename Function>
int boundary(char* error, size_t error_capacity, Function&& function) noexcept {
    error_text(error, error_capacity, "");
    try {
        cuda_check(cudaSetDevice(0));
        function();
        cuda_check(cudaStreamSynchronize(cudaStreamLegacy));
        return 0;
    } catch (const std::exception& exception) {
        synchronize_failure();
        error_text(error, error_capacity, exception.what());
    } catch (...) {
        synchronize_failure();
        error_text(error, error_capacity, "Unknown native FLAME failure");
    }
    return 1;
}

}  // namespace

extern "C" void* ennx_flame_create(const EnnxFlameConfig* config, uint32_t max_tokens,
                                    char* error, size_t error_capacity) {
    std::unique_ptr<Engine> engine;
    const int status = boundary(error, error_capacity, [&] {
        require(config != nullptr, "FLAME config pointer is null");
        engine = std::make_unique<Engine>();
        engine->initialize(*config, max_tokens);
    });
    return status == 0 ? engine.release() : nullptr;
}

extern "C" void ennx_flame_destroy(void* engine) {
    try { delete static_cast<Engine*>(engine); }
    catch (...) { synchronize_failure(); }
}

extern "C" uint64_t ennx_flame_workspace(const void* engine) {
    try { return engine ? static_cast<const Engine*>(engine)->workspace_bytes : 0; }
    catch (...) { return 0; }
}

extern "C" uint64_t ennx_flame_weights_len(const void* engine) {
    try { return engine ? static_cast<const Engine*>(engine)->weight_count : 0; }
    catch (...) { return 0; }
}

extern "C" int ennx_flame_logits(void* opaque, const uint16_t* weights, size_t weights_len,
                                  const int32_t* host_tokens, uint32_t length,
                                  float* host_logits, char* error, size_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        require(opaque != nullptr, "FLAME engine pointer is null");
        require(host_logits != nullptr, "FLAME output logits pointer is null");
        Engine& engine = *static_cast<Engine*>(opaque);
        engine.validate_weights(weights, weights_len);
        engine.validate_sequence(host_tokens, nullptr, length, false);
        (void)multiply(multiply(length, engine.c.vocab), sizeof(float));
        cuda_check(cudaMemcpyAsync(engine.tokens, host_tokens, static_cast<size_t>(length) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, cudaStreamLegacy));
        engine.forward(weights, length);
        engine.output(weights, length, 0, host_logits, nullptr);
    });
}

extern "C" int ennx_flame_losses(void* opaque, const uint16_t* weights, size_t weights_len,
                                  const int32_t* host_tokens, const uint8_t* host_masks,
                                  const uint32_t* host_lengths, size_t batch,
                                  float* host_losses, char* error, size_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        require(opaque != nullptr, "FLAME engine pointer is null");
        require(host_tokens && host_masks && host_lengths && host_losses,
                "FLAME loss inputs or output pointer is null");
        require(batch > 0, "FLAME batch must be nonempty");
        (void)multiply(batch, sizeof(float));
        Engine& engine = *static_cast<Engine*>(opaque);
        engine.validate_weights(weights, weights_len);
        size_t offset = 0;
        // Validate the entire batch before submitting work or changing outputs.
        for (size_t problem = 0; problem < batch; ++problem) {
            const size_t end = add(offset, host_lengths[problem]);
            (void)multiply(end, sizeof(int32_t));
            engine.validate_sequence(host_tokens + offset, host_masks + offset,
                                     host_lengths[problem], true);
            offset = end;
        }
        offset = 0;
        for (size_t problem = 0; problem < batch; ++problem) {
            const uint32_t rows = host_lengths[problem];
            const int scored = engine.validate_sequence(host_tokens + offset, host_masks + offset, rows, true);
            cuda_check(cudaMemcpyAsync(engine.tokens, host_tokens + offset, rows * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, cudaStreamLegacy));
            cuda_check(cudaMemcpyAsync(engine.masks, host_masks + offset, rows * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, cudaStreamLegacy));
            engine.forward(weights, rows);
            engine.output(weights, rows, scored, nullptr, host_losses + problem);
            offset += rows;
        }
    });
}
