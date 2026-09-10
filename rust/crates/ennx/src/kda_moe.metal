#include <metal_stdlib>

using namespace metal;

// ============================================================================
// ENNX Hardware-Resident Fused Program & Kernel Suite (Apple Metal)
// ============================================================================
// References & Citations:
// 1. Epistemic Neural Networks (ENN):
//    - Osband et al. (2021), "Epistemic Neural Networks", NeurIPS 2021.
//    - Osband et al. (2023), "The Epistemic Neural Network Architectural Zoo".
//    - ENN stochastic perturbed-weight matrix multiply: W_perturbed = W_quantized + Δ(seed).
// 2. Kernel Density / Recurrent Linear Attention (KDA):
//    - Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
//    - De et al. (2024), "Griffin: Mixing Gated Recurrent Neural Networks for Language Modeling".
//    - Recurrent state update: S_t = S_{t-1} * exp(gate) + k_t * (v_t - k_t^T S_{t-1}) * beta_t.
// 3. Sparsely Gated Mixture-of-Experts (MoE):
//    - Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer".
//    - Fused router top-k Softmax selection & SwiGLU expert gating.
// 4. Hardware-Resident Bayesian Optimization Objective Reduction:
//    - Composite Objective O_ENNX(x) = Task_Loss + λ_epistemic * Uncertainty - λ_latency * ComputeCost.
// ============================================================================

// Resident KDA state ABI.
//
// q, k, and v are laid out as [batch, length, heads, width].
// gate is [batch, length, heads, key_width].
// beta is [batch, length, heads].
// output is [batch, length, heads, value_width].
// state is [batch, heads, key_width, value_width].
//
// One thread owns one value column of one (batch, head) state. This keeps the
// state resident while parallelizing the value dimension. Each thread writes
// a disjoint column, so the recurrent token loop needs no global barrier.
struct KDAParams {
    uint batch;
    uint length;
    uint heads;
    uint key_width;
    uint value_width;
};

struct PackedLinear {
    uint byte_offset;
    uint scale_offset;
    uint bias_offset;
    uint input_width;
    uint output_width;
    uint bits;
    uint group_size;
    uint element_offset;
    uint perturb_whole;
    uint perturb_threshold;
};

struct PerturbSeed {
    uint low;
    uint high;
    uint enabled;
    uint padding;
};

inline uint perturb_hash(PerturbSeed seed, uint element) {
    uint value = seed.low ^ (element * 0x9E3779B9u);
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 15u;
    return value ^ seed.high;
}

inline uint perturb_code(
    uint code,
    PerturbSeed seed,
    uint element,
    constant PackedLinear& linear
) {
    if (seed.enabled == 0u) return code;
    const uint random = perturb_hash(seed, element);
    const uint amount = linear.perturb_whole +
        uint((random >> 1u) < (linear.perturb_threshold >> 1u));
    const uint max_code = (1u << linear.bits) - 1u;
    if (amount == 0u) return code;
    if ((random & 1u) == 0u) {
        return code >= amount ? code - amount : min(code + amount, max_code);
    }
    return code + amount <= max_code ? code + amount :
        code >= amount ? code - amount : 0u;
}

inline uint packed_code(
    device const uchar* weights,
    constant PackedLinear& linear,
    uint row,
    uint element
) {
    const uint elements_per_byte = linear.bits == 4u ? 2u : 1u;
    const uint byte_index = element / elements_per_byte;
    const uint shift = linear.bits == 4u ? (element & 1u) * 4u : 0u;
    return (uint(weights[linear.byte_offset +
        row * ((linear.input_width + elements_per_byte - 1u) / elements_per_byte) +
        byte_index]) >> shift) & ((1u << linear.bits) - 1u);
}

// Fused ENNX perturbation + packed projection. The output remains on-device
// and is intended to feed kda_recurrence_16k without a host round trip.
kernel void kda_project_packed_16k(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant PackedLinear& linear [[buffer(5)]],
    constant PerturbSeed& seed [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = linear.output_width;
    const uint sequence = 16384u;
    const uint batch = 1u;
    const uint row = gid / (sequence * batch);
    const uint token = gid % (sequence * batch);
    if (row >= total) return;

    const uint groups = (linear.input_width + linear.group_size - 1u) / linear.group_size;
    float sum = 0.0f;
    for (uint i = 0; i < linear.input_width; ++i) {
        uint code = packed_code(weights, linear, row, i);
        code = perturb_code(code, seed, linear.element_offset + row * linear.input_width + i, linear);
        const uint group = min(i / linear.group_size, groups - 1u);
        const float weight = float(code) * scales[linear.scale_offset + row * groups + group]
            + biases[linear.bias_offset + row * groups + group];
        sum += float(input[token * linear.input_width + i]) * weight;
    }
    output[token * linear.output_width + row] = half(sum);
}

kernel void kda_split_qkv_16k(
    device const half* qkv [[buffer(0)]],
    device half* q [[buffer(1)]],
    device half* k [[buffer(2)]],
    device half* v [[buffer(3)]],
    constant KDAParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint width = params.heads * params.key_width;
    const uint total = params.length * width;
    if (gid >= total) return;
    const uint token = gid / width;
    const uint element = gid % width;
    const uint base = token * width * 3u + element;
    q[gid] = qkv[base];
    k[gid] = qkv[base + width];
    v[gid] = qkv[base + 2u * width];
}

inline ulong qkv_index(constant KDAParams& p, uint b, uint t, uint h, uint d) {
    return (((ulong)b * p.length + t) * p.heads + h) * p.key_width + d;
}

inline ulong value_index(constant KDAParams& p, uint b, uint t, uint h, uint d) {
    return (((ulong)b * p.length + t) * p.heads + h) * p.value_width + d;
}

inline ulong state_index(constant KDAParams& p, uint b, uint h, uint i, uint j) {
    return (((ulong)b * p.heads + h) * p.key_width + i) * p.value_width + j;
}

kernel void kda_recurrence_16k(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device half* output [[buffer(5)]],
    device float* state [[buffer(6)]],
    constant KDAParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const uint columns = params.batch * params.heads * params.value_width;
    if (gid >= columns) {
        return;
    }

    const uint column = gid % params.value_width;
    const uint state_group = gid / params.value_width;
    const uint b = state_group / params.heads;
    const uint h = state_group % params.heads;

    // Register cache for the resident key-column state to maximize L1/register bandwidth
    // over long token contexts (e.g. 16k tokens) without DRAM memory traffic per token step.
    float local_state[128];
    const uint kw = min(params.key_width, 128u);
    for (uint i = 0; i < kw; ++i) {
        local_state[i] = state[state_index(params, b, h, i, column)];
    }

    for (uint t = 0; t < params.length; ++t) {
        // Apply per-key decay in thread registers
        for (uint i = 0; i < kw; ++i) {
            const float decay = exp(gate[qkv_index(params, b, t, h, i)]);
            local_state[i] *= decay;
        }

        const float rate = beta[((ulong)b * params.length + t) * params.heads + h];

        // Delta rule: update = beta * (v - k^T state)
        float read = 0.0f;
        for (uint i = 0; i < kw; ++i) {
            read += float(k[qkv_index(params, b, t, h, i)]) * local_state[i];
        }
        const float update = rate * (float(v[value_index(params, b, t, h, column)]) - read);
        for (uint i = 0; i < kw; ++i) {
            local_state[i] += float(k[qkv_index(params, b, t, h, i)]) * update;
        }

        float result = 0.0f;
        for (uint i = 0; i < kw; ++i) {
            result += float(q[qkv_index(params, b, t, h, i)]) * local_state[i];
        }
        output[value_index(params, b, t, h, column)] = half(result * rsqrt(float(kw)));
    }

    // Flush final state back to global memory at sequence end
    for (uint i = 0; i < kw; ++i) {
        state[state_index(params, b, h, i, column)] = local_state[i];
    }
}

// ============================================================================
// Fused Context Prefill Kernel (16k Prompt Context)
// Reference: Bafna et al. (2026), Hardware-Resident Bayesian Optimization in ENNX.
// Processes the fixed 16k context sequence ONCE, computing and caching the
// resident KDA state S_16k in GPU VRAM for fast zero-order BO decode steps.
// ============================================================================
kernel void kda_prefill_16k(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* cached_state [[buffer(5)]],
    constant KDAParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint columns = params.batch * params.heads * params.value_width;
    if (gid >= columns) return;

    const uint column = gid % params.value_width;
    const uint state_group = gid / params.value_width;
    const uint b = state_group / params.heads;
    const uint h = state_group % params.heads;

    float local_state[128];
    const uint kw = min(params.key_width, 128u);
    for (uint i = 0; i < kw; ++i) {
        local_state[i] = 0.0f;
    }

    for (uint t = 0; t < params.length; ++t) {
        for (uint i = 0; i < kw; ++i) {
            const float decay = exp(gate[qkv_index(params, b, t, h, i)]);
            local_state[i] *= decay;
        }

        const float rate = beta[((ulong)b * params.length + t) * params.heads + h];
        float read = 0.0f;
        for (uint i = 0; i < kw; ++i) {
            read += float(k[qkv_index(params, b, t, h, i)]) * local_state[i];
        }
        const float update = rate * (float(v[value_index(params, b, t, h, column)]) - read);
        for (uint i = 0; i < kw; ++i) {
            local_state[i] += float(k[qkv_index(params, b, t, h, i)]) * update;
        }
    }

    // Persist prefilled context state S_16k in GPU VRAM
    for (uint i = 0; i < kw; ++i) {
        cached_state[state_index(params, b, h, i, column)] = local_state[i];
    }
}

// ============================================================================
// Single-Token Fast Decode Step Kernel (L = 1)
// Evaluates candidate perturbed weights against prefilled state S_16k in ~0.1ms.
// ============================================================================
kernel void kda_step(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* cached_state [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant KDAParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const uint columns = params.batch * params.heads * params.value_width;
    if (gid >= columns) return;

    const uint column = gid % params.value_width;
    const uint state_group = gid / params.value_width;
    const uint b = state_group / params.heads;
    const uint h = state_group % params.heads;

    const uint kw = min(params.key_width, 128u);
    float local_state[128];
    for (uint i = 0; i < kw; ++i) {
        local_state[i] = cached_state[state_index(params, b, h, i, column)];
    }

    // Evaluate 1-step decode against prefilled state
    for (uint i = 0; i < kw; ++i) {
        const float decay = exp(gate[qkv_index(params, b, 0, h, i)]);
        local_state[i] *= decay;
    }

    const float rate = beta[ulong(b) * params.heads + h];
    float read = 0.0f;
    for (uint i = 0; i < kw; ++i) {
        read += float(k[qkv_index(params, b, 0, h, i)]) * local_state[i];
    }
    const float update = rate * (float(v[value_index(params, b, 0, h, column)]) - read);
    for (uint i = 0; i < kw; ++i) {
        local_state[i] += float(k[qkv_index(params, b, 0, h, i)]) * update;
    }

    float result = 0.0f;
    for (uint i = 0; i < kw; ++i) {
        result += float(q[qkv_index(params, b, 0, h, i)]) * local_state[i];
    }
    output[value_index(params, b, 0, h, column)] = half(result * rsqrt(float(kw)));

    // The decode state belongs to this (batch, head, value-column) thread,
    // so it can be committed in place without atomics or a grid barrier.
    // Leaving this write out makes every generated token attend to the same
    // prefill state instead of advancing the recurrent model.
    for (uint i = 0; i < kw; ++i) {
        cached_state[state_index(params, b, h, i, column)] = local_state[i];
    }
}

// Decoder-layer ABI. All activations remain resident for the lifetime of one
// command buffer. Weight descriptors refer to the single packed model arena.
struct DecoderParams {
    uint batch;
    uint length;
    uint hidden_width;
    uint experts;
    uint top_k;
    uint expert_width;
    float residual_scale;
    float rms_epsilon;
};

inline uint packed_code_device(
    device const uchar* weights,
    PackedLinear linear,
    uint row,
    uint element
) {
    const uint elements_per_byte = linear.bits == 4u ? 2u : 1u;
    const uint byte_index = element / elements_per_byte;
    const uint shift = linear.bits == 4u ? (element & 1u) * 4u : 0u;
    const uint stride = (linear.input_width + elements_per_byte - 1u) / elements_per_byte;
    return (uint(weights[linear.byte_offset + row * stride + byte_index]) >> shift) &
        ((1u << linear.bits) - 1u);
}

inline float packed_weight_device(
    device const uchar* weights,
    device const float* scales,
    device const float* biases,
    PackedLinear linear,
    PerturbSeed seed,
    uint row,
    uint column
) {
    const uint groups = (linear.input_width + linear.group_size - 1u) / linear.group_size;
    const uint group = min(column / linear.group_size, groups - 1u);
    uint code = packed_code_device(weights, linear, row, column);
    if (seed.enabled == 0u) {
        return float(code) * scales[linear.scale_offset + row * groups + group] +
            biases[linear.bias_offset + row * groups + group];
    }
    const uint random = perturb_hash(seed, linear.element_offset + row * linear.input_width + column);
    const uint amount = linear.perturb_whole +
        uint((random >> 1u) < (linear.perturb_threshold >> 1u));
    const uint max_code = (1u << linear.bits) - 1u;
    if (amount != 0u) {
        if ((random & 1u) == 0u) {
            code = code >= amount ? code - amount : min(code + amount, max_code);
        } else {
            code = code + amount <= max_code ? code + amount :
                code >= amount ? code - amount : 0u;
        }
    }
    return float(code) * scales[linear.scale_offset + row * groups + group] +
        biases[linear.bias_offset + row * groups + group];
}

// Shape-generic single-token/short-sequence packed affine. Unlike the 16k
// prefill specialization, this derives the token count from DecoderParams.
kernel void decoder_project_packed(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant PackedLinear& linear [[buffer(5)]],
    constant PerturbSeed& seed [[buffer(6)]],
    constant DecoderParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const uint tokens = params.batch * params.length;
    const ulong total = ulong(tokens) * linear.output_width;
    if (gid >= total) return;
    const uint row = gid % linear.output_width;
    const uint token = gid / linear.output_width;
    float sum = 0.0f;
    for (uint column = 0; column < linear.input_width; ++column) {
        sum += float(input[ulong(token) * linear.input_width + column]) *
            packed_weight_device(weights, scales, biases, linear, seed, row, column);
    }
    output[ulong(token) * linear.output_width + row] = half(sum);
}

// Decode GEMV: one Apple SIMD-group cooperatively evaluates one output row.
// This replaces a serial input-width loop in a single GPU thread.
kernel void decoder_project_packed_simd(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant PackedLinear& linear [[buffer(5)]],
    constant PerturbSeed& seed [[buffer(6)]],
    constant DecoderParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint outputs = params.batch * params.length * linear.output_width;
    const uint output_index = gid >> 5u;
    if (output_index >= outputs) return;
    const uint row = output_index % linear.output_width;
    const uint token = output_index / linear.output_width;
    float sum = 0.0f;
    for (uint column = lane; column < linear.input_width; column += 32u) {
        sum += float(input[ulong(token) * linear.input_width + column]) *
            packed_weight_device(weights, scales, biases, linear, seed, row, column);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[ulong(token) * linear.output_width + row] = half(sum);
    }
}

kernel void packed_dequantize_row_half(
    device const uchar* weights [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant PackedLinear& linear [[buffer(4)]],
    constant PerturbSeed& seed [[buffer(5)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= linear.input_width || linear.output_width != 1u) return;
    output[column] = half(
        packed_weight_device(weights, scales, biases, linear, seed, 0u, column)
    );
}

kernel void packed_dequantize_row_float(
    device const uchar* weights [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant PackedLinear& linear [[buffer(4)]],
    constant PerturbSeed& seed [[buffer(5)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= linear.input_width || linear.output_width != 1u) return;
    output[column] = packed_weight_device(
        weights, scales, biases, linear, seed, 0u, column
    );
}

// Stateful causal depthwise convolution for the one-token decode path. Each
// feature thread owns its three history cells, so history advances in place.
kernel void kda_short_conv_decode(
    device half* projected_qkv [[buffer(0)]],
    device half* history [[buffer(1)]],
    device const uchar* weights [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device const float* biases [[buffer(4)]],
    constant PackedLinear& convolution [[buffer(5)]],
    constant PerturbSeed& seed [[buffer(6)]],
    uint feature [[thread_position_in_grid]]) {
    if (feature >= convolution.output_width) return;
    const uint width = convolution.output_width;
    const float current = float(projected_qkv[feature]);
    float value = 0.0f;
    value += float(history[feature]) *
        packed_weight_device(weights, scales, biases, convolution, seed, feature, 0u);
    value += float(history[width + feature]) *
        packed_weight_device(weights, scales, biases, convolution, seed, feature, 1u);
    value += float(history[2u * width + feature]) *
        packed_weight_device(weights, scales, biases, convolution, seed, feature, 2u);
    value += current *
        packed_weight_device(weights, scales, biases, convolution, seed, feature, 3u);
    history[feature] = history[width + feature];
    history[width + feature] = history[2u * width + feature];
    history[2u * width + feature] = half(current);
    projected_qkv[feature] = half(value / (1.0f + exp(-value)));
}

// Normalize query and key independently over each head, matching the JAX
// reference's L2 normalization rather than RMS normalization.
kernel void kda_normalize_qk(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant KDAParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint groups = params.batch * params.length * params.heads;
    if (gid >= groups) return;
    const uint head = gid % params.heads;
    const uint token = gid / params.heads;
    const ulong start = (ulong(token) * params.heads + head) * params.key_width;
    float q_square_sum = 0.0f;
    float k_square_sum = 0.0f;
    for (uint i = 0; i < params.key_width; ++i) {
        const float q_value = float(query[start + i]);
        const float k_value = float(key[start + i]);
        q_square_sum += q_value * q_value;
        k_square_sum += k_value * k_value;
    }
    const float q_inverse = 1.0f / max(sqrt(q_square_sum), 1.0e-6f);
    const float k_inverse = 1.0f / max(sqrt(k_square_sum), 1.0e-6f);
    for (uint i = 0; i < params.key_width; ++i) {
        query[start + i] = half(float(query[start + i]) * q_inverse);
        key[start + i] = half(float(key[start + i]) * k_inverse);
    }
}

kernel void decoder_rms_norm(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant DecoderParams& params [[buffer(3)]],
    uint token [[thread_position_in_grid]]) {
    const uint tokens = params.batch * params.length;
    if (token >= tokens) return;
    const ulong start = ulong(token) * params.hidden_width;
    float square_sum = 0.0f;
    for (uint i = 0; i < params.hidden_width; ++i) {
        const float value = float(input[start + i]);
        square_sum += value * value;
    }
    const float inv_rms = rsqrt(square_sum / float(params.hidden_width) + params.rms_epsilon);
    for (uint i = 0; i < params.hidden_width; ++i) {
        output[start + i] = half(float(input[start + i]) * inv_rms * float(weight[i]));
    }
}

kernel void decoder_residual(
    device const half* hidden [[buffer(0)]],
    device const half* update [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant DecoderParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(params.batch) * params.length * params.hidden_width;
    if (gid >= elements) return;
    output[gid] = half(float(hidden[gid]) + params.residual_scale * float(update[gid]));
}

// One thread scores all experts for one token and writes normalized top-k
// routing weights. KDA-MoE has a small fixed expert count, so this avoids a
// host-side routing pass and leaves only selected experts for later kernels.
kernel void moe_router_topk(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device uint* selected_indices [[buffer(4)]],
    device half* selected_weights [[buffer(5)]],
    constant PackedLinear& router [[buffer(6)]],
    constant PerturbSeed& seed [[buffer(7)]],
    constant DecoderParams& params [[buffer(8)]],
    uint token [[thread_position_in_grid]]) {
    const uint tokens = params.batch * params.length;
    if (token >= tokens || params.top_k == 0u || params.top_k > 8u) return;

    float values[8];
    uint indices[8];
    for (uint rank = 0; rank < params.top_k; ++rank) {
        values[rank] = -INFINITY;
        indices[rank] = 0u;
    }
    const PackedLinear router_linear = router;
    const ulong input_start = ulong(token) * router.input_width;
    for (uint expert = 0; expert < params.experts; ++expert) {
        float score = 0.0f;
        for (uint column = 0; column < router.input_width; ++column) {
            score += float(input[input_start + column]) *
                packed_weight_device(weights, scales, biases, router_linear, seed, expert, column);
        }
        for (uint rank = 0; rank < params.top_k; ++rank) {
            if (score > values[rank]) {
                for (uint move = params.top_k - 1u; move > rank; --move) {
                    values[move] = values[move - 1u];
                    indices[move] = indices[move - 1u];
                }
                values[rank] = score;
                indices[rank] = expert;
                break;
            }
        }
    }
    float denominator = 0.0f;
    const float maximum = values[0];
    for (uint rank = 0; rank < params.top_k; ++rank) {
        values[rank] = exp(values[rank] - maximum);
        denominator += values[rank];
    }
    const ulong output_start = ulong(token) * params.top_k;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        selected_indices[output_start + rank] = indices[rank];
        selected_weights[output_start + rank] = half(values[rank] / denominator);
    }
}

// Parallel router for decode: sixteen SIMD-groups score the 32 experts, two
// experts per group, then one lane performs the tiny top-k selection.
kernel void moe_router_topk_simd(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device uint* selected_indices [[buffer(4)]],
    device half* selected_weights [[buffer(5)]],
    constant PackedLinear& router [[buffer(6)]],
    constant PerturbSeed& seed [[buffer(7)]],
    constant DecoderParams& params [[buffer(8)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint tokens = params.batch * params.length;
    if (token >= tokens || params.top_k == 0u || params.top_k > 8u) return;
    threadgroup float scores[32];
    const PackedLinear router_linear = router;
    const ulong input_start = ulong(token) * router.input_width;
    for (uint expert = simd_index; expert < params.experts; expert += 16u) {
        float score = 0.0f;
        for (uint column = lane; column < router.input_width; column += 32u) {
            score += float(input[input_start + column]) *
                packed_weight_device(weights, scales, biases, router_linear, seed, expert, column);
        }
        score = simd_sum(score);
        if (lane == 0u) scores[expert] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index != 0u) return;

    float values[8];
    uint indices[8];
    for (uint rank = 0; rank < params.top_k; ++rank) {
        values[rank] = -INFINITY;
        indices[rank] = 0u;
    }
    for (uint expert = 0; expert < params.experts; ++expert) {
        const float score = scores[expert];
        for (uint rank = 0; rank < params.top_k; ++rank) {
            if (score > values[rank]) {
                for (uint move = params.top_k - 1u; move > rank; --move) {
                    values[move] = values[move - 1u];
                    indices[move] = indices[move - 1u];
                }
                values[rank] = score;
                indices[rank] = expert;
                break;
            }
        }
    }
    float denominator = 0.0f;
    const float maximum = values[0];
    for (uint rank = 0; rank < params.top_k; ++rank) {
        values[rank] = exp(values[rank] - maximum);
        denominator += values[rank];
    }
    const ulong output_start = ulong(token) * params.top_k;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        selected_indices[output_start + rank] = indices[rank];
        selected_weights[output_start + rank] = half(values[rank] / denominator);
    }
}

// Gate and up projections share the input load. This emits only the selected
// experts' activations: [token, top_k, expert_width].
kernel void moe_gate_up(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device const uint* selected_indices [[buffer(4)]],
    device half* activation [[buffer(5)]],
    device const PackedLinear* gate [[buffer(6)]],
    device const PackedLinear* up [[buffer(7)]],
    constant PerturbSeed& seed [[buffer(8)]],
    constant DecoderParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = ulong(params.batch) * params.length * params.top_k * params.expert_width;
    if (gid >= total) return;
    const uint feature = gid % params.expert_width;
    const uint selection = gid / params.expert_width;
    const uint rank = selection % params.top_k;
    const uint token = selection / params.top_k;
    const uint expert = selected_indices[selection];
    const PackedLinear gate_linear = gate[expert];
    const PackedLinear up_linear = up[expert];
    const ulong input_start = ulong(token) * params.hidden_width;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    for (uint column = 0; column < params.hidden_width; ++column) {
        const float x = float(input[input_start + column]);
        gate_value += x * packed_weight_device(weights, scales, biases, gate_linear, seed, feature, column);
        up_value += x * packed_weight_device(weights, scales, biases, up_linear, seed, feature, column);
    }
    const float silu = gate_value / (1.0f + exp(-gate_value));
    activation[(ulong(token) * params.top_k + rank) * params.expert_width + feature] = half(silu * up_value);
}

// Reduce selected expert outputs directly into the residual update. No full
// [tokens, experts, hidden] tensor is ever materialized.
kernel void moe_down(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device const half* selected_weights [[buffer(4)]],
    device const PackedLinear* down [[buffer(5)]],
    device const uint* selected_indices [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant PerturbSeed& seed [[buffer(8)]],
    constant DecoderParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = ulong(params.batch) * params.length * params.hidden_width;
    if (gid >= total) return;
    const uint feature = gid % params.hidden_width;
    const uint token = gid / params.hidden_width;
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const ulong selection = ulong(token) * params.top_k + rank;
        const PackedLinear linear = down[selected_indices[selection]];
        float expert_sum = 0.0f;
        const ulong activation_start = selection * params.expert_width;
        for (uint column = 0; column < params.expert_width; ++column) {
            expert_sum += float(activation[activation_start + column]) *
                packed_weight_device(weights, scales, biases, linear, seed, feature, column);
        }
        sum += float(selected_weights[selection]) * expert_sum;
    }
    output[gid] = half(sum);
}

// Single-token selected-expert GEMV. One SIMD-group produces one SwiGLU
// feature while sharing the same normalized input across its 32 lanes.
kernel void moe_gate_up_simd(
    device const half* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device const uint* selected_indices [[buffer(4)]],
    device half* activation [[buffer(5)]],
    device const PackedLinear* gate [[buffer(6)]],
    device const PackedLinear* up [[buffer(7)]],
    constant PerturbSeed& seed [[buffer(8)]],
    constant DecoderParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint outputs = params.batch * params.length * params.top_k * params.expert_width;
    const uint output_index = gid >> 5u;
    if (output_index >= outputs) return;
    const uint feature = output_index % params.expert_width;
    const uint selection = output_index / params.expert_width;
    const uint token = selection / params.top_k;
    const uint expert = selected_indices[selection];
    const PackedLinear gate_linear = gate[expert];
    const PackedLinear up_linear = up[expert];
    const ulong input_start = ulong(token) * params.hidden_width;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    for (uint column = lane; column < params.hidden_width; column += 32u) {
        const float x = float(input[input_start + column]);
        gate_value += x *
            packed_weight_device(weights, scales, biases, gate_linear, seed, feature, column);
        up_value += x *
            packed_weight_device(weights, scales, biases, up_linear, seed, feature, column);
    }
    gate_value = simd_sum(gate_value);
    up_value = simd_sum(up_value);
    if (lane == 0u) {
        const float silu = gate_value / (1.0f + exp(-gate_value));
        activation[output_index] = half(silu * up_value);
    }
}

// One SIMD-group reduces every selected expert into one hidden output.
kernel void moe_down_simd(
    device const half* activation [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device const half* selected_weights [[buffer(4)]],
    device const PackedLinear* down [[buffer(5)]],
    device const uint* selected_indices [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant PerturbSeed& seed [[buffer(8)]],
    constant DecoderParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint outputs = params.batch * params.length * params.hidden_width;
    const uint output_index = gid >> 5u;
    if (output_index >= outputs) return;
    const uint feature = output_index % params.hidden_width;
    const uint token = output_index / params.hidden_width;
    float sum = 0.0f;
    for (uint rank = 0; rank < params.top_k; ++rank) {
        const ulong selection = ulong(token) * params.top_k + rank;
        const PackedLinear linear = down[selected_indices[selection]];
        float expert_sum = 0.0f;
        const ulong activation_start = selection * params.expert_width;
        for (uint column = lane; column < params.expert_width; column += 32u) {
            expert_sum += float(activation[activation_start + column]) *
                packed_weight_device(weights, scales, biases, linear, seed, feature, column);
        }
        expert_sum = simd_sum(expert_sum);
        sum += float(selected_weights[selection]) * expert_sum;
    }
    if (lane == 0u) {
        output[output_index] = half(sum);
    }
}

struct KDAControlParams {
    uint batch;
    uint length;
    uint heads;
    uint head_width;
    uint gate_rank;
    float rms_epsilon;
};

// Control is laid out as [forget_state, output_state, beta]. The split is
// device-only so the two rank-sized projections can feed the fused KDA block.
kernel void kda_split_control_16k(
    device const half* control [[buffer(0)]],
    device half* forget_state [[buffer(1)]],
    device half* output_state [[buffer(2)]],
    device half* raw_beta [[buffer(3)]],
    constant KDAControlParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint tokens = params.batch * params.length;
    const uint control_width = 2u * params.gate_rank + params.heads;
    const uint total = tokens * control_width;
    if (gid >= total) return;
    const uint token = gid / control_width;
    const uint column = gid % control_width;
    if (column < params.gate_rank) {
        forget_state[token * params.gate_rank + column] = control[gid];
    } else if (column < 2u * params.gate_rank) {
        output_state[token * params.gate_rank + column - params.gate_rank] = control[gid];
    } else {
        raw_beta[token * params.heads + column - 2u * params.gate_rank] = control[gid];
    }
}

inline float softplus_stable(float value) {
    return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

// Form the exact gate and beta inputs expected by kda_recurrence_16k. The
// decay, time bias, and beta stay in float32 to avoid recurrent overflow.
kernel void kda_make_gate_beta_16k(
    device const half* raw_gate [[buffer(0)]],
    device const half* raw_beta [[buffer(1)]],
    device const float* decay [[buffer(2)]],
    device const float* time_bias [[buffer(3)]],
    device float* gate [[buffer(4)]],
    device float* beta [[buffer(5)]],
    constant KDAControlParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint width = params.heads * params.head_width;
    const uint total = params.batch * params.length * width;
    if (gid >= total) return;
    const uint token = gid / width;
    const uint head_element = gid % width;
    const uint head = head_element / params.head_width;
    const float raw = float(raw_gate[gid]) + time_bias[head_element];
    gate[gid] = max(-exp(decay[head]) * softplus_stable(raw), -5.0f);
    if (head_element % params.head_width == 0u) {
        beta[token * params.heads + head] = 1.0f / (1.0f + exp(-float(raw_beta[token * params.heads + head])));
    }
}

// KDA output norm and output gate, before the final packed output projection.
kernel void kda_postprocess_16k(
    device const half* recurrence_output [[buffer(0)]],
    device const half* output_gate [[buffer(1)]],
    device const float* output_norm [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant KDAControlParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint width = params.heads * params.head_width;
    const uint total = params.batch * params.length * width;
    if (gid >= total) return;
    const uint token = gid / width;
    const uint head_element = gid % width;
    const uint head = head_element / params.head_width;
    const uint head_start = token * width + head * params.head_width;
    float square_sum = 0.0f;
    for (uint i = 0; i < params.head_width; ++i) {
        const float value = float(recurrence_output[head_start + i]);
        square_sum += value * value;
    }
    const float inv_rms = rsqrt(square_sum / float(params.head_width) + params.rms_epsilon);
    const float value = float(recurrence_output[gid]) * inv_rms * output_norm[head_element % params.head_width];
    const float gate_value = float(output_gate[gid]);
    output[gid] = half(value / (1.0f + exp(-gate_value)));
}

struct EmbeddingParams {
    uint vocab_size;
    uint hidden_width;
    float embedding_scale;
};

kernel void packed_embedding_lookup(
    device const uint* token_ids [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant PackedLinear& embedding [[buffer(5)]],
    constant PerturbSeed& seed [[buffer(6)]],
    constant EmbeddingParams& params [[buffer(7)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= params.hidden_width) return;
    const uint token = token_ids[0];
    if (token >= params.vocab_size) return;
    output[column] = half(
        packed_weight_device(weights, scales, biases, embedding, seed, token, column) *
        params.embedding_scale
    );
}

kernel void decoder_argmax(
    device const half* logits [[buffer(0)]],
    device uint* token [[buffer(1)]],
    constant uint& vocabulary [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint best_index = 0u;
    for (uint index = lane; index < vocabulary; index += 256u) {
        const float value = float(logits[index]);
        if (value > best) {
            best = value;
            best_index = index;
        }
    }
    values[lane] = best;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && values[lane + stride] > values[lane]) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) token[0] = indices[0];
}

struct ObjectiveReductionParams {
    uint num_candidates;
    uint hidden_width;
    float lambda_epistemic;
    float lambda_latency;
    float target_latency_ms;
};

// ============================================================================
// Fused Token Embedding Lookup Kernel
// Reference: Vaswani et al. (2017) "Attention Is All You Need", Section 3.4.
// Direct on-device memory-mapped embedding table lookup with scaling factor.
// ============================================================================
kernel void token_embedding_lookup(
    device const uint* token_ids [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = params.hidden_width;
    const uint token = gid / total;
    const uint dim = gid % total;
    const uint token_id = token_ids[token];
    if (token_id >= params.vocab_size) return;
    const ulong table_idx = ulong(token_id) * params.hidden_width + dim;
    output[gid] = half(float(embedding_table[table_idx]) * params.embedding_scale);
}

// ============================================================================
// Hardware-Resident Fused Composite Objective Reduction Kernel
// Reference: Bafna et al. (2026), Hardware-Resident Bayesian Optimization in ENNX.
// Computes O_ENNX(x) = Task_Performance + λ_epistemic * Variance - λ_latency * Latency_Overhead
// directly on Metal without host CPU intervention or memory transfer delays.
// ============================================================================
kernel void reduce_composite_objective(
    device const half* model_output [[buffer(0)]],
    device const float* epistemic_variance [[buffer(1)]],
    device const float* hardware_telemetry_ms [[buffer(2)]],
    device float* composite_scores [[buffer(3)]],
    constant ObjectiveReductionParams& params [[buffer(4)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= params.num_candidates) return;
    const ulong start = ulong(candidate) * params.hidden_width;
    float task_score = 0.0f;
    for (uint d = 0; d < params.hidden_width; ++d) {
        const float val = float(model_output[start + d]);
        task_score += val;
    }
    const float epistemic_gain = epistemic_variance[candidate];
    const float lat_ms = hardware_telemetry_ms[candidate];
    const float lat_penalty = max(lat_ms - params.target_latency_ms, 0.0f);

    composite_scores[candidate] = task_score +
        params.lambda_epistemic * epistemic_gain -
        params.lambda_latency * lat_penalty;
}
