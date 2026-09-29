// Opt-in vector selector. Existing scalar selector and its ABI are unchanged.
struct ObjectiveParams {
    float outcomes[8][128], variances[8][128], draws[8][128];
    float scales[8], weights[8], minima[8], maxima[8];
    ulong priorities[4];
    uint width, mode;
    float alpha;
    uint clip;
};
struct ObjectiveSelectionReport {
    float values[4][8], weights[8];
    uint width, nondominated_mask, selected, pad;
};
inline float objective_chebyshev(thread const float *values, constant ObjectiveParams &o) {
    float minimum = INFINITY, sum = 0.0f;
    for (uint metric = 0; metric < o.width; metric++) {
        float range = o.maxima[metric] - o.minima[metric];
        float normalized = range <= 0.0f ? 0.5f : (values[metric] - o.minima[metric]) / range;
        if (o.clip != 0) normalized = clamp(normalized, 0.0f, 1.0f);
        float weighted = normalized * o.weights[metric];
        minimum = min(minimum, weighted);
        sum += weighted;
    }
    return minimum + o.alpha * sum;
}

inline void objective_geometry(
    thread const float *distances,
    constant SelectionParams &q,
    uint excluded_index,
    thread float *scaled_distances,
    thread uint *nearest) {
    if (q.distance_scaling != 0) {
        float query_neighbors[128];
        uint query_count = 0;
        for (uint i = 0; i < q.history; i++) {
            if (i == excluded_index) continue;
            uint position = query_count;
            while (position > 0 && distances[i] < query_neighbors[position - 1]) {
                query_neighbors[position] = query_neighbors[position - 1];
                position--;
            }
            query_neighbors[position] = distances[i];
            query_count++;
        }
        uint local_rank = min(q.local_scale_neighbors, query_count);
        float query_scale = local_rank == 0
            ? 1.0f
            : max(query_neighbors[local_rank - 1], 1.0e-12f);
        for (uint i = 0; i < q.history; i++) {
            float history_scale = max(q.local_scales[i], 1.0e-12f);
            scaled_distances[i] = distances[i] / (sqrt(query_scale) * sqrt(history_scale));
        }
    } else {
        for (uint i = 0; i < q.history; i++) scaled_distances[i] = distances[i];
    }
    for (uint i = 0; i < q.history; i++) {
        uint position = i;
        while (position > 0
            && scaled_distances[i] < scaled_distances[nearest[position - 1]]) {
            nearest[position] = nearest[position - 1];
            position--;
        }
        nearest[position] = i;
    }
}

inline float objective_acquisition_score(
    thread const float *scaled_distances,
    thread const uint *nearest,
    constant SelectionParams &q,
    constant float *outcomes,
    constant float *variances,
    constant float *draws,
    float scale,
    thread float &mean_out,
    thread float &standard_error_out) {
    uint count = min(q.neighbors, q.history);
    float sum = 0.0f;
    float value = 0.0f;
    float reference = 1.17549435e-38f;
    float y_scale_sq = max(scale * scale, 1.0e-12f);
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + variances[index] / y_scale_sq;
        float weight = 1.0f / max(variance, 1.0e-12f);
        sum += weight;
        value += weight * outcomes[index];
        reference = max(reference, weight);
    }
    float mean = value / max(sum, 1.0e-12f);
    float aleatoric = 0.0f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + variances[index] / y_scale_sq;
        float weight = (1.0f / max(variance, 1.0e-12f)) / max(sum, 1.0e-12f);
        aleatoric += weight * (q.aleatoric_scale + variances[index] / y_scale_sq);
    }
    float standard_error = sqrt(1.0f / max(sum, 1.0e-12f) + aleatoric) * scale;
    mean_out = mean;
    standard_error_out = standard_error;
    if (q.acquisition == 1) {
        float noise = 0.0f;
        float squared = 0.0f;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float variance = 1.0e-9f
                + q.epistemic_scale * scaled_distances[index]
                + q.aleatoric_scale
                + variances[index] / y_scale_sq;
            float weight = (1.0f / max(variance, 1.0e-12f)) / reference;
            noise += weight * draws[index];
            squared += weight * weight;
        }
        return mean + standard_error * (noise / max(sqrt(squared), 1.0e-12f));
    }
    if (q.acquisition == 2) return mean + standard_error;
    return mean + q.beta * standard_error;
}

kernel void bf16_select_objectives(
    device const Partial *aggregates [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device float *pool_distances [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    constant ObjectiveParams &o [[buffer(4)]],
    device ObjectiveSelectionReport *report [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    float distances[4][128];
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool changed[4] = {false, false, false, false};
    bool invalid[4] = {false, false, false, false};
    for (uint candidate = 0; candidate < 4; candidate++) {
        proposal_norms[candidate] = aggregates[candidate].squared;
        for (uint row = 0; row < q.history; row++) {
            distances[candidate][row] = q.exact_history != 0
                ? pool_distances[candidate * q.history + row]
                : q.implicit_history != 0
                ? q.base_distances[row] + proposal_norms[candidate]
                : 0.0f;
            if (q.program != 0 && q.exact_history == 0)
                distances[candidate][row] += pool_distances[candidate * q.history + row];
            if (q.axis_enabled != 0) {
                float delta = q.axis_candidates[candidate] - q.axis_history[row];
                distances[candidate][row] += q.axis_weight * delta * delta;
            }
        }
        if (q.exact_history != 0) {
            proposal_norms[candidate] = distances[candidate][q.incumbent_index];
            changed[candidate] = proposal_norms[candidate] > 0.0f;
            for (uint row = 0; row < q.history; row++)
                invalid[candidate] = invalid[candidate]
                    || !isfinite(distances[candidate][row]);
        }
    }
    if (q.exact_history == 0)
    for (uint candidate = 0; candidate < 4; candidate++) {
        for (uint row = 0; row < q.resident_history; row += 2) {
            Partial aggregate = aggregates[(row / 2) * 4 + candidate];
            float first = aggregate.anchor;
            float second = aggregate.rejected;
            changed[candidate] = changed[candidate] || aggregate.changed != 0;
            invalid[candidate] = invalid[candidate] || aggregate.invalid != 0;
            uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
            if (q.exact_history == 0) distances[candidate][first_index] = first;
            if (row + 1 < q.resident_history) {
                uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
                if (q.exact_history == 0) distances[candidate][second_index] = second;
            }
            invalid[candidate] = invalid[candidate] || !isfinite(first)
                || (row + 1 < q.resident_history && !isfinite(second));
        }
    }
    bool any_changed = false;
    for (uint candidate = 0; candidate < 4; candidate++) {
        changed[candidate] = changed[candidate] || (q.axis_enabled != 0
            && q.axis_candidates[candidate] != q.axis_base);
        any_changed = any_changed || changed[candidate];
    }
    uint selected = 4;
    float best = -INFINITY;
    float candidate_means[4];
    float candidate_standard_errors[4];
    float incumbent_distances[128];
    for (uint row = 0; row < q.history; row++)
        incumbent_distances[row] = q.base_distances[row]
            + (q.axis_enabled != 0
                ? q.axis_weight * (q.axis_base - q.axis_history[row])
                    * (q.axis_base - q.axis_history[row])
                : 0.0f)
            + (q.program != 0 ? pool_distances[4 * q.history + row] : 0.0f);
    float incumbent_mean;
    float incumbent_standard_error;
    acquisition_score(
        incumbent_distances,
        q,
        q.incumbent_index,
        incumbent_mean,
        incumbent_standard_error);
    bool eligible[4] = {false, false, false, false};
    float vectors[4][8];
    float scalar_scores[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (uint candidate = 0; candidate < 4; candidate++) {
        for (uint metric = 0; metric < 8; metric++) vectors[candidate][metric] = NAN;
        if (candidate < q.candidate_floor || invalid[candidate]
            || (any_changed && !changed[candidate])) continue;
        float scaled_distances[128];
        uint nearest[128];
        objective_geometry(distances[candidate], q, 128u, scaled_distances, nearest);
        float control_score = objective_acquisition_score(scaled_distances, nearest, q,
            q.outcomes, q.variances, q.draws, q.y_scale,
            candidate_means[candidate], candidate_standard_errors[candidate]);
        eligible[candidate] = isfinite(control_score);
        for (uint metric = 0; metric < o.width; metric++) {
            float mean, error;
            vectors[candidate][metric] = objective_acquisition_score(
                scaled_distances, nearest, q, o.outcomes[metric], o.variances[metric],
                o.draws[metric], o.scales[metric], mean, error);
            eligible[candidate] = eligible[candidate] && isfinite(vectors[candidate][metric]);
        }
        if (o.mode == 1) scalar_scores[candidate] = objective_chebyshev(vectors[candidate], o);
        if (o.mode == 1) eligible[candidate] = eligible[candidate] && isfinite(scalar_scores[candidate]);
    }
    uint nondominated_mask = 0;
    for (uint candidate = 0; candidate < 4; candidate++) {
        if (!eligible[candidate]) continue;
        bool dominated = false;
        for (uint other = 0; other < 4; other++) {
            if (other == candidate || !eligible[other]) continue;
            bool no_worse = true, strictly_better = false;
            for (uint metric = 0; metric < o.width; metric++) {
                no_worse = no_worse && vectors[other][metric] >= vectors[candidate][metric];
                strictly_better = strictly_better || vectors[other][metric] > vectors[candidate][metric];
            }
            dominated = dominated || (no_worse && strictly_better);
        }
        if (!dominated) nondominated_mask |= 1u << candidate;
        if (q.forced_candidate < 4) {
            if (candidate == q.forced_candidate) {
                selected = candidate;
                best = o.mode == 1 ? scalar_scores[candidate] : 0.0f;
            }
            continue;
        }
        if (o.mode == 1 && (selected == 4 || scalar_scores[candidate] > best)) {
            selected = candidate;
            best = scalar_scores[candidate];
        }
    }
    if (o.mode == 0 && q.forced_candidate >= 4) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            if ((nondominated_mask & (1u << candidate)) == 0) continue;
            if (selected == 4 || o.priorities[candidate] > o.priorities[selected]) selected = candidate;
        }
        best = 0.0f; // Pareto has no scalar objective score.
    }
    for (uint candidate = 0; candidate < 4; candidate++)
        for (uint metric = 0; metric < 8; metric++) report->values[candidate][metric] = vectors[candidate][metric];
    for (uint metric = 0; metric < 8; metric++) report->weights[metric] = o.weights[metric];
    report->width = o.width;
    report->nondominated_mask = nondominated_mask;
    report->selected = selected;
    report->pad = 0;
    decision->index = selected;
    decision->valid = selected < 4 ? 1u : 0u;
    decision->root_seed = q.root_seed;
    decision->seed = selected < 4 ? candidate_seed(q.root_seed, selected) : 0ul;
    decision->basis_seed = q.basis_seed;
    decision->radius = selected < 4
#if defined(PROCEDURAL_SLOTS) && PROCEDURAL_SLOTS == 1
        ? q.radius
#else
        ? ((selected & 1u) == 0 ? q.radius : q.alternate_radius)
#endif
        : 0.0f;
    decision->score = selected < 4 ? best : 0.0f;
    decision->mode = q.mode;
    decision->program = q.program;
    decision->predicted_mean = selected < 4 ? candidate_means[selected] : 0.0f;
    decision->predicted_standard_error = selected < 4
        ? candidate_standard_errors[selected]
        : 0.0f;
    decision->incumbent_mean = incumbent_mean;
    decision->incumbent_standard_error = incumbent_standard_error;
    for (uint candidate = 0; candidate < 4; candidate++)
        for (uint row = 0; row < q.history; row++)
            pool_distances[candidate * q.history + row] = distances[candidate][row];
}
