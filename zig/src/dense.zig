const std = @import("std");

pub const Leaf = extern struct {
    key: u64,
    offset: usize,
    len: usize,
    scale: f32,
};

pub const Term = extern struct {
    seed: u64,
    coefficient: f32,
};

pub const View = extern struct {
    key: u64,
    start: u64,
    scale: f32,
};

pub const Error = error{
    InvalidArgument,
    InvalidLayout,
    InvalidValue,
};

pub fn sign(seed: u64, leaf_key: u64, element: u64) f32 {
    const leaf = mix64(leaf_key ^ 0xd6e8_feb8_6659_fd93);
    const coordinate = mix64(element ^ 0xa076_1d64_78bd_642f);
    return if ((mix64(seed ^ leaf ^ coordinate) & 1) == 0) -1.0 else 1.0;
}

pub fn apply(
    base: []const f32,
    leaves: []const Leaf,
    terms: []const Term,
    out: []f32,
) Error!usize {
    if (base.len == 0 or out.len != base.len or leaves.len == 0 or terms.len == 0) {
        return error.InvalidArgument;
    }

    try validateTerms(terms);
    if (!hasDirection(terms)) return error.InvalidArgument;

    try validateLeaves(leaves, base.len);

    var changed: usize = 0;
    for (leaves) |leaf| {
        for (0..leaf.len) |local_index| {
            const index = leaf.offset + local_index;
            const value = base[index];
            if (!std.math.isFinite(value)) return error.InvalidValue;
            out[index] = try perturbUnchecked(
                value,
                .{ .key = leaf.key, .start = 0, .scale = leaf.scale },
                @intCast(local_index),
                terms,
            );
            changed += @intFromBool(out[index] != value);
        }
    }

    return changed;
}

pub fn linear(
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    weight_view: View,
    bias_view: View,
    terms: []const Term,
    out: []f32,
) Error!void {
    if (input.len == 0 or
        weight.len == 0 or
        weight.len % input.len != 0 or
        terms.len == 0)
    {
        return error.InvalidArgument;
    }
    const rows = weight.len / input.len;
    if (out.len != rows or (bias != null and bias.?.len != rows)) {
        return error.InvalidArgument;
    }
    try validateView(weight_view, weight.len);
    if (bias != null) try validateView(bias_view, rows);
    try validateTerms(terms);
    if (!hasDirection(terms)) return error.InvalidArgument;
    for (input) |value| {
        if (!std.math.isFinite(value)) return error.InvalidValue;
    }

    for (0..rows) |row| {
        var sum: f32 = 0;
        for (0..input.len) |column| {
            const index = row * input.len + column;
            const base = weight[index];
            if (!std.math.isFinite(base)) return error.InvalidValue;
            const value = try perturbUnchecked(
                base,
                weight_view,
                @intCast(index),
                terms,
            );
            sum = @mulAdd(f32, input[column], value, sum);
        }
        if (bias) |values| {
            const base = values[row];
            if (!std.math.isFinite(base)) return error.InvalidValue;
            sum += try perturbUnchecked(
                base,
                bias_view,
                @intCast(row),
                terms,
            );
        }
        if (!std.math.isFinite(sum)) return error.InvalidValue;
        out[row] = sum;
    }
}

pub fn dist2(
    leaves: []const Leaf,
    left: []const Term,
    right: []const Term,
) Error!f64 {
    if (leaves.len == 0) return error.InvalidArgument;

    var end: usize = 0;
    var energy: f64 = 0;
    for (leaves) |leaf| {
        if (leaf.len == 0 or
            !std.math.isFinite(leaf.scale) or
            leaf.scale <= 0 or
            leaf.offset != end)
        {
            return error.InvalidLayout;
        }
        end = std.math.add(usize, leaf.offset, leaf.len) catch
            return error.InvalidLayout;
        const scale = @as(f64, leaf.scale);
        energy += @as(f64, @floatFromInt(leaf.len)) * scale * scale;
    }

    try validateTerms(left);
    try validateTerms(right);

    var coefficient_distance: f64 = 0;
    for (left, 0..) |term, index| {
        if (seenSeed(left[0..index], term.seed)) continue;
        const delta = coefficient(left, term.seed) -
            coefficient(right, term.seed);
        coefficient_distance += delta * delta;
    }
    for (right, 0..) |term, index| {
        if (seenSeed(right[0..index], term.seed) or
            seenSeed(left, term.seed))
        {
            continue;
        }
        const value = coefficient(right, term.seed);
        coefficient_distance += value * value;
    }

    return energy * coefficient_distance;
}

fn validateLeaves(leaves: []const Leaf, num_values: usize) Error!void {
    var end: usize = 0;
    for (leaves) |leaf| {
        if (leaf.len == 0 or
            !std.math.isFinite(leaf.scale) or
            leaf.scale <= 0 or
            leaf.offset != end)
        {
            return error.InvalidLayout;
        }
        end = std.math.add(usize, leaf.offset, leaf.len) catch
            return error.InvalidLayout;
        if (end > num_values) return error.InvalidLayout;
    }
    if (end != num_values) return error.InvalidLayout;
}

fn validateTerms(terms: []const Term) Error!void {
    for (terms) |term| {
        if (!std.math.isFinite(term.coefficient)) return error.InvalidValue;
    }
}

fn validateView(view: View, len: usize) Error!void {
    if (!std.math.isFinite(view.scale) or view.scale <= 0) {
        return error.InvalidLayout;
    }
    _ = std.math.add(u64, view.start, @as(u64, @intCast(len))) catch
        return error.InvalidLayout;
}

fn perturbUnchecked(
    base: f32,
    view: View,
    element: u64,
    terms: []const Term,
) Error!f32 {
    const coordinate = std.math.add(u64, view.start, element) catch
        return error.InvalidLayout;
    var coefficient_sum: f32 = 0;
    var strongest: f32 = 0;
    var fallback_positive = true;
    for (terms) |term| {
        if (term.coefficient == 0) continue;
        const direction_sign = sign(term.seed, view.key, coordinate);
        coefficient_sum += term.coefficient * direction_sign;
        if (@abs(term.coefficient) > strongest) {
            strongest = @abs(term.coefficient);
            fallback_positive =
                (term.coefficient > 0) == (direction_sign > 0);
        }
    }

    const candidate = base + view.scale * coefficient_sum;
    if (coefficient_sum == 0 or candidate == base) {
        return nextFinite(base, fallback_positive);
    }
    if (!std.math.isFinite(candidate)) return error.InvalidValue;
    return candidate;
}

fn hasDirection(terms: []const Term) bool {
    for (terms, 0..) |term, index| {
        if (seenSeed(terms[0..index], term.seed)) continue;
        if (coefficient(terms, term.seed) != 0) return true;
    }
    return false;
}

fn coefficient(terms: []const Term, seed: u64) f64 {
    var value: f64 = 0;
    for (terms) |term| {
        if (term.seed == seed) value += term.coefficient;
    }
    return value;
}

fn seenSeed(terms: []const Term, seed: u64) bool {
    for (terms) |term| {
        if (term.seed == seed) return true;
    }
    return false;
}

fn nextFinite(value: f32, positive: bool) f32 {
    var bits: u32 = @bitCast(value);
    if (value == 0) {
        return @bitCast(if (positive) @as(u32, 1) else @as(u32, 0x8000_0001));
    }

    if ((value > 0) == positive) {
        bits +%= 1;
    } else {
        bits -%= 1;
    }
    const candidate: f32 = @bitCast(bits);
    if (std.math.isFinite(candidate)) return candidate;

    bits = @bitCast(value);
    if ((value > 0) == positive) {
        bits -%= 1;
    } else {
        bits +%= 1;
    }
    return @bitCast(bits);
}

fn mix64(input: u64) u64 {
    var value = input +% 0x9e37_79b9_7f4a_7c15;
    value = (value ^ (value >> 30)) *% 0xbf58_476d_1ce4_e5b9;
    value = (value ^ (value >> 27)) *% 0x94d0_49bb_1331_11eb;
    return value ^ (value >> 31);
}

test "direction signs have a stable cross-backend sequence" {
    const expected = [_]f32{
        1,  1, 1,  -1, 1, 1, 1, -1,
        -1, 1, -1, 1,  1, 1, 1, -1,
    };
    for (expected, 0..) |expected_sign, element| {
        try std.testing.expectEqual(
            expected_sign,
            sign(0x1234_5678_9abc_def0, 11, element),
        );
    }
}

test "a direction addresses billion and trillion scale coordinates" {
    try std.testing.expectEqual(
        @as(f32, -1),
        sign(7, 73, 1_334_625_279),
    );
    try std.testing.expectEqual(
        @as(f32, -1),
        sign(7, 73, 999_999_999_999),
    );
}

test "one seed perturbs every scalar in every leaf" {
    const base = [_]f32{
        0.5, -1.0, 2.0,  0.25,
        4.0, -2.0, 0.75, -0.125,
    };
    const leaves = [_]Leaf{
        .{ .key = 11, .offset = 0, .len = 4, .scale = 0.5 },
        .{ .key = 29, .offset = 4, .len = 4, .scale = 1.25 },
    };
    const terms = [_]Term{
        .{ .seed = 0x1234_5678_9abc_def0, .coefficient = 0.01 },
    };
    var out: [base.len]f32 = undefined;

    const changed = try apply(&base, &leaves, &terms, &out);

    try std.testing.expectEqual(base.len, changed);
    for (base, out) |before, after| {
        try std.testing.expect(before != after);
    }
}

test "the same description regenerates identical bits" {
    const base = [_]f32{ 0.25, -0.5, 1.0, 2.0, -4.0, 8.0 };
    const leaves = [_]Leaf{
        .{ .key = 73, .offset = 0, .len = base.len, .scale = 0.75 },
    };
    const terms = [_]Term{
        .{ .seed = 7, .coefficient = 0.02 },
        .{ .seed = 19, .coefficient = -0.005 },
    };
    var first: [base.len]f32 = undefined;
    var second: [base.len]f32 = undefined;

    _ = try apply(&base, &leaves, &terms, &first);
    _ = try apply(&base, &leaves, &terms, &second);

    try std.testing.expectEqualSlices(f32, &first, &second);
}

test "application may update a resident model in place" {
    var values = [_]f32{ 0.25, -0.5, 1.0, 2.0 };
    const before = values;
    const leaves = [_]Leaf{
        .{ .key = 37, .offset = 0, .len = values.len, .scale = 0.5 },
    };
    const terms = [_]Term{
        .{ .seed = 13, .coefficient = 0.01 },
    };

    const changed = try apply(&values, &leaves, &terms, &values);

    try std.testing.expectEqual(values.len, changed);
    for (before, values) |old, new| try std.testing.expect(old != new);
}

test "linear consumes procedural weights without materializing them" {
    const input = [_]f32{ 0.25, -0.5, 1.5, 2.0 };
    const weight = [_]f32{
        0.5,  -1.0, 0.75, 0.25,
        -0.5, 2.0,  1.25, -0.75,
    };
    const bias = [_]f32{ 0.125, -0.25 };
    const weight_view = View{ .key = 11, .start = 0, .scale = 0.02 };
    const bias_view = View{ .key = 29, .start = 0, .scale = 0.01 };
    const terms = [_]Term{
        .{ .seed = 0x1234_5678_9abc_def0, .coefficient = 0.5 },
        .{ .seed = 91, .coefficient = -0.125 },
    };
    var actual: [bias.len]f32 = undefined;
    try linear(
        &input,
        &weight,
        &bias,
        weight_view,
        bias_view,
        &terms,
        &actual,
    );

    var moved_weight: [weight.len]f32 = undefined;
    var moved_bias: [bias.len]f32 = undefined;
    for (weight, 0..) |value, index| {
        moved_weight[index] = try perturbUnchecked(
            value,
            weight_view,
            @intCast(index),
            &terms,
        );
    }
    for (bias, 0..) |value, index| {
        moved_bias[index] = try perturbUnchecked(
            value,
            bias_view,
            @intCast(index),
            &terms,
        );
    }
    for (0..bias.len) |row| {
        var expected = moved_bias[row];
        for (0..input.len) |column| {
            expected = @mulAdd(
                f32,
                input[column],
                moved_weight[row * input.len + column],
                expected,
            );
        }
        try std.testing.expectApproxEqAbs(
            expected,
            actual[row],
            2 * std.math.floatEps(f32),
        );
    }
}

test "local cancellation and rounding still change every scalar" {
    const base = [_]f32{1.0} ** 64;
    const leaves = [_]Leaf{
        .{ .key = 101, .offset = 0, .len = base.len, .scale = 1.0e-20 },
    };
    const terms = [_]Term{
        .{ .seed = 41, .coefficient = 1.0e-20 },
        .{ .seed = 97, .coefficient = 1.0e-20 },
    };
    var out: [base.len]f32 = undefined;
    var local_cancellations: usize = 0;
    for (0..base.len) |element| {
        const sum = sign(41, leaves[0].key, element) +
            sign(97, leaves[0].key, element);
        local_cancellations += @intFromBool(sum == 0);
    }

    const changed = try apply(&base, &leaves, &terms, &out);

    try std.testing.expect(local_cancellations > 0);
    try std.testing.expectEqual(base.len, changed);
    for (base, out) |before, after| {
        try std.testing.expect(before != after);
        try std.testing.expect(std.math.isFinite(after));
    }
}

test "a globally cancelled direction is rejected" {
    const base = [_]f32{ 1, 2 };
    const leaves = [_]Leaf{
        .{ .key = 1, .offset = 0, .len = base.len, .scale = 1 },
    };
    const terms = [_]Term{
        .{ .seed = 7, .coefficient = 0.5 },
        .{ .seed = 7, .coefficient = -0.5 },
    };
    var out: [base.len]f32 = undefined;

    try std.testing.expectError(
        error.InvalidArgument,
        apply(&base, &leaves, &terms, &out),
    );
}

test "expected distance follows the procedural basis coefficients" {
    const leaves = [_]Leaf{
        .{ .key = 3, .offset = 0, .len = 8, .scale = 0.5 },
        .{ .key = 5, .offset = 8, .len = 4, .scale = 2.0 },
    };
    const left = [_]Term{
        .{ .seed = 10, .coefficient = 0.5 },
        .{ .seed = 20, .coefficient = -0.25 },
    };
    const right = [_]Term{
        .{ .seed = 10, .coefficient = -0.5 },
        .{ .seed = 30, .coefficient = 0.75 },
    };
    const energy = 8.0 * 0.25 + 4.0 * 4.0;
    const coefficient_distance = 1.0 + 0.0625 + 0.5625;

    try std.testing.expectApproxEqAbs(
        energy * coefficient_distance,
        try dist2(&leaves, &left, &right),
        1e-12,
    );
}

test "leaves must cover the complete flat pytree" {
    const base = [_]f32{ 1, 2, 3, 4 };
    const leaves = [_]Leaf{
        .{ .key = 1, .offset = 1, .len = 3, .scale = 1 },
    };
    const terms = [_]Term{
        .{ .seed = 1, .coefficient = 0.1 },
    };
    var out: [base.len]f32 = undefined;

    try std.testing.expectError(
        error.InvalidLayout,
        apply(&base, &leaves, &terms, &out),
    );
}
