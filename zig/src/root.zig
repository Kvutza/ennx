const std = @import("std");

pub const dense = @import("dense.zig");

pub export fn ennx_abi_version() callconv(.c) u32 {
    return 1;
}

pub export fn ennx_dense_apply_f32(
    base_ptr: ?[*]const f32,
    num_values: usize,
    leaves_ptr: ?[*]const dense.Leaf,
    num_leaves: usize,
    terms_ptr: ?[*]const dense.Term,
    num_terms: usize,
    out_ptr: ?[*]f32,
    out_changed: ?*usize,
) callconv(.c) i32 {
    const changed = out_changed orelse return 1;
    changed.* = 0;
    if (num_values == 0 or
        num_leaves == 0 or
        num_terms == 0 or
        base_ptr == null or
        leaves_ptr == null or
        terms_ptr == null or
        out_ptr == null)
    {
        return 1;
    }

    changed.* = dense.apply(
        base_ptr.?[0..num_values],
        leaves_ptr.?[0..num_leaves],
        terms_ptr.?[0..num_terms],
        out_ptr.?[0..num_values],
    ) catch return 1;
    return 0;
}

pub export fn ennx_dense_dist2(
    leaves_ptr: ?[*]const dense.Leaf,
    num_leaves: usize,
    left_ptr: ?[*]const dense.Term,
    num_left: usize,
    right_ptr: ?[*]const dense.Term,
    num_right: usize,
    out_distance: ?*f64,
) callconv(.c) i32 {
    const distance = out_distance orelse return 1;
    distance.* = 0;
    if (num_leaves == 0 or leaves_ptr == null or
        (num_left != 0 and left_ptr == null) or
        (num_right != 0 and right_ptr == null))
    {
        return 1;
    }

    const no_terms: [0]dense.Term = .{};
    const left = if (num_left == 0)
        no_terms[0..]
    else
        left_ptr.?[0..num_left];
    const right = if (num_right == 0)
        no_terms[0..]
    else
        right_ptr.?[0..num_right];

    distance.* = dense.dist2(
        leaves_ptr.?[0..num_leaves],
        left,
        right,
    ) catch return 1;
    return 0;
}

pub export fn ennx_dense_linear_f32(
    input_ptr: ?[*]const f32,
    columns: usize,
    weight_ptr: ?[*]const f32,
    rows: usize,
    bias_ptr: ?[*]const f32,
    weight_view: dense.View,
    bias_view: dense.View,
    terms_ptr: ?[*]const dense.Term,
    num_terms: usize,
    out_ptr: ?[*]f32,
) callconv(.c) i32 {
    if (columns == 0 or
        rows == 0 or
        num_terms == 0 or
        input_ptr == null or
        weight_ptr == null or
        terms_ptr == null or
        out_ptr == null)
    {
        return 1;
    }
    const weight_len = std.math.mul(usize, rows, columns) catch return 1;
    const bias = if (bias_ptr) |values| values[0..rows] else null;
    dense.linear(
        input_ptr.?[0..columns],
        weight_ptr.?[0..weight_len],
        bias,
        weight_view,
        bias_view,
        terms_ptr.?[0..num_terms],
        out_ptr.?[0..rows],
    ) catch return 1;
    return 0;
}

test "C ABI perturbs the complete input" {
    const base = [_]f32{ 1, 2, 3, 4 };
    const leaves = [_]dense.Leaf{
        .{ .key = 9, .offset = 0, .len = base.len, .scale = 0.5 },
    };
    const terms = [_]dense.Term{
        .{ .seed = 17, .coefficient = 0.01 },
    };
    var out: [base.len]f32 = undefined;
    var changed: usize = 0;

    const status = ennx_dense_apply_f32(
        &base,
        base.len,
        &leaves,
        leaves.len,
        &terms,
        terms.len,
        &out,
        &changed,
    );

    try std.testing.expectEqual(@as(i32, 0), status);
    try std.testing.expectEqual(base.len, changed);
}

test "C ABI exposes the coefficient-space distance" {
    const leaves = [_]dense.Leaf{
        .{ .key = 9, .offset = 0, .len = 4, .scale = 0.5 },
    };
    const left = [_]dense.Term{
        .{ .seed = 17, .coefficient = 0.5 },
    };
    const right = [_]dense.Term{
        .{ .seed = 17, .coefficient = -0.5 },
    };
    var distance: f64 = 0;

    const status = ennx_dense_dist2(
        &leaves,
        leaves.len,
        &left,
        left.len,
        &right,
        right.len,
        &distance,
    );

    try std.testing.expectEqual(@as(i32, 0), status);
    try std.testing.expectEqual(@as(f64, 1), distance);
}

test "C ABI exposes fused dense linear" {
    const input = [_]f32{ 0.5, -1.0 };
    const weight = [_]f32{ 0.25, 0.75, -0.5, 1.25 };
    const terms = [_]dense.Term{
        .{ .seed = 17, .coefficient = 0.01 },
    };
    var out: [2]f32 = undefined;

    const status = ennx_dense_linear_f32(
        &input,
        input.len,
        &weight,
        out.len,
        null,
        .{ .key = 9, .start = 0, .scale = 0.5 },
        .{ .key = 0, .start = 0, .scale = 1 },
        &terms,
        terms.len,
        &out,
    );

    try std.testing.expectEqual(@as(i32, 0), status);
    try std.testing.expect(out[0] != 0 and out[1] != 0);
}
