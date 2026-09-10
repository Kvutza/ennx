const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const engine = b.addModule("ennx", .{
        .root_source_file = b.path("zig/src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const static = b.addLibrary(.{
        .name = "ennx",
        .root_module = engine,
    });
    b.installArtifact(static);

    const shared = b.addLibrary(.{
        .name = "ennx",
        .linkage = .dynamic,
        .root_module = engine,
    });
    b.installArtifact(shared);

    const header = b.addInstallHeaderFile(
        b.path("zig/include/ennx.h"),
        "ennx.h",
    );
    b.getInstallStep().dependOn(&header.step);

    const tests = b.addTest(.{
        .root_module = engine,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run ENNX Zig tests");
    test_step.dependOn(&run_tests.step);
}
