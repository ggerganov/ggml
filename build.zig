const std = @import("std");
const builtin = @import("builtin");

// Zig Version: 0.11.0
// Zig Build Command: zig build
// Zig Run Command: zig build -h
//     zig build run_gpt-j
//     zig build run_mnist
//     zig build run_magika
//     zig build run_test-grad0
//     zig build run_test-mul-mat0
//     zig build run_test-mul-mat2
//     zig build run_test-opt
//     zig build run_test-vec1
//     zig build run_test0
//     zig build run_test1
//     zig build run_test2
//     zig build run_test3
//     zig build run_zig_test0
//     zig build run_zig_test1
//     zig build run_zig_test2
//     zig build run_zig_test3
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib = b.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });
    lib.addIncludePath(b.path("./include"));
    lib.addIncludePath(b.path("./include/ggml"));
    lib.addCSourceFiles(.{ .files = &.{
        "src/ggml.c",
        "src/ggml-alloc.c",
        "src/ggml-backend.c",
        "src/ggml-quants.c",
    }, .flags = &.{
        "-std=c11",
        "-D_GNU_SOURCE",
        "-D_XOPEN_SOURCE=600",
    } });
    lib.linkLibC();
    lib.linkLibCpp();
    b.installArtifact(lib);

    // examples
    const examples = .{
        "gpt-j",
        "magika",
        "mnist",
        // "whisper",
    };
    inline for (examples) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath(b.path("./include"));
        exe.addIncludePath(b.path("./include/ggml"));
        exe.addIncludePath(b.path("./examples"));
        // exe.addIncludePath("./examples/whisper");
        exe.addCSourceFiles(.{
            .files = &.{
                std.fmt.comptimePrint("examples/{s}/main.cpp", .{name}),
                "examples/common.cpp",
                "examples/common-ggml.cpp",
                // "examples/whisper/whisper.cpp",
            },
            .flags = &.{"-std=c++11"},
        });
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_" ++ name, "Run examples");
        run_step.dependOn(&run_cmd.step);
    }

    // tests
    const tests = if (builtin.target.cpu.arch == .x86_64) .{
        // "test-blas0",
        // "test-grad0",
        "test-mul-mat0",
        // "test-mul-mat1",
        "test-mul-mat2",
        // "test-opt",
        // "test-svd0",
        "test-vec0",
        "test-vec1",
        // "test-vec2",
        "test0",
        "test1",
        "test2",
        "test3",
    } else .{
        // "test-blas0",
        // "test-grad0",
        "test-mul-mat0",
        // "test-mul-mat1",
        "test-mul-mat2",
        // "test-opt",
        // "test-svd0",
        // "test-vec0",
        // "test-vec1",
        // "test-vec2",
        "test0",
        "test1",
        "test2",
        "test3",
    };
    inline for (tests) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath(b.path("./include"));
        exe.addIncludePath(b.path("./include/ggml"));
        exe.addCSourceFiles(.{ .files = &.{
            std.fmt.comptimePrint("tests/{s}.c", .{name}),
        }, .flags = &.{
            "-std=c11",
        } });
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_" ++ name, "Run tests");
        run_step.dependOn(&run_cmd.step);
    }

    // zig_tests
    const zig_tests = .{
        "test0",
        "test1",
        "test2",
        "test3",
    };
    inline for (zig_tests) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_source_file = b.path(std.fmt.comptimePrint("tests/{s}.zig", .{name})),
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath(b.path("./include"));
        exe.addIncludePath(b.path("./include/ggml"));
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_zig_" ++ name, "Run zig_tests");
        run_step.dependOn(&run_cmd.step);
    }
}
