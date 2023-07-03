const std = @import("std");

// Zig Version: 0.11.0-dev.3886+0c1bfe271
// Zig Build Command: zig build
// Zig Run Command: zig build -h
//     zig build run_dolly-v2
//     zig build run_gpt-2
//     zig build run_gpt-j
//     zig build run_gpt-neox
//     zig build run_mnist
//     zig build run_mpt
//     zig build run_replit
//     zig build run_starcoder
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
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib = b.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });
    lib.addIncludePath("./include");
    lib.addIncludePath("./include/ggml");
    lib.addCSourceFiles(&.{
        "src/ggml.c",
    }, &.{"-std=c11"});
    lib.linkLibC();
    lib.linkLibCpp();
    b.installArtifact(lib);

    // examples
    const examples = .{
        "dolly-v2",
        "gpt-2",
        "gpt-j",
        "gpt-neox",
        "mnist",
        "mpt",
        "replit",
        "starcoder",
        // "whisper",
    };
    inline for (examples) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath("./include");
        exe.addIncludePath("./include/ggml");
        exe.addIncludePath("./examples");
        // exe.addIncludePath("./examples/whisper");
        exe.addCSourceFiles(&.{
            std.fmt.comptimePrint("examples/{s}/main.cpp", .{name}),
            "examples/common.cpp",
            "examples/common-ggml.cpp",
            // "examples/whisper/whisper.cpp",
        }, &.{"-std=c++11"});
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_" ++ name, "Run examples");
        run_step.dependOn(&run_cmd.step);
    }

    // tests
    const tests = .{
        // "test-blas0",
        "test-grad0",
        "test-mul-mat0",
        // "test-mul-mat1",
        "test-mul-mat2",
        "test-opt",
        // "test-svd0",
        // "test-vec0",
        "test-vec1",
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
        exe.addIncludePath("./include");
        exe.addIncludePath("./include/ggml");
        exe.addCSourceFiles(&.{
            std.fmt.comptimePrint("tests/{s}.c", .{name}),
        }, &.{"-std=c11"});
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
            .root_source_file = .{ .path = std.fmt.comptimePrint("tests/{s}.zig", .{name}) },
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath("./include");
        exe.addIncludePath("./include/ggml");
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_zig_" ++ name, "Run zig_tests");
        run_step.dependOn(&run_cmd.step);
    }
}
