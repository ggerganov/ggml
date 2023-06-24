const std = @import("std");

// Zig Version: 0.11.0-dev.3379+629f0d23b
pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib = b.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    lib.linkLibCpp();
    lib.addIncludePath(".");
    lib.addIncludePath("./include");
    lib.addIncludePath("./include/ggml");
    lib.addIncludePath("./examples");
    lib.addCSourceFiles(&.{
        "src/ggml.c",
    }, &.{"-std=c11"});
    b.installArtifact(lib);

    const examples = .{
        "dolly-v2",
        "gpt-2",
        "gpt-j",
        "gpt-neox",
        "mnist",
        "mpt",
        "replit",
        "starcoder",
    };

    inline for (examples) |example_name| {
        const exe = b.addExecutable(.{
            .name = example_name,
            .target = target,
            .optimize = optimize,
        });
        exe.addIncludePath(".");
        exe.addIncludePath("./include");
        exe.addIncludePath("./include/ggml");
        exe.addIncludePath("./examples");
        exe.addCSourceFiles(&.{
            std.fmt.comptimePrint("examples/{s}/{s}.cpp", .{example_name, "main"}),
            "examples/common.cpp",
            "examples/common-ggml.cpp",
        }, &.{"-std=c++11"});
        exe.linkLibrary(lib);
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("run_" ++ example_name, "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}