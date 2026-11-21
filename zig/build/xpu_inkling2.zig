//! Inkling 2 Mini on XPU: the one-card decode runner `xpu-inkling2` (needs ocloc), apart from the Qwen27 kernel build.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const ocloc = b.option([]const u8, "ink2-ocloc", "Intel offline compiler for the Inkling 2 XPU kernels") orelse "ocloc";
    const files = b.addWriteFiles();
    // the same source twice: 128 registers a thread (the default, twice the resident threads) and 256 (INK2_GRF=256)
    for ([_][2][]const u8{ .{ "ink2", "-cl-std=CL3.0" }, .{ "ink2_256", "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread" } }) |v| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
        c.addFileArg(b.path("zig/kernels/xpu/inkling2/ink2.cl"));
        c.addArgs(&.{ "-device", "bmg", "-options", v[1], "-output_no_suffix", "-output", v[0], "-out_dir" });
        const dir = c.addOutputDirectoryArg(b.fmt("ocloc-{s}", .{v[0]}));
        _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{v[0]})), b.fmt("{s}.bin", .{v[0]}));
    }
    const kernels = b.createModule(.{ .root_source_file = files.add("xpu_inkling2_kernels.zig",
        \\pub const ink2 align(8) = @embedFile("ink2.bin").*;
        \\pub const ink2_256 align(8) = @embedFile("ink2_256.bin").*;
        \\
    ) });
    const ze = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const gguf = b.createModule(.{ .root_source_file = b.path("zig/src/core/gguf.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const main = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    main.addImport("ze", ze);
    main.addImport("gguf", gguf);
    main.addImport("xpu_inkling2_kernels", kernels);
    const exe = b.addExecutable(.{ .name = "xpu-inkling2", .root_module = main });
    b.step("xpu-inkling2", "Build the Inkling 2 Mini one-card XPU decode runner (needs ocloc)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
