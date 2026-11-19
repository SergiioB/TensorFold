//! Low-level XPU device plumbing for the Qwen3.8-27B runner: safetensors shards, the Level Zero device, profiling.
const std = @import("std");
const ze = @import("ze");
const safetensors = @import("safetensors.zig");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const a8 = fwd.a8;
const alloc = q.alloc;

// ---------------------------------------------------------------- safetensors (mmap through libc)

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

pub const Tensor = safetensors.Tensor;

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,
    maps: std.ArrayList([]const u8) = .empty, // the shards, mapped as long as the tensors' bytes are read
    headers: std.ArrayList(std.heap.ArenaAllocator) = .empty, // the tensors' names live in them

    pub fn addShard(c: *Checkpoint, path: [:0]const u8) !void {
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const end = lseek(fd, 0, 2);
        if (end < 8) return error.BadSafetensors;
        const size: usize = @intCast(end);
        const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed; // PROT_READ, MAP_PRIVATE
        if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed; // MAP_FAILED
        const bytes = @as([*]const u8, @ptrCast(base))[0..size];
        errdefer _ = munmap(base, size);
        try c.maps.ensureUnusedCapacity(alloc, 1);
        try c.headers.ensureUnusedCapacity(alloc, 1);
        c.headers.appendAssumeCapacity(try safetensors.parse(alloc, bytes, &c.tensors));
        c.maps.appendAssumeCapacity(bytes);
    }

    pub fn deinit(c: *Checkpoint) void {
        for (c.headers.items) |h| h.deinit();
        for (c.maps.items) |m| _ = munmap(@ptrCast(@constCast(m.ptr)), m.len);
        c.tensors.deinit(alloc);
        c.maps.deinit(alloc);
        c.headers.deinit(alloc);
        c.* = .{};
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "model.language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
    pub fn has(c: *const Checkpoint, name: []const u8) bool {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name)) return true;
        return false;
    }
};

// ---------------------------------------------------------------- device

/// A list that will not drain may still read or write host memory its callers free: the process ends (tests swap it).
pub var lost: *const fn (rc: u32) void = exitLost;

fn exitLost(rc: u32) void {
    std.log.err("XPU: a command list did not drain (Level Zero 0x{x}): its copies may still use host memory, so the process ends", .{rc});
    std.process.exit(1);
}

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    rank: u8 = 0, // tensor-parallel rank of this card: XPU_PROFILE keeps one event pool per rank
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    allocated: usize = 0, // device bytes allocated (frees not counted: buffers live as long as the runner)
    k: struct {
        embed: ze.H,
        add_rmsnorm_dec: ze.H,
        add_rmsnorm: ze.H,
        gdn_pre: ze.H,
        gdn_pre_rows: ze.H,
        gated_norm: ze.H,
        attn_prep: ze.H,
        attn_gate: ze.H,
        swiglu: ze.H,
        argmax: ze.H,
        argmax_part: ze.H,
        argmax_part_masked: ze.H,
        tap: ze.H,
        conv_commit: ze.H,
        repack: ze.H,
        gdn_step: ze.H,
        attn_decode: ze.H,
        attn_merge: ze.H,
        kv_move: ze.H,
        exl3_gemv: ze.H,
        rot_heads: ze.H,
        attn_prompt: ze.H,
        attn_prompt8: ze.H,
        embed_dense: ze.H,
        dense_rows: ze.H,
        repack_gptq: ze.H,
        quantize_mlx: ze.H,
        dec_wide_g128: ze.H,
        dec_wide64_g128: ze.H,
        dec_narrow_g128: ze.H,
        ngen_wide_g128: ze.H,
        ngen_narrow_g128: ze.H,
        ngen_swiglu: ze.H,
        ngen_swiglu_g128: ze.H,
        rowsums_ff: ze.H,
        fakequant_rows: ze.H,
        a8: ze.H,
        a8_swiglu: ze.H,
        quant_a8: ze.H,
        a4: ze.H,
        quant_a4: ze.H,
        dec_wide: ze.H,
        dec_wide64: ze.H,
        dec_narrow: ze.H,
        gdn_prep: ze.H,
        gdn_state: ze.H,
        gdn_out: ze.H,
        ngen_wide: ze.H,
        ngen_narrow: ze.H,
    },

    /// Waits for the queue, then gives the card back everything made on it (Gpu.close).
    pub fn deinit(d: *Dev) void {
        d.settle(); // the runner's host tables and staging go next: they must be idle
        d.g.close();
    }

    /// Waits for every command on the list; callers settle before host memory a queued copy reads or writes goes.
    pub fn settle(d: *Dev) void {
        const rc = d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64));
        if (rc != 0) lost(rc);
    }

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (any memory, mmap included) to the device, through the USM staging buffer.
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, null, 0, null));
            try d.sync();
            done += n;
        }
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        const ev: ze.H = if (q.prof.on) q.prof.take(d.rank, k) else null;
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }

    pub fn sync(d: *Dev) !void {
        d.settle();
        if (q.prof.on) try q.prof.drain(d);
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }
};

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
}

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// Timestamped events a rank can hold between two drains (one forward needs about 1,500 on a rank).
pub const PROF_EVENTS = 8192;
const NAMES = 80;

/// One rank's XPU_PROFILE state: its own event pool (events cannot cross cards), its kernel handles, and per kernel
/// the GPU time and the idle time on this rank's in-order list right before it (a wait on the peer shows up there).
pub const ProfRank = struct {
    events: [PROF_EVENTS]ze.H = @splat(null),
    kernel: [PROF_EVENTS]ze.H = @splat(null),
    used: usize = 0,
    dropped: usize = 0, // launches left unprofiled because the pool was full: a mid-forward drain could deadlock on a peer
    names: [NAMES]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [NAMES + 1]f64 = @splat(0), // slot NAMES: kernels with no name here (the drafter's own kernels)
    gap: [NAMES + 1]f64 = @splat(0),
    calls: [NAMES + 1]usize = @splat(0),
    span: f64 = 0, // first kernel start to last kernel end of each drain, summed

    pub fn name(p: *ProfRank, k: ze.H, label: []const u8) void {
        for (&p.names) |*n| if (n.k == null) {
            n.* = .{ .k = k, .name = label };
            return;
        };
    }

    fn slot(p: *const ProfRank, k: ze.H) usize {
        for (p.names, 0..) |n, i| if (n.k == k) return i;
        return NAMES;
    }
};

/// XPU_PROFILE=1: GPU time per kernel and rank, summed over the run.
pub const Prof = struct {
    on: bool = false,
    ranks: usize = 1,
    rank: [4]ProfRank = @splat(.{}),

    /// The event to signal for launch `k` on `rank`, or null when the pool is full (never drains mid-forward).
    pub fn take(p: *Prof, rank: u8, k: ze.H) ze.H {
        const pr = &p.rank[rank];
        if (pr.used == PROF_EVENTS) {
            pr.dropped += 1;
            return null;
        }
        const e = pr.events[pr.used];
        pr.kernel[pr.used] = k;
        pr.used += 1;
        return e;
    }

    /// Reads and resets the events of `d`'s rank; `d`'s list must be idle (Dev.sync settles it first).
    pub fn drain(p: *Prof, d: *Dev) !void {
        const pr = &p.rank[d.rank];
        const hz: f64 = @floatFromInt(d.g.props.timerResolution);
        var first: u64 = 0;
        var prev: ?u64 = null;
        for (pr.events[0..pr.used], pr.kernel[0..pr.used]) |e, k| {
            var t: ze.KernelTimestampResult = .{};
            try ze.check(d.g.api.zeEventQueryKernelTimestamp(e, &t));
            try ze.check(d.g.api.zeEventHostReset(e));
            const i = pr.slot(k);
            pr.total[i] += @as(f64, @floatFromInt(t.global_end -% t.global_start)) / hz;
            pr.calls[i] += 1;
            if (prev) |end| {
                if (t.global_start > end) pr.gap[i] += @as(f64, @floatFromInt(t.global_start - end)) / hz;
            } else first = t.global_start;
            prev = t.global_end;
        }
        if (prev) |end| pr.span += @as(f64, @floatFromInt(end -% first)) / hz;
        pr.used = 0;
    }
};

pub fn profReport(label: []const u8) void {
    if (!q.prof.on) return;
    for (q.prof.rank[0..q.prof.ranks], 0..) |*pr, rk| {
        var sum: f64 = 0;
        var idle: f64 = 0;
        for (pr.total, pr.gap) |t, g| {
            sum += t;
            idle += g;
        }
        if (q.prof.ranks == 1) {
            std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
        } else {
            std.debug.print("GPU time, {s}, rank {d}: {d:.1} ms busy, {d:.1} ms idle between kernels, {d:.1} ms span, {d} launches unprofiled\n", .{ label, rk, sum * 1e3, idle * 1e3, pr.span * 1e3, pr.dropped });
        }
        for (0..NAMES + 1) |i| {
            if (pr.calls[i] > 0) {
                const nm = if (i == NAMES) "(other)" else pr.names[i].name;
                std.debug.print("  {s:<14} {d:8.2} ms {d:5.1} %  {d:7} calls  {d:8.2} ms idle before\n", .{ nm, pr.total[i] * 1e3, pr.total[i] / sum * 100, pr.calls[i], pr.gap[i] * 1e3 });
            }
            pr.total[i] = 0;
            pr.gap[i] = 0;
            pr.calls[i] = 0;
        }
        pr.span = 0;
        pr.dropped = 0;
    }
}
