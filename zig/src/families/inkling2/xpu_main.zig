//! Inkling 2 Mini (inkling2_loop) one-card XPU decode from a llama.cpp GGUF: greedy tokens, timings, per-visit dumps.
//!   xpu-inkling2 MODEL.gguf IDS.json N_NEW [CTX]
//! INK2_DUMP=dir writes each visit's output row (`v<visit>_p<pos>.f32`) and the logits of each step for the host check.
const std = @import("std");
const ze = @import("ze");
const gguf = @import("gguf");
const kbin = @import("xpu_inkling2_kernels");

const alloc = std.heap.c_allocator;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;

fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

fn mapFile(path: []const u8) ![]const u8 {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = open(z.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    const end = lseek(fd, 0, 2);
    if (end <= 0) return error.EmptyFile;
    const size: usize = @intCast(end);
    const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed;
    if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed;
    return @as([*]const u8, @ptrCast(base))[0..size];
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = creat(z.ptr, 0o644);
    if (fd < 0) return error.CannotCreate;
    defer _ = close(fd);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.ShortWrite;
        done += @intCast(n);
    }
}

// ---------------------------------------------------------------- the model's fixed shape (checked against the GGUF)
const C = 2304;
const HD = 128;
const NH = 24;
const NKV = 8;
const NR = 16;
const FF = 1152;
const NE = 128;
const TOP = 8;
const LAYERS = 24;
const VISITS = 48;
const CORE0 = 6;
const CORE1 = 18;
const PASSES = 3;
const BLOCK = 6;
const EXTENT = 1024;
const WINDOW = 1024;
const HIST = 6656;
const PROWS = 3072 + 384 + 3072 + 1024 + 1024; // q | rel | gate | k | v
const MAXS = 8;
const SLOTS = 32;
const TMAX = 64; // prompt rows a chunk
const RING = WINDOW + TMAX; // sliding caches: a chunk never overwrites keys its rows still read

fn layerOf(il: usize) usize {
    if (il < CORE0) return il;
    if (il < CORE0 + PASSES * (CORE1 - CORE0)) return CORE0 + (il - CORE0) % (CORE1 - CORE0);
    return il - (PASSES - 1) * (CORE1 - CORE0);
}

fn isGlobal(ip: usize) bool {
    return (ip + 1) % 6 == 0;
}

// ---------------------------------------------------------------- device
/// INK2_PROFILE=1: GPU time of every launch from timestamped events, summed per kernel.
const Prof = struct {
    events: [4096]ze.H = @splat(null),
    kernel: [4096]ze.H = @splat(null),
    used: usize = 0,
    names: [32]ze.H = @splat(null),
    total: [32]f64 = @splat(0),
    count: [32]u64 = @splat(0),
};

const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{},
    bytes: usize = 0,
    prof: ?*Prof = null,

    fn alloc_(d: *Dev, n: usize) !*anyopaque {
        d.bytes += n;
        return d.g.deviceAlloc(@max(n, 64));
    }

    fn sync(d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)));
        if (d.prof) |p| {
            for (p.events[0..p.used], p.kernel[0..p.used]) |e, k| {
                const t = try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                for (&p.names, 0..) |*n, i| {
                    if (n.* == null) n.* = k;
                    if (n.* == k) {
                        p.total[i] += t;
                        p.count[i] += 1;
                        break;
                    }
                }
            }
            p.used = 0;
        }
    }

    fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
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

    fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    fn read(d: *Dev, dst: []u8, src: *anyopaque) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, d.staging.ptr, src, dst.len, null, 0, null));
        try d.sync();
        @memcpy(dst, d.staging[0..dst.len]);
    }

    fn zero(d: *Dev, p: *anyopaque, n: usize) !void {
        const z = try alloc.alloc(u8, n);
        defer alloc.free(z);
        @memset(z, 0);
        try d.copyIn(p, z);
    }

    fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var ev: ze.H = null;
        if (d.prof) |p| {
            if (p.used == p.events.len) try d.sync();
            ev = p.events[p.used];
            p.kernel[p.used] = k;
            p.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }
};

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

// ---------------------------------------------------------------- host repacks
fn f16bits(b: []const u8) u16 {
    return std.mem.readInt(u16, b[0..2], .little);
}

/// Q8_0 blocks (fp16 d, 32 int8) to int8 rows and fp16 scales.
fn repackQ8(src: []const u8, qs: []i8, sc: []u16) void {
    const nb = src.len / 34;
    for (0..nb) |i| {
        const blk = src[i * 34 ..][0..34];
        sc[i] = f16bits(blk[0..2]);
        @memcpy(std.mem.sliceAsBytes(qs[i * 32 ..][0..32]), blk[2..34]);
    }
}

/// NVFP4 blocks (4 UE4M3 scales, 32 bytes for 4 sub-blocks of 16) to per-sub-block 8 nibble bytes and one scale byte.
fn repackNvfp4(src: []const u8, qs: []u8, sc: []u8) void {
    const nb = src.len / 36;
    for (0..nb) |i| {
        const blk = src[i * 36 ..][0..36];
        for (0..4) |s| {
            sc[i * 4 + s] = blk[s];
            @memcpy(qs[(i * 4 + s) * 8 ..][0..8], blk[4 + s * 8 ..][0..8]);
        }
    }
}

fn bf16ToF32(src: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) << 16);
}

fn ue4m3(x: u8) f32 {
    if (x == 0 or x == 0x7F) return 0;
    const e: i32 = (x >> 3) & 0xF;
    const m: f32 = @floatFromInt(x & 7);
    const raw = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(1.0 + m / 8.0, e - 7);
    return raw * 0.5;
}

// ---------------------------------------------------------------- weights
const Q8 = struct { q: *anyopaque, s: *anyopaque, rows: usize, k: usize };
const Experts = struct { nvfp4: bool, gu_q: *anyopaque, gu_s: *anyopaque, d_q: *anyopaque, d_s: *anyopaque, gu_scale: ?*anyopaque = null, d_scale: ?*anyopaque = null };

const Layer = struct {
    big: Q8, // q | rel | gate | k | v
    wo: Q8,
    small: *anyopaque, // fp32 [sink 24 | kdyn 32 | vdyn 32]
    small_rows: u32,
    kconv: *anyopaque,
    vconv: *anyopaque,
    aconv: *anyopaque,
    fconv: *anyopaque,
    qclip: *anyopaque,
    rclip: *anyopaque,
    relb: *anyopaque,
    norms: *anyopaque, // [3][C] on global layers, [C] on sliding ones
    attn_q: ?*anyopaque, // [nq][C], none on layer 0
    ffn_q: *anyopaque,
    ffn_norm: *anyopaque,
    router: *anyopaque,
    probs_b: *anyopaque,
    exp: Experts,
    loop_down: [4]?*anyopaque = @splat(null), // views into the stacked [4] tensors below
    loop_gain: [4]?*anyopaque = @splat(null),
    loop_forget: [4]?*anyopaque = @splat(null),
    loop_down4: ?*anyopaque = null, // [attn, k, v, mlp][64][2C]
    loop_gain4: ?*anyopaque = null, // [4][C][64]
    loop_forget4: ?*anyopaque = null,
};

/// fp32 tensors of equal size stacked into one device buffer.
fn uploadStack(d: *Dev, parts: []const gguf.Tensor) !*anyopaque {
    const n = parts[0].bytes.len;
    const buf = try alloc.alloc(u8, n * parts.len);
    defer alloc.free(buf);
    for (parts, 0..) |p, i| {
        if (p.kind != .f32 or p.bytes.len != n) return error.UnexpectedTensorType;
        @memcpy(buf[i * n .. (i + 1) * n], p.bytes);
    }
    return d.upload(buf);
}

const Model = struct {
    doc: gguf.Document,
    embed: Q8,
    embed_norm: *anyopaque,
    out: Q8,
    out_norm: *anyopaque,
    out_q: *anyopaque,
    loop_q: *anyopaque,
    loop_kv_q: *anyopaque,
    layers: [LAYERS]Layer,
    lut: *anyopaque,
    lut16: *anyopaque, // lut x 2^15 for the fp16-field NVFP4 decode
    eps: f32,
    vocab: usize,

    fn t(m: *const Model, comptime fmt: []const u8, args: anytype) !gguf.Tensor {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return m.doc.getTensor(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
    }
};

fn uploadF32(d: *Dev, t: gguf.Tensor) !*anyopaque {
    return switch (t.kind) {
        .f32 => d.upload(t.bytes),
        .bf16 => blk: {
            const out = try alloc.alloc(f32, t.bytes.len / 2);
            defer alloc.free(out);
            bf16ToF32(t.bytes, out);
            break :blk d.upload(std.mem.sliceAsBytes(out));
        },
        else => error.UnexpectedTensorType,
    };
}

/// Q8_0 tensors stacked by rows (all with the same row width) into one int8 matrix and its scales.
fn uploadQ8(d: *Dev, parts: []const gguf.Tensor) !Q8 {
    const k: usize = @intCast(parts[0].shape[0]);
    var rows: usize = 0;
    for (parts) |p| {
        if (p.kind != .q8_0 or p.shape[0] != k) return error.UnexpectedTensorType;
        rows += p.bytes.len / 34 * 32 / k;
    }
    const qs = try alloc.alloc(i8, rows * k);
    defer alloc.free(qs);
    const sc = try alloc.alloc(u16, rows * k / 32);
    defer alloc.free(sc);
    var at: usize = 0;
    for (parts) |p| {
        const n = p.bytes.len / 34 * 32;
        repackQ8(p.bytes, qs[at .. at + n], sc[at / 32 .. (at + n) / 32]);
        at += n;
    }
    return .{ .q = try d.upload(std.mem.sliceAsBytes(qs)), .s = try d.upload(std.mem.sliceAsBytes(sc)), .rows = rows, .k = k };
}

fn uploadExperts(d: *Dev, m: *const Model, ip: usize) !Experts {
    const gu = try m.t("blk.{d}.ffn_gate_up_exps.weight", .{ip});
    const dn = try m.t("blk.{d}.ffn_down_exps.weight", .{ip});
    if (gu.kind != dn.kind) return error.UnexpectedTensorType;
    if (gu.kind == .q8_0) {
        const a = try uploadQ8(d, &.{gu});
        const b = try uploadQ8(d, &.{dn});
        return .{ .nvfp4 = false, .gu_q = a.q, .gu_s = a.s, .d_q = b.q, .d_s = b.s };
    }
    if (gu.kind != .nvfp4) return error.UnexpectedTensorType;
    var e: Experts = .{ .nvfp4 = true, .gu_q = undefined, .gu_s = undefined, .d_q = undefined, .d_s = undefined };
    inline for (.{ .{ gu, "gu" }, .{ dn, "d" } }) |pair| {
        const src = pair[0];
        const nb = src.bytes.len / 36;
        const qs = try alloc.alloc(u8, nb * 32);
        defer alloc.free(qs);
        const sc = try alloc.alloc(u8, nb * 4);
        defer alloc.free(sc);
        repackNvfp4(src.bytes, qs, sc);
        @field(e, pair[1] ++ "_q") = try d.upload(qs);
        @field(e, pair[1] ++ "_s") = try d.upload(sc);
    }
    e.gu_scale = try uploadF32(d, try m.t("blk.{d}.ffn_gate_up_exps.scale", .{ip}));
    e.d_scale = try uploadF32(d, try m.t("blk.{d}.ffn_down_exps.scale", .{ip}));
    return e;
}

fn metaInt(doc: *const gguf.Document, key: []const u8) !u64 {
    return (doc.get(key) orelse return error.MissingMetadata).unsigned();
}

fn checkShape(doc: *const gguf.Document) !void {
    const want = [_]struct { []const u8, u64 }{
        .{ "inkling2.block_count", VISITS },       .{ "inkling2.embedding_length", C },
        .{ "inkling2.expert_count", NE },          .{ "inkling2.expert_used_count", TOP },
        .{ "inkling2.expert_feed_forward_length", FF }, .{ "inkling2.num_loops", PASSES },
        .{ "inkling2.loop.core_start", CORE0 },    .{ "inkling2.loop.core_end", CORE1 },
        .{ "inkling2.attn_res.block_size", BLOCK }, .{ "inkling2.attention.relative_length", NR },
        .{ "inkling2.attention.relative_extent", EXTENT }, .{ "inkling2.attention.sliding_window", WINDOW },
        .{ "inkling2.shortconv.l_cache", 4 },      .{ "inkling2.attention.key_length", HD },
    };
    for (want) |w| if (try metaInt(doc, w[0]) != w[1]) {
        std.debug.print("{s} is not {d}: this runner is built for Inkling 2 Mini\n", .{ w[0], w[1] });
        return error.UnsupportedModel;
    };
}

fn metaF32(doc: *const gguf.Document, key: []const u8) !f32 {
    return @floatCast(try (doc.get(key) orelse return error.MissingMetadata).float());
}

fn load(d: *Dev, path: []const u8) !Model {
    const bytes = try mapFile(path);
    var m: Model = undefined;
    m.doc = try gguf.Document.parse(alloc, bytes, .{});
    try checkShape(&m.doc);
    m.eps = try metaF32(&m.doc, "inkling2.attention.layer_norm_rms_epsilon");
    m.vocab = try metaInt(&m.doc, "inkling2.vocab_size_unpadded");
    m.embed = try uploadQ8(d, &.{try m.t("token_embd.weight", .{})});
    m.embed_norm = try uploadF32(d, try m.t("token_embd_norm.weight", .{}));
    m.out = try uploadQ8(d, &.{try m.t("output.weight", .{})});
    m.out_norm = try uploadF32(d, try m.t("output_norm.weight", .{}));
    m.out_q = try uploadF32(d, try m.t("output_res_query.weight", .{}));
    m.loop_q = try uploadF32(d, try m.t("loop_res_query.weight", .{}));
    m.loop_kv_q = try uploadF32(d, try m.t("loop_kv_res_query.weight", .{}));
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*v, i| v.* = ue4m3(@intCast(i));
    m.lut = try d.upload(std.mem.sliceAsBytes(&lut));
    for (&lut) |*v| v.* *= 32768.0;
    m.lut16 = try d.upload(std.mem.sliceAsBytes(&lut));
    for (0..LAYERS) |ip| {
        const g = isGlobal(ip);
        var l: Layer = undefined;
        l.big = try uploadQ8(d, &.{
            try m.t("blk.{d}.attn_q.weight", .{ip}),    try m.t("blk.{d}.attn_rel.weight", .{ip}),
            try m.t("blk.{d}.attn_gate.weight", .{ip}), try m.t("blk.{d}.attn_k.weight", .{ip}),
            try m.t("blk.{d}.attn_v.weight", .{ip}),
        });
        if (l.big.rows != PROWS) return error.UnexpectedShape;
        l.wo = try uploadQ8(d, &.{try m.t("blk.{d}.attn_output.weight", .{ip})});
        {
            const sink = try m.t("blk.{d}.attn_sink_proj.weight", .{ip});
            var rows: usize = NH;
            const buf = try alloc.alloc(f32, 88 * C);
            defer alloc.free(buf);
            bf16ToF32(sink.bytes, buf[0 .. NH * C]);
            if (g) {
                bf16ToF32((try m.t("blk.{d}.attn_k_conv_proj.weight", .{ip})).bytes, buf[24 * C .. 56 * C]);
                bf16ToF32((try m.t("blk.{d}.attn_v_conv_proj.weight", .{ip})).bytes, buf[56 * C .. 88 * C]);
                rows = 88;
            }
            l.small = try d.upload(std.mem.sliceAsBytes(buf[0 .. rows * C]));
            l.small_rows = @intCast(rows);
        }
        l.kconv = try uploadF32(d, try m.t("blk.{d}.attn_k_conv.weight", .{ip}));
        l.vconv = try uploadF32(d, try m.t("blk.{d}.attn_v_conv.weight", .{ip}));
        l.aconv = try uploadF32(d, try m.t("blk.{d}.attn_post_conv.weight", .{ip}));
        l.fconv = try uploadF32(d, try m.t("blk.{d}.ffn_post_conv.weight", .{ip}));
        l.qclip = try uploadF32(d, try m.t("blk.{d}.attn_q_clip.weight", .{ip}));
        l.rclip = try uploadF32(d, try m.t("blk.{d}.attn_rel_clip.weight", .{ip}));
        l.relb = try uploadF32(d, try m.t("blk.{d}.attn_rel_bias.weight", .{ip}));
        if (g) {
            var buf: [3 * C]f32 = undefined;
            for ([_][]const u8{ "q", "k", "v" }, 0..) |n, j| {
                const tt = try m.t("blk.{d}.attn_{s}_inp_norm.weight", .{ ip, n });
                @memcpy(std.mem.sliceAsBytes(buf[j * C .. (j + 1) * C]), tt.bytes);
            }
            l.norms = try d.upload(std.mem.sliceAsBytes(&buf));
        } else l.norms = try uploadF32(d, try m.t("blk.{d}.attn_norm.weight", .{ip}));
        l.attn_q = if (ip > 0) try uploadF32(d, try m.t("blk.{d}.attn_res_query.weight", .{ip})) else null;
        l.ffn_q = try uploadF32(d, try m.t("blk.{d}.ffn_res_query.weight", .{ip}));
        l.ffn_norm = try uploadF32(d, try m.t("blk.{d}.ffn_norm.weight", .{ip}));
        l.router = try uploadF32(d, try m.t("blk.{d}.ffn_gate_inp.weight", .{ip}));
        l.probs_b = try uploadF32(d, try m.t("blk.{d}.exp_probs_b.bias", .{ip}));
        l.exp = try uploadExperts(d, &m, ip);
        l.loop_down = @splat(null);
        l.loop_gain = @splat(null);
        l.loop_forget = @splat(null);
        l.loop_down4 = null;
        l.loop_gain4 = null;
        l.loop_forget4 = null;
        if (ip >= CORE0 and ip < CORE1) {
            var dn: [4]gguf.Tensor = undefined;
            var gn: [4]gguf.Tensor = undefined;
            var fg: [4]gguf.Tensor = undefined;
            for ([_][]const u8{ "attn", "k", "v", "mlp" }, 0..) |n, j| {
                dn[j] = try m.t("blk.{d}.loop_{s}_down.weight", .{ ip, n });
                gn[j] = try m.t("blk.{d}.loop_{s}_gain.weight", .{ ip, n });
                fg[j] = try m.t("blk.{d}.loop_{s}_forget.weight", .{ ip, n });
            }
            l.loop_down4 = try uploadStack(d, &dn);
            l.loop_gain4 = try uploadStack(d, &gn);
            l.loop_forget4 = try uploadStack(d, &fg);
            for (0..4) |j| {
                l.loop_down[j] = off(l.loop_down4.?, j * 64 * 2 * C * 4);
                l.loop_gain[j] = off(l.loop_gain4.?, j * C * 64 * 4);
                l.loop_forget[j] = off(l.loop_forget4.?, j * C * 64 * 4);
            }
        }
        m.layers[ip] = l;
        std.debug.print("\rloaded layer {d}/{d}, {d:.2} GB on the card", .{ ip + 1, LAYERS, @as(f64, @floatFromInt(d.bytes)) / 1e9 });
    }
    std.debug.print("\n", .{});
    return m;
}

// ---------------------------------------------------------------- kernels and activations
const K = struct {
    embed: ze.H,
    mix: ze.H,
    norm: ze.H,
    ctl_in: ze.H,
    ctl_out: ze.H,
    gemv_f32: ze.H,
    gemv_q8: ze.H,
    prep: ze.H,
    attn: ze.H,
    conv_res: ze.H,
    topk: ze.H,
    gateup_nvfp4: ze.H,
    down_nvfp4: ze.H,
    gateup_q8: ze.H,
    down_q8: ze.H,
    argmax_part: ze.H,
    argmax_final: ze.H,
    mix3: ze.H,
    ctl_down: ze.H,
    ctl_out2: ze.H,
    gemv_big: ze.H,
    attn2: ze.H,
    gemv_conv: ze.H,
    router: ze.H,
    gateup_nvfp4b: ze.H,
    down_nvfp4_conv: ze.H,
    down_q8_conv: ze.H,
    gemv_q8n: ze.H,
    sample_part: ze.H,
    gateup_nvfp4c: ze.H,
    down_nvfp4c: ze.H,
    mix3n: ze.H,
    gemv_bigx: ze.H,
    embed_b: ze.H,
    mix3_b: ze.H,
    norm_b: ze.H,
    ctl_down_b: ze.H,
    ctl_out_b: ze.H,
    gemv_q8_b: ze.H,
    gemv_f32_b: ze.H,
    prep_b: ze.H,
    attn_b: ze.H,
    conv_res_b: ze.H,
    topk_b: ze.H,
    route_lists: ze.H,
    gateup_nvfp4_b: ze.H,
    gateup_q8_b: ze.H,
    down_nvfp4_b: ze.H,
    down_q8_b: ze.H,
    combine_b: ze.H,
    attn_split: ze.H,
    attn_merge: ze.H,
};

const Slots = extern struct { n: i32, idx: [MAXS]i32 };
const Prep = extern struct { qk_clip: f32, rel_clip: f32, eps: f32, tau: f32, global_layer: i32, slot: i32 };
const Span = extern struct { pos: i32, p0: i32, ring: i32, extent: i32 };

const Acts = struct {
    S: *anyopaque, // [SLOTS][C]
    mix: *anyopaque, // [3][C]
    msg_a: *anyopaque, // [2][C]
    msg_k: *anyopaque, // [2][C]
    inp: *anyopaque, // [3][C] after the controllers
    innorm: *anyopaque, // [3][C]
    p: *anyopaque,
    dsm: *anyopaque,
    q: *anyopaque,
    r: *anyopaque,
    o: *anyopaque,
    y: *anyopaque,
    hin: *anyopaque,
    hdown: *anyopaque,
    rlog: *anyopaque,
    ids: *anyopaque,
    wts: *anyopaque,
    act: *anyopaque,
    logits: *anyopaque,
    pv: *anyopaque,
    pi: *anyopaque,
    tok: *anyopaque,
    scratch: *anyopaque,
    hist: *anyopaque, // [VISITS][3][HIST]
    hist2: *anyopaque, // [VISITS][2][3][HIST]: v2 reads [pos % 2], writes the other
    msg4: *anyopaque, // [attn, k, v, mlp][C]
    hd: *anyopaque, // [3][64]
    xn: *anyopaque,
    counter: *anyopaque,
    // prompt chunks: [TMAX] rows each
    Sb: *anyopaque, // [SLOTS][TMAX][C]
    ids_b: *anyopaque,
    mix_b: *anyopaque, // [4][TMAX][C]
    msg_b: *anyopaque, // [4][TMAX][C]
    inp_b: *anyopaque, // [3][TMAX][C]
    xn_b: *anyopaque, // [3][TMAX][C]
    hd_b: *anyopaque, // [3][TMAX][64]
    p_b: *anyopaque, // [TMAX][PROWS]
    d_b: *anyopaque, // [TMAX][88]
    q_b: *anyopaque,
    r_b: *anyopaque,
    o_b: *anyopaque,
    y_b: *anyopaque,
    rlog_b: *anyopaque,
    ids8_b: *anyopaque,
    wts_b: *anyopaque,
    start_b: *anyopaque,
    list_b: *anyopaque,
    act_b: *anyopaque, // [TMAX * TOP][FF]
    ye_b: *anyopaque, // [TMAX * TOP][C]
    scratch_b: *anyopaque, // [TMAX][24][ctx]
    part_o: *anyopaque, // [24][ctx / 256][128]
    part_ml: *anyopaque, // [24][ctx / 256][2]
    kc: [VISITS]*anyopaque,
    vc: [VISITS]*anyopaque,
    ring: [VISITS]u32,
};

const Runner = struct {
    d: *Dev,
    m: *Model,
    k: K,
    a: Acts,
    ctx: usize,
    qk_clip: f32,
    rel_clip: f32,
    floor: f32,
    alpha: f32,
    situ_b: f32,
    situ_lb: f32,
    wscale: f32,
    cap: f32,
    dump: ?[]const u8,
    v1: bool,
    onerow: bool, // INK2_ONEROW=1: prompts through the one-row decode path
    expk_b: bool, // INK2_EXPK=b: the version-b expert kernels
    pro: bool, // INK2_PRO=1: norms inside the GEMV prologues (the first v2 layout)
    split_at: usize = 2048, // global visits with more keys take the chunked attention (INK2_SPLIT=N)
    temp: f32 = 0, // sampling temperature of the request; 0: greedy
    seed: u64 = 0,

    fn kernels(d: *Dev) !K {
        const g256 = if (getenv("INK2_GRF")) |v| std.mem.eql(u8, std.mem.span(v), "256") else false;
        const bin: []const u8 = if (g256) kbin.ink2_256[0..] else kbin.ink2[0..];
        const mod = try d.g.module(bin);
        return .{
            .embed = try d.g.kernel(mod, "ink2_embed"),
            .mix = try d.g.kernel(mod, "ink2_mix"),
            .norm = try d.g.kernel(mod, "ink2_norm"),
            .ctl_in = try d.g.kernel(mod, "ink2_ctl_in"),
            .ctl_out = try d.g.kernel(mod, "ink2_ctl_out"),
            .gemv_f32 = try d.g.kernel(mod, "ink2_gemv_f32"),
            .gemv_q8 = try d.g.kernel(mod, "ink2_gemv_q8"),
            .prep = try d.g.kernel(mod, "ink2_prep"),
            .attn = try d.g.kernel(mod, "ink2_attn"),
            .conv_res = try d.g.kernel(mod, "ink2_conv_res"),
            .topk = try d.g.kernel(mod, "ink2_topk"),
            .gateup_nvfp4 = try d.g.kernel(mod, "ink2_gateup_nvfp4"),
            .down_nvfp4 = try d.g.kernel(mod, "ink2_down_nvfp4"),
            .gateup_q8 = try d.g.kernel(mod, "ink2_gateup_q8"),
            .down_q8 = try d.g.kernel(mod, "ink2_down_q8"),
            .argmax_part = try d.g.kernel(mod, "ink2_softcap_argmax_part"),
            .argmax_final = try d.g.kernel(mod, "ink2_argmax_final"),
            .mix3 = try d.g.kernel(mod, "ink2_mix3"),
            .ctl_down = try d.g.kernel(mod, "ink2_ctl_down"),
            .ctl_out2 = try d.g.kernel(mod, "ink2_ctl_out2"),
            .gemv_big = try d.g.kernel(mod, "ink2_gemv_big"),
            .attn2 = try d.g.kernel(mod, "ink2_attn2"),
            .gemv_conv = try d.g.kernel(mod, "ink2_gemv_conv"),
            .router = try d.g.kernel(mod, "ink2_router"),
            .gateup_nvfp4b = try d.g.kernel(mod, "ink2_gateup_nvfp4b"),
            .down_nvfp4_conv = try d.g.kernel(mod, "ink2_down_nvfp4_conv"),
            .down_q8_conv = try d.g.kernel(mod, "ink2_down_q8_conv"),
            .gemv_q8n = try d.g.kernel(mod, "ink2_gemv_q8n"),
            .sample_part = try d.g.kernel(mod, "ink2_sample_part"),
            .gateup_nvfp4c = try d.g.kernel(mod, "ink2_gateup_nvfp4c"),
            .down_nvfp4c = try d.g.kernel(mod, "ink2_down_nvfp4c"),
            .mix3n = try d.g.kernel(mod, "ink2_mix3n"),
            .gemv_bigx = try d.g.kernel(mod, "ink2_gemv_bigx"),
            .embed_b = try d.g.kernel(mod, "ink2_embed_b"),
            .mix3_b = try d.g.kernel(mod, "ink2_mix3_b"),
            .norm_b = try d.g.kernel(mod, "ink2_norm_b"),
            .ctl_down_b = try d.g.kernel(mod, "ink2_ctl_down_b"),
            .ctl_out_b = try d.g.kernel(mod, "ink2_ctl_out_b"),
            .gemv_q8_b = try d.g.kernel(mod, "ink2_gemv_q8_b"),
            .gemv_f32_b = try d.g.kernel(mod, "ink2_gemv_f32_b"),
            .prep_b = try d.g.kernel(mod, "ink2_prep_b"),
            .attn_b = try d.g.kernel(mod, "ink2_attn_b"),
            .conv_res_b = try d.g.kernel(mod, "ink2_conv_res_b"),
            .topk_b = try d.g.kernel(mod, "ink2_topk_b"),
            .route_lists = try d.g.kernel(mod, "ink2_route_lists"),
            .gateup_nvfp4_b = try d.g.kernel(mod, "ink2_gateup_nvfp4_b"),
            .gateup_q8_b = try d.g.kernel(mod, "ink2_gateup_q8_b"),
            .down_nvfp4_b = try d.g.kernel(mod, "ink2_down_nvfp4_b"),
            .down_q8_b = try d.g.kernel(mod, "ink2_down_q8_b"),
            .combine_b = try d.g.kernel(mod, "ink2_combine_b"),
            .attn_split = try d.g.kernel(mod, "ink2_attn_split"),
            .attn_merge = try d.g.kernel(mod, "ink2_attn_merge"),
        };
    }

    fn init(d: *Dev, m: *Model, ctx: usize) !Runner {
        var r: Runner = undefined;
        r.d = d;
        r.m = m;
        r.ctx = ctx;
        r.k = try kernels(d);
        const doc = &m.doc;
        r.qk_clip = try metaF32(doc, "inkling2.attention.qk_clip");
        r.rel_clip = try metaF32(doc, "inkling2.attention.relative_clip");
        r.floor = @floatFromInt(try metaInt(doc, "inkling2.attention.log_scale_floor"));
        r.alpha = try metaF32(doc, "inkling2.attention.log_scale_alpha");
        r.situ_b = try metaF32(doc, "inkling2.activation.situ_beta");
        r.situ_lb = try metaF32(doc, "inkling2.activation.situ_linear_beta");
        r.wscale = try metaF32(doc, "inkling2.expert_weights_scale");
        r.cap = try metaF32(doc, "inkling2.final_logit_softcapping");
        r.dump = if (getenv("INK2_DUMP")) |p| std.mem.span(p) else null;
        const f = @sizeOf(f32);
        var a: Acts = undefined;
        a.S = try d.alloc_(SLOTS * C * f);
        a.mix = try d.alloc_(3 * C * f);
        a.msg_a = try d.alloc_(2 * C * f);
        a.msg_k = try d.alloc_(2 * C * f);
        a.inp = try d.alloc_(3 * C * f);
        a.innorm = try d.alloc_(3 * C * f);
        a.p = try d.alloc_(PROWS * f);
        a.dsm = try d.alloc_(88 * f);
        a.q = try d.alloc_(NH * HD * f);
        a.r = try d.alloc_(NH * NR * f);
        a.o = try d.alloc_(NH * HD * f);
        a.y = try d.alloc_(C * f);
        a.hin = try d.alloc_(2 * C * f);
        a.hdown = try d.alloc_(64 * f);
        a.rlog = try d.alloc_(NE * f);
        a.ids = try d.alloc_(TOP * 4);
        a.wts = try d.alloc_(TOP * f);
        a.act = try d.alloc_(TOP * FF * f);
        a.logits = try d.alloc_(m.out.rows * f);
        a.pv = try d.alloc_(1024 * f);
        a.pi = try d.alloc_(1024 * 4);
        a.tok = try d.alloc_(64);
        a.scratch = try d.alloc_(NH * ctx * f);
        a.hist = try d.alloc_(VISITS * 3 * HIST * f);
        a.hist2 = try d.alloc_(VISITS * 2 * 3 * HIST * f);
        a.msg4 = try d.alloc_(4 * C * f);
        a.hd = try d.alloc_(3 * 64 * f);
        a.xn = try d.alloc_(C * f);
        a.counter = try d.alloc_(64);
        a.Sb = try d.alloc_(SLOTS * TMAX * C * f);
        a.ids_b = try d.alloc_(TMAX * 4);
        a.mix_b = try d.alloc_(4 * TMAX * C * f);
        a.msg_b = try d.alloc_(4 * TMAX * C * f);
        a.inp_b = try d.alloc_(3 * TMAX * C * f);
        a.xn_b = try d.alloc_(3 * TMAX * C * f);
        a.hd_b = try d.alloc_(3 * TMAX * 64 * f);
        a.p_b = try d.alloc_(TMAX * PROWS * f);
        a.d_b = try d.alloc_(TMAX * 88 * f);
        a.q_b = try d.alloc_(TMAX * NH * HD * f);
        a.r_b = try d.alloc_(TMAX * NH * NR * f);
        a.o_b = try d.alloc_(TMAX * NH * HD * f);
        a.y_b = try d.alloc_(TMAX * C * f);
        a.rlog_b = try d.alloc_(TMAX * NE * f);
        a.ids8_b = try d.alloc_(TMAX * TOP * 4);
        a.wts_b = try d.alloc_(TMAX * TOP * f);
        a.start_b = try d.alloc_((NE + 1) * 4);
        a.list_b = try d.alloc_(TMAX * TOP * 4);
        a.act_b = try d.alloc_(TMAX * TOP * FF * f);
        a.ye_b = try d.alloc_(TMAX * TOP * C * f);
        a.scratch_b = try d.alloc_(TMAX * NH * ctx * f);
        a.part_o = try d.alloc_(NH * (ctx / 256 + 1) * HD * f);
        a.part_ml = try d.alloc_(NH * (ctx / 256 + 1) * 2 * f);
        for (0..VISITS) |il| {
            const ring: usize = if (isGlobal(layerOf(il))) ctx else RING;
            a.ring[il] = @intCast(ring);
            a.kc[il] = try d.alloc_(ring * NKV * HD * 2);
            a.vc[il] = try d.alloc_(ring * NKV * HD * 2);
        }
        r.a = a;
        r.v1 = getenv("INK2_V1") != null;
        r.onerow = r.v1 or getenv("INK2_ONEROW") != null;
        r.expk_b = getenv("INK2_EXPK") != null;
        r.pro = getenv("INK2_PRO") != null;
        r.split_at = if (getenv("INK2_SPLIT")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
        r.temp = 0;
        r.seed = 0;
        try d.zero(a.counter, 64);
        try r.reset();
        return r;
    }

    fn reset(r: *Runner) !void {
        try r.d.zero(r.a.hist, VISITS * 3 * HIST * 4);
        try r.d.zero(r.a.hist2, VISITS * 2 * 3 * HIST * 4);
        try r.d.sync();
    }

    fn slot(r: *Runner, s: usize) *anyopaque {
        return off(r.a.S, s * C * 4);
    }

    fn mix(r: *Runner, states: []const usize, query: *anyopaque, nq: i32, out: *anyopaque) !void {
        var sl: Slots = .{ .n = @intCast(states.len), .idx = @splat(0) };
        if (states.len > MAXS) return error.TooManyStates;
        for (states, 0..) |s, i| sl.idx[i] = @intCast(s);
        try r.d.launch(r.k.mix, .{ .x = 1 }, .{ r.a.S, sl, query, nq, out });
    }

    fn control(r: *Runner, l: *const Layer, mod: usize, cur: *anyopaque, msg: *anyopaque, out: *anyopaque) !void {
        try r.d.launch(r.k.ctl_in, .{ .x = 1 }, .{ cur, msg, r.a.hin });
        try r.d.launch(r.k.gemv_f32, .{ .x = 4 }, .{ l.loop_down[mod].?, @as(i32, 64), @as(i32, 2 * C), r.a.hin, r.a.hin, r.a.hin, @as(i32, 64), @as(i32, 64), r.a.hdown });
        try r.d.launch(r.k.ctl_out, .{ .x = C / 16 }, .{ l.loop_gain[mod].?, l.loop_forget[mod].?, r.a.hdown, cur, msg, out });
    }

    /// A free slot: none of `used` holds it.
    fn fresh(used: []const usize) usize {
        var s: usize = 0;
        while (s < SLOTS) : (s += 1) {
            if (std.mem.indexOfScalar(usize, used, s) == null) return s;
        }
        unreachable;
    }

    fn step(r: *Runner, tok: u32, pos: usize) !u32 {
        return if (r.v1) r.stepV1(tok, pos) else r.stepV2(tok, pos);
    }

    fn mix3(r: *Runner, states: []const usize, q1: *anyopaque, n1: i32, q2: ?*anyopaque, n2: i32, q3: ?*anyopaque, n3: i32, out: *anyopaque) !void {
        var sl: Slots = .{ .n = @intCast(states.len), .idx = @splat(0) };
        if (states.len > MAXS) return error.TooManyStates;
        for (states, 0..) |s_, i| sl.idx[i] = @intCast(s_);
        try r.d.launch(r.k.mix3, .{ .x = 1 }, .{ r.a.S, sl, q1, n1, q2 orelse q1, n2, q3 orelse q1, n3, out });
    }

    fn mix3n(r: *Runner, states: []const usize, q1: *anyopaque, n1: i32, out: *anyopaque, gamma: *anyopaque, gstride: i32) !void {
        var sl: Slots = .{ .n = @intCast(states.len), .idx = @splat(0) };
        if (states.len > MAXS) return error.TooManyStates;
        for (states, 0..) |s_, i| sl.idx[i] = @intCast(s_);
        try r.d.launch(r.k.mix3n, .{ .x = 1 }, .{ r.a.S, sl, q1, n1, q1, @as(i32, 0), q1, @as(i32, 0), out, gamma, gstride, r.m.eps });
    }

    /// v2: about 8 launches a plain visit, 13 a message visit; returns the greedy next token.
    fn stepV2(r: *Runner, tok: u32, pos: usize) !u32 {
        const d = r.d;
        const m = r.m;
        const a = &r.a;
        var blocks_buf: [16]usize = undefined;
        var blocks: std.ArrayList(usize) = .initBuffer(&blocks_buf);
        var prelude_buf: [16]usize = undefined;
        var prelude: std.ArrayList(usize) = .initBuffer(&prelude_buf);
        var bank_buf: [16]usize = undefined;
        var bank: std.ArrayList(usize) = .initBuffer(&bank_buf);
        var prelude_cur: usize = 0;
        var bank_last: usize = 0;
        var cur: usize = 0;
        try d.launch(r.k.embed, .{ .x = 1 }, .{ m.embed.q, m.embed.s, @as(i32, @intCast(tok)), m.embed_norm, m.eps, r.slot(cur) });
        const tau: f32 = 1.0 + r.alpha * @log(@max((@as(f32, @floatFromInt(pos)) + 1.0) / r.floor, 1.0));
        const n_core = CORE1 - CORE0;
        const par = pos % 2;
        const hsz = 3 * HIST * 4;
        for (0..VISITS) |il| {
            const ip = layerOf(il);
            const l = &m.layers[ip];
            const g = isGlobal(ip);
            const hin = off(a.hist2, (il * 2 + par) * hsz);
            const hout = off(a.hist2, (il * 2 + 1 - par) * hsz);
            var msgs = false;
            if (ip >= CORE0 and ip < CORE1) {
                const pas = (il - CORE0) / n_core;
                if (ip == CORE0) {
                    if (pas == 0) {
                        prelude.clearRetainingCapacity();
                        prelude.appendSliceAssumeCapacity(blocks.items);
                        prelude_cur = cur;
                    } else {
                        bank.clearRetainingCapacity();
                        bank.appendSliceAssumeCapacity(blocks.items[prelude.items.len + 1 ..]);
                        bank_last = cur;
                        blocks.clearRetainingCapacity();
                        blocks.appendSliceAssumeCapacity(prelude.items);
                        cur = prelude_cur;
                    }
                }
                if (pas > 0) {
                    msgs = true;
                    const row = 2 * (ip - CORE0);
                    var st_buf: [16]usize = undefined;
                    var st: std.ArrayList(usize) = .initBuffer(&st_buf);
                    st.appendSliceAssumeCapacity(bank.items);
                    st.appendAssumeCapacity(bank_last);
                    try r.mix3(st.items, off(m.loop_q, row * C * 4), 1, off(m.loop_kv_q, row * C * 4), 2, off(m.loop_q, (row + 1) * C * 4), 1, a.msg4);
                }
            }
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            const nsel: i32 = if (g or msgs) 3 else 1;
            if (!r.pro) {
                // inputs normalised once (in the mix when no controller follows), then the projections read them
                if (l.attn_q != null and !msgs) {
                    try r.mix3n(st.items, l.attn_q.?, if (g) 3 else 1, a.innorm, l.norms, @as(i32, if (g) C else 0));
                } else {
                    var in_ptr: *anyopaque = r.slot(cur);
                    var in_stride: i32 = 0;
                    if (l.attn_q) |q| {
                        try r.mix3(st.items, q, if (g) 3 else 1, null, 0, null, 0, a.mix);
                        in_ptr = a.mix;
                        in_stride = if (g) C else 0;
                    }
                    if (msgs) {
                        try d.launch(r.k.ctl_down, .{ .x = 4, .y = 3 }, .{ in_ptr, in_stride, a.msg4, @as(i32, C), l.loop_down4.?, @as(i32, 0), a.hd });
                        try d.launch(r.k.ctl_out2, .{ .x = C / 16, .y = 3 }, .{ l.loop_gain4.?, l.loop_forget4.?, @as(i32, 0), a.hd, in_ptr, in_stride, a.msg4, @as(i32, C), a.inp });
                        in_ptr = a.inp;
                        in_stride = C;
                    }
                    try d.launch(r.k.norm, .{ .x = @as(u32, @intCast(nsel)) }, .{ in_ptr, l.norms, m.eps, a.innorm, in_stride, @as(i32, if (g) C else 0) });
                }
                const x1 = if (nsel == 3) off(a.innorm, C * 4) else a.innorm;
                const x2 = if (nsel == 3) off(a.innorm, 2 * C * 4) else a.innorm;
                try d.launch(r.k.gemv_bigx, .{ .x = (PROWS + l.small_rows + 15) / 16 }, .{ l.big.q, l.big.s, l.small, @as(i32, @intCast(l.small_rows)), a.innorm, x1, x2, nsel, a.p, a.dsm });
            } else {
                var in_ptr: *anyopaque = r.slot(cur);
                var in_stride: i32 = 0;
                if (l.attn_q) |q| {
                    try r.mix3(st.items, q, if (g) 3 else 1, null, 0, null, 0, a.mix);
                    in_ptr = a.mix;
                    in_stride = if (g) C else 0;
                }
                if (msgs) {
                    try d.launch(r.k.ctl_down, .{ .x = 4, .y = 3 }, .{ in_ptr, in_stride, a.msg4, @as(i32, C), l.loop_down4.?, @as(i32, 0), a.hd });
                    try d.launch(r.k.ctl_out2, .{ .x = C / 16, .y = 3 }, .{ l.loop_gain4.?, l.loop_forget4.?, @as(i32, 0), a.hd, in_ptr, in_stride, a.msg4, @as(i32, C), a.inp });
                    in_ptr = a.inp;
                    in_stride = C;
                }
                const st_ = @as(usize, @intCast(in_stride)) * 4;
                try d.launch(r.k.gemv_big, .{ .x = (PROWS + l.small_rows + 15) / 16 }, .{ l.big.q, l.big.s, l.small, @as(i32, @intCast(l.small_rows)), in_ptr, off(in_ptr, st_), off(in_ptr, 2 * st_), nsel, l.norms, @as(i32, if (g) C else 0), m.eps, a.p, a.dsm });
            }
            const ring = a.ring[il];
            const pp: Prep = .{ .qk_clip = r.qk_clip, .rel_clip = r.rel_clip, .eps = m.eps, .tau = tau, .global_layer = @intFromBool(g), .slot = @intCast(pos % ring) };
            const p0: usize = if (g) 0 else pos -| (WINDOW - 1);
            const sp: Span = .{ .pos = @intCast(pos), .p0 = @intCast(p0), .ring = @intCast(ring), .extent = EXTENT };
            const nkeys = pos - p0 + 1;
            if (g and nkeys > r.split_at) {
                const nch: u32 = @intCast((nkeys + 255) / 256);
                try d.launch(r.k.prep, .{ .x = 64 }, .{ a.p, a.dsm, l.kconv, l.vconv, hin, hout, l.qclip, l.rclip, pp, a.q, a.r, a.kc[il], a.vc[il] });
                try d.launch(r.k.attn_split, .{ .x = NKV, .y = nch }, .{ a.q, a.r, a.dsm, a.kc[il], a.vc[il], l.relb, sp, a.part_o, a.part_ml });
                try d.launch(r.k.attn_merge, .{ .x = NH }, .{ a.part_o, a.part_ml, @as(i32, @intCast(nch)), a.p, a.o });
            } else try d.launch(r.k.attn2, .{ .x = NH }, .{ a.p, a.dsm, l.kconv, l.vconv, hin, hout, l.qclip, l.rclip, pp, a.kc[il], a.vc[il], l.relb, sp, a.scratch, a.o });
            const boundary = ip % BLOCK == 0;
            var used_buf: [48]usize = undefined;
            var used: std.ArrayList(usize) = .initBuffer(&used_buf);
            used.appendSliceAssumeCapacity(blocks.items);
            used.appendSliceAssumeCapacity(prelude.items);
            used.appendSliceAssumeCapacity(bank.items);
            used.appendSliceAssumeCapacity(&.{ prelude_cur, bank_last, cur });
            const attn_out = fresh(used.items);
            try d.launch(r.k.gemv_conv, .{ .x = C / 16 }, .{ l.wo.q, l.wo.s, @as(i32, NH * HD), a.o, l.aconv, hin, hout, @as(i32, 2048), r.slot(cur), @as(i32, @intFromBool(!boundary)), r.slot(attn_out) });
            if (boundary) blocks.appendAssumeCapacity(cur);
            st.clearRetainingCapacity();
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(attn_out);
            try r.mix3(st.items, l.ffn_q, 1, null, 0, null, 0, a.mix);
            var xin: *anyopaque = a.mix;
            if (msgs) {
                const mm = off(a.msg4, 3 * C * 4);
                try d.launch(r.k.ctl_down, .{ .x = 4, .y = 1 }, .{ a.mix, @as(i32, 0), mm, @as(i32, 0), l.loop_down4.?, @as(i32, 3), a.hd });
                try d.launch(r.k.ctl_out2, .{ .x = C / 16, .y = 1 }, .{ l.loop_gain4.?, l.loop_forget4.?, @as(i32, 3), a.hd, a.mix, @as(i32, 0), mm, @as(i32, 0), a.inp });
                xin = a.inp;
            }
            try d.launch(r.k.router, .{ .x = NE / 16 }, .{ xin, l.ffn_norm, m.eps, l.router, a.xn, a.rlog, l.probs_b, r.wscale, a.ids, a.wts, a.counter });
            used.appendAssumeCapacity(attn_out);
            used.appendSliceAssumeCapacity(blocks.items);
            const out = fresh(used.items);
            if (l.exp.nvfp4 and !r.expk_b) {
                try d.launch(r.k.gateup_nvfp4c, .{ .x = FF / 32, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, l.exp.gu_scale.?, a.ids, a.xn, m.lut16, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_nvfp4c, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, l.exp.d_scale.?, a.ids, a.wts, a.act, m.lut16, l.fconv, hin, hout, @as(i32, 4352), r.slot(attn_out), r.slot(out) });
            } else if (l.exp.nvfp4) {
                try d.launch(r.k.gateup_nvfp4b, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, l.exp.gu_scale.?, a.ids, a.xn, m.lut, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_nvfp4_conv, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, l.exp.d_scale.?, a.ids, a.wts, a.act, m.lut, l.fconv, hin, hout, @as(i32, 4352), r.slot(attn_out), r.slot(out) });
            } else {
                try d.launch(r.k.gateup_q8, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, a.ids, a.xn, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_q8_conv, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, a.ids, a.wts, a.act, l.fconv, hin, hout, @as(i32, 4352), r.slot(attn_out), r.slot(out) });
            }
            cur = out;
            if (r.dump) |dir| try r.dumpRow(dir, "v", il, pos, r.slot(cur), C);
        }
        {
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            if (r.pro) try r.mix3(st.items, m.out_q, 1, null, 0, null, 0, a.mix) else try r.mix3n(st.items, m.out_q, 1, a.innorm, m.out_norm, 0);
        }
        const rows: u32 = @intCast(m.out.rows);
        if (r.pro) {
            try d.launch(r.k.gemv_q8n, .{ .x = (rows + 15) / 16 }, .{ m.out.q, m.out.s, @as(i32, @intCast(rows)), a.mix, m.out_norm, m.eps, a.logits });
        } else try d.launch(r.k.gemv_q8, .{ .x = (rows + 15) / 16 }, .{ m.out.q, m.out.s, @as(i32, @intCast(rows)), @as(i32, C), a.innorm, a.innorm, a.innorm, @as(i32, @intCast(rows)), @as(i32, @intCast(rows)), a.logits });
        return r.finish(pos);
    }

    fn finish(r: *Runner, pos: usize) !u32 {
        const d = r.d;
        const a = &r.a;
        const n: u32 = @intCast(r.m.vocab);
        const parts = (n + 255) / 256;
        const seed_lo: u32 = @truncate(r.seed);
        const seed_hi: u32 = @truncate(r.seed >> 32);
        try d.launch(r.k.sample_part, .{ .x = parts }, .{ a.logits, @as(i32, @intCast(n)), r.cap, r.temp, seed_lo, seed_hi, @as(u32, @intCast(pos)), a.pv, a.pi });
        try d.launch(r.k.argmax_final, .{ .x = 1 }, .{ a.pv, a.pi, @as(i32, @intCast(parts)), a.tok });
        var out: [4]u8 = undefined;
        try d.read(&out, a.tok);
        if (r.dump) |dir| try r.dumpRow(dir, "logits", 0, pos, a.logits, n);
        return std.mem.readInt(u32, &out, .little);
    }

    /// The prompt from position 0; returns the token sampled after its last row.
    fn prefill(r: *Runner, ids: []const u32) !u32 {
        if (r.onerow) {
            var next: u32 = 0;
            for (ids, 0..) |tok, pos| next = try r.step(tok, pos);
            return next;
        }
        var p0: usize = 0;
        while (true) {
            const n = @min(TMAX, ids.len - p0);
            const last = p0 + n == ids.len;
            const tok = try r.chunk(ids[p0 .. p0 + n], p0, last);
            if (last) return tok;
            p0 += n;
        }
    }

    fn rowsOf(r: *Runner, s_: usize) *anyopaque {
        return off(r.a.Sb, s_ * TMAX * C * 4);
    }

    fn mix3b(r: *Runner, states: []const usize, q1: *anyopaque, n1: i32, q2: ?*anyopaque, n2: i32, q3: ?*anyopaque, n3: i32, out: *anyopaque, T: u32) !void {
        var sl: Slots = .{ .n = @intCast(states.len), .idx = @splat(0) };
        if (states.len > MAXS) return error.TooManyStates;
        for (states, 0..) |s_, i| sl.idx[i] = @intCast(s_);
        try r.d.launch(r.k.mix3_b, .{ .x = T }, .{ r.a.Sb, sl, q1, n1, q2 orelse q1, n2, q3 orelse q1, n3, out });
    }

    /// `ids` (at most TMAX rows) at positions p0.. through every visit; with `last`, the token sampled after the last row.
    fn chunk(r: *Runner, ids: []const u32, p0: usize, last: bool) !u32 {
        const d = r.d;
        const m = r.m;
        const a = &r.a;
        const T: u32 = @intCast(ids.len);
        const RC: i64 = TMAX * C; // floats between the selections of a [j][TMAX][C] buffer
        var idb: [TMAX]i32 = undefined;
        for (ids, 0..) |t, i| idb[i] = @intCast(t);
        try d.copyIn(a.ids_b, std.mem.sliceAsBytes(idb[0..ids.len]));
        var blocks_buf: [16]usize = undefined;
        var blocks: std.ArrayList(usize) = .initBuffer(&blocks_buf);
        var prelude_buf: [16]usize = undefined;
        var prelude: std.ArrayList(usize) = .initBuffer(&prelude_buf);
        var bank_buf: [16]usize = undefined;
        var bank: std.ArrayList(usize) = .initBuffer(&bank_buf);
        var prelude_cur: usize = 0;
        var bank_last: usize = 0;
        var cur: usize = 0;
        try d.launch(r.k.embed_b, .{ .x = T }, .{ m.embed.q, m.embed.s, a.ids_b, m.embed_norm, m.eps, r.rowsOf(cur) });
        const n_core = CORE1 - CORE0;
        const hsz = 3 * HIST * 4;
        for (0..VISITS) |il| {
            const ip = layerOf(il);
            const l = &m.layers[ip];
            const g = isGlobal(ip);
            const hin = off(a.hist2, (il * 2 + p0 % 2) * hsz);
            const hout = off(a.hist2, (il * 2 + (p0 + T) % 2) * hsz);
            var msgs = false;
            if (ip >= CORE0 and ip < CORE1) {
                const pas = (il - CORE0) / n_core;
                if (ip == CORE0) {
                    if (pas == 0) {
                        prelude.clearRetainingCapacity();
                        prelude.appendSliceAssumeCapacity(blocks.items);
                        prelude_cur = cur;
                    } else {
                        bank.clearRetainingCapacity();
                        bank.appendSliceAssumeCapacity(blocks.items[prelude.items.len + 1 ..]);
                        bank_last = cur;
                        blocks.clearRetainingCapacity();
                        blocks.appendSliceAssumeCapacity(prelude.items);
                        cur = prelude_cur;
                    }
                }
                if (pas > 0) {
                    msgs = true;
                    const row = 2 * (ip - CORE0);
                    var st_buf: [16]usize = undefined;
                    var st: std.ArrayList(usize) = .initBuffer(&st_buf);
                    st.appendSliceAssumeCapacity(bank.items);
                    st.appendAssumeCapacity(bank_last);
                    try r.mix3b(st.items, off(m.loop_q, row * C * 4), 1, off(m.loop_kv_q, row * C * 4), 2, off(m.loop_q, (row + 1) * C * 4), 1, a.msg_b, T);
                }
            }
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            var in_ptr: *anyopaque = r.rowsOf(cur);
            var in_stride: i64 = 0;
            if (l.attn_q) |q| {
                try r.mix3b(st.items, q, if (g) 3 else 1, null, 0, null, 0, a.mix_b, T);
                in_ptr = a.mix_b;
                in_stride = if (g) RC else 0;
            }
            if (msgs) {
                try d.launch(r.k.ctl_down_b, .{ .x = 4, .y = 3, .z = T }, .{ in_ptr, in_stride, a.msg_b, RC, l.loop_down4.?, @as(i32, 0), a.hd_b });
                try d.launch(r.k.ctl_out_b, .{ .x = C / 16, .y = 3, .z = T }, .{ l.loop_gain4.?, l.loop_forget4.?, @as(i32, 0), a.hd_b, in_ptr, in_stride, a.msg_b, RC, a.inp_b });
                in_ptr = a.inp_b;
                in_stride = RC;
            }
            const nsel: u32 = if (g or msgs) 3 else 1;
            try d.launch(r.k.norm_b, .{ .x = nsel, .y = T }, .{ in_ptr, in_stride, l.norms, @as(i32, if (g) C else 0), m.eps, a.xn_b });
            const xsel: i64 = if (nsel == 1) 0 else RC;
            try d.launch(r.k.gemv_q8_b, .{ .x = PROWS / 16 }, .{ l.big.q, l.big.s, @as(i32, PROWS), @as(i32, C), a.xn_b, xsel, @as(i32, 6528), @as(i32, 7552), @as(i32, @intCast(T)), a.p_b, @as(i32, PROWS) });
            try d.launch(r.k.gemv_f32_b, .{ .x = (l.small_rows + 15) / 16 }, .{ l.small, @as(i32, @intCast(l.small_rows)), @as(i32, C), a.xn_b, xsel, @as(i32, 24), @as(i32, 56), @as(i32, @intCast(T)), a.d_b, @as(i32, 88) });
            const ring = a.ring[il];
            const pp: Prep = .{ .qk_clip = r.qk_clip, .rel_clip = r.rel_clip, .eps = m.eps, .tau = 1, .global_layer = @intFromBool(g), .slot = 0 };
            try d.launch(r.k.prep_b, .{ .x = 64 }, .{ a.p_b, a.d_b, l.kconv, l.vconv, hin, hout, l.qclip, l.rclip, pp, @as(i32, @intCast(p0)), @as(i32, @intCast(T)), @as(i32, @intCast(ring)), r.floor, r.alpha, a.q_b, a.r_b, a.kc[il], a.vc[il] });
            try d.launch(r.k.attn_b, .{ .x = NH, .y = T }, .{ a.q_b, a.r_b, a.d_b, a.kc[il], a.vc[il], l.relb, @as(i32, @intCast(p0)), @as(i32, if (g) 0 else WINDOW), @as(i32, @intCast(ring)), @as(i32, EXTENT), a.scratch_b, a.p_b, a.o_b });
            try d.launch(r.k.gemv_q8_b, .{ .x = C / 16 }, .{ l.wo.q, l.wo.s, @as(i32, C), @as(i32, NH * HD), a.o_b, @as(i64, 0), @as(i32, C), @as(i32, C), @as(i32, @intCast(T)), a.y_b, @as(i32, C) });
            const boundary = ip % BLOCK == 0;
            var used_buf: [48]usize = undefined;
            var used: std.ArrayList(usize) = .initBuffer(&used_buf);
            used.appendSliceAssumeCapacity(blocks.items);
            used.appendSliceAssumeCapacity(prelude.items);
            used.appendSliceAssumeCapacity(bank.items);
            used.appendSliceAssumeCapacity(&.{ prelude_cur, bank_last, cur });
            const attn_out = fresh(used.items);
            try d.launch(r.k.conv_res_b, .{ .x = C / 256 }, .{ a.y_b, l.aconv, hin, hout, @as(i32, 2048), r.rowsOf(cur), @as(i32, @intFromBool(!boundary)), @as(i32, @intCast(T)), r.rowsOf(attn_out) });
            if (boundary) blocks.appendAssumeCapacity(cur);
            st.clearRetainingCapacity();
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(attn_out);
            try r.mix3b(st.items, l.ffn_q, 1, null, 0, null, 0, a.mix_b, T);
            var xin: *anyopaque = a.mix_b;
            if (msgs) {
                const mm = off(a.msg_b, 3 * TMAX * C * 4);
                try d.launch(r.k.ctl_down_b, .{ .x = 4, .y = 1, .z = T }, .{ a.mix_b, @as(i64, 0), mm, @as(i64, 0), l.loop_down4.?, @as(i32, 3), a.hd_b });
                try d.launch(r.k.ctl_out_b, .{ .x = C / 16, .y = 1, .z = T }, .{ l.loop_gain4.?, l.loop_forget4.?, @as(i32, 3), a.hd_b, a.mix_b, @as(i64, 0), mm, @as(i64, 0), a.inp_b });
                xin = a.inp_b;
            }
            try d.launch(r.k.norm_b, .{ .x = 1, .y = T }, .{ xin, @as(i64, 0), l.ffn_norm, @as(i32, 0), m.eps, a.xn_b });
            try d.launch(r.k.gemv_f32_b, .{ .x = NE / 16 }, .{ l.router, @as(i32, NE), @as(i32, C), a.xn_b, @as(i64, 0), @as(i32, NE), @as(i32, NE), @as(i32, @intCast(T)), a.rlog_b, @as(i32, NE) });
            try d.launch(r.k.topk_b, .{ .x = T }, .{ a.rlog_b, l.probs_b, r.wscale, a.ids8_b, a.wts_b });
            try d.launch(r.k.route_lists, .{ .x = 1 }, .{ a.ids8_b, @as(i32, @intCast(T)), a.start_b, a.list_b });
            if (l.exp.nvfp4) {
                try d.launch(r.k.gateup_nvfp4_b, .{ .x = FF / 16, .y = NE }, .{ l.exp.gu_q, l.exp.gu_s, l.exp.gu_scale.?, a.start_b, a.list_b, a.xn_b, m.lut, r.situ_b, r.situ_lb, a.act_b });
                try d.launch(r.k.down_nvfp4_b, .{ .x = C / 16, .y = NE }, .{ l.exp.d_q, l.exp.d_s, l.exp.d_scale.?, a.start_b, a.list_b, a.act_b, m.lut, a.ye_b });
            } else {
                try d.launch(r.k.gateup_q8_b, .{ .x = FF / 16, .y = NE }, .{ l.exp.gu_q, l.exp.gu_s, a.start_b, a.list_b, a.xn_b, r.situ_b, r.situ_lb, a.act_b });
                try d.launch(r.k.down_q8_b, .{ .x = C / 16, .y = NE }, .{ l.exp.d_q, l.exp.d_s, a.start_b, a.list_b, a.act_b, a.ye_b });
            }
            used.appendAssumeCapacity(attn_out);
            used.appendSliceAssumeCapacity(blocks.items);
            const out = fresh(used.items);
            try d.launch(r.k.combine_b, .{ .x = C / 256 }, .{ a.ye_b, a.wts_b, l.fconv, hin, hout, @as(i32, 4352), r.rowsOf(attn_out), @as(i32, @intCast(T)), r.rowsOf(out) });
            cur = out;
            if (r.dump) |dir| for (0..T) |t| try r.dumpRow(dir, "v", il, p0 + t, off(r.rowsOf(cur), t * C * 4), C);
        }
        if (!last) return 0;
        {
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            try r.mix3b(st.items, m.out_q, 1, null, 0, null, 0, a.mix_b, T);
        }
        const nrows: u32 = @intCast(m.out.rows);
        try d.launch(r.k.gemv_q8n, .{ .x = (nrows + 15) / 16 }, .{ m.out.q, m.out.s, @as(i32, @intCast(nrows)), off(a.mix_b, (T - 1) * C * 4), m.out_norm, m.eps, a.logits });
        return r.finish(p0 + T - 1);
    }

    /// v1: about 15 launches a visit (kept for same-binary comparison, INK2_V1=1).
    fn stepV1(r: *Runner, tok: u32, pos: usize) !u32 {
        const d = r.d;
        const m = r.m;
        const a = &r.a;
        var blocks_buf: [16]usize = undefined;
        var blocks: std.ArrayList(usize) = .initBuffer(&blocks_buf);
        var prelude_buf: [16]usize = undefined;
        var prelude: std.ArrayList(usize) = .initBuffer(&prelude_buf);
        var bank_buf: [16]usize = undefined;
        var bank: std.ArrayList(usize) = .initBuffer(&bank_buf);
        var prelude_cur: usize = 0;
        var bank_last: usize = 0;
        var cur: usize = 0;
        try d.launch(r.k.embed, .{ .x = 1 }, .{ m.embed.q, m.embed.s, @as(i32, @intCast(tok)), m.embed_norm, m.eps, r.slot(cur) });
        if (pos == 0) try r.dbg("emb", r.slot(cur), C * 4);
        const tau: f32 = 1.0 + r.alpha * @log(@max((@as(f32, @floatFromInt(pos)) + 1.0) / r.floor, 1.0));
        const n_core = CORE1 - CORE0;
        for (0..VISITS) |il| {
            const ip = layerOf(il);
            const l = &m.layers[ip];
            const g = isGlobal(ip);
            var msgs = false;
            if (ip >= CORE0 and ip < CORE1) {
                const pas = (il - CORE0) / n_core;
                if (ip == CORE0) {
                    if (pas == 0) {
                        prelude.clearRetainingCapacity();
                        prelude.appendSliceAssumeCapacity(blocks.items);
                        prelude_cur = cur;
                    } else {
                        bank.clearRetainingCapacity();
                        bank.appendSliceAssumeCapacity(blocks.items[prelude.items.len + 1 ..]);
                        bank_last = cur;
                        blocks.clearRetainingCapacity();
                        blocks.appendSliceAssumeCapacity(prelude.items);
                        cur = prelude_cur;
                    }
                }
                if (pas > 0) {
                    msgs = true;
                    const row = 2 * (ip - CORE0);
                    var st_buf: [16]usize = undefined;
                    var st: std.ArrayList(usize) = .initBuffer(&st_buf);
                    st.appendSliceAssumeCapacity(bank.items);
                    st.appendAssumeCapacity(bank_last);
                    try r.mix(st.items, off(m.loop_q, row * C * 4), 2, a.msg_a);
                    try r.mix(st.items, off(m.loop_kv_q, row * C * 4), 2, a.msg_k);
                }
            }
            // attention inputs
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            var in_ptr: *anyopaque = r.slot(cur); // [nin][C] (stride C) or one vector (stride 0)
            var in_stride: i32 = 0;
            if (l.attn_q) |q| {
                try r.mix(st.items, q, if (g) 3 else 1, a.mix);
                in_ptr = a.mix;
                in_stride = if (g) C else 0;
            }
            if (msgs) {
                const msg = [3]*anyopaque{ a.msg_a, a.msg_k, off(a.msg_k, C * 4) };
                for (0..3) |j| try r.control(l, j, off(in_ptr, @as(usize, @intCast(in_stride)) * j * 4), msg[j], off(a.inp, j * C * 4));
                in_ptr = a.inp;
                in_stride = C;
            }
            const nin: u32 = if (g or msgs) 3 else 1;
            try d.launch(r.k.norm, .{ .x = nin }, .{ in_ptr, l.norms, m.eps, a.innorm, in_stride, @as(i32, if (g) C else 0) });
            if (il == 0 and pos == 0) try r.dbg("attn_xn", a.innorm, C * 4);
            const x0 = a.innorm;
            const x1 = if (nin == 3) off(a.innorm, C * 4) else a.innorm;
            const x2 = if (nin == 3) off(a.innorm, 2 * C * 4) else a.innorm;
            try d.launch(r.k.gemv_q8, .{ .x = PROWS / 16 }, .{ l.big.q, l.big.s, @as(i32, PROWS), @as(i32, C), x0, x1, x2, @as(i32, 6528), @as(i32, 7552), a.p });
            if (il == 0 and pos == 0) try r.dbg("p", a.p, PROWS * 4);
            try d.launch(r.k.gemv_f32, .{ .x = (l.small_rows + 15) / 16 }, .{ l.small, @as(i32, @intCast(l.small_rows)), @as(i32, C), x0, x1, x2, @as(i32, 24), @as(i32, 56), a.dsm });
            if (il == 0 and pos == 0) try r.dbg("dsm", a.dsm, 24 * 4);
            const ring = a.ring[il];
            const hist = off(a.hist, il * 3 * HIST * 4);
            const pp: Prep = .{ .qk_clip = r.qk_clip, .rel_clip = r.rel_clip, .eps = m.eps, .tau = tau, .global_layer = @intFromBool(g), .slot = @intCast(pos % ring) };
            try d.launch(r.k.prep, .{ .x = 64 }, .{ a.p, a.dsm, l.kconv, l.vconv, hist, hist, l.qclip, l.rclip, pp, a.q, a.r, a.kc[il], a.vc[il] });
            if (il == 0 and pos == 0) try r.dbg("q", a.q, NH * HD * 4);
            if (il == 0 and pos == 0) try r.dbg("r", a.r, NH * NR * 4);
            if (il == 0 and pos == 0) try r.dbg("k0", a.kc[il], 1024 * 2);
            if (il == 0 and pos == 0) try r.dbg("v0", a.vc[il], 1024 * 2);
            const p0: usize = if (g) 0 else pos -| (WINDOW - 1);
            const sp: Span = .{ .pos = @intCast(pos), .p0 = @intCast(p0), .ring = @intCast(ring), .extent = EXTENT };
            try d.launch(r.k.attn, .{ .x = NH }, .{ a.q, a.r, a.dsm, a.kc[il], a.vc[il], l.relb, sp, a.scratch, a.p, a.o });
            if (il == 0 and pos == 0) try r.dbg("o", a.o, NH * HD * 4);
            try d.launch(r.k.gemv_q8, .{ .x = C / 16 }, .{ l.wo.q, l.wo.s, @as(i32, C), @as(i32, NH * HD), a.o, a.o, a.o, @as(i32, C), @as(i32, C), a.y });
            if (il == 0 and pos == 0) try r.dbg("attn_proj", a.y, C * 4);
            // the first layer of a block starts a new residual stream; its input joins the blocks
            const boundary = ip % BLOCK == 0;
            var used_buf: [48]usize = undefined;
            var used: std.ArrayList(usize) = .initBuffer(&used_buf);
            used.appendSliceAssumeCapacity(blocks.items);
            used.appendSliceAssumeCapacity(prelude.items);
            used.appendSliceAssumeCapacity(bank.items);
            used.appendSliceAssumeCapacity(&.{ prelude_cur, bank_last, cur });
            const attn_out = fresh(used.items);
            try d.launch(r.k.conv_res, .{ .x = C / 256 }, .{ a.y, l.aconv, hist, @as(i32, 2048), r.slot(cur), @as(i32, @intFromBool(!boundary)), r.slot(attn_out) });
            if (il == 0 and pos == 0) try r.dbg("attn_out", r.slot(attn_out), C * 4);
            if (boundary) blocks.appendAssumeCapacity(cur);
            // MLP input
            st.clearRetainingCapacity();
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(attn_out);
            try r.mix(st.items, l.ffn_q, 1, a.mix);
            var xin: *anyopaque = a.mix;
            if (msgs) {
                try r.control(l, 3, a.mix, off(a.msg_a, C * 4), a.inp);
                xin = a.inp;
            }
            try d.launch(r.k.norm, .{ .x = 1 }, .{ xin, l.ffn_norm, m.eps, a.innorm, @as(i32, 0), @as(i32, 0) });
            if (il == 0 and pos == 0) try r.dbg("ffn_mix", xin, C * 4);
            if (il == 0 and pos == 0) try r.dbg("ffn_xn", a.innorm, C * 4);
            try d.launch(r.k.gemv_f32, .{ .x = NE / 16 }, .{ l.router, @as(i32, NE), @as(i32, C), a.innorm, a.innorm, a.innorm, @as(i32, NE), @as(i32, NE), a.rlog });
            try d.launch(r.k.topk, .{ .x = 1 }, .{ a.rlog, l.probs_b, r.wscale, a.ids, a.wts });
            if (il == 0 and pos == 0) try r.dbg("rlog", a.rlog, NE * 4);
            if (il == 0 and pos == 0) try r.dbg("ids", a.ids, TOP * 4);
            if (il == 0 and pos == 0) try r.dbg("wts", a.wts, TOP * 4);
            if (l.exp.nvfp4) {
                try d.launch(r.k.gateup_nvfp4, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, l.exp.gu_scale.?, a.ids, a.innorm, m.lut, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_nvfp4, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, l.exp.d_scale.?, a.ids, a.wts, a.act, m.lut, a.y });
            if (il == 0 and pos == 0) try r.dbg("act", a.act, TOP * FF * 4);
            if (il == 0 and pos == 0) try r.dbg("moe", a.y, C * 4);
            } else {
                try d.launch(r.k.gateup_q8, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, a.ids, a.innorm, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_q8, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, a.ids, a.wts, a.act, a.y });
            }
            used.appendAssumeCapacity(attn_out);
            used.appendSliceAssumeCapacity(blocks.items);
            const out = fresh(used.items);
            try d.launch(r.k.conv_res, .{ .x = C / 256 }, .{ a.y, l.fconv, hist, @as(i32, 4352), r.slot(attn_out), @as(i32, 1), r.slot(out) });
            if (il == 0 and pos == 0) try r.dbg("out", r.slot(out), C * 4);
            cur = out;
            if (r.dump) |dir| try r.dumpRow(dir, "v", il, pos, r.slot(cur), C);
        }
        st_final: {
            var st_buf: [16]usize = undefined;
            var st: std.ArrayList(usize) = .initBuffer(&st_buf);
            st.appendSliceAssumeCapacity(blocks.items);
            st.appendAssumeCapacity(cur);
            try r.mix(st.items, m.out_q, 1, a.mix);
            break :st_final;
        }
        try d.launch(r.k.norm, .{ .x = 1 }, .{ a.mix, m.out_norm, m.eps, a.innorm, @as(i32, 0), @as(i32, 0) });
        const rows: u32 = @intCast(m.out.rows);
        try d.launch(r.k.gemv_q8, .{ .x = (rows + 15) / 16 }, .{ m.out.q, m.out.s, @as(i32, @intCast(rows)), @as(i32, C), a.innorm, a.innorm, a.innorm, @as(i32, @intCast(rows)), @as(i32, @intCast(rows)), a.logits });
        const n: u32 = @intCast(m.vocab);
        const parts = (n + 255) / 256;
        try d.launch(r.k.argmax_part, .{ .x = parts }, .{ a.logits, @as(i32, @intCast(n)), r.cap, a.pv, a.pi });
        try d.launch(r.k.argmax_final, .{ .x = 1 }, .{ a.pv, a.pi, @as(i32, @intCast(parts)), a.tok });
        var out: [4]u8 = undefined;
        try d.read(&out, a.tok);
        if (r.dump) |dir| try r.dumpRow(dir, "logits", 0, pos, a.logits, n);
        return std.mem.readInt(u32, &out, .little);
    }

    /// INK2_DUMPALL=dir: one buffer of visit 0 at position 0 (v1 path), raw bytes, for the host comparison.
    fn dbg(r: *Runner, name: []const u8, src: *anyopaque, nbytes: usize) !void {
        const dir = std.mem.span(getenv("INK2_DUMPALL") orelse return);
        const buf = try alloc.alloc(u8, nbytes);
        defer alloc.free(buf);
        try r.d.read(buf, src);
        const path = try std.fmt.allocPrint(alloc, "{s}/{s}.bin", .{ dir, name });
        defer alloc.free(path);
        try writeFile(path, buf);
    }

    fn dumpRow(r: *Runner, dir: []const u8, tag: []const u8, il: usize, pos: usize, src: *anyopaque, n: usize) !void {
        const buf = try alloc.alloc(u8, n * 4);
        defer alloc.free(buf);
        try r.d.read(buf, src);
        const path = try std.fmt.allocPrint(alloc, "{s}/{s}{d}_p{d}.f32", .{ dir, tag, il, pos });
        defer alloc.free(path);
        try writeFile(path, buf);
    }
};

fn profReport(d: *Dev, k: *const K, steps: usize) void {
    const p = d.prof orelse return;
    const names = [_][]const u8{ "embed", "mix", "norm", "ctl_in", "ctl_out", "gemv_f32", "gemv_q8", "prep", "attn", "conv_res", "topk", "gateup_nvfp4", "down_nvfp4", "gateup_q8", "down_q8", "argmax_part", "argmax_final", "mix3", "ctl_down", "ctl_out2", "gemv_big", "attn2", "gemv_conv", "router", "gateup_nvfp4b", "down_nvfp4_conv", "down_q8_conv", "gemv_q8n", "gateup_nvfp4c", "down_nvfp4c", "mix3_b", "sample_part", "attn_split", "attn_merge", "prep", "mix3n", "gemv_bigx" };
    const hs = [_]ze.H{ k.embed, k.mix, k.norm, k.ctl_in, k.ctl_out, k.gemv_f32, k.gemv_q8, k.prep, k.attn, k.conv_res, k.topk, k.gateup_nvfp4, k.down_nvfp4, k.gateup_q8, k.down_q8, k.argmax_part, k.argmax_final, k.mix3, k.ctl_down, k.ctl_out2, k.gemv_big, k.attn2, k.gemv_conv, k.router, k.gateup_nvfp4b, k.down_nvfp4_conv, k.down_q8_conv, k.gemv_q8n, k.gateup_nvfp4c, k.down_nvfp4c, k.mix3_b, k.sample_part, k.attn_split, k.attn_merge, k.prep, k.mix3n, k.gemv_bigx };
    var sum: f64 = 0;
    var launches: u64 = 0;
    for (p.total, p.count) |t, c| {
        sum += t;
        launches += c;
    }
    const n: f64 = @floatFromInt(steps);
    std.debug.print("GPU busy {d:.3} ms a step, {d:.0} launches a step\n", .{ sum * 1e3 / n, @as(f64, @floatFromInt(launches)) / n });
    for (p.names, p.total, p.count) |h, t, c| {
        if (h == null) continue;
        var name: []const u8 = "?";
        for (hs, names) |x, nm| if (x == h) {
            name = nm;
        };
        std.debug.print("  {s:<14} {d:8.3} ms/step {d:5.1} % {d:6.1} launches/step {d:7.2} us each\n", .{ name, t * 1e3 / n, t / sum * 100, @as(f64, @floatFromInt(c)) / n, t * 1e6 / @as(f64, @floatFromInt(c)) });
    }
}

fn writeOut(bytes: []const u8) void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = write(1, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return;
        done += @intCast(n);
    }
}

/// --serve: one request a line on stdin, "<max_new> <temp> <seed> <id> <id> ...", each from a fresh state (no prefix
/// cache). Replies "t <id>" a token, then "e <prompt> <generated> <prefill_s> <decode_s>". Stops at token `eos`.
fn serve(r: *Runner, eos: u32) !void {
    var buf: std.ArrayList(u8) = .empty;
    var chunk: [65536]u8 = undefined;
    writeOut("ready\n");
    while (true) {
        const nl = std.mem.indexOfScalar(u8, buf.items, '\n');
        if (nl == null) {
            const n = read(0, &chunk, chunk.len);
            if (n <= 0) return;
            try buf.appendSlice(alloc, chunk[0..@intCast(n)]);
            continue;
        }
        const line = try alloc.dupe(u8, buf.items[0..nl.?]);
        defer alloc.free(line);
        try buf.replaceRange(alloc, 0, nl.? + 1, &.{});
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        const max_new = try std.fmt.parseInt(usize, it.next() orelse continue, 10);
        r.temp = try std.fmt.parseFloat(f32, it.next() orelse return error.BadRequest);
        r.seed = try std.fmt.parseInt(u64, it.next() orelse return error.BadRequest, 10);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(alloc);
        while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
        var msg: [96]u8 = undefined;
        if (ids.items.len == 0 or ids.items.len + max_new > r.ctx) {
            writeOut(try std.fmt.bufPrint(&msg, "x context {d}\n", .{r.ctx}));
            continue;
        }
        try r.reset();
        const t0 = now();
        var next = try r.prefill(ids.items);
        const t1 = now();
        var n_gen: usize = 0;
        while (n_gen < max_new) {
            writeOut(try std.fmt.bufPrint(&msg, "t {d}\n", .{next}));
            n_gen += 1;
            if (next == eos or n_gen == max_new) break;
            next = try r.step(next, ids.items.len + n_gen - 1);
        }
        const t2 = now();
        writeOut(try std.fmt.bufPrint(&msg, "e {d} {d} {d:.6} {d:.6}\n", .{ ids.items.len, n_gen, t1 - t0, t2 - t1 }));
    }
}

fn readIds(path: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, try mapFile(path), " \n\r\t,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    return ids.items;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len >= 4 and std.mem.eql(u8, argv[2], "--serve")) {
        const ctx = try std.fmt.parseInt(usize, argv[3], 10);
        var d: Dev = .{ .g = try ze.Gpu.open(0xe223), .list = undefined };
        defer d.g.close();
        d.list = try d.g.immediate(0, 0);
        var m = try load(&d, argv[1]);
        var r = try Runner.init(&d, &m, ctx);
        return serve(&r, 200006);
    }
    if (argv.len < 4) {
        std.debug.print("usage: xpu-inkling2 MODEL.gguf IDS.json N_NEW [CTX] | MODEL.gguf --serve CTX\n", .{});
        return error.Usage;
    }
    const ids = try readIds(argv[2]);
    const n_new = try std.fmt.parseInt(usize, argv[3], 10);
    const ctx = if (argv.len > 4) try std.fmt.parseInt(usize, argv[4], 10) else 4096;
    if (ids.len == 0 or ids.len + n_new > ctx) return error.Usage;
    var d: Dev = .{ .g = try ze.Gpu.open(0xe223), .list = undefined };
    defer d.g.close();
    d.list = try d.g.immediate(0, 0);
    const t0 = now();
    var m = try load(&d, argv[1]);
    const t1 = now();
    std.debug.print("load {d:.1} s, {d:.2} GB on the card\n", .{ t1 - t0, @as(f64, @floatFromInt(d.bytes)) / 1e9 });
    var r = try Runner.init(&d, &m, ctx);
    var prof: Prof = .{};
    const profiling = getenv("INK2_PROFILE") != null;
    const t2 = now();
    var next = try r.prefill(ids);
    const t3 = now();
    std.debug.print("prompt {d} tokens ({s}): {d:.3} s\n", .{ ids.len, if (r.onerow) "one row at a time" else "chunks of 64 rows", t3 - t2 });
    var gen: std.ArrayList(u32) = .empty;
    try gen.append(alloc, next);
    if (profiling) {
        prof.events = try d.g.events(4096);
        d.prof = &prof;
    }
    const t4 = now();
    for (1..n_new) |i| {
        next = try r.step(next, ids.len + i - 1);
        try gen.append(alloc, next);
    }
    const t5 = now();
    std.debug.print("generated:", .{});
    for (gen.items) |g| std.debug.print(" {d}", .{g});
    std.debug.print("\n", .{});
    if (profiling and n_new > 1) profReport(&d, &r.k, n_new - 1);
    if (n_new > 1) std.debug.print("decode {d} steps: {d:.3} s, {d:.2} tok/s (host sync every step, no graph replay)\n", .{ n_new - 1, t5 - t4, @as(f64, @floatFromInt(n_new - 1)) / (t5 - t4) });
}
