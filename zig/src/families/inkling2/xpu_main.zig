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

fn layerOf(il: usize) usize {
    if (il < CORE0) return il;
    if (il < CORE0 + PASSES * (CORE1 - CORE0)) return CORE0 + (il - CORE0) % (CORE1 - CORE0);
    return il - (PASSES - 1) * (CORE1 - CORE0);
}

fn isGlobal(ip: usize) bool {
    return (ip + 1) % 6 == 0;
}

// ---------------------------------------------------------------- device
const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{},
    bytes: usize = 0,

    fn alloc_(d: *Dev, n: usize) !*anyopaque {
        d.bytes += n;
        return d.g.deviceAlloc(@max(n, 64));
    }

    fn sync(d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)));
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
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, null, 0, null));
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
    loop_down: [4]?*anyopaque = @splat(null),
    loop_gain: [4]?*anyopaque = @splat(null),
    loop_forget: [4]?*anyopaque = @splat(null),
};

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
        if (ip >= CORE0 and ip < CORE1) {
            for ([_][]const u8{ "attn", "k", "v", "mlp" }, 0..) |n, j| {
                l.loop_down[j] = try uploadF32(d, try m.t("blk.{d}.loop_{s}_down.weight", .{ ip, n }));
                l.loop_gain[j] = try uploadF32(d, try m.t("blk.{d}.loop_{s}_gain.weight", .{ ip, n }));
                l.loop_forget[j] = try uploadF32(d, try m.t("blk.{d}.loop_{s}_forget.weight", .{ ip, n }));
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

    fn kernels(d: *Dev) !K {
        const mod = try d.g.module(&kbin.ink2);
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
        for (0..VISITS) |il| {
            const ring: usize = if (isGlobal(layerOf(il))) ctx else WINDOW;
            a.ring[il] = @intCast(ring);
            a.kc[il] = try d.alloc_(ring * NKV * HD * 2);
            a.vc[il] = try d.alloc_(ring * NKV * HD * 2);
        }
        r.a = a;
        try r.reset();
        return r;
    }

    fn reset(r: *Runner) !void {
        try r.d.zero(r.a.hist, VISITS * 3 * HIST * 4);
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

    /// One token at `pos` through every visit; returns the greedy next token.
    fn step(r: *Runner, tok: u32, pos: usize) !u32 {
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
            const x0 = a.innorm;
            const x1 = if (nin == 3) off(a.innorm, C * 4) else a.innorm;
            const x2 = if (nin == 3) off(a.innorm, 2 * C * 4) else a.innorm;
            try d.launch(r.k.gemv_q8, .{ .x = PROWS / 16 }, .{ l.big.q, l.big.s, @as(i32, PROWS), @as(i32, C), x0, x1, x2, @as(i32, 6528), @as(i32, 7552), a.p });
            try d.launch(r.k.gemv_f32, .{ .x = (l.small_rows + 15) / 16 }, .{ l.small, @as(i32, @intCast(l.small_rows)), @as(i32, C), x0, x1, x2, @as(i32, 24), @as(i32, 56), a.dsm });
            const ring = a.ring[il];
            const hist = off(a.hist, il * 3 * HIST * 4);
            const pp: Prep = .{ .qk_clip = r.qk_clip, .rel_clip = r.rel_clip, .eps = m.eps, .tau = tau, .global_layer = @intFromBool(g), .slot = @intCast(pos % ring) };
            try d.launch(r.k.prep, .{ .x = 64 }, .{ a.p, a.dsm, l.kconv, l.vconv, hist, l.qclip, l.rclip, pp, a.q, a.r, a.kc[il], a.vc[il] });
            const p0: usize = if (g) 0 else pos -| (WINDOW - 1);
            const sp: Span = .{ .pos = @intCast(pos), .p0 = @intCast(p0), .ring = @intCast(ring), .extent = EXTENT };
            try d.launch(r.k.attn, .{ .x = NH }, .{ a.q, a.r, a.dsm, a.kc[il], a.vc[il], l.relb, sp, a.scratch, a.p, a.o });
            try d.launch(r.k.gemv_q8, .{ .x = C / 16 }, .{ l.wo.q, l.wo.s, @as(i32, C), @as(i32, NH * HD), a.o, a.o, a.o, @as(i32, C), @as(i32, C), a.y });
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
            try d.launch(r.k.gemv_f32, .{ .x = NE / 16 }, .{ l.router, @as(i32, NE), @as(i32, C), a.innorm, a.innorm, a.innorm, @as(i32, NE), @as(i32, NE), a.rlog });
            try d.launch(r.k.topk, .{ .x = 1 }, .{ a.rlog, l.probs_b, r.wscale, a.ids, a.wts });
            if (l.exp.nvfp4) {
                try d.launch(r.k.gateup_nvfp4, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, l.exp.gu_scale.?, a.ids, a.innorm, m.lut, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_nvfp4, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, l.exp.d_scale.?, a.ids, a.wts, a.act, m.lut, a.y });
            } else {
                try d.launch(r.k.gateup_q8, .{ .x = FF / 16, .y = TOP }, .{ l.exp.gu_q, l.exp.gu_s, a.ids, a.innorm, r.situ_b, r.situ_lb, a.act });
                try d.launch(r.k.down_q8, .{ .x = C / 16 }, .{ l.exp.d_q, l.exp.d_s, a.ids, a.wts, a.act, a.y });
            }
            used.appendAssumeCapacity(attn_out);
            used.appendSliceAssumeCapacity(blocks.items);
            const out = fresh(used.items);
            try d.launch(r.k.conv_res, .{ .x = C / 256 }, .{ a.y, l.fconv, hist, @as(i32, 4352), r.slot(attn_out), @as(i32, 1), r.slot(out) });
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

    fn dumpRow(r: *Runner, dir: []const u8, tag: []const u8, il: usize, pos: usize, src: *anyopaque, n: usize) !void {
        const buf = try alloc.alloc(u8, n * 4);
        defer alloc.free(buf);
        try r.d.read(buf, src);
        const path = try std.fmt.allocPrint(alloc, "{s}/{s}{d}_p{d}.f32", .{ dir, tag, il, pos });
        defer alloc.free(path);
        try writeFile(path, buf);
    }
};

fn readIds(path: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, try mapFile(path), " \n\r\t,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    return ids.items;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 4) {
        std.debug.print("usage: xpu-inkling2 MODEL.gguf IDS.json N_NEW [CTX]\n", .{});
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
    var next: u32 = 0;
    const t2 = now();
    for (ids, 0..) |tok, pos| next = try r.step(tok, pos);
    const t3 = now();
    std.debug.print("prompt {d} tokens one at a time: {d:.3} s\n", .{ ids.len, t3 - t2 });
    var gen: std.ArrayList(u32) = .empty;
    try gen.append(alloc, next);
    const t4 = now();
    for (1..n_new) |i| {
        next = try r.step(next, ids.len + i - 1);
        try gen.append(alloc, next);
    }
    const t5 = now();
    std.debug.print("generated:", .{});
    for (gen.items) |g| std.debug.print(" {d}", .{g});
    std.debug.print("\n", .{});
    if (n_new > 1) std.debug.print("decode {d} steps: {d:.3} s, {d:.2} tok/s (host sync every step, no graph replay)\n", .{ n_new - 1, t5 - t4, @as(f64, @floatFromInt(n_new - 1)) / (t5 - t4) });
}
