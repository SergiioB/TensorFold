//! XPU_SPLIT=N: decode rows run N x 1,024 of every MLP's neurons on a second B70 and the rest on the first.
//! The peer holds only MLP weight slices: no K/V, no GDN state, no drafter. An exchange is plain stores into the
//! peer's memory and one flag word (kernels/xpu/qwen27/split.cl). Prompt chunks stay on the first card.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const loader = @import("load.zig");
const Checkpoint = dev.Checkpoint;
const D = q.D;
const Dev = dev.Dev;
const FF = q.FF;
const LAYERS = q.LAYERS;
const Proj = loader.Proj;
const alloc = q.alloc;
const off = dev.off;

const ROWS = 16; // decode and verification rows
const CAP: i32 = 2_000_000; // polls of one wait (about 0.3 to 0.5 s), under the xe job timeout

pub const Split = struct {
    b: Dev, // the peer card
    ffa: u32, // neurons on the first card
    ffb: u32, // neurons on the peer
    on: bool = true,
    epoch: i32 = 0,
    gu_a: [LAYERS]Proj = undefined,
    down_a: [LAYERS]Proj = undefined,
    gu_b: [LAYERS]Proj = undefined,
    down_b: [LAYERS]Proj = undefined,
    // on the peer: the inbox the first card writes, then the peer's own activations
    in_h: *anyopaque,
    in_hi: *anyopaque,
    in_lo: *anyopaque,
    bh: *anyopaque,
    bahi: *anyopaque,
    balo: *anyopaque,
    bp: *anyopaque,
    bact: *anyopaque,
    bahi2: *anyopaque,
    balo2: *anyopaque,
    br: *anyopaque,
    bpscr: *anyopaque,
    bflag: *anyopaque,
    bstat: *anyopaque,
    // on the first card: the peer's part of the down projection
    rb: *anyopaque,
    aflag: *anyopaque,
    astat: *anyopaque,
    host_stat: [2]u32 = .{ 0, 0 },

    pub fn init(d: *Dev, c: *const Checkpoint, n: u32) !*Split {
        if (q.GS != 64 or q.rot_on != 0) return error.SplitNeedsPlainMlxWeights;
        if (n == 0 or n >= FF / 1024) return error.SplitOutOfRange;
        const s = try alloc.create(Split);
        const ffb = n * 1024;
        s.* = .{
            .b = .{ .g = try d.g.openPeer(), .list = undefined, .k = d.k, .noprof = true },
            .ffa = FF - ffb,
            .ffb = ffb,
            .in_h = undefined,
            .in_hi = undefined,
            .in_lo = undefined,
            .bh = undefined,
            .bahi = undefined,
            .balo = undefined,
            .bp = undefined,
            .bact = undefined,
            .bahi2 = undefined,
            .balo2 = undefined,
            .br = undefined,
            .bpscr = undefined,
            .bflag = undefined,
            .bstat = undefined,
            .rb = undefined,
            .aflag = undefined,
            .astat = undefined,
        };
        const b = &s.b;
        b.list = try b.g.immediate(0, 0);
        const kb = q.kernel_bins;
        const mg = try b.g.module(&kb.glue);
        const mm = try b.g.module(&kb.gemm);
        const ms = try b.g.module(&kb.split);
        b.k.repack = try b.kernel(mg, "repack", 16);
        b.k.dec_wide = try b.kernel(mm, "w4k1_decode", 64);
        b.k.dec_wide64 = try b.kernel(mm, "w4k1_d64", 64);
        b.k.dec_narrow = try b.kernel(mm, "w4k_dsplit", 64);
        b.k.peer_copy = try b.kernel(ms, "peer_copy", 256);
        b.k.peer_flag = try b.kernel(ms, "peer_flag", 16);
        b.k.peer_wait = try b.kernel(ms, "peer_wait", 16);
        b.k.add_half = try b.kernel(ms, "add_half", 256);
        b.k.swiglu_n = try b.kernel(ms, "swiglu_n", 256);
        s.in_h = try b.alloc_(ROWS * D * 2);
        s.in_hi = try b.alloc_(ROWS * (D / 64) * 2);
        s.in_lo = try b.alloc_(ROWS * (D / 64) * 2);
        s.bh = try b.alloc_(ROWS * D * 2);
        s.bahi = try b.alloc_(ROWS * (D / 64) * 2);
        s.balo = try b.alloc_(ROWS * (D / 64) * 2);
        s.bp = try b.alloc_(ROWS * 2 * ffb * 2);
        s.bact = try b.alloc_(ROWS * ffb * 2);
        s.bahi2 = try b.alloc_(ROWS * (ffb / 64) * 2);
        s.balo2 = try b.alloc_(ROWS * (ffb / 64) * 2);
        s.br = try b.alloc_(ROWS * D * 2);
        s.bpscr = try b.alloc_(ROWS * D * 4);
        s.bflag = try b.alloc_(4096);
        s.bstat = try b.alloc_(4096);
        s.rb = try d.alloc_(ROWS * D * 2);
        s.aflag = try d.alloc_(4096);
        s.astat = try d.alloc_(4096);
        try b.fill(s.bflag, @as(u32, 0), 4096);
        try b.fill(s.bstat, @as(u32, 0), 4096);
        try d.fill(s.aflag, @as(u32, 0), 4096);
        try d.fill(s.astat, @as(u32, 0), 4096);
        const stage_a = try d.alloc_(128 << 20);
        const stage_b = try b.alloc_(128 << 20);
        for (0..LAYERS) |i| {
            const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.mlp.", .{i});
            defer alloc.free(p);
            s.gu_a[i] = try slice(d, c, p, &.{ "gate_proj", "up_proj" }, true, true, stage_a, 0, s.ffa, 0, D);
            s.down_a[i] = try slice(d, c, p, &.{"down_proj"}, false, false, stage_a, 0, D, 0, s.ffa);
            s.gu_b[i] = try slice(b, c, p, &.{ "gate_proj", "up_proj" }, true, true, stage_b, s.ffa, ffb, 0, D);
            s.down_b[i] = try slice(b, c, p, &.{"down_proj"}, false, false, stage_b, 0, D, s.ffa, ffb);
        }
        try d.sync();
        try b.sync();
        d.g.free(stage_a);
        d.allocated -= 128 << 20;
        b.g.free(stage_b);
        std.debug.print("XPU_SPLIT: MLP neurons {d} on the first card, {d} on the peer ({d:.2} GiB there)\n", .{ s.ffa, ffb, @as(f64, @floatFromInt(b.allocated - (128 << 20))) / (1 << 30) });
        return s;
    }

    /// One layer's MLP for `T` decode rows: a.h (the normed residual) in, a.r (the down projection) out.
    pub fn mlp(s: *Split, d: *Dev, a: *fwd.Acts, i: usize, T: u32) !void {
        const b = &s.b;
        s.epoch += 1;
        const e = s.epoch;
        const gs: i32 = 64;
        const nh: i32 = @intCast(T * D / 2); // 4-byte words of T rows of h
        const ng: i32 = @intCast(T * (D / 64) / 2);
        const zero: i32 = 0;
        const cg: ze.GroupCount = .{ .x = (T + 1) / 2 };
        const one: ze.GroupCount = .{ .x = 1 };
        // first card: the MLP input to the peer
        try d.launch(d.k.peer_copy, cg, .{ a.h, s.in_h, nh, a.ahi, s.in_hi, ng, a.alo, s.in_lo, ng });
        try d.launch(d.k.peer_flag, one, .{ s.bflag, e, s.astat });
        // peer: its slice of gate|up, SwiGLU and down, and the result into the first card's memory
        try b.launch(b.k.peer_wait, one, .{ s.bflag, e, CAP, s.bstat });
        try b.launch(b.k.peer_copy, cg, .{ s.in_h, s.bh, nh, s.in_hi, s.bahi, ng, s.in_lo, s.balo, ng });
        try fwd.gemmRows(b, s.bpscr, s.bh, s.bahi, s.balo, s.gu_b[i], s.bp, T, false);
        try b.launch(b.k.swiglu_n, .{ .x = T, .y = s.ffb / 1024 }, .{ s.bp, s.bact, s.bahi2, s.balo2, gs, @as(i32, @intCast(s.ffb)) });
        try fwd.gemmRows(b, s.bpscr, s.bact, s.bahi2, s.balo2, s.down_b[i], s.br, T, false);
        try b.launch(b.k.peer_copy, cg, .{ s.br, s.rb, nh, s.br, s.rb, zero, s.br, s.rb, zero });
        try b.launch(b.k.peer_flag, one, .{ s.aflag, e, s.bstat });
        // first card: its own slice, then the peer's part added
        try fwd.gemmRows(d, a.pscr, a.h, a.ahi, a.alo, s.gu_a[i], a.p, T, false);
        try d.launch(d.k.swiglu_n, .{ .x = T, .y = s.ffa / 1024 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, @intCast(s.ffa)) });
        try fwd.gemmRows(d, a.pscr, a.h, a.ahi, a.alo, s.down_a[i], a.r, T, false);
        try d.launch(d.k.peer_wait, one, .{ s.aflag, e, CAP, s.astat });
        try d.launch(d.k.add_half, cg, .{ a.r, s.rb, @as(i32, @intCast(T * D)) });
    }

    /// Waits for both cards and fails closed when an exchange timed out.
    pub fn finish(s: *Split, d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &s.host_stat, s.astat, 8, null, 0, null));
        try d.sync();
        s.b.settle();
        if (s.host_stat[0] != 0) return error.PeerExchangeTimedOut;
    }
};

/// Rows r0 .. r0 + rn of each part and columns c0 .. c0 + cn (multiples of 64) of an MLX projection, in the GEMM layout.
fn slice(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, ilv: bool, stage: *anyopaque, r0: u32, rn: u32, c0: u32, cn: u32) !Proj {
    const n_total: u32 = @intCast(parts.len * rn);
    if (n_total % 256 != 0 or cn % 256 != 0 or c0 % 64 != 0) return error.BadSplitShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, cn / 8) * n_total * 4), .s = try d.alloc_(@as(usize, cn / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, cn / 64) * n_total * 2), .n = n_total, .k = cn, .wide = wide, .ilv = ilv };
    const wrow: usize = (cn / 8) * 4;
    const srow: usize = (cn / 64) * 2;
    const tw = try alloc.alloc(u8, rn * wrow);
    defer alloc.free(tw);
    const ts = try alloc.alloc(u8, rn * srow);
    defer alloc.free(ts);
    const tb = try alloc.alloc(u8, rn * srow);
    defer alloc.free(tb);
    var n0: u32 = 0;
    for (parts, 0..) |part, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, part }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, part }));
        const bi = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, part }));
        const k: usize = @intCast(w.shape[1] * 8);
        if (r0 + rn > w.shape[0] or c0 + cn > k) return error.BadSplitShape;
        for (0..rn) |r| {
            const row = r0 + r;
            @memcpy(tw[r * wrow ..][0..wrow], w.bytes[(row * (k / 8) + c0 / 8) * 4 ..][0..wrow]);
            @memcpy(ts[r * srow ..][0..srow], sc.bytes[(row * (k / 64) + c0 / 64) * 2 ..][0..srow]);
            @memcpy(tb[r * srow ..][0..srow], bi.bytes[(row * (k / 64) + c0 / 64) * 2 ..][0..srow]);
        }
        const ss = off(stage, tw.len);
        const sb = off(ss, ts.len);
        try d.copyIn(stage, tw);
        try d.copyIn(ss, ts);
        try d.copyIn(sb, tb);
        try d.launch(d.k.repack, .{ .x = (rn + 15) / 16, .y = cn / 8 }, .{ stage, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(rn)), @as(i32, @intCast(cn)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += rn;
    }
    return out;
}
