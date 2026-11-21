// Inkling 2 Mini (inkling2_loop) one-row decode on Intel Xe2 (OpenCL C), after llama.cpp src/models/inkling2.cpp.
// Activations fp32; dense weights Q8_0 as int8 rows + fp16 scales per 32; experts NVFP4 as nibbles + UE4M3 bytes per 16.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define C 2304
#define HD 128
#define NR 16
#define FF 1152
#define NE 128
#define TOP 8
#define NCONV 4
#define HIST 6656 // conv history channels a visit: k 1024, v 1024, attention out 2304, ffn out 2304
#define MAXS 8

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float sigmoid1(float x) { return 1.0f / (1.0f + exp(-x)); }

// Sum over a work-group of N threads (N/16 subgroups) in fixed order.
static float wg_sum(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  const int sg = get_sub_group_id(), ns = get_num_sub_groups();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < ns; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static float wg_max(float v, local float *red) {
  const float s = sub_group_reduce_max(v);
  const int sg = get_sub_group_id(), ns = get_num_sub_groups();
  if (get_sub_group_local_id() == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = -INFINITY;
  for (int i = 0; i < ns; i++) t = fmax(t, red[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// ---------------------------------------------------------------- embedding: Q8_0 row, rms norm with gamma
KERNEL(256) void ink2_embed(global const char *q, global const half *s, int tok, global const float *gamma, float eps,
                            global float *out) {
  local float red[16];
  const int t = get_local_id(0);
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    const int c = t + i * 256;
    v[i] = (float)q[(long)tok * C + c] * (float)s[(long)tok * (C / 32) + c / 32];
    ss += v[i] * v[i];
  }
  const float r = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) out[t + i * 256] = v[i] * r * gamma[t + i * 256];
}

// ---------------------------------------------------------------- depth mix: softmax over rms-normed states with queries
typedef struct {
  int n;
  int idx[MAXS];
} Slots;

// out[qi][c] = sum_s p[qi][s] * rms(S[idx[s]])[c]; p = softmax_s(rms(S_s) . query[qi] / sqrt(C)). One work-group.
KERNEL(512) void ink2_mix(global const float *S, Slots sl, global const float *query, int nq, global float *out) {
  local float red[32];
  local float rs[MAXS], sc[3][MAXS];
  const int t = get_local_id(0);
  for (int s = 0; s < sl.n; s++) {
    global const float *x = S + (long)sl.idx[s] * C;
    float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    for (int c = t; c < C; c += 512) {
      const float v = x[c];
      ss += v * v;
      d0 += v * query[c];
      if (nq > 1) {
        d1 += v * query[C + c];
        d2 += v * query[2 * C + c];
      }
    }
    ss = wg_sum(ss, red);
    d0 = wg_sum(d0, red);
    if (nq > 1) {
      d1 = wg_sum(d1, red);
      d2 = wg_sum(d2, red);
    }
    if (t == 0) {
      const float r = rsqrt(ss / C + 1e-6f);
      rs[s] = r;
      sc[0][s] = d0 * r * rsqrt((float)C);
      sc[1][s] = d1 * r * rsqrt((float)C);
      sc[2][s] = d2 * r * rsqrt((float)C);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < nq) {
    float m = -INFINITY;
    for (int s = 0; s < sl.n; s++) m = fmax(m, sc[t][s]);
    float z = 0.0f;
    for (int s = 0; s < sl.n; s++) {
      sc[t][s] = exp(sc[t][s] - m);
      z += sc[t][s];
    }
    for (int s = 0; s < sl.n; s++) sc[t][s] = sc[t][s] / z * rs[s];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int c = t; c < C; c += 512) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int s = 0; s < sl.n; s++) {
      const float v = S[(long)sl.idx[s] * C + c];
      a0 += sc[0][s] * v;
      a1 += sc[1][s] * v;
      a2 += sc[2][s] * v;
    }
    out[c] = a0;
    if (nq > 1) {
      out[C + c] = a1;
      out[2 * C + c] = a2;
    }
  }
}

// ---------------------------------------------------------------- rms norm with gamma, one vector a work-group
KERNEL(256) void ink2_norm(global const float *x, global const float *gamma, float eps, global float *y, int stride_x,
                           int stride_g) {
  local float red[16];
  const int t = get_local_id(0), w = get_group_id(0);
  x += (long)w * stride_x;
  gamma += (long)w * stride_g;
  y += (long)w * C;
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    v[i] = x[t + i * 256];
    ss += v[i] * v[i];
  }
  const float r = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) y[t + i * 256] = v[i] * r * gamma[t + i * 256];
}

// ---------------------------------------------------------------- loop message controller
// h_in = [rms(cur); rms(msg)] (2C), one work-group.
KERNEL(256) void ink2_ctl_in(global const float *cur, global const float *msg, global float *h) {
  local float red[16];
  const int t = get_local_id(0);
  float a = 0.0f, b = 0.0f;
  for (int c = t; c < C; c += 256) {
    a += cur[c] * cur[c];
    b += msg[c] * msg[c];
  }
  const float ra = rsqrt(wg_sum(a, red) / C + 1e-6f), rb = rsqrt(wg_sum(b, red) / C + 1e-6f);
  for (int c = t; c < C; c += 256) {
    h[c] = cur[c] * ra;
    h[C + c] = msg[c] * rb;
  }
}

// out[r] = (1 - tanh(F[r].h)) * cur[r] + tanh(G[r].h) * msg[r]; G, F [C][64] fp32; a subgroup a row.
KERNEL(256) void ink2_ctl_out(global const float *G, global const float *Fw, global const float *h, global const float *cur,
                              global const float *msg, global float *out) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  const float4 hv = vload4(lane, h);
  const float4 g = vload4(lane, G + (long)r * 64), f = vload4(lane, Fw + (long)r * 64);
  const float gs = sub_group_reduce_add(dot(g, hv)), fs = sub_group_reduce_add(dot(f, hv));
  if (lane == 0) out[r] = (1.0f - tanh(fs)) * cur[r] + tanh(gs) * msg[r];
}

// ---------------------------------------------------------------- fp32 rows: y[r] = W[r] . x_sel(r), K a multiple of 64
// Rows below r1 read x0, below r2 read x1, the rest x2. A subgroup a row, 16 rows a work-group.
KERNEL(256) void ink2_gemv_f32(global const float *W, int rows, int K, global const float *x0, global const float *x1,
                               global const float *x2, int r1, int r2, global float *y) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r >= rows) return;
  global const float *x = r < r1 ? x0 : (r < r2 ? x1 : x2);
  global const float *w = W + (long)r * K;
  float acc = 0.0f;
  for (int k = lane * 4; k < K; k += 64) acc += dot(vload4(0, w + k), vload4(0, x + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) y[r] = acc;
}

// ---------------------------------------------------------------- Q8_0 rows: y[r] = sum_k q[r][k] * s[r][k/32] * x(r)[k]
// K a multiple of 256; each lane reads 16 bytes (half a block) per step. Input selection as ink2_gemv_f32.
KERNEL(256) void ink2_gemv_q8(global const char *q, global const half *s, int rows, int K, global const float *x0,
                              global const float *x1, global const float *x2, int r1, int r2, global float *y) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r >= rows) return;
  global const float *x = r < r1 ? x0 : (r < r2 ? x1 : x2);
  global const char *qr = q + (long)r * K;
  global const half *sr = s + (long)r * (K / 32);
  float acc = 0.0f;
  for (int hb = lane; hb < K / 16; hb += 16) {
    const char16 w = vload16(hb, qr);
    const float16 xv = vload16(hb, x);
    const float16 wf = convert_float16(w);
    float d = dot(wf.lo.lo, xv.lo.lo) + dot(wf.lo.hi, xv.lo.hi) + dot(wf.hi.lo, xv.hi.lo) + dot(wf.hi.hi, xv.hi.hi);
    acc += d * (float)sr[hb >> 1];
  }
  acc = sub_group_reduce_add(acc);
  if (lane == 0) y[r] = acc;
}

// ---------------------------------------------------------------- attention prologue
// P: [q 3072 | rel 384 | gate 3072 | k 1024 | v 1024] projections; D: [sink 24 | kdyn 32 | vdyn 32] (global layers).
// Tasks (work-groups of 128): 0..23 q heads, 24..31 k heads, 32..39 v heads, 40..63 rel heads (16 lanes used).
// k, v: short conv over the history (static + dynamic taps on global layers), then k clip / v rms, stored as fp16.
typedef struct {
  float qk_clip, rel_clip, eps, tau;
  int global_layer, slot; // slot: cache row of this position
} Prep;

KERNEL(128) void ink2_prep(global const float *P, global const float *D, global const float *kconv,
                           global const float *vconv, global float *hist, global const float *qclip,
                           global const float *rclip, Prep pp, global float *Q, global float *R, global half *Kc,
                           global half *Vc) {
  local float red[8];
  const int t = get_local_id(0), task = get_group_id(0);
  if (task < 24) {
    const int h = task;
    float v = P[h * HD + t] * qclip[h * HD + t];
    const float n = sqrt(wg_sum(v * v, red)) + 1e-6f;
    v *= clamp(pp.qk_clip / n, 0.0f, 1.0f);
    if (pp.global_layer) v *= pp.tau;
    Q[h * HD + t] = v;
  } else if (task < 40) {
    const int isv = task >= 32, h = task - (isv ? 32 : 24);
    const int c = h * HD + t; // channel within k (or v)
    const float x = P[(isv ? 7552 : 6528) + c];
    global float *hs = hist + (isv ? 1024 : 0) + c; // [3][HIST] rows oldest first
    const float h0 = hs[0], h1 = hs[HIST], h2 = hs[2 * HIST];
    global const float *w = (isv ? vconv : kconv) + c * NCONV;
    float o = h0 * w[0] + h1 * w[1] + h2 * w[2] + x * w[3];
    float scale = 0.2236068f; // 1/sqrt(5)
    if (pp.global_layer) {
      global const float *dy = D + 24 + (isv ? 32 : 0) + h * NCONV; // [head][lag]
      o += x * dy[0] + h2 * dy[1] + h1 * dy[2] + h0 * dy[3];
      scale = 1.0f / 3.0f; // 1/sqrt(9)
    }
    hs[0] = h1;
    hs[HIST] = h2;
    hs[2 * HIST] = x;
    float v = (x + o) * scale;
    if (!isv) {
      const float n = sqrt(wg_sum(v * v, red)) + 1e-6f;
      v *= clamp(pp.qk_clip / n, 0.0f, 1.0f);
      Kc[(long)pp.slot * 1024 + c] = convert_half(v);
    } else {
      const float r = rsqrt(wg_sum(v * v, red) / HD + pp.eps);
      Vc[(long)pp.slot * 1024 + c] = convert_half(v * r);
    }
  } else {
    const int h = task - 40;
    float v = t < NR ? P[3072 + h * NR + t] * rclip[h * NR + t] : 0.0f;
    const float n = sqrt(wg_sum(v * v, red)) + 1e-6f;
    v *= clamp(pp.rel_clip / n, 0.0f, 1.0f);
    if (pp.global_layer) v *= pp.tau;
    if (t < NR) R[h * NR + t] = v;
  }
}

// ---------------------------------------------------------------- attention, one query head a work-group (v1)
// Keys at positions p0..pos; cache row of p = p % ring. Bias = relb[min(pos-p, extent)] . R[h] (row extent is zero), minus
// the sink logit t[h]; one extra logit 0 (the sink) joins the softmax. Output times sigmoid(gate).
typedef struct {
  int pos, p0, ring, extent;
} Span;

KERNEL(256) void ink2_attn(global const float *Q, global const float *R, global const float *D, global const half *Kc,
                           global const half *Vc, global const float *relb, Span sp, global float *scratch,
                           global const float *P, global float *O) {
  local float red[16];
  local float qs[HD], rv[NR], part[2][HD];
  const int t = get_local_id(0), h = get_group_id(0), kvh = h / 3;
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (t < HD) qs[t] = Q[h * HD + t];
  if (t < NR) rv[t] = R[h * NR + t];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sink = D[h];
  const int n = sp.pos - sp.p0 + 1;
  global float *sc = scratch + (long)h * sp.ring;
  float m = 0.0f; // the sink logit 0 is always in the softmax
  for (int j = sg; j < n; j += 16) {
    const int p = sp.p0 + j, d = sp.pos - p, row = p % sp.ring;
    global const half *k = Kc + (long)row * 1024 + kvh * HD;
    float a = 0.0f;
    for (int i = 0; i < 8; i++) a += (float)k[lane * 8 + i] * qs[lane * 8 + i];
    a = sub_group_reduce_add(a);
    const int bi = d < sp.extent ? d : sp.extent;
    float b = bi < sp.extent ? relb[bi * NR + lane] * rv[lane] : 0.0f;
    b = sub_group_reduce_add(b);
    const float s = a * 0.08838834764831845f + b - sink;
    if (lane == 0) sc[j] = s;
    m = fmax(m, s);
  }
  barrier(CLK_GLOBAL_MEM_FENCE); // the scores in scratch are read by other work-items below
  m = wg_max(m, red);
  float z = 0.0f;
  for (int j = t; j < n; j += 256) z += exp(sc[j] - m);
  z = wg_sum(z, red) + exp(-m);
  const int d = t & (HD - 1), half_ = t >> 7;
  float acc = 0.0f;
  for (int j = half_; j < n; j += 2) {
    const int row = (sp.p0 + j) % sp.ring;
    acc += exp(sc[j] - m) * (float)Vc[(long)row * 1024 + kvh * HD + d];
  }
  part[half_][d] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < HD) {
    const float o = (part[0][t] + part[1][t]) / z;
    O[h * HD + t] = o * sigmoid1(P[3456 + h * HD + t]);
  }
}

// ---------------------------------------------------------------- post conv (static) + residual
// y: projection out [C]; hist row offset `site` in [3][HIST]; out = (y + conv(y)) / sqrt(5) (+ res when res != 0).
KERNEL(256) void ink2_conv_res(global const float *y, global const float *w, global float *hist, int site,
                               global const float *res, int has_res, global float *out) {
  const int c = get_global_id(0);
  if (c >= C) return;
  global float *hs = hist + site + c;
  const float x = y[c], h0 = hs[0], h1 = hs[HIST], h2 = hs[2 * HIST];
  global const float *ww = w + c * NCONV;
  const float o = h0 * ww[0] + h1 * ww[1] + h2 * ww[2] + x * ww[3];
  hs[0] = h1;
  hs[HIST] = h2;
  hs[2 * HIST] = x;
  float v = (x + o) * 0.2236068f;
  if (has_res) v += res[c];
  out[c] = v;
}

// ---------------------------------------------------------------- router: sigmoid, selection bias, top 8, normalise x scale
KERNEL(128) void ink2_topk(global const float *logits, global const float *bias, float wscale, global int *ids,
                           global float *wts) {
  local float sel[NE], prob[NE];
  local int pick[TOP];
  const int t = get_local_id(0);
  const float p = sigmoid1(logits[t]);
  prob[t] = p;
  sel[t] = p + bias[t];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    float sum = 0.0f;
    for (int k = 0; k < TOP; k++) {
      int best = -1;
      float bv = -INFINITY;
      for (int e = 0; e < NE; e++)
        if (sel[e] > bv) {
          bv = sel[e];
          best = e;
        }
      pick[k] = best;
      sel[best] = -INFINITY;
      sum += prob[best];
    }
    sum += 1e-20f;
    for (int k = 0; k < TOP; k++) {
      ids[k] = pick[k];
      wts[k] = prob[pick[k]] / sum * wscale;
    }
  }
}

// ---------------------------------------------------------------- experts
// NVFP4 rows: qs [rows][K/2] (per 16 values: 8 bytes, low nibble = value j, high = value j+8), sc [rows][K/16] UE4M3.
// lut: 256 UE4M3 decodes (x 0.5); e2m1 values doubled, as ggml's kvalues.
constant float E2M1[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static inline float nvfp4_dot16(uint2 q, float sc, global const float *x) {
  float a = 0.0f;
  for (int j = 0; j < 4; j++) {
    const uint b0 = (q.x >> (8 * j)) & 0xFF, b1 = (q.y >> (8 * j)) & 0xFF;
    a += E2M1[b0 & 15] * x[j] + E2M1[b0 >> 4] * x[8 + j];
    a += E2M1[b1 & 15] * x[4 + j] + E2M1[b1 >> 4] * x[12 + j];
  }
  return a * sc;
}

static float nvfp4_row(global const uchar *qs, global const uchar *sc, int K, global const float *x,
                       constant float *lut) {
  const int lane = get_sub_group_local_id();
  float acc = 0.0f;
  for (int b = lane; b < K / 16; b += 16)
    acc += nvfp4_dot16(vload2(b, (global const uint *)qs), lut[sc[b]], x + b * 16);
  return sub_group_reduce_add(acc);
}

static float q8_row(global const char *q, global const half *s, int K, global const float *x) {
  const int lane = get_sub_group_local_id();
  float acc = 0.0f;
  for (int hb = lane; hb < K / 16; hb += 16) {
    const float16 wf = convert_float16(vload16(hb, q));
    const float16 xv = vload16(hb, x);
    acc += (dot(wf.lo.lo, xv.lo.lo) + dot(wf.lo.hi, xv.lo.hi) + dot(wf.hi.lo, xv.hi.lo) + dot(wf.hi.hi, xv.hi.hi)) *
           (float)s[hb >> 1];
  }
  return sub_group_reduce_add(acc);
}

static inline float situ(float g, float u, float b, float lb) {
  return b * tanh(g / b) * sigmoid1(g) * (lb * tanh(u / lb));
}

// gate_up, NVFP4: act[e][j] = situ(s_e * W[e][j] . x, s_e * W[e][FF + j] . x); grid (FF / 16, TOP).
KERNEL(256) void ink2_gateup_nvfp4(global const uchar *qs, global const uchar *sc, global const float *escale,
                                   global const int *ids, global const float *x, constant float *lut, float b, float lb,
                                   global float *act) {
  const int j = get_group_id(0) * 16 + get_sub_group_id(), slot = get_group_id(1);
  const int e = ids[slot];
  const long rows = 2 * FF;
  global const uchar *qe = qs + (long)e * rows * (C / 2);
  global const uchar *se = sc + (long)e * rows * (C / 16);
  const float g = nvfp4_row(qe + (long)j * (C / 2), se + (long)j * (C / 16), C, x, lut);
  const float u = nvfp4_row(qe + (long)(FF + j) * (C / 2), se + (long)(FF + j) * (C / 16), C, x, lut);
  if (get_sub_group_local_id() == 0) act[slot * FF + j] = situ(g * escale[e], u * escale[e], b, lb);
}

// down, NVFP4: y[r] = sum_slot w[slot] * s_e * W[e][r] . act[slot]; grid (C / 16).
KERNEL(256) void ink2_down_nvfp4(global const uchar *qs, global const uchar *sc, global const float *escale,
                                 global const int *ids, global const float *wts, global const float *act,
                                 constant float *lut, global float *y) {
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  float acc = 0.0f;
  for (int slot = 0; slot < TOP; slot++) {
    const int e = ids[slot];
    global const uchar *qe = qs + ((long)e * C + r) * (FF / 2);
    global const uchar *se = sc + ((long)e * C + r) * (FF / 16);
    acc += wts[slot] * escale[e] * nvfp4_row(qe, se, FF, act + slot * FF, lut);
  }
  if (get_sub_group_local_id() == 0) y[r] = acc;
}

KERNEL(256) void ink2_gateup_q8(global const char *q, global const half *s, global const int *ids, global const float *x,
                                float b, float lb, global float *act) {
  const int j = get_group_id(0) * 16 + get_sub_group_id(), slot = get_group_id(1);
  const int e = ids[slot];
  const long rows = 2 * FF;
  global const char *qe = q + (long)e * rows * C;
  global const half *se = s + (long)e * rows * (C / 32);
  const float g = q8_row(qe + (long)j * C, se + (long)j * (C / 32), C, x);
  const float u = q8_row(qe + (long)(FF + j) * C, se + (long)(FF + j) * (C / 32), C, x);
  if (get_sub_group_local_id() == 0) act[slot * FF + j] = situ(g, u, b, lb);
}

KERNEL(256) void ink2_down_q8(global const char *q, global const half *s, global const int *ids, global const float *wts,
                              global const float *act, global float *y) {
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  float acc = 0.0f;
  for (int slot = 0; slot < TOP; slot++) {
    const int e = ids[slot];
    acc += wts[slot] * q8_row(q + ((long)e * C + r) * FF, s + ((long)e * C + r) * (FF / 32), FF, act + slot * FF);
  }
  if (get_sub_group_local_id() == 0) y[r] = acc;
}

// ---------------------------------------------------------------- logits: softcap, then argmax in two stages
KERNEL(256) void ink2_softcap_argmax_part(global float *logits, int n, float cap, global float *pv, global int *pi) {
  local float red[16];
  local int redi[16];
  const int t = get_local_id(0), g = get_group_id(0);
  const int i = g * 256 + t;
  float v = -INFINITY;
  if (i < n) {
    v = tanh(logits[i] / cap) * cap;
    logits[i] = v;
  }
  // lowest id on ties: max value, then min index among equals
  const float m = sub_group_reduce_max(v);
  int idx = v == m && i < n ? i : 0x7fffffff;
  idx = sub_group_reduce_min(idx);
  const int sg = get_sub_group_id();
  if (get_sub_group_local_id() == 0) {
    red[sg] = m;
    redi[sg] = idx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    float bm = -INFINITY;
    int bi = 0x7fffffff;
    for (int k = 0; k < 16; k++)
      if (red[k] > bm || (red[k] == bm && redi[k] < bi)) {
        bm = red[k];
        bi = redi[k];
      }
    pv[g] = bm;
    pi[g] = bi;
  }
}

KERNEL(256) void ink2_argmax_final(global const float *pv, global const int *pi, int n, global int *out) {
  local float red[256];
  local int redi[256];
  const int t = get_local_id(0);
  float bm = -INFINITY;
  int bi = 0x7fffffff;
  for (int k = t; k < n; k += 256)
    if (pv[k] > bm || (pv[k] == bm && pi[k] < bi)) {
      bm = pv[k];
      bi = pi[k];
    }
  red[t] = bm;
  redi[t] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    for (int k = 1; k < 256; k++)
      if (red[k] > bm || (red[k] == bm && redi[k] < bi)) {
        bm = red[k];
        bi = redi[k];
      }
    out[0] = bi;
  }
}
