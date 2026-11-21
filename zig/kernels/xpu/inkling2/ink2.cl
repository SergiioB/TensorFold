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
                           global const float *vconv, global const float *hist, global float *hout, global const float *qclip,
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
    global const float *hs = hist + (isv ? 1024 : 0) + c; // [3][HIST] rows oldest first
    global float *ho = hout + (isv ? 1024 : 0) + c;
    const float h0 = hs[0], h1 = hs[HIST], h2 = hs[2 * HIST];
    global const float *w = (isv ? vconv : kconv) + c * NCONV;
    float o = h0 * w[0] + h1 * w[1] + h2 * w[2] + x * w[3];
    float scale = 0.2236068f; // 1/sqrt(5)
    if (pp.global_layer) {
      global const float *dy = D + 24 + (isv ? 32 : 0) + h * NCONV; // [head][lag]
      o += x * dy[0] + h2 * dy[1] + h1 * dy[2] + h0 * dy[3];
      scale = 1.0f / 3.0f; // 1/sqrt(9)
    }
    ho[0] = h1;
    ho[HIST] = h2;
    ho[2 * HIST] = x;
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

// ================================================================ v2: fewer launches a visit
// History buffers are double: a step reads [pos % 2] and writes [(pos + 1) % 2], so kernels never race on them.

// Depth mix with up to three query sets (outputs in order): out[i][c] for i < n1 + n2 + n3 (at most 4).
// One pass over the states: every thread keeps (sum of squares, dot with each query) per state, then one reduction.
static void mix_core(global const float *S, long sstride, long row_off, Slots sl, global const float *q1, int n1,
                     global const float *q2, int n2, global const float *q3, int n3, global float *out, long ostride,
                     local float *part, local float *tot, global const float *gamma, int gstride, float eps) {
  const int t = get_local_id(0), nq = n1 + n2 + n3, ns = sl.n;
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();
  global const float *qp[4];
  for (int i = 0; i < 4; i++) qp[i] = i < n1 ? q1 + i * C : (i < n1 + n2 ? q2 + (i - n1) * C : q3 + (i - n1 - n2) * C);
  float acc[MAXS][5];
  for (int s = 0; s < MAXS; s++)
    for (int k = 0; k < 5; k++) acc[s][k] = 0.0f;
  for (int c = t; c < C; c += 512) {
    float qv[4];
    for (int i = 0; i < 4; i++) qv[i] = i < nq ? qp[i][c] : 0.0f;
    for (int s = 0; s < MAXS; s++) {
      if (s >= ns) break;
      const float v = S[(long)sl.idx[s] * sstride + row_off + c];
      acc[s][0] += v * v;
      for (int i = 0; i < 4; i++) acc[s][1 + i] += v * qv[i];
    }
  }
  for (int s = 0; s < MAXS; s++) {
    if (s >= ns) break;
    for (int k = 0; k < 5; k++) {
      const float r = sub_group_reduce_add(acc[s][k]);
      if (lane == 0) part[sg * (MAXS * 5) + s * 5 + k] = r;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < ns * 5) {
    float a = 0.0f;
    for (int g = 0; g < nsg; g++) a += part[g * (MAXS * 5) + t];
    tot[t] = a;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  local float *w = part; // reuse: w[i * MAXS + s] = softmax weight times the state's rms factor
  if (t < nq) {
    float sc[MAXS], m = -INFINITY;
    for (int s = 0; s < ns; s++) {
      const float r = rsqrt(tot[s * 5] / C + 1e-6f);
      sc[s] = tot[s * 5 + 1 + t] * r * rsqrt((float)C);
      m = fmax(m, sc[s]);
    }
    float z = 0.0f;
    for (int s = 0; s < ns; s++) {
      sc[s] = exp(sc[s] - m);
      z += sc[s];
    }
    for (int s = 0; s < ns; s++) w[t * MAXS + s] = sc[s] / z * rsqrt(tot[s * 5] / C + 1e-6f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float a[5][4];
  float ss[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < 5; k++) {
    const int c = t + k * 512;
    for (int i = 0; i < 4; i++) a[k][i] = 0.0f;
    if (c >= C) continue;
    for (int s = 0; s < MAXS; s++) {
      if (s >= ns) break;
      const float v = S[(long)sl.idx[s] * sstride + row_off + c];
      for (int i = 0; i < 4; i++) a[k][i] += w[i * MAXS + s] * v;
    }
    for (int i = 0; i < 4; i++) ss[i] += a[k][i] * a[k][i];
  }
  float rn[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  if (gamma) { // rms norm of each output with its gamma (gamma + i * gstride), one more reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < nq; i++) {
      const float r = sub_group_reduce_add(ss[i]);
      if (lane == 0) part[sg * 4 + i] = r;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < nq; i++) {
      float z = 0.0f;
      for (int g = 0; g < nsg; g++) z += part[g * 4 + i];
      rn[i] = rsqrt(z / C + eps);
    }
  }
  for (int k = 0; k < 5; k++) {
    const int c = t + k * 512;
    if (c >= C) break;
    for (int i = 0; i < nq; i++) out[i * ostride + c] = gamma ? a[k][i] * rn[i] * gamma[i * gstride + c] : a[k][i];
  }
}

KERNEL(512) void ink2_mix3(global const float *S, Slots sl, global const float *q1, int n1, global const float *q2, int n2,
                           global const float *q3, int n3, global float *out) {
  local float part[32 * MAXS * 5];
  local float tot[MAXS * 5];
  mix_core(S, C, 0, sl, q1, n1, q2, n2, q3, n3, out, C, part, tot, 0, 0, 0.0f);
}

// mix3 with the rms norm (gamma + i * gstride) of each output: the GEMVs after it read ready inputs.
KERNEL(512) void ink2_mix3n(global const float *S, Slots sl, global const float *q1, int n1, global const float *q2,
                            int n2, global const float *q3, int n3, global float *out, global const float *gamma,
                            int gstride, float eps) {
  local float part[32 * MAXS * 5];
  local float tot[MAXS * 5];
  mix_core(S, C, 0, sl, q1, n1, q2, n2, q3, n3, out, C, part, tot, gamma, gstride, eps);
}

// Controller, part 1: hd[j][r] = Wd[m0+j][r] . [rms(cur_j); rms(msg_j)]; grid (4, controllers).
KERNEL(256) void ink2_ctl_down(global const float *cur, int cur_stride, global const float *msg, int msg_stride,
                               global const float *Wd, int m0, global float *hd) {
  local float xs[2 * C];
  local float red[16];
  const int t = get_local_id(0), j = get_group_id(1);
  global const float *c = cur + j * cur_stride;
  global const float *g = msg + j * msg_stride;
  float a = 0.0f, b = 0.0f;
  for (int k = t; k < C; k += 256) {
    a += c[k] * c[k];
    b += g[k] * g[k];
  }
  const float ra = rsqrt(wg_sum(a, red) / C + 1e-6f), rb = rsqrt(wg_sum(b, red) / C + 1e-6f);
  for (int k = t; k < C; k += 256) {
    xs[k] = c[k] * ra;
    xs[C + k] = g[k] * rb;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  global const float *w = Wd + ((long)(m0 + j) * 64 + r) * (2 * C);
  float acc = 0.0f;
  for (int k = lane * 4; k < 2 * C; k += 64) acc += dot(vload4(0, w + k), vload4(0, xs + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) hd[j * 64 + r] = acc;
}

// Controller, part 1 with 8 rows a work-group of 128: grid (8, controllers).
KERNEL(128) void ink2_ctl_down8(global const float *cur, int cur_stride, global const float *msg, int msg_stride,
                               global const float *Wd, int m0, global float *hd) {
  local float xs[2 * C];
  local float red[16];
  const int t = get_local_id(0), j = get_group_id(1);
  global const float *c = cur + j * cur_stride;
  global const float *g = msg + j * msg_stride;
  float a = 0.0f, b = 0.0f;
  for (int k = t; k < C; k += 128) {
    a += c[k] * c[k];
    b += g[k] * g[k];
  }
  const float ra = rsqrt(wg_sum(a, red) / C + 1e-6f), rb = rsqrt(wg_sum(b, red) / C + 1e-6f);
  for (int k = t; k < C; k += 128) {
    xs[k] = c[k] * ra;
    xs[C + k] = g[k] * rb;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 8 + get_sub_group_id();
  global const float *w = Wd + ((long)(m0 + j) * 64 + r) * (2 * C);
  float acc = 0.0f;
  for (int k = lane * 4; k < 2 * C; k += 64) acc += dot(vload4(0, w + k), vload4(0, xs + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) hd[j * 64 + r] = acc;
}

// Controller, part 2: out[j][r] = (1 - tanh(F.h)) cur + tanh(G.h) msg; grid (C / 16, controllers).
KERNEL(256) void ink2_ctl_out2(global const float *G, global const float *Fw, int m0, global const float *hd,
                               global const float *cur, int cur_stride, global const float *msg, int msg_stride,
                               global float *out) {
  const int lane = get_sub_group_local_id(), j = get_group_id(1);
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  const long row = ((long)(m0 + j) * C + r) * 64;
  const float4 hv = vload4(lane, hd + j * 64);
  const float gs = sub_group_reduce_add(dot(vload4(lane, G + row), hv));
  const float fs = sub_group_reduce_add(dot(vload4(lane, Fw + row), hv));
  if (lane == 0) out[j * C + r] = (1.0f - tanh(fs)) * cur[j * cur_stride + r] + tanh(gs) * msg[j * msg_stride + r];
}

// Normalised inputs (rms with gamma) of nsel vectors into local memory, then rows: Q8_0 [0, PROWS) into P, fp32 small
// rows into D. Selection: q/rel/gate rows read input 0, k rows input 1, v rows input 2 (one input when nsel == 1).
#define PROWS 8576
KERNEL(256) void ink2_gemv_big(global const char *q, global const half *s, global const float *Wsm, int nsmall,
                               global const float *x0, global const float *x1, global const float *x2, int nsel,
                               global const float *gamma, int gstride, float eps, global float *P, global float *D) {
  local float xs[3][C];
  local float red[16];
  const int t = get_local_id(0);
  for (int j = 0; j < nsel; j++) {
    global const float *x = j == 0 ? x0 : (j == 1 ? x1 : x2);
    float v[9], ss = 0.0f;
    for (int i = 0; i < 9; i++) {
      v[i] = x[t + i * 256];
      ss += v[i] * v[i];
    }
    const float r = rsqrt(wg_sum(ss, red) / C + eps);
    for (int i = 0; i < 9; i++) xs[j][t + i * 256] = v[i] * r * gamma[j * gstride + t + i * 256];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r < PROWS) {
    const int sel = nsel == 1 ? 0 : (r < 6528 ? 0 : (r < 7552 ? 1 : 2));
    local const float *x = xs[sel];
    global const char *qr = q + (long)r * C;
    global const half *sr = s + (long)r * (C / 32);
    float acc = 0.0f;
    for (int hb = lane; hb < C / 16; hb += 16) {
      const float16 wf = convert_float16(vload16(hb, qr));
      const float16 xv = vload16(hb, x);
      acc += (dot(wf.lo.lo, xv.lo.lo) + dot(wf.lo.hi, xv.lo.hi) + dot(wf.hi.lo, xv.hi.lo) + dot(wf.hi.hi, xv.hi.hi)) *
             (float)sr[hb >> 1];
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) P[r] = acc;
  } else if (r < PROWS + nsmall) {
    const int rr = r - PROWS;
    const int sel = nsel == 1 ? 0 : (rr < 24 ? 0 : (rr < 56 ? 1 : 2));
    local const float *x = xs[sel];
    global const float *w = Wsm + (long)rr * C;
    float acc = 0.0f;
    for (int k = lane * 4; k < C; k += 64) acc += dot(vload4(0, w + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (lane == 0) D[rr] = acc;
  }
}

// Attention with its prologue, one query head a work-group: q, rel of the head; k, v of the current row for its key head
// (the first query head of the group stores them in the cache and the history); keys p0..pos-1 from the cache.
KERNEL(256) void ink2_attn2(global const float *P, global const float *D, global const float *kconv,
                            global const float *vconv, global const float *hin, global float *hout,
                            global const float *qclip, global const float *rclip, Prep pp, global half *Kc,
                            global half *Vc, global const float *relb, Span sp, global float *scratch, global float *O) {
  local float red[16];
  local float qs[HD], rv[NR], kcur[HD], vcur[HD], part[2][HD];
  const int t = get_local_id(0), h = get_group_id(0), kvh = h / 3, owner = h % 3 == 0;
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  // q and rel of this head
  float qv = t < HD ? P[h * HD + t] * qclip[h * HD + t] : 0.0f;
  const float qn = sqrt(wg_sum(qv * qv, red)) + 1e-6f;
  float rr = t < NR ? P[3072 + h * NR + t] * rclip[h * NR + t] : 0.0f;
  const float rn = sqrt(wg_sum(rr * rr, red)) + 1e-6f;
  const float tau = pp.global_layer ? pp.tau : 1.0f;
  if (t < HD) qs[t] = qv * clamp(pp.qk_clip / qn, 0.0f, 1.0f) * tau;
  if (t < NR) rv[t] = rr * clamp(pp.rel_clip / rn, 0.0f, 1.0f) * tau;
  // k (threads 0..127) and v (128..255) of the current row
  const int isv = t >= HD, c = kvh * HD + (t & (HD - 1));
  const float x = P[(isv ? 7552 : 6528) + c];
  const int site = (isv ? 1024 : 0) + c;
  const float h0 = hin[site], h1 = hin[HIST + site], h2 = hin[2 * HIST + site];
  global const float *w = (isv ? vconv : kconv) + c * NCONV;
  float o = h0 * w[0] + h1 * w[1] + h2 * w[2] + x * w[3];
  float scale = 0.2236068f;
  if (pp.global_layer) {
    global const float *dy = D + 24 + (isv ? 32 : 0) + kvh * NCONV;
    o += x * dy[0] + h2 * dy[1] + h1 * dy[2] + h0 * dy[3];
    scale = 1.0f / 3.0f;
  }
  float val = (x + o) * scale;
  const float ks = wg_sum(isv ? 0.0f : val * val, red), vs = wg_sum(isv ? val * val : 0.0f, red);
  val *= isv ? rsqrt(vs / HD + pp.eps) : clamp(pp.qk_clip / (sqrt(ks) + 1e-6f), 0.0f, 1.0f);
  const half hv = convert_half(val);
  if (isv) vcur[t - HD] = (float)hv;
  else kcur[t] = (float)hv;
  if (owner) {
    hout[site] = h1;
    hout[HIST + site] = h2;
    hout[2 * HIST + site] = x;
    (isv ? Vc : Kc)[(long)pp.slot * 1024 + c] = hv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sink = D[h];
  const int n = sp.pos - sp.p0 + 1;
  global float *sc = scratch + (long)h * sp.ring;
  float m = 0.0f;
  for (int j = sg; j < n; j += 16) {
    const int p = sp.p0 + j, d = sp.pos - p, row = p % sp.ring;
    float a = 0.0f;
    if (j == n - 1) {
      for (int i = 0; i < 8; i++) a += kcur[lane * 8 + i] * qs[lane * 8 + i];
    } else {
      const half8 k = vload8(0, Kc + (long)row * 1024 + kvh * HD + lane * 8);
      const float8 kf = convert_float8(k);
      const float8 qf = vload8(0, qs + lane * 8);
      a = dot(kf.lo, qf.lo) + dot(kf.hi, qf.hi);
    }
    a = sub_group_reduce_add(a);
    const int bi = d < sp.extent ? d : sp.extent;
    float b = bi < sp.extent ? relb[bi * NR + lane] * rv[lane] : 0.0f;
    b = sub_group_reduce_add(b);
    const float s = a * 0.08838834764831845f + b - sink;
    if (lane == 0) sc[j] = s;
    m = fmax(m, s);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  m = wg_max(m, red);
  float z = 0.0f;
  for (int j = t; j < n; j += 256) z += exp(sc[j] - m);
  z = wg_sum(z, red) + exp(-m);
  const int d = t & (HD - 1), half_ = t >> 7;
  float acc = 0.0f;
  for (int j = half_; j < n - 1; j += 2) {
    const int row = (sp.p0 + j) % sp.ring;
    acc += exp(sc[j] - m) * (float)Vc[(long)row * 1024 + kvh * HD + d];
  }
  if (half_ == ((n - 1) & 1)) acc += exp(sc[n - 1] - m) * vcur[d];
  part[half_][d] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < HD) O[h * HD + t] = (part[0][t] + part[1][t]) / z * sigmoid1(P[3456 + h * HD + t]);
}

// Post conv of one channel (static taps) from the double history; returns (y + conv(y)) / sqrt(5).
static inline float conv_post(float y, global const float *w, global const float *hin, global float *hout, int idx) {
  const float h0 = hin[idx], h1 = hin[HIST + idx], h2 = hin[2 * HIST + idx];
  const float o = h0 * w[0] + h1 * w[1] + h2 * w[2] + y * w[3];
  hout[idx] = h1;
  hout[HIST + idx] = h2;
  hout[2 * HIST + idx] = y;
  return (y + o) * 0.2236068f;
}

// Q8_0 rows with the post conv and the residual: out[r] = conv(W[r] . x) (+ res[r]); C rows, grid C / 16.
KERNEL(256) void ink2_gemv_conv(global const char *q, global const half *s, int K, global const float *x,
                                global const float *w, global const float *hin, global float *hout, int site,
                                global const float *res, int has_res, global float *out) {
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  const float y = q8_row(q + (long)r * K, s + (long)r * (K / 32), K, x);
  if (get_sub_group_local_id() == 0) {
    float v = conv_post(y, w + r * NCONV, hin, hout, site + r);
    if (has_res) v += res[r];
    out[r] = v;
  }
}

// Router: rms norm with gamma (work-group 0 writes the normalised row for the experts), fp32 rows, then the last
// work-group to finish picks the top 8 (sigmoid, selection bias, weights normalised and scaled). Grid NE / 16.
KERNEL(256) void ink2_router(global const float *x, global const float *gamma, float eps, global const float *W,
                             global float *xn, global float *logits, global const float *bias, float wscale,
                             global int *ids, global float *wts, global int *counter) {
  local float xs[C];
  local float red[16];
  local int redi[16];
  local int last;
  local float sel[NE], prob[NE], pw[TOP];
  local int pid[TOP];
  const int t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    v[i] = x[t + i * 256];
    ss += v[i] * v[i];
  }
  const float rn = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) {
    const float y = v[i] * rn * gamma[t + i * 256];
    xs[t + i * 256] = y;
    if (get_group_id(0) == 0) xn[t + i * 256] = y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int r = get_group_id(0) * 16 + sg;
  float acc = 0.0f;
  for (int k = lane * 4; k < C; k += 64) acc += dot(vload4(0, W + (long)r * C + k), vload4(0, xs + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) logits[r] = acc;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (t == 0) {
    const int old = atomic_fetch_add_explicit((volatile global atomic_int *)counter, 1, memory_order_acq_rel,
                                              memory_scope_device);
    last = old == (int)get_num_groups(0) - 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!last) return;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  if (t < NE) {
    const float p = sigmoid1(((volatile global float *)logits)[t]);
    prob[t] = p;
    sel[t] = p + bias[t];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  for (int k = 0; k < TOP; k++) {
    const float sv = t < NE ? sel[t] : -INFINITY;
    const float mx = sub_group_reduce_max(sv);
    int idx = sv == mx && t < NE ? t : 0x7fffffff;
    idx = sub_group_reduce_min(idx);
    if (lane == 0) {
      red[sg] = mx;
      redi[sg] = idx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float bm = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = 0; i < 16; i++)
      if (red[i] > bm || (red[i] == bm && redi[i] < bi)) {
        bm = red[i];
        bi = redi[i];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      pid[k] = bi;
      pw[k] = prob[bi];
      sel[bi] = -INFINITY;
    }
    sum += prob[bi];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t < TOP) {
    ids[t] = pid[t];
    wts[t] = pw[t] / (sum + 1e-20f) * wscale;
  }
  if (t == 0) *counter = 0;
}

// Router with 8 rows a work-group of 128 (grid NE / 8); otherwise as ink2_router.
// Router: rms norm with gamma (work-group 0 writes the normalised row for the experts), fp32 rows, then the last
// work-group to finish picks the top 8 (sigmoid, selection bias, weights normalised and scaled). Grid NE / 16.
KERNEL(128) void ink2_router8(global const float *x, global const float *gamma, float eps, global const float *W,
                             global float *xn, global float *logits, global const float *bias, float wscale,
                             global int *ids, global float *wts, global int *counter) {
  local float xs[C];
  local float red[16];
  local int redi[16];
  local int last;
  local float sel[NE], prob[NE], pw[TOP];
  local int pid[TOP];
  const int t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  float v[18], ss = 0.0f;
  for (int i = 0; i < 18; i++) {
    v[i] = x[t + i * 128];
    ss += v[i] * v[i];
  }
  const float rn = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 18; i++) {
    const float y = v[i] * rn * gamma[t + i * 128];
    xs[t + i * 128] = y;
    if (get_group_id(0) == 0) xn[t + i * 128] = y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int r = get_group_id(0) * 8 + sg;
  float acc = 0.0f;
  for (int k = lane * 4; k < C; k += 64) acc += dot(vload4(0, W + (long)r * C + k), vload4(0, xs + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) logits[r] = acc;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (t == 0) {
    const int old = atomic_fetch_add_explicit((volatile global atomic_int *)counter, 1, memory_order_acq_rel,
                                              memory_scope_device);
    last = old == (int)get_num_groups(0) - 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!last) return;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  if (t < NE) {
    const float p = sigmoid1(((volatile global float *)logits)[t]);
    prob[t] = p;
    sel[t] = p + bias[t];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  for (int k = 0; k < TOP; k++) {
    const float sv = t < NE ? sel[t] : -INFINITY;
    const float mx = sub_group_reduce_max(sv);
    int idx = sv == mx && t < NE ? t : 0x7fffffff;
    idx = sub_group_reduce_min(idx);
    if (lane == 0) {
      red[sg] = mx;
      redi[sg] = idx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float bm = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = 0; i < 8; i++)
      if (red[i] > bm || (red[i] == bm && redi[i] < bi)) {
        bm = red[i];
        bi = redi[i];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      pid[k] = bi;
      pw[k] = prob[bi];
      sel[bi] = -INFINITY;
    }
    sum += prob[bi];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t < TOP) {
    ids[t] = pid[t];
    wts[t] = pw[t] / (sum + 1e-20f) * wscale;
  }
  if (t == 0) *counter = 0;
}

// E2M1 (doubled, as ggml's kvalues) from a nibble with integer ops only.
static inline float e2m1(uint n) {
  const uint m = n & 7;
  const float v = m < 2 ? (float)m : (float)((2 + (m & 1)) << ((m >> 1) - 1));
  return (n & 8) ? -v : v;
}

static inline float nvfp4_dot16b(uint2 q, float sc, global const float *x) {
  const float16 xv = vload16(0, x);
  float a = 0.0f;
  a += e2m1(q.x) * xv.s0 + e2m1(q.x >> 4) * xv.s8 + e2m1(q.x >> 8) * xv.s1 + e2m1(q.x >> 12) * xv.s9;
  a += e2m1(q.x >> 16) * xv.s2 + e2m1(q.x >> 20) * xv.sa + e2m1(q.x >> 24) * xv.s3 + e2m1(q.x >> 28) * xv.sb;
  a += e2m1(q.y) * xv.s4 + e2m1(q.y >> 4) * xv.sc + e2m1(q.y >> 8) * xv.s5 + e2m1(q.y >> 12) * xv.sd;
  a += e2m1(q.y >> 16) * xv.s6 + e2m1(q.y >> 20) * xv.se + e2m1(q.y >> 24) * xv.s7 + e2m1(q.y >> 28) * xv.sf;
  return a * sc;
}

static float nvfp4_row2(global const uchar *qs, global const uchar *sc, int K, global const float *x,
                        constant float *lut) {
  const int lane = get_sub_group_local_id();
  float acc = 0.0f;
  for (int b = lane; b < K / 16; b += 16)
    acc += nvfp4_dot16b(vload2(b, (global const uint *)qs), lut[sc[b]], x + b * 16);
  return sub_group_reduce_add(acc);
}

KERNEL(256) void ink2_gateup_nvfp4b(global const uchar *qs, global const uchar *sc, global const float *escale,
                                    global const int *ids, global const float *x, constant float *lut, float b,
                                    float lb, global float *act) {
  const int j = get_group_id(0) * 16 + get_sub_group_id(), slot = get_group_id(1);
  const int e = ids[slot];
  const long rows = 2 * FF;
  global const uchar *qe = qs + (long)e * rows * (C / 2);
  global const uchar *se = sc + (long)e * rows * (C / 16);
  const float g = nvfp4_row2(qe + (long)j * (C / 2), se + (long)j * (C / 16), C, x, lut);
  const float u = nvfp4_row2(qe + (long)(FF + j) * (C / 2), se + (long)(FF + j) * (C / 16), C, x, lut);
  if (get_sub_group_local_id() == 0) act[slot * FF + j] = situ(g * escale[e], u * escale[e], b, lb);
}

// down with the post conv and the residual; grid C / 16.
KERNEL(256) void ink2_down_nvfp4_conv(global const uchar *qs, global const uchar *sc, global const float *escale,
                                      global const int *ids, global const float *wts, global const float *act,
                                      constant float *lut, global const float *w, global const float *hin,
                                      global float *hout, int site, global const float *res, global float *out) {
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  float acc = 0.0f;
  for (int slot = 0; slot < TOP; slot++) {
    const int e = ids[slot];
    global const uchar *qe = qs + ((long)e * C + r) * (FF / 2);
    global const uchar *se = sc + ((long)e * C + r) * (FF / 16);
    acc += wts[slot] * escale[e] * nvfp4_row2(qe, se, FF, act + slot * FF, lut);
  }
  if (get_sub_group_local_id() == 0) out[r] = conv_post(acc, w + r * NCONV, hin, hout, site + r) + res[r];
}

KERNEL(256) void ink2_down_q8_conv(global const char *q, global const half *s, global const int *ids,
                                   global const float *wts, global const float *act, global const float *w,
                                   global const float *hin, global float *hout, int site, global const float *res,
                                   global float *out) {
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  float acc = 0.0f;
  for (int slot = 0; slot < TOP; slot++) {
    const int e = ids[slot];
    acc += wts[slot] * q8_row(q + ((long)e * C + r) * FF, s + ((long)e * C + r) * (FF / 32), FF, act + slot * FF);
  }
  if (get_sub_group_local_id() == 0) out[r] = conv_post(acc, w + r * NCONV, hin, hout, site + r) + res[r];
}

// The logits rows with the output norm in the prologue (Q8_0, one input).
KERNEL(256) void ink2_gemv_q8n(global const char *q, global const half *s, int rows, global const float *x,
                               global const float *gamma, float eps, global float *y) {
  local float xs[C];
  local float red[16];
  const int t = get_local_id(0);
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    v[i] = x[t + i * 256];
    ss += v[i] * v[i];
  }
  const float rn = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) xs[t + i * 256] = v[i] * rn * gamma[t + i * 256];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r >= rows) return;
  global const char *qr = q + (long)r * C;
  global const half *sr = s + (long)r * (C / 32);
  float acc = 0.0f;
  for (int hb = lane; hb < C / 16; hb += 16) {
    const float16 wf = convert_float16(vload16(hb, qr));
    const float16 xv = vload16(hb, xs);
    acc += (dot(wf.lo.lo, xv.lo.lo) + dot(wf.lo.hi, xv.lo.hi) + dot(wf.hi.lo, xv.hi.lo) + dot(wf.hi.hi, xv.hi.hi)) *
           (float)sr[hb >> 1];
  }
  acc = sub_group_reduce_add(acc);
  if (lane == 0) y[r] = acc;
}

// ---------------------------------------------------------------- sampling: softcap, then the max of l / T + Gumbel noise
// (T <= 0: plain argmax). Noise from a counter hash of (seed, step, id): one draw per id and step, reproducible by seed.
static inline uint mix32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

KERNEL(256) void ink2_sample_part(global float *logits, int n, float cap, float temp, uint seed_lo, uint seed_hi,
                                  uint step, global float *pv, global int *pi) {
  local float red[16];
  local int redi[16];
  const int t = get_local_id(0), g = get_group_id(0);
  const int i = g * 256 + t;
  float v = -INFINITY;
  if (i < n) {
    const float l = tanh(logits[i] / cap) * cap;
    logits[i] = l;
    v = l;
    if (temp > 0.0f) {
      const uint h = mix32(mix32(mix32((uint)i ^ seed_lo) + step * 0x9e3779b9U) ^ seed_hi);
      const float u = ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
      v = l / temp - log(-log(u));
    }
  }
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

// ================================================================ prompt rows: chunks of T <= TMAX rows
// Row-major activations X[t][C]; a slot of the state pool holds TMAX rows. Histories as v2 (read [p0 % 2] at the start,
// the chunk's last three inputs written to [(p0 + T) % 2]; the same thread reads all three before it writes any).
#define TMAX 64
#define TB 8

KERNEL(256) void ink2_embed_b(global const char *q, global const half *s, global const int *ids,
                              global const float *gamma, float eps, global float *out) {
  local float red[16];
  const int t = get_local_id(0), row = get_group_id(0), tok = ids[row];
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    const int c = t + i * 256;
    v[i] = (float)q[(long)tok * C + c] * (float)s[(long)tok * (C / 32) + c / 32];
    ss += v[i] * v[i];
  }
  const float r = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) out[(long)row * C + t + i * 256] = v[i] * r * gamma[t + i * 256];
}

// mix3 a row a work-group (grid T); states S[slot][TMAX][C]; out[i][TMAX][C].
KERNEL(512) void ink2_mix3_b(global const float *S, Slots sl, global const float *q1, int n1, global const float *q2,
                             int n2, global const float *q3, int n3, global float *out) {
  local float part[32 * MAXS * 5];
  local float tot[MAXS * 5];
  const long row = get_group_id(0);
  mix_core(S, (long)TMAX * C, row * C, sl, q1, n1, q2, n2, q3, n3, out + row * C, (long)TMAX * C, part, tot, 0, 0, 0.0f);
}

// rms norm with gamma of nsel inputs (stride TMAX*C apart, or one input when xstride == 0), grid (nsel, T).
KERNEL(256) void ink2_norm_b(global const float *x, long xstride, global const float *gamma, int gstride, float eps,
                             global float *y) {
  local float red[16];
  const int t = get_local_id(0), j = get_group_id(0), row = get_group_id(1);
  global const float *xr = x + j * xstride + (long)row * C;
  float v[9], ss = 0.0f;
  for (int i = 0; i < 9; i++) {
    v[i] = xr[t + i * 256];
    ss += v[i] * v[i];
  }
  const float r = rsqrt(wg_sum(ss, red) / C + eps);
  for (int i = 0; i < 9; i++) y[((long)j * TMAX + row) * C + t + i * 256] = v[i] * r * gamma[j * gstride + t + i * 256];
}

// Controller rows (grid (4, controllers, T)) and outputs (grid (C / 16, controllers, T)); cur/msg [j][TMAX][C].
KERNEL(256) void ink2_ctl_down_b(global const float *cur, long cur_stride, global const float *msg, long msg_stride,
                                 global const float *Wd, int m0, global float *hd) {
  local float xs[2 * C];
  local float red[16];
  const int t = get_local_id(0), j = get_group_id(1), row = get_group_id(2);
  global const float *c = cur + j * cur_stride + (long)row * C;
  global const float *g = msg + j * msg_stride + (long)row * C;
  float a = 0.0f, b = 0.0f;
  for (int k = t; k < C; k += 256) {
    a += c[k] * c[k];
    b += g[k] * g[k];
  }
  const float ra = rsqrt(wg_sum(a, red) / C + 1e-6f), rb = rsqrt(wg_sum(b, red) / C + 1e-6f);
  for (int k = t; k < C; k += 256) {
    xs[k] = c[k] * ra;
    xs[C + k] = g[k] * rb;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  global const float *w = Wd + ((long)(m0 + j) * 64 + r) * (2 * C);
  float acc = 0.0f;
  for (int k = lane * 4; k < 2 * C; k += 64) acc += dot(vload4(0, w + k), vload4(0, xs + k));
  acc = sub_group_reduce_add(acc);
  if (lane == 0) hd[((long)j * TMAX + row) * 64 + r] = acc;
}

KERNEL(256) void ink2_ctl_out_b(global const float *G, global const float *Fw, int m0, global const float *hd,
                                global const float *cur, long cur_stride, global const float *msg, long msg_stride,
                                global float *out) {
  const int lane = get_sub_group_local_id(), j = get_group_id(1), row = get_group_id(2);
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  const long w = ((long)(m0 + j) * C + r) * 64;
  const float4 hv = vload4(lane, hd + ((long)j * TMAX + row) * 64);
  const float gs = sub_group_reduce_add(dot(vload4(lane, G + w), hv));
  const float fs = sub_group_reduce_add(dot(vload4(lane, Fw + w), hv));
  if (lane == 0)
    out[((long)j * TMAX + row) * C + r] =
        (1.0f - tanh(fs)) * cur[j * cur_stride + (long)row * C + r] + tanh(gs) * msg[j * msg_stride + (long)row * C + r];
}

// Q8_0 rows for T input rows: y[t][r] (row stride ystride) = W[r] . x_sel(r)[t]; x_sel = x + sel * xsel_stride, sel by
// r1/r2 as ink2_gemv_q8. A subgroup a weight row, input rows in groups of TB from L1.
KERNEL(256) void ink2_gemv_q8_b(global const char *q, global const half *s, int rows, int K, global const float *x,
                                long xsel_stride, int r1, int r2, int T, global float *y, int ystride) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r >= rows) return;
  global const float *xb = x + (r < r1 ? 0 : (r < r2 ? xsel_stride : 2 * xsel_stride));
  global const char *qr = q + (long)r * K;
  global const half *sr = s + (long)r * (K / 32);
  for (int t0 = 0; t0 < T; t0 += TB) {
    float acc[TB];
    for (int i = 0; i < TB; i++) acc[i] = 0.0f;
    for (int hb = lane; hb < K / 16; hb += 16) {
      const float16 wf = convert_float16(vload16(hb, qr)) * (float)sr[hb >> 1];
      for (int i = 0; i < TB; i++) {
        if (t0 + i >= T) break;
        const float16 xv = vload16(hb, xb + (long)(t0 + i) * K);
        acc[i] += dot(wf.lo.lo, xv.lo.lo) + dot(wf.lo.hi, xv.lo.hi) + dot(wf.hi.lo, xv.hi.lo) + dot(wf.hi.hi, xv.hi.hi);
      }
    }
    for (int i = 0; i < TB; i++) {
      if (t0 + i >= T) break;
      const float a = sub_group_reduce_add(acc[i]);
      if (lane == 0) y[(long)(t0 + i) * ystride + r] = a;
    }
  }
}

// fp32 rows for T input rows, the same selection (K a multiple of 64).
KERNEL(256) void ink2_gemv_f32_b(global const float *W, int rows, int K, global const float *x, long xsel_stride, int r1,
                                 int r2, int T, global float *y, int ystride) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r >= rows) return;
  global const float *xb = x + (r < r1 ? 0 : (r < r2 ? xsel_stride : 2 * xsel_stride));
  global const float *w = W + (long)r * K;
  for (int t = 0; t < T; t++) {
    float acc = 0.0f;
    for (int k = lane * 4; k < K; k += 64) acc += dot(vload4(0, w + k), vload4(0, xb + (long)t * K + k));
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[(long)t * ystride + r] = acc;
  }
}

// Attention prologue for T rows (positions p0..p0+T-1): q, rel of each row and head; k, v of each row through the short
// conv (rows in order, history rolled in registers), stored in the cache. Work-groups of 128: 0..23 q heads,
// 24..31 k heads, 32..39 v heads, 40..63 rel heads. P [T][PROWS], D [T][88]; Q [T][24][128], R [T][24][16].
KERNEL(128) void ink2_prep_b(global const float *P, global const float *D, global const float *kconv,
                             global const float *vconv, global const float *hin, global float *hout,
                             global const float *qclip, global const float *rclip, Prep pp, int p0, int T, int ring,
                             float log_floor, float log_alpha, global float *Q, global float *R, global half *Kc,
                             global half *Vc) {
  local float red[8];
  const int t = get_local_id(0), task = get_group_id(0);
  if (task < 24 || task >= 40) {
    const int isr = task >= 40, h = isr ? task - 40 : task, n = isr ? NR : HD;
    for (int row = 0; row < T; row++) {
      const float tau = pp.global_layer ? 1.0f + log_alpha * log(fmax(((float)(p0 + row) + 1.0f) / log_floor, 1.0f)) : 1.0f;
      float v = t < n ? P[(long)row * PROWS + (isr ? 3072 + h * NR : h * HD) + t] * (isr ? rclip[h * NR + t] : qclip[h * HD + t]) : 0.0f;
      const float nn = sqrt(wg_sum(v * v, red)) + 1e-6f;
      v *= clamp((isr ? pp.rel_clip : pp.qk_clip) / nn, 0.0f, 1.0f) * tau;
      if (t < n) (isr ? R : Q)[((long)row * 24 + h) * n + t] = v;
    }
    return;
  }
  const int isv = task >= 32, h = task - (isv ? 32 : 24), c = h * HD + t, site = (isv ? 1024 : 0) + c;
  float h0 = hin[site], h1 = hin[HIST + site], h2 = hin[2 * HIST + site];
  global const float *w = (isv ? vconv : kconv) + c * NCONV;
  for (int row = 0; row < T; row++) {
    const float x = P[(long)row * PROWS + (isv ? 7552 : 6528) + c];
    float o = h0 * w[0] + h1 * w[1] + h2 * w[2] + x * w[3];
    float scale = 0.2236068f;
    if (pp.global_layer) {
      global const float *dy = D + (long)row * 88 + 24 + (isv ? 32 : 0) + h * NCONV;
      o += x * dy[0] + h2 * dy[1] + h1 * dy[2] + h0 * dy[3];
      scale = 1.0f / 3.0f;
    }
    float val = (x + o) * scale;
    const float ss = wg_sum(val * val, red);
    val *= isv ? rsqrt(ss / HD + pp.eps) : clamp(pp.qk_clip / (sqrt(ss) + 1e-6f), 0.0f, 1.0f);
    (isv ? Vc : Kc)[(long)((p0 + row) % ring) * 1024 + c] = convert_half(val);
    h0 = h1;
    h1 = h2;
    h2 = x;
  }
  hout[site] = h0;
  hout[HIST + site] = h1;
  hout[2 * HIST + site] = h2;
}

// Attention of row `row` (grid (24, T)), keys from the cache only (the prologue stored the chunk's rows first).
KERNEL(256) void ink2_attn_b(global const float *Q, global const float *R, global const float *D, global const half *Kc,
                             global const half *Vc, global const float *relb, int p0c, int window, int ring, int extent,
                             global float *scratch, global const float *P, global float *O) {
  local float red[16];
  local float qs[HD], rv[NR], part[2][HD];
  const int t = get_local_id(0), h = get_group_id(0), row = get_group_id(1), kvh = h / 3;
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const int pos = p0c + row, p0 = window > 0 ? max(0, pos - window + 1) : 0;
  if (t < HD) qs[t] = Q[((long)row * 24 + h) * HD + t];
  if (t < NR) rv[t] = R[((long)row * 24 + h) * NR + t];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float sink = D[(long)row * 88 + h];
  const int n = pos - p0 + 1;
  global float *sc = scratch + ((long)row * 24 + h) * ring;
  float m = 0.0f;
  for (int j = sg; j < n; j += 16) {
    const int p = p0 + j, d = pos - p, rr = p % ring;
    const float8 kf = convert_float8(vload8(0, Kc + (long)rr * 1024 + kvh * HD + lane * 8));
    const float8 qf = vload8(0, qs + lane * 8);
    float a = sub_group_reduce_add(dot(kf.lo, qf.lo) + dot(kf.hi, qf.hi));
    const int bi = d < extent ? d : extent;
    float b = bi < extent ? relb[bi * NR + lane] * rv[lane] : 0.0f;
    b = sub_group_reduce_add(b);
    const float s = a * 0.08838834764831845f + b - sink;
    if (lane == 0) sc[j] = s;
    m = fmax(m, s);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  m = wg_max(m, red);
  float z = 0.0f;
  for (int j = t; j < n; j += 256) z += exp(sc[j] - m);
  z = wg_sum(z, red) + exp(-m);
  const int d = t & (HD - 1), half_ = t >> 7;
  float acc = 0.0f;
  for (int j = half_; j < n; j += 2) acc += exp(sc[j] - m) * (float)Vc[(long)((p0 + j) % ring) * 1024 + kvh * HD + d];
  part[half_][d] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < HD)
    O[((long)row * 24 + h) * HD + t] = (part[0][t] + part[1][t]) / z * sigmoid1(P[(long)row * PROWS + 3456 + h * HD + t]);
}

// Post conv over T rows per channel in order, plus the residual: out[t][c] = conv(y[t][c]) (+ res[t][c]); grid C / 256.
KERNEL(256) void ink2_conv_res_b(global const float *y, global const float *w, global const float *hin,
                                 global float *hout, int site, global const float *res, int has_res, int T,
                                 global float *out) {
  const int c = get_global_id(0);
  const int idx = site + c;
  float h0 = hin[idx], h1 = hin[HIST + idx], h2 = hin[2 * HIST + idx];
  global const float *ww = w + c * NCONV;
  for (int t = 0; t < T; t++) {
    const float x = y[(long)t * C + c];
    const float o = h0 * ww[0] + h1 * ww[1] + h2 * ww[2] + x * ww[3];
    float v = (x + o) * 0.2236068f;
    if (has_res) v += res[(long)t * C + c];
    out[(long)t * C + c] = v;
    h0 = h1;
    h1 = h2;
    h2 = x;
  }
  hout[idx] = h0;
  hout[HIST + idx] = h1;
  hout[2 * HIST + idx] = h2;
}

// Top 8 of each row (grid T): sigmoid, selection bias, weights normalised and scaled; and the expert lists of the chunk
// are built by ink2_route_lists after it.
KERNEL(128) void ink2_topk_b(global const float *logits, global const float *bias, float wscale, global int *ids,
                             global float *wts) {
  local float sel[NE], prob[NE];
  local int pick[TOP];
  const int t = get_local_id(0), row = get_group_id(0);
  const float p = sigmoid1(logits[row * NE + t]);
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
      ids[row * TOP + k] = pick[k];
      wts[row * TOP + k] = prob[pick[k]] / sum * wscale;
    }
  }
}

// Per-expert lists of the chunk's (row, slot) pairs in row order (one work-group): start[NE + 1], list[T * TOP] holds
// row * TOP + slot; entry e of list is where that pair's expert output goes.
KERNEL(128) void ink2_route_lists(global const int *ids, int T, global int *start, global int *list) {
  local int cnt[NE], off_[NE + 1];
  const int t = get_local_id(0);
  cnt[t] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    for (int i = 0; i < T * TOP; i++) cnt[ids[i]]++;
    off_[0] = 0;
    for (int e = 0; e < NE; e++) off_[e + 1] = off_[e] + cnt[e];
    for (int e = 0; e <= NE; e++) start[e] = off_[e];
    for (int i = 0; i < T * TOP; i++) list[off_[ids[i]]++] = i;
  }
}

// gate_up of every (row, slot) pair of one expert (grid (FF / 16, NE)): act[pair][j] (pair = row * TOP + slot).
KERNEL(256) void ink2_gateup_nvfp4_b(global const uchar *qs, global const uchar *sc, global const float *escale,
                                     global const int *start, global const int *list, global const float *x,
                                     constant float *lut, float b, float lb, global float *act) {
  const int j = get_group_id(0) * 16 + get_sub_group_id(), e = get_group_id(1), lane = get_sub_group_local_id();
  const int s0 = start[e], s1 = start[e + 1];
  if (s0 == s1) return;
  const long rows = 2 * FF;
  global const uchar *qg = qs + ((long)e * rows + j) * (C / 2), *qu = qs + ((long)e * rows + FF + j) * (C / 2);
  global const uchar *sg_ = sc + ((long)e * rows + j) * (C / 16), *su = sc + ((long)e * rows + FF + j) * (C / 16);
  for (int i0 = s0; i0 < s1; i0 += TB) {
    float ag[TB], au[TB];
    for (int i = 0; i < TB; i++) ag[i] = au[i] = 0.0f;
    for (int bk = lane; bk < C / 16; bk += 16) {
      const uint2 wg = vload2(bk, (global const uint *)qg), wu = vload2(bk, (global const uint *)qu);
      const float fg = lut[sg_[bk]], fu = lut[su[bk]];
      for (int i = 0; i < TB; i++) {
        if (i0 + i >= s1) break;
        global const float *xr = x + (long)(list[i0 + i] / TOP) * C + bk * 16;
        ag[i] += nvfp4_dot16b(wg, fg, xr);
        au[i] += nvfp4_dot16b(wu, fu, xr);
      }
    }
    for (int i = 0; i < TB; i++) {
      if (i0 + i >= s1) break;
      const float g = sub_group_reduce_add(ag[i]), u = sub_group_reduce_add(au[i]);
      if (lane == 0) act[(long)list[i0 + i] * FF + j] = situ(g * escale[e], u * escale[e], b, lb);
    }
  }
}

KERNEL(256) void ink2_gateup_q8_b(global const char *q, global const half *s, global const int *start,
                                  global const int *list, global const float *x, float b, float lb, global float *act) {
  const int j = get_group_id(0) * 16 + get_sub_group_id(), e = get_group_id(1), lane = get_sub_group_local_id();
  const int s0 = start[e], s1 = start[e + 1];
  if (s0 == s1) return;
  const long rows = 2 * FF;
  global const char *qg = q + ((long)e * rows + j) * C, *qu = q + ((long)e * rows + FF + j) * C;
  global const half *sg_ = s + ((long)e * rows + j) * (C / 32), *su = s + ((long)e * rows + FF + j) * (C / 32);
  for (int i0 = s0; i0 < s1; i0 += TB) {
    float ag[TB], au[TB];
    for (int i = 0; i < TB; i++) ag[i] = au[i] = 0.0f;
    for (int hb = lane; hb < C / 16; hb += 16) {
      const float16 wg = convert_float16(vload16(hb, qg)) * (float)sg_[hb >> 1];
      const float16 wu = convert_float16(vload16(hb, qu)) * (float)su[hb >> 1];
      for (int i = 0; i < TB; i++) {
        if (i0 + i >= s1) break;
        const float16 xv = vload16(hb, x + (long)(list[i0 + i] / TOP) * C);
        ag[i] += dot(wg.lo.lo, xv.lo.lo) + dot(wg.lo.hi, xv.lo.hi) + dot(wg.hi.lo, xv.hi.lo) + dot(wg.hi.hi, xv.hi.hi);
        au[i] += dot(wu.lo.lo, xv.lo.lo) + dot(wu.lo.hi, xv.lo.hi) + dot(wu.hi.lo, xv.hi.lo) + dot(wu.hi.hi, xv.hi.hi);
      }
    }
    for (int i = 0; i < TB; i++) {
      if (i0 + i >= s1) break;
      const float g = sub_group_reduce_add(ag[i]), u = sub_group_reduce_add(au[i]);
      if (lane == 0) act[(long)list[i0 + i] * FF + j] = situ(g, u, b, lb);
    }
  }
}

// down of every pair of one expert (grid (C / 16, NE)): ye[pair][r] = s_e * W[e][r] . act[pair].
KERNEL(256) void ink2_down_nvfp4_b(global const uchar *qs, global const uchar *sc, global const float *escale,
                                   global const int *start, global const int *list, global const float *act,
                                   constant float *lut, global float *ye) {
  const int r = get_group_id(0) * 16 + get_sub_group_id(), e = get_group_id(1), lane = get_sub_group_local_id();
  const int s0 = start[e], s1 = start[e + 1];
  if (s0 == s1) return;
  global const uchar *qe = qs + ((long)e * C + r) * (FF / 2);
  global const uchar *se = sc + ((long)e * C + r) * (FF / 16);
  for (int i0 = s0; i0 < s1; i0 += TB) {
    float a[TB];
    for (int i = 0; i < TB; i++) a[i] = 0.0f;
    for (int bk = lane; bk < FF / 16; bk += 16) {
      const uint2 w = vload2(bk, (global const uint *)qe);
      const float f = lut[se[bk]];
      for (int i = 0; i < TB; i++) {
        if (i0 + i >= s1) break;
        a[i] += nvfp4_dot16b(w, f, act + (long)list[i0 + i] * FF + bk * 16);
      }
    }
    for (int i = 0; i < TB; i++) {
      if (i0 + i >= s1) break;
      const float v = sub_group_reduce_add(a[i]);
      if (lane == 0) ye[(long)list[i0 + i] * C + r] = v * escale[e];
    }
  }
}

KERNEL(256) void ink2_down_q8_b(global const char *q, global const half *s, global const int *start,
                                global const int *list, global const float *act, global float *ye) {
  const int r = get_group_id(0) * 16 + get_sub_group_id(), e = get_group_id(1), lane = get_sub_group_local_id();
  const int s0 = start[e], s1 = start[e + 1];
  if (s0 == s1) return;
  global const char *qe = q + ((long)e * C + r) * FF;
  global const half *se = s + ((long)e * C + r) * (FF / 32);
  for (int i0 = s0; i0 < s1; i0 += TB) {
    float a[TB];
    for (int i = 0; i < TB; i++) a[i] = 0.0f;
    for (int hb = lane; hb < FF / 16; hb += 16) {
      const float16 w = convert_float16(vload16(hb, qe)) * (float)se[hb >> 1];
      for (int i = 0; i < TB; i++) {
        if (i0 + i >= s1) break;
        const float16 xv = vload16(hb, act + (long)list[i0 + i] * FF);
        a[i] += dot(w.lo.lo, xv.lo.lo) + dot(w.lo.hi, xv.lo.hi) + dot(w.hi.lo, xv.hi.lo) + dot(w.hi.hi, xv.hi.hi);
      }
    }
    for (int i = 0; i < TB; i++) {
      if (i0 + i >= s1) break;
      const float v = sub_group_reduce_add(a[i]);
      if (lane == 0) ye[(long)list[i0 + i] * C + r] = v;
    }
  }
}

// Weighted sum of each row's 8 expert outputs (slot order), then the post conv over the rows and the residual; grid C/256.
KERNEL(256) void ink2_combine_b(global const float *ye, global const float *wts, global const float *w,
                                global const float *hin, global float *hout, int site, global const float *res, int T,
                                global float *out) {
  const int c = get_global_id(0);
  const int idx = site + c;
  float h0 = hin[idx], h1 = hin[HIST + idx], h2 = hin[2 * HIST + idx];
  global const float *ww = w + c * NCONV;
  for (int t = 0; t < T; t++) {
    float x = 0.0f;
    for (int k = 0; k < TOP; k++) x += wts[t * TOP + k] * ye[(long)(t * TOP + k) * C + c];
    const float o = h0 * ww[0] + h1 * ww[1] + h2 * ww[2] + x * ww[3];
    out[(long)t * C + c] = (x + o) * 0.2236068f + res[(long)t * C + c];
    h0 = h1;
    h1 = h2;
    h2 = x;
  }
  hout[idx] = h0;
  hout[HIST + idx] = h1;
  hout[2 * HIST + idx] = h2;
}

// ---------------------------------------------------------------- long-context decode attention: key chunks, then merge
// Grid (8 key heads, chunks): a work-group takes CH keys of its key head for its 3 query heads (keys read once), one key
// a thread for the scores, then (head, dim) pairs for the values. Partials: m, l, o[128] per (query head, chunk).
#define CH 256
KERNEL(256) void ink2_attn_split(global const float *Q, global const float *R, global const float *D,
                                 global const half *Kc, global const half *Vc, global const float *relb, Span sp,
                                 global float *part_o, global float *part_ml) {
  local float qs[3][HD], rv[3][NR], sk[3], sc[3][CH], mx[3], sm[3];
  local float red[16];
  const int t = get_local_id(0), kvh = get_group_id(0), ch = get_group_id(1), nch = get_num_groups(1);
  for (int i = t; i < 3 * HD; i += 256) qs[i / HD][i % HD] = Q[(kvh * 3 + i / HD) * HD + i % HD];
  if (t < 3 * NR) rv[t / NR][t % NR] = R[(kvh * 3 + t / NR) * NR + t % NR];
  if (t < 3) sk[t] = D[kvh * 3 + t];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n = sp.pos - sp.p0 + 1, j = ch * CH + t;
  float s0 = -INFINITY, s1 = -INFINITY, s2 = -INFINITY;
  if (j < n) {
    const int p = sp.p0 + j, d = sp.pos - p;
    global const half *k = Kc + (long)(p % sp.ring) * 1024 + kvh * HD;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int i = 0; i < HD; i += 8) {
      const float8 kf = convert_float8(vload8(0, k + i));
      a0 += dot(kf.lo, vload4(0, &qs[0][i])) + dot(kf.hi, vload4(0, &qs[0][i + 4]));
      a1 += dot(kf.lo, vload4(0, &qs[1][i])) + dot(kf.hi, vload4(0, &qs[1][i + 4]));
      a2 += dot(kf.lo, vload4(0, &qs[2][i])) + dot(kf.hi, vload4(0, &qs[2][i + 4]));
    }
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    if (d < sp.extent) {
      global const float *rb = relb + d * NR;
      for (int i = 0; i < NR; i++) {
        b0 += rb[i] * rv[0][i];
        b1 += rb[i] * rv[1][i];
        b2 += rb[i] * rv[2][i];
      }
    }
    s0 = a0 * 0.08838834764831845f + b0 - sk[0];
    s1 = a1 * 0.08838834764831845f + b1 - sk[1];
    s2 = a2 * 0.08838834764831845f + b2 - sk[2];
  }
  sc[0][t] = s0;
  sc[1][t] = s1;
  sc[2][t] = s2;
  const float m0 = wg_max(s0, red), m1 = wg_max(s1, red), m2 = wg_max(s2, red);
  const float e0 = j < n ? exp(s0 - m0) : 0.0f, e1 = j < n ? exp(s1 - m1) : 0.0f, e2 = j < n ? exp(s2 - m2) : 0.0f;
  sc[0][t] = e0;
  sc[1][t] = e1;
  sc[2][t] = e2;
  const float l0 = wg_sum(e0, red), l1 = wg_sum(e1, red), l2 = wg_sum(e2, red);
  if (t == 0) {
    mx[0] = m0; mx[1] = m1; mx[2] = m2;
    sm[0] = l0; sm[1] = l1; sm[2] = l2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nk = min(CH, n - ch * CH);
  for (int pr = t; pr < 3 * HD; pr += 256) {
    const int hh = pr / HD, d = pr % HD;
    float acc = 0.0f;
    for (int jj = 0; jj < nk; jj++)
      acc += sc[hh][jj] * (float)Vc[(long)((sp.p0 + ch * CH + jj) % sp.ring) * 1024 + kvh * HD + d];
    part_o[((long)(kvh * 3 + hh) * nch + ch) * HD + d] = acc;
  }
  if (t < 3) {
    part_ml[((long)(kvh * 3 + t) * nch + ch) * 2] = mx[t];
    part_ml[((long)(kvh * 3 + t) * nch + ch) * 2 + 1] = sm[t];
  }
}

// Merge the chunks of each query head in order with the sink logit 0, times sigmoid(gate). Grid 24, 128 threads.
KERNEL(128) void ink2_attn_merge(global const float *part_o, global const float *part_ml, int nch, global const float *P,
                                 global float *O) {
  const int h = get_group_id(0), d = get_local_id(0);
  float m = 0.0f;
  for (int c = 0; c < nch; c++) m = fmax(m, part_ml[((long)h * nch + c) * 2]);
  float l = exp(-m), o = 0.0f;
  for (int c = 0; c < nch; c++) {
    const float f = exp(part_ml[((long)h * nch + c) * 2] - m);
    l += part_ml[((long)h * nch + c) * 2 + 1] * f;
    o += part_o[((long)h * nch + c) * HD + d] * f;
  }
  O[h * HD + d] = o / l * sigmoid1(P[3456 + h * HD + d]);
}

// ================================================================ decode experts, version c
// NVFP4 nibbles straight into fp16 fields: E2M1 bits s e e m -> fp16 sign | exponent low bits | mantissa top bit, which is
// the E2M1 value times 2^-14 (subnormal for e = 0 included); the factor joins the UE4M3 scale. Two values per half2.
#define NV_LO(w) as_half2((((w) & 0x00070007u) << 9) | (((w) & 0x00080008u) << 12))
static inline float nv_dot8(uint w, float4 xa, float4 xb) {
  // w: 4 bytes of a 16-value group; byte j low nibble = value j (or j + 4), high = value j + 8 (or j + 12).
  // xa = x[j0 .. j0+3], xb = x[j0+8 .. j0+11]
  const float2 l02 = convert_float2(NV_LO(w)), l13 = convert_float2(NV_LO(w >> 8));
  const float2 h02 = convert_float2(NV_LO(w >> 4)), h13 = convert_float2(NV_LO(w >> 12));
  return l02.x * xa.x + l13.x * xa.y + l02.y * xa.z + l13.y * xa.w + h02.x * xb.x + h13.x * xb.y + h02.y * xb.z +
         h13.y * xb.w;
}

// 32 values (two groups of 16) of one row against x[0..31]: q = 16 bytes, s0/s1 = the two UE4M3 scales already decoded.
static inline float nv_dot32(uint4 q, float s0, float s1, global const float *x) {
  const float16 xv = vload16(0, x), xw = vload16(0, x + 16);
  const float a = nv_dot8(q.x, xv.s0123, xv.s89ab) + nv_dot8(q.y, xv.s4567, xv.scdef);
  const float b = nv_dot8(q.z, xw.s0123, xw.s89ab) + nv_dot8(q.w, xw.s4567, xw.scdef);
  return a * s0 + b * s1;
}

// gate_up, two (gate, up) row pairs a subgroup; grid (FF / 32, TOP), 256 threads. lut16 = lut x 2^15 (fp16 value x 2^15
// = ggml's doubled E2M1, and lut carries the x 0.5).
KERNEL(256) void ink2_gateup_nvfp4c(global const uchar *qs, global const uchar *sc, global const float *escale,
                                    global const int *ids, global const float *x, constant float *lut16, float b,
                                    float lb, global float *act) {
  const int lane = get_sub_group_local_id(), slot = get_group_id(1);
  const int j0 = (get_group_id(0) * 16 + get_sub_group_id()) * 2;
  const int e = ids[slot];
  const long rows = 2 * FF;
  global const uchar *qe = qs + (long)e * rows * (C / 2);
  global const uchar *se = sc + (long)e * rows * (C / 16);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int rr[4] = {j0, j0 + 1, FF + j0, FF + j0 + 1};
  for (int c = lane; c < C / 32; c += 16) { // 72 chunks of 32 values
    global const float *xc = x + c * 32;
    for (int i = 0; i < 4; i++) {
      const uint4 q = vload4(c, (global const uint *)(qe + (long)rr[i] * (C / 2)));
      const uchar2 s = vload2(c, se + (long)rr[i] * (C / 16));
      acc[i] += nv_dot32(q, lut16[s.x], lut16[s.y], xc);
    }
  }
  for (int i = 0; i < 4; i++) acc[i] = sub_group_reduce_add(acc[i]);
  if (lane < 2) {
    const float s = escale[e];
    act[slot * FF + j0 + lane] = situ((lane ? acc[1] : acc[0]) * s, (lane ? acc[3] : acc[2]) * s, b, lb);
  }
}

// down + post conv + residual: one row a subgroup, lanes = 8 experts x 2; grid C / 16.
KERNEL(256) void ink2_down_nvfp4c(global const uchar *qs, global const uchar *sc, global const float *escale,
                                  global const int *ids, global const float *wts, global const float *act,
                                  constant float *lut16, global const float *w, global const float *hin,
                                  global float *hout, int site, global const float *res, global float *out) {
  const int lane = get_sub_group_local_id(), r = get_group_id(0) * 16 + get_sub_group_id();
  const int slot = lane >> 1, e = ids[slot];
  global const uchar *qe = qs + ((long)e * C + r) * (FF / 2);
  global const uchar *se = sc + ((long)e * C + r) * (FF / 16);
  global const float *ae = act + slot * FF;
  float acc = 0.0f;
  for (int c = lane & 1; c < FF / 32; c += 2) { // 36 chunks of 32 values
    const uint4 q = vload4(c, (global const uint *)qe);
    const uchar2 s = vload2(c, se);
    acc += nv_dot32(q, lut16[s.x], lut16[s.y], ae + c * 32);
  }
  acc *= wts[slot] * escale[e];
  acc = sub_group_reduce_add(acc);
  if (lane == 0) out[r] = conv_post(acc, w + r * NCONV, hin, hout, site + r) + res[r];
}

// The attention projections from ready (normalised) inputs in global memory; rows as ink2_gemv_big.
KERNEL(256) void ink2_gemv_bigx(global const char *q, global const half *s, global const float *Wsm, int nsmall,
                                global const float *x0, global const float *x1, global const float *x2, int nsel,
                                global float *P, global float *D) {
  const int lane = get_sub_group_local_id();
  const int r = get_group_id(0) * 16 + get_sub_group_id();
  if (r < PROWS) {
    const int sel = nsel == 1 ? 0 : (r < 6528 ? 0 : (r < 7552 ? 1 : 2));
    const float y = q8_row(q + (long)r * C, s + (long)r * (C / 32), C, sel == 0 ? x0 : (sel == 1 ? x1 : x2));
    if (lane == 0) P[r] = y;
  } else if (r < PROWS + nsmall) {
    const int rr = r - PROWS;
    const int sel = nsel == 1 ? 0 : (rr < 24 ? 0 : (rr < 56 ? 1 : 2));
    global const float *x = sel == 0 ? x0 : (sel == 1 ? x1 : x2);
    global const float *w = Wsm + (long)rr * C;
    float acc = 0.0f;
    for (int k = lane * 4; k < C; k += 64) acc += dot(vload4(0, w + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (lane == 0) D[rr] = acc;
  }
}
