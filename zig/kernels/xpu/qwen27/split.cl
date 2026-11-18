// XPU_SPLIT: the MLP of a decode row split between two cards (B70 pair). Each card holds a slice of the MLP's
// neurons; the first card sends the MLP input to the peer and gets the peer's part of the down projection back.
// The exchange is a plain store into the peer's memory and one flag word; peer atomics are not needed.
#pragma OPENCL FP_CONTRACT OFF

#ifndef ASC
#define ASC 256.0f
#endif
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline float rbf(float f) { return bf2f(intel_convert_bfloat16_as_ushort(f)); }
static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// Up to three buffers as 4-byte words, strided over the launch (16-byte peer stores were slower and wrong: RL-264).
KERNEL(256) void peer_copy(global const uint *S0, global uint *D0, int n0, global const uint *S1, global uint *D1,
                           int n1, global const uint *S2, global uint *D2, int n2) {
  const int gid = get_global_id(0), all = get_global_size(0);
  for (int i = gid; i < n0; i += all) D0[i] = S0[i];
  for (int i = gid; i < n1; i += all) D1[i] = S1[i];
  for (int i = gid; i < n2; i += all) D2[i] = S2[i];
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
}

// Raises the peer's flag to `epoch` once the copy before it on this list is done; not after a timeout here.
KERNEL(16) void peer_flag(global uint *FLAG, int epoch, global const uint *STAT) {
  if (get_global_id(0) != 0) return;
  if (STAT[0] != 0) return;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_all_svm_devices);
  *(volatile global uint *)FLAG = (uint)epoch;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
}

// Waits on this card's flag for the peer's `epoch`, at most `cap` polls; STAT[0] = 1 on a timeout (kept), STAT[1]
// the polls of the last wait. The payload is read by the NEXT kernel: one that spins does not see fresh plain loads.
KERNEL(16) void peer_wait(global uint *FLAG, int epoch, int cap, global uint *STAT) {
  if (get_global_id(0) != 0) return;
  volatile atomic_uint *f = (volatile atomic_uint *)FLAG;
  uint v = atomic_load_explicit(f, memory_order_relaxed, memory_scope_all_svm_devices);
  int spins = 0;
  while (v < (uint)epoch && spins < cap) {
    spins++;
    v = atomic_load_explicit(f, memory_order_relaxed, memory_scope_all_svm_devices);
  }
  if (v < (uint)epoch) STAT[0] = 1;
  STAT[1] = (uint)spins;
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
}

// R = fp16(R + RB): the two cards' parts of the down projection, always in this order.
KERNEL(256) void add_half(global half *R, global const half *RB, int n) {
  const int all = get_global_size(0);
  for (int i = get_global_id(0); i < n; i += all) R[i] = convert_half_rte((float)R[i] + (float)RB[i]);
}

// swiglu (glue_attn.cl) for a slice of `ff` neurons, gate|up interleaved by 16, weights not rotated.
KERNEL(256) void swiglu_n(global const half *GU, global half *H, global half *AHI, global half *ALO, int gs, int ff) {
  local half hl[1024];
  const int t = get_group_id(0), blk = get_group_id(1), lid = get_local_id(0);
  {
    const int e = lid * 4, c = blk * 1024 + e;
    const long gc = (c / 16) * 32 + c % 16, uc = gc + 16;
    const half4 gv = vload4(0, GU + (long)t * (2 * ff) + gc), uv = vload4(0, GU + (long)t * (2 * ff) + uc);
    half4 o;
    for (int i = 0; i < 4; i++) o[i] = convert_half_rte(rbf(silu((float)gv[i]) * (float)uv[i]));
    vstore4(o, 0, H + (long)t * ff + c);
    vstore4(o, 0, hl + e);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), per = gs / 16, ng = 1024 / gs;
  for (int g = get_sub_group_id(); g < ng; g += 16) {
    float s = 0.0f;
    for (int k = 0; k < per; k++) s += (float)hl[g * gs + lane * per + k];
    s = sub_group_reduce_add(s) * (1.0f / gs);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)t * (ff / gs) + blk * ng + g] = hi;
      ALO[(long)t * (ff / gs) + blk * ng + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}
