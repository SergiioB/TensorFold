#!/usr/bin/env python3
"""Inkling 2 Mini (inkling2_loop) float32 reference forward over a GGUF, after llama.cpp src/models/inkling2.cpp.

  inkling2_ref.py MODEL.gguf generate IDS_JSON N_NEW OUT.json   (INK2_DUMP=dir: visit outputs and logits as raw fp32)

Dense weights are dequantized once to float32; routed experts are dequantized on use (LRU cache).
Greedy argmax over the softcapped logits; numpy + gguf-py only (gguf-py from a llama.cpp tree on PYTHONPATH).
"""
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import gguf
from gguf import quants

F = np.float32
DUMP = os.environ.get("INK2_DUMP")  # directory: each visit's output row and each step's logits, as raw fp32


def rms(x, eps):
    x = x.astype(F)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + F(eps))


def sigmoid(x):
    return F(1) / (F(1) + np.exp(-x))


class Model:
    def __init__(self, path, expert_cache=160):
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        kv = {f.name: f for f in self.r.fields.values()}

        def val(k):
            f = kv[k]
            return f.contents()

        a = "inkling2."
        self.n_visit = int(val(a + "block_count"))
        self.n_embd = int(val(a + "embedding_length"))
        self.n_head = list(val(a + "attention.head_count"))
        self.n_head_kv = list(val(a + "attention.head_count_kv"))
        self.head_dim = int(val(a + "attention.key_length"))
        self.eps = float(val(a + "attention.layer_norm_rms_epsilon"))
        self.n_exp = int(val(a + "expert_count"))
        self.n_used = int(val(a + "expert_used_count"))
        self.n_ff = int(val(a + "expert_feed_forward_length"))
        self.n_pass = int(val(a + "num_loops"))
        self.core0 = int(val(a + "loop.core_start"))
        self.core1 = int(val(a + "loop.core_end"))
        self.vocab = int(val(a + "vocab_size_unpadded"))
        self.swa = int(val(a + "attention.sliding_window"))
        self.swa_pat = [bool(x) for x in val(a + "attention.sliding_window_pattern")]
        self.n_conv = int(val(a + "shortconv.l_cache"))
        self.w_scale = float(val(a + "expert_weights_scale"))
        self.situ_b = float(val(a + "activation.situ_beta"))
        self.situ_lb = float(val(a + "activation.situ_linear_beta"))
        self.block = int(val(a + "attn_res.block_size"))
        self.cap = float(val(a + "final_logit_softcapping"))
        self.n_rel = int(val(a + "attention.relative_length"))
        self.rel_extent = int(val(a + "attention.relative_extent"))
        self.qk_clip = float(val(a + "attention.qk_clip"))
        self.rel_clip = float(val(a + "attention.relative_clip"))
        self.log_floor = float(val(a + "attention.log_scale_floor"))
        self.log_alpha = float(val(a + "attention.log_scale_alpha"))
        self.n_core = self.core1 - self.core0
        self.n_extra = (self.n_pass - 1) * self.n_core
        self._dense = {}
        self._exp = OrderedDict()
        self._exp_cap = expert_cache

    # visit il -> stored layer ip (llama_hparams::loop_layer)
    def layer(self, il):
        if self.n_pass < 2 or il < self.core0:
            return il
        if il < self.core0 + self.n_pass * self.n_core:
            return self.core0 + (il - self.core0) % self.n_core
        return il - self.n_extra

    def has(self, name):
        return name in self.t

    def w(self, name):
        """A tensor as float32 numpy, ggml ne reversed (rows = outputs)."""
        if name not in self._dense:
            t = self.t[name]
            x = quants.dequantize(t.data, t.tensor_type).astype(F)
            shape = [int(s) for s in reversed(list(t.shape))]
            self._dense[name] = x.reshape(shape)
        return self._dense[name]

    def embed_row(self, tok):
        t = self.t["token_embd.weight"]
        return quants.dequantize(t.data[tok:tok + 1], t.tensor_type).astype(F).reshape(self.n_embd)

    def expert(self, ip, e):
        key = (ip, e)
        if key in self._exp:
            self._exp.move_to_end(key)
            return self._exp[key]
        out = []
        for part in ("ffn_gate_up_exps", "ffn_down_exps"):
            t = self.t[f"blk.{ip}.{part}.weight"]
            x = quants.dequantize(t.data[e], t.tensor_type).astype(F)
            rows = int(t.shape[1])
            x = x.reshape(rows, -1)
            s = f"blk.{ip}.{part}.scale"
            if s in self.t:
                x *= F(self.w(s)[e])
            out.append(x)
        self._exp[key] = out
        if len(self._exp) > self._exp_cap:
            self._exp.popitem(last=False)
        return out


class State:
    """Per-visit caches: K, V rows with positions, and the four short-conv histories."""

    def __init__(self, m):
        self.k = [[] for _ in range(m.n_visit)]
        self.v = [[] for _ in range(m.n_visit)]
        self.pos = [[] for _ in range(m.n_visit)]
        self.hist = [dict() for _ in range(m.n_visit)]


def mix(blocks, cur, query):
    """Depth attention: softmax over rms-normed states (blocks + cur) with learned queries; returns [nq, T, C]."""
    states = np.stack(list(blocks) + [cur], 0)  # [S, T, C]
    states = rms(states, 1e-6)
    q = query.reshape(-1, states.shape[-1])  # [nq, C]
    c = states.shape[-1]
    sc = np.einsum("stc,qc->qts", states, q) / F(math.sqrt(c))
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("qts,stc->qtc", p, states).astype(F)


def control(m, ip, cur, msg, mod):
    names = ("attn", "k", "v", "mlp")[mod]
    h = np.concatenate([rms(cur, 1e-6), rms(msg, 1e-6)], -1)
    h = h @ m.w(f"blk.{ip}.loop_{names}_down.weight").T
    gain = np.tanh(h @ m.w(f"blk.{ip}.loop_{names}_gain.weight").T)
    forget = np.tanh(h @ m.w(f"blk.{ip}.loop_{names}_forget.weight").T)
    return ((F(1) - forget) * cur + gain * msg).astype(F)


def clip(x, scale, thr):
    if scale is not None:
        x = x * scale
    n = np.sqrt((x * x).sum(-1, keepdims=True)) + F(1e-6)
    return x * np.clip(F(thr) / n, 0, 1)


def convolve(m, st, il, site, cur, weight, dyn):
    """Causal short conv over T rows with the visit's history; weight [C, n_conv]; dyn [T, heads, n_conv] or None."""
    T, C = cur.shape
    nc = m.n_conv
    past = st.hist[il].get(site)
    if past is None:
        past = np.zeros((nc - 1, C), F)
    x = np.concatenate([past, cur], 0)  # [nc-1+T, C]
    st.hist[il][site] = x[-(nc - 1):].copy()
    out = np.zeros_like(cur)
    for k in range(nc):
        out += x[k:k + T] * weight[:, k]
    if dyn is not None:
        heads = dyn.shape[1]
        for lag in range(nc):
            win = x[nc - 1 - lag:nc - 1 - lag + T].reshape(T, heads, C // heads)
            out += (win * dyn[:, :, lag:lag + 1]).reshape(T, C)
    return ((cur + out) / F(math.sqrt((2 if dyn is not None else 1) * nc + 1))).astype(F)


def attention(m, st, il, ip, pos, inputs, is_swa):
    nh, nkv, hd = m.n_head[il], m.n_head_kv[il], m.head_dim
    T = len(pos)
    p = f"blk.{ip}."
    q = inputs[0] @ m.w(p + "attn_q.weight").T
    k = inputs[1] @ m.w(p + "attn_k.weight").T
    v = inputs[2] @ m.w(p + "attn_v.weight").T
    r = inputs[0] @ m.w(p + "attn_rel.weight").T
    kd = vd = None
    if not is_swa:
        kd = (inputs[1] @ m.w(p + "attn_k_conv_proj.weight").T).reshape(T, nkv, m.n_conv)
        vd = (inputs[2] @ m.w(p + "attn_v_conv_proj.weight").T).reshape(T, nkv, m.n_conv)
    k = convolve(m, st, il, "k", k, m.w(p + "attn_k_conv.weight"), kd)
    v = convolve(m, st, il, "v", v, m.w(p + "attn_v_conv.weight"), vd)
    q = clip(q.reshape(T, nh, hd), m.w(p + "attn_q_clip.weight"), m.qk_clip)
    k = clip(k.reshape(T, nkv, hd), None, m.qk_clip)
    v = rms(v.reshape(T, nkv, hd), m.eps)
    r = clip(r.reshape(T, nh, m.n_rel), m.w(p + "attn_rel_clip.weight"), m.rel_clip)
    if not is_swa:
        tau = 1 + m.log_alpha * np.log(np.maximum((np.asarray(pos, F) + 1) / F(m.log_floor), 1))
        q = q * tau[:, None, None].astype(F)
        r = r * tau[:, None, None].astype(F)
    sinks = inputs[0] @ m.w(p + "attn_sink_proj.weight").T  # [T, nh]
    extent = m.swa if is_swa else m.rel_extent
    bias = m.w(p + "attn_rel_bias.weight")  # [extent, n_rel]
    table = np.concatenate([bias, np.zeros((1, m.n_rel), F)], 0)  # [extent+1, n_rel]
    tab = np.einsum("dr,thr->thd", table, r) - sinks[:, :, None]  # [T, nh, extent+1]
    st.k[il].extend(list(k))
    st.v[il].extend(list(v))
    st.pos[il].extend(list(pos))
    K = np.stack(st.k[il], 0)
    V = np.stack(st.v[il], 0)
    kp = np.asarray(st.pos[il])
    out = np.zeros((T, nh, hd), F)
    g = nh // nkv
    for t in range(T):
        d = pos[t] - kp
        ok = d >= 0
        if is_swa:
            ok &= d < m.swa
        idx = np.where((d >= 0) & (d < extent), d, extent)
        for h in range(nh):
            s = (K[:, h // g] @ q[t, h]) / F(math.sqrt(hd)) + tab[t, h, idx]
            s = np.where(ok, s, -np.inf)
            mx = max(s.max(), 0.0)
            e = np.exp(s - mx)
            den = e.sum() + math.exp(0.0 - mx)
            out[t, h] = (e[:, None] * V[:, h // g]).sum(0) / F(den)
    o = out.reshape(T, nh * hd) * sigmoid(inputs[0] @ m.w(p + "attn_gate.weight").T)
    o = o @ m.w(p + "attn_output.weight").T
    return convolve(m, st, il, "ao", o, m.w(p + "attn_post_conv.weight"), None)


def moe(m, ip, x):
    p = f"blk.{ip}."
    logits = x @ m.w(p + "ffn_gate_inp.weight").T
    probs = sigmoid(logits)
    sel = probs + m.w(p + "exp_probs_b.bias")
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        top = np.argsort(-sel[t], kind="stable")[: m.n_used]
        wts = probs[t, top]
        wts = wts / (wts.sum() + F(1e-20)) * F(m.w_scale)
        for e, we in zip(top, wts):
            gu, dn = m.expert(ip, int(e))
            h = gu @ x[t]
            gate, up = h[: m.n_ff], h[m.n_ff:]
            act = F(m.situ_b) * np.tanh(gate / F(m.situ_b)) * sigmoid(gate)
            act = act * (F(m.situ_lb) * np.tanh(up / F(m.situ_lb)))
            out[t] += we * (dn @ act)
    return out


def forward(m, st, ids, pos):
    """Rows `ids` at positions `pos` through all visits; returns the softcapped logits of the last row."""
    T = len(ids)
    cur = np.stack([m.embed_row(i) for i in ids], 0)
    cur = rms(cur, m.eps) * m.w("token_embd_norm.weight")
    blocks = []
    prelude_blocks = prelude_cur = None
    bank = bank_last = None
    for il in range(m.n_visit):
        ip = m.layer(il)
        p = f"blk.{ip}."
        msgs = [None] * 4
        if m.n_pass > 1 and m.core0 <= ip < m.core1:
            pas = (il - m.core0) // m.n_core
            if ip == m.core0:
                if pas == 0:
                    prelude_blocks = list(blocks)
                    prelude_cur = cur
                else:
                    bank = blocks[len(prelude_blocks) + 1:]
                    bank_last = cur
                    blocks = list(prelude_blocks)
                    cur = prelude_cur
            if pas > 0:
                row = 2 * (ip - m.core0)
                qa = m.w("loop_res_query.weight")[row:row + 2]
                qk = m.w("loop_kv_res_query.weight")[row:row + 2]
                ma = mix(bank, bank_last, qa)
                mk = mix(bank, bank_last, qk)
                msgs = [ma[0], mk[0], mk[1], ma[1]]
        is_swa = m.swa_pat[il]
        inpSA = cur
        inputs = [cur, cur, cur]
        if m.has(p + "attn_res_query.weight"):
            mixed = mix(blocks, cur, m.w(p + "attn_res_query.weight"))
            inputs = [mixed[0 if is_swa else j] for j in range(3)]
        if msgs[0] is not None:
            inputs = [control(m, ip, inputs[j], msgs[j], j) for j in range(3)]
        if is_swa and msgs[0] is not None:
            inputs = [rms(x, m.eps) * m.w(p + "attn_norm.weight") for x in inputs]
        elif is_swa:
            x = rms(inputs[0], m.eps) * m.w(p + "attn_norm.weight")
            inputs = [x, x, x]
        else:
            inputs = [rms(inputs[j], m.eps) * m.w(p + f"attn_{n}_inp_norm.weight") for j, n in enumerate("qkv")]
        cur = attention(m, st, il, ip, pos, inputs, is_swa)
        if ip % m.block == 0:
            blocks.append(inpSA)
        else:
            cur = inpSA + cur
        ffn_inp = cur
        x = mix(blocks, cur, m.w(p + "ffn_res_query.weight"))[0]
        if msgs[3] is not None:
            x = control(m, ip, x, msgs[3], 3)
        x = rms(x, m.eps) * m.w(p + "ffn_norm.weight")
        x = moe(m, ip, x)
        x = convolve(m, st, il, "fo", x, m.w(p + "ffn_post_conv.weight"), None)
        cur = ffn_inp + x
        if DUMP:
            for t, p_ in enumerate(pos):
                cur[t].astype(F).tofile(os.path.join(DUMP, f"v{il}_p{p_}.f32"))
    x = mix(blocks, cur, m.w("output_res_query.weight"))[0]
    x = rms(x, m.eps) * m.w("output_norm.weight")
    logits = x[-1] @ m.w("output.weight").T
    logits = np.tanh(logits / F(m.cap)) * F(m.cap)
    if DUMP:
        logits[: m.vocab].astype(F).tofile(os.path.join(DUMP, f"logits0_p{pos[-1]}.f32"))
    return logits[: m.vocab]


def main():
    path, mode = sys.argv[1], sys.argv[2]
    m = Model(path)
    st = State(m)
    if mode == "generate":
        ids = json.load(open(sys.argv[3]))
        n_new = int(sys.argv[4])
        out_path = sys.argv[5]
        logits = forward(m, st, ids, list(range(len(ids))))
        gen, margins = [], []
        for i in range(n_new):
            order = np.argsort(-logits)[:2]
            tok = int(order[0])
            gen.append(tok)
            margins.append(float(logits[order[0]] - logits[order[1]]))
            print(i, tok, round(margins[-1], 4), flush=True)
            json.dump({"prompt": ids, "generated": gen, "top2_margin": margins}, open(out_path, "w"))
            if i + 1 < n_new:
                logits = forward(m, st, [tok], [len(ids) + i])
    else:
        raise SystemExit("unknown mode " + mode)


if __name__ == "__main__":
    main()
