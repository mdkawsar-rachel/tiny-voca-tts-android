"""Convert Underdog Woof's ``woof_int8.npz`` weights to ONNX plus a ``tokens.txt`` vocabulary.

Woof is an autoregressive TTS: a causal speech transformer emits 32-dim latents at 25 Hz, and a
Mimi-style decoder (transformer + SEANet conv stack) upsamples them 960x to 24 kHz audio. Because
generation keeps a growing KV cache, a faithful export is three graphs rather than one:

    woof_encoder.onnx   text tokens + voice id  ->  conditioning (s), EOS logit, prefix KV cache
    woof_step.onnx      one autoregressive step ->  next latent, new s/EOS, updated KV cache
    woof_decoder.onnx   raw latents (T, 32)     ->  24 kHz mono waveform (T*960,)

``tokens.txt`` lists the 95 character-level tokens (id 0 pad, 1 bos, 2 unk, 3..94 the charset) in
``symbol id`` form, one per line. The space character (id 3) is written literally; read it back with
``line.rsplit(" ", 1)`` so the symbol may itself be a space.

Only numpy + onnx are needed to export; onnxruntime is needed to run/verify. The weights are
dequantized to float32 once here, so the ONNX files are self-contained (no npz required at runtime).

Usage:
    python export_onnx.py                          # export to ./onnx/ and verify against numpy
    python export_onnx.py --out onnx --no-check    # export only
    python export_onnx.py --demo "Call 415-555-0199 before 5:30 p.m." --voice mira --seed 1 -o mira.wav
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from underdog_woof.runtime import (CHARS, EPS, VOCAB, TorchRandn, WoofNP, deq,
                                   encode_text, lin_w, unflatten, write_wav)

OPSET = 17
MAX_LEN = 1100
SR = 24000
VOICES = ("juno", "mira")


# ------------------------------------------------------------------------------------------------- onnx builder
class Builder:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def name(self, hint="t"):
        self.n += 1
        return f"{hint}_{self.n}"

    def const(self, arr, hint="c"):
        arr = np.asarray(arr)
        name = self.name(hint)
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def op(self, typ, inputs, hint="t", nout=1, **attrs):
        outs = [self.name(hint) for _ in range(nout)]
        self.nodes.append(helper.make_node(typ, list(inputs), outs, name=self.name(typ), **attrs))
        return outs[0] if nout == 1 else outs

    # -- convenience wrappers -------------------------------------------------------------------
    def matmul(self, a, b):
        return self.op("MatMul", [a, b], "mm")

    def add(self, a, b):
        return self.op("Add", [a, b], "add")

    def mul(self, a, b):
        return self.op("Mul", [a, b], "mul")

    def concat(self, xs, axis):
        return self.op("Concat", xs, "cat", axis=axis)

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", [x, self.const(np.array(axes, np.int64), "ax")], "uns")

    def squeeze(self, x, axes):
        return self.op("Squeeze", [x, self.const(np.array(axes, np.int64), "ax")], "sq")

    def slice(self, x, starts, ends, axes, steps=None):
        ins = [x, self.const(np.array(starts, np.int64), "st"),
               self.const(np.array(ends, np.int64), "en"),
               self.const(np.array(axes, np.int64), "ax")]
        if steps is not None:
            ins.append(self.const(np.array(steps, np.int64), "sp"))
        return self.op("Slice", ins, "sl")

    def gather(self, x, idx, axis=0):
        return self.op("Gather", [x, idx], "ga", axis=axis)

    def reduce_mean(self, x, axis=-1):
        return self.op("ReduceMean", [x], "mean", axes=[axis], keepdims=1)

    def reshape(self, x, shape):
        return self.op("Reshape", [x, self.const(np.array(shape, np.int64), "sh")], "rs")

    def linear(self, x, w, b=None):
        y = self.matmul(x, w)
        return self.add(y, b) if b is not None else y

    def output(self, x, name):
        self.nodes.append(helper.make_node("Identity", [x], [name], name=self.name("out")))
        return name


def rms_norm(g, x, eps):
    sq = g.mul(x, x)
    m = g.reduce_mean(sq)
    denom = g.op("Sqrt", [g.add(m, g.const(np.float32(eps), "eps"))], "sqrt")
    return g.op("Div", [x, denom], "rms")


def rinv(g, x, eps):
    """Scalar (per-vector) 1/sqrt(mean(x^2)+eps), matching runtime.Speech._rinv."""
    m = g.reduce_mean(g.mul(x, x))
    denom = g.op("Sqrt", [g.add(m, g.const(np.float32(eps), "eps"))], "sqrt")
    return g.op("Div", [g.const(np.float32(1), "one"), denom], "rinv")


def layer_norm_noaffine(g, x, eps, dim=None):
    if dim is not None:
        return g.op("LayerNormalization", [x, g.const(np.ones(dim, np.float32), "ln_s")],
                    "ln", axis=-1, epsilon=float(eps))
    mu = g.reduce_mean(x)
    c = g.op("Sub", [x, mu], "sub")
    var = g.reduce_mean(g.mul(c, c))
    denom = g.op("Sqrt", [g.add(var, g.const(np.float32(eps), "eps"))], "sqrt")
    return g.op("Div", [c, denom], "ln")


def silu(g, x):
    return g.mul(x, g.op("Sigmoid", [x], "sig"))


def gelu(g, x):
    half = g.const(np.float32(0.5), "half")
    rsqrt2 = g.const(np.float32(1 / np.sqrt(2)), "rsqrt2")
    e = g.op("Erf", [g.mul(x, rsqrt2)], "erf")
    return g.mul(g.mul(half, x), g.add(g.const(np.float32(1), "one"), e))


def rope_table(inv_freq, max_len=MAX_LEN):
    """Precompute speech-model RoPE tables (cos, sin) of shape (max_len, dh)."""
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    ang = pos * inv_freq[None, :].astype(np.float32)
    cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    return np.concatenate([cs, cs], -1), np.concatenate([-sn, sn], -1)


def rope_apply(g, qk, cos, sin, dh, decoder=False):
    x1, x2 = g.op("Split", [qk], "r_sp", nout=2, axis=-1)
    rot = g.concat([g.op("Neg", [x2], "negx2"), x1] if decoder else [x2, x1], -1)
    return g.add(g.mul(qk, cos), g.mul(rot, sin))


def causal_mask(g, n, window=None):
    idx = g.op("Range", [g.const(np.int64(0), "zero"), n, g.const(np.int64(1), "one")], "idx")
    row = g.unsqueeze(idx, [1])                            # (n, 1) query i
    col = g.unsqueeze(idx, [0])                            # (1, n) key j
    bad = g.op("Greater", [col, row], "future")            # j > i
    if window is not None:
        too_far = g.op("LessOrEqual", [col, g.op("Sub", [row, g.const(np.int64(window), "win")], "sub")], "far")
        bad = g.op("Or", [bad, too_far], "or")
    neg = g.const(np.float32(-1e9), "neg")
    zero = g.const(np.float32(0), "zero")
    return g.op("Where", [bad, neg, zero], "mask")


# ------------------------------------------------------------------------------------------------- weight loading
def load_state(path):
    with np.load(path) as z:
        return unflatten(z)


def speech_weights(st):
    """Dequantized speech weights, linear matrices stored as (in, out) for right-multiplication."""
    s = st["speech"]
    c = s["cfg"]
    W = {"cfg": c}
    W["layers"] = [[(lin_w(e)[0].T.copy(), None) for e in layer] for layer in s["layers"]]
    for key in ("ce", "ada", "lat_in", "h_inp", "h_out"):
        w, b = lin_w(s[key])
        W[key] = (w.T.copy(), b)
    W["hblocks"] = [[(lin_w(e)[0].T.copy(), lin_w(e)[1]) for e in blk] for blk in s["hblocks"]]
    W["text_emb"] = s["text_emb"].astype(np.float32)
    W["bos"] = s["bos"].astype(np.float32)
    W["voices"] = np.stack([st["voices"][v].astype(np.float32) for v in VOICES])  # (2, 16, d)
    Dh = c["d"] // c["H"]
    W["inv_freq"] = (1.0 / (c["rope_base"] ** (np.arange(0, Dh, 2, dtype=np.float32) / np.float32(Dh)))).astype(np.float32)
    return W


# ------------------------------------------------------------------------------------------------- tokens.txt
def write_tokens(path):
    syms = {0: "<pad>", 1: "<bos>", 2: "<unk>"}
    for ch, i in VOCAB.items():
        syms[i] = ch
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(CHARS) + 3):
            f.write(f"{syms[i]} {i}\n")


def read_tokens(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sym, idx = line.rsplit(" ", 1)
            vocab[sym] = int(idx)
    return vocab


# ------------------------------------------------------------------------------------------------- encoder graph
def build_encoder(W, split_kv=True):
    g = Builder()
    c = W["cfg"]
    L, d, H, ff, hw = c["L"], c["d"], c["H"], c["ff"], c["hw"]
    Dh = d // H
    scale = g.const(np.float32(Dh ** -0.5), "scale")
    cos_arr, sin_arr = rope_table(W["inv_freq"])
    cos_tbl = g.const(cos_arr, "cos_tbl")
    sin_tbl = g.const(sin_arr, "sin_tbl")
    sp_qkv = g.const(np.array([2 * H, H], np.int64), "sp_qkv")
    sp_ce = g.const(np.array([hw, 1], np.int64), "sp_ce")
    text_emb = g.const(W["text_emb"], "text_emb")
    bos = g.const(W["bos"], "bos")
    voices = g.const(W["voices"], "voices")

    tokens_vi = helper.make_tensor_value_info("tokens", TensorProto.INT64, ["N"])
    voice_vi = helper.make_tensor_value_info("voice_id", TensorProto.INT64, [])
    x = g.gather(text_emb, "tokens")                        # (N, d)
    voice = g.gather(voices, "voice_id")                    # (16, d)
    x = g.concat([voice, x, g.unsqueeze(bos, [0])], 0)      # (p0, d)
    p0 = g.op("Shape", [x], "p0shape")
    p0 = g.squeeze(g.slice(p0, [0], [1], [0]), [0])         # scalar p0

    pos = g.op("Range", [g.const(np.int64(0), "z"), p0, g.const(np.int64(1), "o")], "pos")
    cos = g.unsqueeze(g.gather(cos_tbl, pos), [1])          # (p0, 1, Dh)
    sin = g.unsqueeze(g.gather(sin_tbl, pos), [1])
    mask = causal_mask(g, p0)

    ks, vs = [], []
    for i in range(L):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = W["layers"][i]
        wqkv = g.const(wqkv, f"wqkv{i}"); wo = g.const(wo, f"wo{i}")
        w1 = g.const(w1, f"w1{i}"); w2 = g.const(w2, f"w2{i}")
        xn = rms_norm(g, x, EPS)
        qkv = g.reshape(g.matmul(xn, wqkv), [-1, 3 * H, Dh])  # (p0, 3*H, Dh)
        qk, v = g.op("Split", [qkv, sp_qkv], f"qkv_sp{i}", nout=2, axis=1)
        qk = rope_apply(g, qk, cos, sin, Dh)
        q, k = g.op("Split", [qk], f"qk_sp{i}", nout=2, axis=1)
        q = g.op("Transpose", [q], "qt", perm=[1, 0, 2])    # (H, p0, Dh)
        k = g.op("Transpose", [k], "kt", perm=[1, 0, 2])
        v = g.op("Transpose", [v], "vt", perm=[1, 0, 2])
        if split_kv:
            ks.append(k); vs.append(v)
        else:
            ks.append(g.unsqueeze(k, [0])); vs.append(g.unsqueeze(v, [0]))
        scores = g.mul(g.matmul(q, g.op("Transpose", [k], "kT", perm=[0, 2, 1])), scale)
        scores = g.add(scores, mask)
        attn = g.op("Softmax", [scores], "sm", axis=-1)
        a = g.matmul(attn, v)                               # (H, p0, Dh)
        a = g.reshape(g.op("Transpose", [a], "at", perm=[1, 0, 2]), [-1, d])
        x = g.add(x, g.matmul(a, wo))
        xn = rms_norm(g, x, EPS)
        gu = g.matmul(xn, w1)                               # (p0, 2*ff)
        gate, up = g.op("Split", [gu], f"gu_sp{i}", nout=2, axis=-1)
        x = g.add(x, g.matmul(g.mul(silu(g, gate), up), w2))

    last = g.gather(x, g.const(np.int64(-1), "lastidx"))    # (d,)
    # conditioning head: ce = (w @ last) * rinv(last) + b
    ce_w, ce_b = W["ce"]
    ce = g.matmul(last, g.const(ce_w, "ce_w"))
    ce = g.add(g.mul(ce, rinv(g, last, EPS)), g.const(ce_b, "ce_b"))
    s_raw, e_raw = g.op("Split", [ce, sp_ce], "ce_sp", nout=2, axis=0)
    s = silu(g, s_raw)
    e = g.squeeze(e_raw, [0])

    g.output(s, "s"); g.output(e, "eos")
    outs = [helper.make_tensor_value_info("s", TensorProto.FLOAT, [hw]),
            helper.make_tensor_value_info("eos", TensorProto.FLOAT, [])]
    if split_kv:
        for i in range(L):
            g.output(ks[i], f"k_cache{i}")
            outs.append(helper.make_tensor_value_info(f"k_cache{i}", TensorProto.FLOAT, [H, "p0", Dh]))
        for i in range(L):
            g.output(vs[i], f"v_cache{i}")
            outs.append(helper.make_tensor_value_info(f"v_cache{i}", TensorProto.FLOAT, [H, "p0", Dh]))
    else:
        g.output(g.concat(ks, 0), "k_cache"); g.output(g.concat(vs, 0), "v_cache")
        outs.extend([helper.make_tensor_value_info("k_cache", TensorProto.FLOAT, [L, H, "p0", Dh]),
                     helper.make_tensor_value_info("v_cache", TensorProto.FLOAT, [L, H, "p0", Dh])])
    graph = helper.make_graph(g.nodes, "woof_encoder", [tokens_vi, voice_vi], outs, g.inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


# ------------------------------------------------------------------------------------------------- step graph
def build_step(W, split_kv=True):
    g = Builder()
    c = W["cfg"]
    L, d, H, ff, hw, depth = c["L"], c["d"], c["H"], c["ff"], c["hw"], c["depth"]
    Dh = d // H
    scale = g.const(np.float32(Dh ** -0.5), "scale")
    cos_arr, sin_arr = rope_table(W["inv_freq"])
    cos_tbl = g.const(cos_arr, "cos_tbl")
    sin_tbl = g.const(sin_arr, "sin_tbl")
    sp_qkv = g.const(np.array([2 * H, H], np.int64), "sp_qkv")
    sp_ce = g.const(np.array([hw, 1], np.int64), "sp_ce")

    s_vi = helper.make_tensor_value_info("s_in", TensorProto.FLOAT, [hw])
    noise_vi = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [c["latent"]])
    pos_vi = helper.make_tensor_value_info("pos", TensorProto.INT64, [])
    step_ins = [s_vi, noise_vi, pos_vi]
    if split_kv:
        step_ins.extend(helper.make_tensor_value_info(f"k_in{i}", TensorProto.FLOAT, [H, "T", Dh]) for i in range(L))
        step_ins.extend(helper.make_tensor_value_info(f"v_in{i}", TensorProto.FLOAT, [H, "T", Dh]) for i in range(L))
    else:
        step_ins.extend([helper.make_tensor_value_info("k_in", TensorProto.FLOAT, [L, H, "T", Dh]),
                         helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [L, H, "T", Dh])])

    # ---- one-step flow head: z = noise + head(s, noise) ----
    ada_w, ada_b = W["ada"]
    A = g.linear("s_in", g.const(ada_w, "ada_w"), g.const(ada_b, "ada_b"))   # (4480,)
    ada_chunks = g.op("Split", [A], "ada_sp", nout=3 * depth + 2, axis=0)
    hi_w, hi_b = W["h_inp"]
    h = g.linear("noise", g.const(hi_w, "h_inp_w"), g.const(hi_b, "h_inp_b"))
    for j, ((w1, b1), (w2, b2)) in enumerate(W["hblocks"]):
        shift, scl, gate = ada_chunks[3 * j], ada_chunks[3 * j + 1], ada_chunks[3 * j + 2]
        y = g.add(shift, g.mul(layer_norm_noaffine(g, h, 1e-5, hw), scl))
        inner = g.linear(silu(g, g.linear(y, g.const(w1, f"hb_w1_{j}"), g.const(b1, f"hb_b1_{j}"))),
                         g.const(w2, f"hb_w2_{j}"), g.const(b2, f"hb_b2_{j}"))
        h = g.add(h, g.mul(gate, inner))
    shift, scl = ada_chunks[3 * depth], ada_chunks[3 * depth + 1]
    ho_w, ho_b = W["h_out"]
    head = g.linear(g.add(shift, g.mul(layer_norm_noaffine(g, h, 1e-5, hw), scl)),
                    g.const(ho_w, "h_out_w"), g.const(ho_b, "h_out_b"))
    z = g.add("noise", head)                                # (latent,)

    # ---- feedback through the causal transformer with KV cache ----
    li_w, li_b = W["lat_in"]
    x = g.linear(z, g.const(li_w, "lat_in_w"), g.const(li_b, "lat_in_b"))   # (d,)
    cos = g.gather(cos_tbl, "pos")                          # (Dh,)
    sin = g.gather(sin_tbl, "pos")                          # (Dh,)

    k_outs, v_outs = [], []
    for i in range(L):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = W["layers"][i]
        wqkv = g.const(wqkv, f"wqkv{i}"); wo = g.const(wo, f"wo{i}")
        w1 = g.const(w1, f"w1{i}"); w2 = g.const(w2, f"w2{i}")
        Ki = f"k_in{i}" if split_kv else g.gather("k_in", g.const(np.int64(i), f"ki{i}"))
        Vi = f"v_in{i}" if split_kv else g.gather("v_in", g.const(np.int64(i), f"vi{i}"))
        qkv = g.mul(g.matmul(x, wqkv), rinv(g, x, EPS))     # (3*H*Dh,)
        qkv = g.reshape(qkv, [3 * H, Dh])
        qk, v = g.op("Split", [qkv, sp_qkv], f"qkv_sp{i}", nout=2, axis=0)
        qk = rope_apply(g, qk, cos, sin, Dh)
        q, k = g.op("Split", [qk], f"qk_sp{i}", nout=2, axis=0)
        K = g.concat([Ki, g.unsqueeze(k, [1])], 1)            # (H, T+1, Dh)
        V = g.concat([Vi, g.unsqueeze(v, [1])], 1)
        if split_kv:
            k_outs.append(K); v_outs.append(V)
        else:
            k_outs.append(g.unsqueeze(K, [0])); v_outs.append(g.unsqueeze(V, [0]))
        qs = g.mul(q, scale)
        scores = g.squeeze(g.matmul(K, g.unsqueeze(qs, [-1])), [-1])   # (H, T+1)
        attn = g.op("Softmax", [scores], "sm", axis=-1)
        a = g.squeeze(g.matmul(g.unsqueeze(attn, [1]), V), [1])       # (H, Dh)
        x = g.add(x, g.matmul(g.reshape(a, [d]), wo))
        gu = g.mul(g.matmul(x, w1), rinv(g, x, EPS))
        gate, up = g.op("Split", [gu], f"gu_sp{i}", nout=2, axis=0)
        x = g.add(x, g.matmul(g.mul(silu(g, gate), up), w2))

    ce_w, ce_b = W["ce"]
    ce = g.matmul(x, g.const(ce_w, "ce_w"))
    ce = g.add(g.mul(ce, rinv(g, x, EPS)), g.const(ce_b, "ce_b"))
    s_raw, e_raw = g.op("Split", [ce, sp_ce], "ce_sp", nout=2, axis=0)
    s_new = silu(g, s_raw)
    e_new = g.squeeze(e_raw, [0])

    g.output(z, "z"); g.output(s_new, "s"); g.output(e_new, "eos")
    outs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, [c["latent"]]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [hw]),
            helper.make_tensor_value_info("eos", TensorProto.FLOAT, [])]
    if split_kv:
        for i in range(L):
            g.output(k_outs[i], f"k_cache{i}")
            outs.append(helper.make_tensor_value_info(f"k_cache{i}", TensorProto.FLOAT, [H, "T1", Dh]))
        for i in range(L):
            g.output(v_outs[i], f"v_cache{i}")
            outs.append(helper.make_tensor_value_info(f"v_cache{i}", TensorProto.FLOAT, [H, "T1", Dh]))
    else:
        g.output(g.concat(k_outs, 0), "k_cache"); g.output(g.concat(v_outs, 0), "v_cache")
        outs.extend([helper.make_tensor_value_info("k_cache", TensorProto.FLOAT, [L, H, "T1", Dh]),
                     helper.make_tensor_value_info("v_cache", TensorProto.FLOAT, [L, H, "T1", Dh])])
    graph = helper.make_graph(g.nodes, "woof_step", step_ins, outs, g.inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


# ------------------------------------------------------------------------------------------------- decoder graph
def build_decoder(st):
    g = Builder()
    d = st["decoder"]
    c = d["cfg"]
    nh, hd, window, eps = c["heads"], c["head_dim"], c["window"], float(c["eps"])
    D = nh * hd
    mean = g.const(st["mean"].astype(np.float32), "mean")
    std = g.const(st["std"].astype(np.float32), "std")
    inp_w = g.const(deq(d["inp"][0]).T.copy(), "inp_w")     # (32, D)
    inp_b = g.const(deq(d["inp"][1]), "inp_b")
    inv_freq = g.const(d["inv_freq"].astype(np.float32), "inv_freq")

    lat_vi = helper.make_tensor_value_info("latents", TensorProto.FLOAT, ["T", st["speech"]["cfg"]["latent"]])
    x = g.add(g.mul("latents", std), mean)                  # denormalize
    h = g.linear(x, inp_w, inp_b)                           # (T, D)
    T = g.squeeze(g.slice(g.op("Shape", [h], "Tshape"), [0], [1], [0]), [0])

    # rotary embeddings for the decoder (sin = [sn, sn], rotate-half = [-x2, x1])
    pos = g.op("Range", [g.const(np.int64(0), "z"), T, g.const(np.int64(1), "o")], "pos")
    posf = g.op("Cast", [pos], "posf", to=TensorProto.FLOAT)
    fr = g.mul(g.unsqueeze(posf, [1]), inv_freq)            # (T, hd/2)
    emb = g.concat([fr, fr], -1)
    cos = g.unsqueeze(g.op("Cos", [emb], "cos"), [0, 1])    # (1, 1, T, hd)
    sin = g.unsqueeze(g.op("Sin", [emb], "sin"), [0, 1])
    mask = causal_mask(g, T, window=window)

    for li, layer in enumerate(d["tf"]):
        ln1 = [g.const(deq(t), f"ln1_{li}_{k}") for k, t in enumerate(layer["ln1"])]
        ln2 = [g.const(deq(t), f"ln2_{li}_{k}") for k, t in enumerate(layer["ln2"])]
        qkv_w = g.const(deq(layer["qkv"]).T.copy(), f"qkv_{li}")
        o_w = g.const(deq(layer["o"]).T.copy(), f"o_{li}")
        fc1 = g.const(deq(layer["fc1"]).T.copy(), f"fc1_{li}")
        fc2 = g.const(deq(layer["fc2"]).T.copy(), f"fc2_{li}")
        y = g.op("LayerNormalization", [h, ln1[0], ln1[1]], "ln1", axis=-1, epsilon=eps)
        qkv = g.reshape(g.matmul(y, qkv_w), [-1, 3, nh, hd])
        qkv = g.op("Transpose", [qkv], "qkvT", perm=[1, 2, 0, 3])   # (3, nh, T, hd)
        q = g.squeeze(g.slice(qkv, [0], [1], [0]), [0])
        k = g.squeeze(g.slice(qkv, [1], [2], [0]), [0])
        v = g.squeeze(g.slice(qkv, [2], [3], [0]), [0])
        qk = g.concat([g.unsqueeze(q, [0]), g.unsqueeze(k, [0])], 0)   # (2, nh, T, hd)
        qk = rope_apply(g, qk, cos, sin, hd, decoder=True)
        q = g.squeeze(g.slice(qk, [0], [1], [0]), [0])
        k = g.squeeze(g.slice(qk, [1], [2], [0]), [0])
        scale = g.const(np.float32(1 / np.sqrt(hd)), "scale")
        scores = g.mul(g.matmul(q, g.op("Transpose", [k], "kT", perm=[0, 2, 1])), scale)
        scores = g.add(scores, mask)
        attn = g.op("Softmax", [scores], "sm", axis=-1)
        a = g.matmul(attn, v)
        a = g.reshape(g.op("Transpose", [a], "at", perm=[1, 0, 2]), [-1, D])
        h = g.add(h, g.matmul(a, o_w))
        y = g.op("LayerNormalization", [h, ln2[0], ln2[1]], "ln2", axis=-1, epsilon=eps)
        h = g.add(h, g.matmul(gelu(g, g.matmul(y, fc1)), fc2))

    # SEANet conv stack, run on (1, C, T); causal convs and stride-s transposed convs (kernel 2s).
    x = g.op("Transpose", [g.unsqueeze(h, [0])], "seanet_in", perm=[0, 2, 1])  # (1, D, T)

    def conv(p, hint):
        w = deq(p["w"])                                     # (Cout, Cin, k)
        k = w.shape[2]
        wname = g.const(w, f"{hint}_w"); bname = g.const(p["b"].astype(np.float32), f"{hint}_b")
        return g.op("Conv", [x, wname, bname], hint, kernel_shape=[k], pads=[k - 1, 0], strides=[1], dilations=[1])

    def convt(p, hint):
        w = deq(p["w"])                                     # (Cin, Cout, k=2s)
        k, s = w.shape[2], p["stride"]
        wname = g.const(w, f"{hint}_w"); bname = g.const(p["b"].astype(np.float32), f"{hint}_b")
        y = g.op("ConvTranspose", [x, wname, bname], hint, kernel_shape=[k], strides=[s],
                 pads=[0, 0], output_padding=[0], dilations=[1])
        return g.slice(y, [0], [-s], [2])                   # drop trailing carry -> exactly T*s

    for idx, (kind, p) in enumerate(d["seanet"]):
        if kind == "elu":
            x = g.op("Elu", [x], f"elu{idx}", alpha=1.0)
        elif kind == "conv":
            x = conv(p, f"conv{idx}")
        elif kind == "convt":
            x = convt(p, f"convt{idx}")
        elif kind == "res":
            res = x
            y = g.op("Elu", [x], f"res{idx}_elu0", alpha=1.0)
            w0 = deq(p[0]["w"]); k0 = w0.shape[2]
            y = g.op("Conv", [y, g.const(w0, f"res{idx}_w0"), g.const(p[0]["b"].astype(np.float32), f"res{idx}_b0")],
                     f"res{idx}_c0", kernel_shape=[k0], pads=[k0 - 1, 0], strides=[1], dilations=[1])
            y = g.op("Elu", [y], f"res{idx}_elu1", alpha=1.0)
            w1 = deq(p[1]["w"]); k1 = w1.shape[2]
            y = g.op("Conv", [y, g.const(w1, f"res{idx}_w1"), g.const(p[1]["b"].astype(np.float32), f"res{idx}_b1")],
                     f"res{idx}_c1", kernel_shape=[k1], pads=[k1 - 1, 0], strides=[1], dilations=[1])
            x = g.add(y, res)
    audio = g.squeeze(x, [0, 1])                            # (T*960,)
    g.output(audio, "audio")
    out = helper.make_tensor_value_info("audio", TensorProto.FLOAT, ["N"])
    graph = helper.make_graph(g.nodes, "woof_decoder", [lat_vi], [out], g.inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


# ------------------------------------------------------------------------------------------------- save / quantize / verify
def fp16_weights(model, min_elems=256):
    """Store large float32 initializers as float16 with a Cast-to-fp32 in front, halving weight bytes
    while all compute stays float32. Skips MatMulNBits scale inputs and small tensors."""
    g = model.graph
    skip = {node.input[2] for node in g.node if node.op_type == "MatMulNBits" and len(node.input) > 2}
    new_inits, casts = [], []
    for init in g.initializer:
        n = int(np.prod(init.dims)) if init.dims else 1
        if init.data_type != TensorProto.FLOAT or n < min_elems or init.name in skip:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init)
        assert np.isfinite(arr).all() and float(np.abs(arr).max()) < 65504, f"fp16 overflow in {init.name}"
        new_inits.append(numpy_helper.from_array(arr.astype(np.float16), init.name + "_h"))
        casts.append(helper.make_node("Cast", [init.name + "_h"], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "_cast"))
    if casts:
        del g.initializer[:]
        g.initializer.extend(new_inits)
        nodes = list(g.node)
        del g.node[:]
        g.node.extend(casts + nodes)
    return model


def quantize_nb8(model, block_size=32, accuracy_level=4, min_elems=4096):
    """Quantize large constant-B MatMul weights to 8-bit blockwise symmetric com.microsoft::MatMulNBits."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    consumers = {}
    for node in g.node:
        for inp in node.input:
            consumers[inp] = consumers.get(inp, 0) + 1
    mm_consumers = {}
    for node in g.node:
        if node.op_type == "MatMul" and len(node.input) >= 2:
            mm_consumers[node.input[1]] = mm_consumers.get(node.input[1], 0) + 1

    packed_cache = {}
    new_nodes, drop_inits = [], set()
    replaced = False
    for node in g.node:
        b_name = node.input[1] if node.op_type == "MatMul" and len(node.input) == 2 else None
        init = inits.get(b_name) if b_name else None
        if (init is None or init.data_type != TensorProto.FLOAT or len(init.dims) != 2
                or int(init.dims[0]) * int(init.dims[1]) < min_elems
                or consumers.get(b_name, 0) != mm_consumers.get(b_name, 0)):
            new_nodes.append(node)
            continue
        if b_name not in packed_cache:
            w = numpy_helper.to_array(init).astype(np.float32)
            rows, cols = w.shape
            k_blocks = (rows + block_size - 1) // block_size
            pad = k_blocks * block_size - rows
            if pad > 0:
                w = np.pad(w, ((0, pad), (0, 0)), mode="constant")
            # MatMulNBits expects uint8 weights packed as (N, k_blocks, block_size) centered at 128
            w_blk = np.ascontiguousarray(w.T).reshape(cols, k_blocks, block_size)
            scales = (np.maximum(np.abs(w_blk).max(axis=-1), 1e-8) / 127.0).astype(np.float32)
            q = (np.clip(np.round(w_blk / scales[..., None]), -127, 127).astype(np.int32) + 128).astype(np.uint8)
            q_name, s_name = b_name + "_q8", b_name + "_scales"
            g.initializer.extend([numpy_helper.from_array(q, q_name), numpy_helper.from_array(scales, s_name)])
            packed_cache[b_name] = (q_name, s_name, rows, cols)
            drop_inits.add(b_name)
        q_name, s_name, rows, cols = packed_cache[b_name]
        attrs = dict(K=rows, N=cols, bits=8, block_size=block_size)
        if accuracy_level:
            attrs["accuracy_level"] = accuracy_level
        new_nodes.append(helper.make_node(
            "MatMulNBits", [node.input[0], q_name, s_name], list(node.output),
            name=(node.name + "_Q8") if node.name else "", domain="com.microsoft", **attrs
        ))
        replaced = True

    if replaced:
        del g.node[:]
        g.node.extend(new_nodes)
        kept = [t for t in g.initializer if t.name not in drop_inits]
        del g.initializer[:]
        g.initializer.extend(kept)
        if not any(op.domain == "com.microsoft" for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return model


def save_model(model, path, fp16=False, quant="none"):
    model.ir_version = 10  # broad onnxruntime compatibility (opset 17 needs IR >= 8)
    model = onnx.shape_inference.infer_shapes(model)
    if quant == "nb8":
        model = quantize_nb8(model)
    if fp16:
        fp16_weights(model)
    onnx.checker.check_model(model)
    if quant == "qint8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        tmp = path + ".fp32.tmp"
        onnx.save(model, tmp)
        try:
            quantize_dynamic(tmp, path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        onnx.save(model, path)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def export(weights, outdir, check=True, fp16=False, quant="none", split_kv=True):
    os.makedirs(outdir, exist_ok=True)
    st = load_state(weights)
    W = speech_weights(st)
    write_tokens(os.path.join(outdir, "tokens.txt"))
    print(f"  wrote {os.path.join(outdir, 'tokens.txt')}  ({len(CHARS) + 3} tokens)")
    save_model(build_encoder(W, split_kv=split_kv), os.path.join(outdir, "woof_encoder.onnx"), fp16=fp16, quant=quant)
    save_model(build_step(W, split_kv=split_kv), os.path.join(outdir, "woof_step.onnx"), fp16=fp16, quant=quant)
    # Keep waveform decoder weights in fp16->fp32: int8/ConvInteger hurts both latency and audio SNR.
    save_model(build_decoder(st), os.path.join(outdir, "woof_decoder.onnx"), fp16=fp16, quant="none")
    if check:
        verify(weights, outdir, tol=3e-2 if fp16 else 1e-4, quantized=(quant != "none"))


# ------------------------------------------------------------------------------------------------- onnx runtime driver
class WoofONNX:
    """Drives the three exported graphs. Any standard-normal noise source works; pass rng=TorchRandn(seed)
    to reproduce the numpy runtime bit-for-bit."""

    def __init__(self, outdir, voice="juno"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        dec_so = ort.SessionOptions()
        dec_so.intra_op_num_threads = min(2, os.cpu_count() or 2)
        self.enc = ort.InferenceSession(os.path.join(outdir, "woof_encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.step = ort.InferenceSession(os.path.join(outdir, "woof_step.onnx"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(outdir, "woof_decoder.onnx"), dec_so, providers=["CPUExecutionProvider"])
        step_ins = [i.name for i in self.step.get_inputs()]
        self.split_kv = "k_in0" in step_ins
        self.layers = sum(1 for n in step_ins if n.startswith("k_in")) if self.split_kv else 0
        self.voice_id = VOICES.index(voice)
        self.latent = 32
        self.max_len = MAX_LEN

    def __call__(self, text, seed=0, temp=0.5, eos=0.9, max_frames=750, rng=None):
        rng = rng if rng is not None else TorchRandn(seed)
        ids = np.array(encode_text(text), np.int64)
        enc_out = self.enc.run(None, {"tokens": ids, "voice_id": np.array(self.voice_id, np.int64)})
        s, e = enc_out[0], enc_out[1]
        kv = enc_out[2:]
        p0 = 16 + len(ids) + 1
        max_frames = min(max_frames, self.max_len - p0)
        eos_t = np.log(eos / (1 - eos))
        lat = []
        for f in range(max_frames):
            if f > 3 and float(e) > eos_t:
                break
            noise = rng.randn(self.latent).astype(np.float32) * np.float32(temp)
            feeds = {"s_in": s, "noise": noise, "pos": np.array(p0 + f, np.int64)}
            if self.split_kv:
                for i in range(self.layers):
                    feeds[f"k_in{i}"] = kv[i]
                    feeds[f"v_in{i}"] = kv[self.layers + i]
            else:
                feeds["k_in"] = kv[0]
                feeds["v_in"] = kv[1]
            step_out = self.step.run(None, feeds)
            z, s, e = step_out[0], step_out[1], step_out[2]
            kv = step_out[3:]
            lat.append(z)
        lat = np.stack(lat).astype(np.float32)
        audio = self.dec.run(None, {"latents": lat})[0]
        return lat, audio.astype(np.float32)


def _assert_close(name, a, b, tol=1e-4, quantized=False):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    err = float(np.abs(a - b).max())
    if quantized and a.size > 1:
        cos = float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  {name:18s} cos={cos:.5f} max|d|={err:.3e}")
        assert cos >= 0.99, f"{name}: cosine similarity {cos} < 0.99"
    else:
        lim = max(tol, 0.3) if quantized else tol
        print(f"  {name:18s} max|d|={err:.3e}")
        assert err < lim, f"{name}: mismatch {err} >= {lim}"


def verify(weights, outdir, text="Please call me back at 415-555-0199 before 5:30 p.m. on Friday.",
           tol=1e-4, quantized=False):
    """Check each graph against the numpy runtime on identical inputs. The autoregressive sampler is
    float32-chaotic, so a full free-running loop diverges from numpy after many steps even though every
    individual step matches to ~1e-6; we therefore verify the three components directly."""
    import onnxruntime as ort
    import underdog_woof.runtime as rt
    print("\nVerifying ONNX against the numpy runtime (per component) ...")
    ref = WoofNP(weights, voice="juno")
    sp = ref.speech
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    prov = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(os.path.join(outdir, "woof_encoder.onnx"), so, providers=prov)
    step = ort.InferenceSession(os.path.join(outdir, "woof_step.onnx"), so, providers=prov)
    dec = ort.InferenceSession(os.path.join(outdir, "woof_decoder.onnx"), so, providers=prov)
    split_kv = any(i.name == "k_in0" for i in step.get_inputs())
    ids = np.asarray(encode_text(text))

    # decoder on reference latents
    old = rt.RNG; rt.RNG = TorchRandn(0)
    lat_ref = sp.generate(ref.voice, encode_text(text), ref.frames_max, 0.5, 0.9)
    rt.RNG = old
    wav_ref = ref.dec(lat_ref * ref.std + ref.mean)
    wav_o = dec.run(None, {"latents": lat_ref.astype(np.float32)})[0]
    _assert_close("decoder audio", wav_ref, wav_o, tol)

    # encoder (prefix) -> conditioning + KV cache
    x = np.concatenate([ref.voice, sp.text_emb[ids], sp.bos[None]], 0).astype(np.float32)
    p0 = x.shape[0]
    for i in range(sp.L):
        x = sp._prefix_block(i, x, p0)
    s_ref, e_ref = sp._cond(np.ascontiguousarray(x[-1]))
    k_ref = sp.kv[:, 0, :, :p0, :].copy(); v_ref = sp.kv[:, 1, :, :p0, :].copy()
    enc_o = enc.run(None, {"tokens": ids.astype(np.int64), "voice_id": np.array(0, np.int64)})
    s_o, e_o = enc_o[0], enc_o[1]
    k_o = np.stack(enc_o[2:2 + sp.L], 0) if split_kv else enc_o[2]
    v_o = np.stack(enc_o[2 + sp.L:], 0) if split_kv else enc_o[3]
    _assert_close("encoder s", s_ref, s_o, tol, quantized)
    _assert_close("encoder eos", e_ref, e_o, tol, quantized)
    _assert_close("encoder k_cache", k_ref, k_o, tol, quantized)
    _assert_close("encoder v_cache", v_ref, v_o, tol, quantized)

    # one autoregressive step fed the reference state
    noise = TorchRandn(0).randn(sp.latent).astype(np.float32) * np.float32(0.5)
    z_ref = noise + sp._head(s_ref, noise)
    xx = sp.lat_in[0] @ z_ref + sp.lat_in[1]
    for i in range(sp.L):
        xx = sp._block1(i, xx, p0)
    sn_ref, en_ref = sp._cond(xx)
    kn_ref = sp.kv[:, 0, :, :p0 + 1, :].copy(); vn_ref = sp.kv[:, 1, :, :p0 + 1, :].copy()
    feeds = {"s_in": s_ref.astype(np.float32), "noise": noise, "pos": np.array(p0, np.int64)}
    if split_kv:
        for i in range(sp.L):
            feeds[f"k_in{i}"] = k_ref[i].astype(np.float32)
            feeds[f"v_in{i}"] = v_ref[i].astype(np.float32)
    else:
        feeds["k_in"] = k_ref.astype(np.float32)
        feeds["v_in"] = v_ref.astype(np.float32)
    step_o = step.run(None, feeds)
    z_o, sn_o, en_o = step_o[0], step_o[1], step_o[2]
    kn_o = np.stack(step_o[3:3 + sp.L], 0) if split_kv else step_o[3]
    vn_o = np.stack(step_o[3 + sp.L:], 0) if split_kv else step_o[4]
    _assert_close("step z", z_ref, z_o, tol, quantized)
    _assert_close("step s", sn_ref, sn_o, tol, quantized)
    _assert_close("step eos", en_ref, en_o, tol, quantized)
    _assert_close("step k_cache", kn_ref, kn_o, tol, quantized)
    _assert_close("step v_cache", vn_ref, vn_o, tol, quantized)

    # full free-running ONNX pipeline sanity across both voices
    for voice, sample, sd in (("juno", text, 0), ("mira", "Your order has shipped today.", 1)):
        onx = WoofONNX(outdir, voice=voice)
        lat, wav = onx(sample, seed=sd)
        assert 0 < len(lat) < 750 and np.isfinite(wav).all() and len(wav) == len(lat) * 960, f"{voice}: {len(lat)} frames"
        print(f"  full pipeline ({voice}): {len(lat)} frames -> {len(wav) / SR:.2f}s of finite audio")
    print("  OK: encoder, step and decoder all match the numpy runtime.")


# ------------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "woof_int8.npz"))
    ap.add_argument("--out", default="onnx", help="output directory for tokens.txt and the .onnx graphs")
    ap.add_argument("--no-check", action="store_true", help="skip the numpy-vs-onnx verification")
    ap.add_argument("--fp16", action="store_true", help="store float weights as float16 (halves size; compute stays fp32)")
    ap.add_argument("--quant", default="none", choices=("none", "nb8", "qint8"),
                    help="weight quantization for encoder/step: nb8 (8-bit MatMulNBits, fastest) or qint8")
    ap.add_argument("--legacy", action="store_true", help="export legacy stacked k_cache/v_cache schema")
    ap.add_argument("--demo", metavar="TEXT", help="synthesize TEXT with the exported ONNX models")
    ap.add_argument("--voice", default="juno", choices=VOICES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--output", default="onnx_demo.wav")
    args = ap.parse_args()

    if args.demo:
        tts = WoofONNX(args.out, voice=args.voice)
        _, wav = tts(args.demo, seed=args.seed)
        write_wav(args.output, wav)
        print(f"Synthesized {len(wav) / SR:.2f}s -> {args.output}")
        return

    tag = ", ".join(x for x in (args.quant if args.quant != "none" else "", "fp16" if args.fp16 else "",
                                "legacy" if args.legacy else "split-kv") if x)
    print(f"Exporting {args.weights} -> {args.out}/ ({tag})")
    export(args.weights, args.out, check=not args.no_check, fp16=args.fp16, quant=args.quant, split_kv=not args.legacy)


if __name__ == "__main__":
    main()