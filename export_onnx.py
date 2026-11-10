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


def layer_norm_noaffine(g, x, eps):
    mu = g.reduce_mean(x)
    c = g.op("Sub", [x, mu], "sub")
    var = g.reduce_mean(g.mul(c, c))
    denom = g.op("Sqrt", [g.add(var, g.const(np.float32(eps), "eps"))], "sqrt")
    return g.op("Div", [c, denom], "ln")


def silu(g, x):
    one = g.const(np.float32(1), "one")
    neg = g.op("Neg", [x], "neg")
    exp = g.op("Exp", [neg], "exp")
    return g.op("Div", [x, g.add(one, exp)], "silu")


def gelu(g, x):
    half = g.const(np.float32(0.5), "half")
    rsqrt2 = g.const(np.float32(1 / np.sqrt(2)), "rsqrt2")
    e = g.op("Erf", [g.mul(x, rsqrt2)], "erf")
    return g.mul(g.mul(half, x), g.add(g.const(np.float32(1), "one"), e))


def rope_angles(g, positions, inv_freq, dh):
    """positions (P,) integer -> cos (P, dh), sin (P, dh) with the speech-model sign convention
    (sin = [-sn, sn]); the decoder flips the rotate-half sign instead, see rope_apply below."""
    posf = g.op("Cast", [positions], "posf", to=TensorProto.FLOAT)
    ang = g.mul(g.unsqueeze(posf, [1]), inv_freq)          # (P, dh/2)
    cs, sn = g.op("Cos", [ang], "cos"), g.op("Sin", [ang], "sin")
    cos = g.concat([cs, cs], -1)
    sin = g.concat([g.op("Neg", [sn], "negsn"), sn], -1)
    return cos, sin


def rope_apply(g, qk, cos, sin, dh, decoder=False):
    h2 = dh // 2
    x1 = g.slice(qk, [0], [h2], [-1])
    x2 = g.slice(qk, [h2], [dh], [-1])
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
def build_encoder(W):
    g = Builder()
    c = W["cfg"]
    L, d, H, ff, hw = c["L"], c["d"], c["H"], c["ff"], c["hw"]
    Dh = d // H
    scale = g.const(np.float32(Dh ** -0.5), "scale")
    inv_freq = g.const(W["inv_freq"], "inv_freq")
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

    cos, sin = rope_angles(g, g.op("Range", [g.const(np.int64(0), "z"), p0, g.const(np.int64(1), "o")], "pos"),
                           inv_freq, Dh)
    cos = g.unsqueeze(cos, [1, 2])                          # (p0, 1, 1, Dh)
    sin = g.unsqueeze(sin, [1, 2])
    mask = causal_mask(g, p0)

    ks, vs = [], []
    for i in range(L):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = W["layers"][i]
        wqkv = g.const(wqkv, f"wqkv{i}"); wo = g.const(wo, f"wo{i}")
        w1 = g.const(w1, f"w1{i}"); w2 = g.const(w2, f"w2{i}")
        xn = rms_norm(g, x, EPS)
        qkv = g.matmul(xn, wqkv)                            # (p0, 3*H*Dh)
        qkv = g.reshape(qkv, [-1, 3, H, Dh])
        q = g.slice(qkv, [0], [1], [1]); k = g.slice(qkv, [1], [2], [1]); v = g.slice(qkv, [2], [3], [1])
        q = g.squeeze(q, [1]); k = g.squeeze(k, [1]); v = g.squeeze(v, [1])   # (p0, H, Dh)
        qk = g.concat([g.unsqueeze(q, [1]), g.unsqueeze(k, [1])], 1)          # (p0, 2, H, Dh)
        qk = rope_apply(g, qk, cos, sin, Dh)
        q = g.squeeze(g.slice(qk, [0], [1], [1]), [1])
        k = g.squeeze(g.slice(qk, [1], [2], [1]), [1])
        q = g.op("Transpose", [q], "qt", perm=[1, 0, 2])    # (H, p0, Dh)
        k = g.op("Transpose", [k], "kt", perm=[1, 0, 2])
        v = g.op("Transpose", [v], "vt", perm=[1, 0, 2])
        ks.append(g.unsqueeze(k, [0])); vs.append(g.unsqueeze(v, [0]))
        scores = g.mul(g.matmul(q, g.op("Transpose", [k], "kT", perm=[0, 2, 1])), scale)
        scores = g.add(scores, mask)
        attn = g.op("Softmax", [scores], "sm", axis=-1)
        a = g.matmul(attn, v)                               # (H, p0, Dh)
        a = g.reshape(g.op("Transpose", [a], "at", perm=[1, 0, 2]), [-1, d])
        x = g.add(x, g.matmul(a, wo))
        xn = rms_norm(g, x, EPS)
        gu = g.matmul(xn, w1)                               # (p0, 2*ff)
        gate = g.slice(gu, [0], [ff], [-1]); up = g.slice(gu, [ff], [2 * ff], [-1])
        x = g.add(x, g.matmul(g.mul(silu(g, gate), up), w2))

    last = g.gather(x, g.const(np.int64(-1), "lastidx"))    # (d,)
    # conditioning head: ce = (w @ last) * rinv(last) + b
    ce_w, ce_b = W["ce"]
    ce = g.matmul(last, g.const(ce_w, "ce_w"))
    ce = g.add(g.mul(ce, rinv(g, last, EPS)), g.const(ce_b, "ce_b"))
    s = silu(g, g.slice(ce, [0], [hw], [0]))
    e = g.squeeze(g.slice(ce, [hw], [hw + 1], [0]), [0])

    k_cache = g.concat(ks, 0)                               # (L, H, p0, Dh)
    v_cache = g.concat(vs, 0)
    g.output(s, "s"); g.output(e, "eos"); g.output(k_cache, "k_cache"); g.output(v_cache, "v_cache")
    outs = [helper.make_tensor_value_info("s", TensorProto.FLOAT, [hw]),
            helper.make_tensor_value_info("eos", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("k_cache", TensorProto.FLOAT, [L, H, "p0", Dh]),
            helper.make_tensor_value_info("v_cache", TensorProto.FLOAT, [L, H, "p0", Dh])]
    graph = helper.make_graph(g.nodes, "woof_encoder", [tokens_vi, voice_vi], outs, g.inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


# ------------------------------------------------------------------------------------------------- step graph
def build_step(W):
    g = Builder()
    c = W["cfg"]
    L, d, H, ff, hw, depth = c["L"], c["d"], c["H"], c["ff"], c["hw"], c["depth"]
    Dh = d // H
    scale = g.const(np.float32(Dh ** -0.5), "scale")
    inv_freq = g.const(W["inv_freq"], "inv_freq")

    s_vi = helper.make_tensor_value_info("s_in", TensorProto.FLOAT, [hw])
    noise_vi = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [c["latent"]])
    pos_vi = helper.make_tensor_value_info("pos", TensorProto.INT64, [])
    k_vi = helper.make_tensor_value_info("k_in", TensorProto.FLOAT, [L, H, "T", Dh])
    v_vi = helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [L, H, "T", Dh])

    # ---- one-step flow head: z = noise + head(s, noise) ----
    ada_w, ada_b = W["ada"]
    A = g.linear("s_in", g.const(ada_w, "ada_w"), g.const(ada_b, "ada_b"))   # (4480,)
    hi_w, hi_b = W["h_inp"]
    h = g.linear("noise", g.const(hi_w, "h_inp_w"), g.const(hi_b, "h_inp_b"))
    for j, ((w1, b1), (w2, b2)) in enumerate(W["hblocks"]):
        o = 3 * hw * j
        shift = g.slice(A, [o], [o + hw], [0])
        scl = g.slice(A, [o + hw], [o + 2 * hw], [0])
        gate = g.slice(A, [o + 2 * hw], [o + 3 * hw], [0])
        y = g.add(shift, g.mul(layer_norm_noaffine(g, h, 1e-5), scl))
        inner = g.linear(silu(g, g.linear(y, g.const(w1, f"hb_w1_{j}"), g.const(b1, f"hb_b1_{j}"))),
                         g.const(w2, f"hb_w2_{j}"), g.const(b2, f"hb_b2_{j}"))
        h = g.add(h, g.mul(gate, inner))
    o = 3 * hw * depth
    shift = g.slice(A, [o], [o + hw], [0]); scl = g.slice(A, [o + hw], [o + 2 * hw], [0])
    ho_w, ho_b = W["h_out"]
    head = g.linear(g.add(shift, g.mul(layer_norm_noaffine(g, h, 1e-5), scl)),
                    g.const(ho_w, "h_out_w"), g.const(ho_b, "h_out_b"))
    z = g.add("noise", head)                                # (latent,)

    # ---- feedback through the causal transformer with KV cache ----
    li_w, li_b = W["lat_in"]
    x = g.linear(z, g.const(li_w, "lat_in_w"), g.const(li_b, "lat_in_b"))   # (d,)
    posv = g.unsqueeze("pos", [0])                          # (1,)
    cos, sin = rope_angles(g, posv, inv_freq, Dh)           # (1, Dh)
    cos = g.squeeze(cos, [0]); sin = g.squeeze(sin, [0])    # (Dh,)

    k_outs, v_outs = [], []
    for i in range(L):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = W["layers"][i]
        wqkv = g.const(wqkv, f"wqkv{i}"); wo = g.const(wo, f"wo{i}")
        w1 = g.const(w1, f"w1{i}"); w2 = g.const(w2, f"w2{i}")
        Ki = g.gather("k_in", g.const(np.int64(i), f"ki{i}"))   # (H, T, Dh)
        Vi = g.gather("v_in", g.const(np.int64(i), f"vi{i}"))
        qkv = g.mul(g.matmul(x, wqkv), rinv(g, x, EPS))     # (3*H*Dh,)
        qkv = g.reshape(qkv, [3, H, Dh])
        q = g.squeeze(g.slice(qkv, [0], [1], [0]), [0])       # (H, Dh)
        k = g.squeeze(g.slice(qkv, [1], [2], [0]), [0])
        v = g.squeeze(g.slice(qkv, [2], [3], [0]), [0])
        qk = g.concat([g.unsqueeze(q, [0]), g.unsqueeze(k, [0])], 0)   # (2, H, Dh)
        qk = rope_apply(g, qk, cos, sin, Dh)
        q = g.squeeze(g.slice(qk, [0], [1], [0]), [0])
        k = g.squeeze(g.slice(qk, [1], [2], [0]), [0])
        K = g.concat([Ki, g.unsqueeze(k, [1])], 1)            # (H, T+1, Dh)
        V = g.concat([Vi, g.unsqueeze(v, [1])], 1)
        k_outs.append(g.unsqueeze(K, [0])); v_outs.append(g.unsqueeze(V, [0]))
        scores = g.squeeze(g.matmul(K, g.unsqueeze(q, [-1])), [-1])   # (H, T+1)
        scores = g.mul(scores, scale)
        attn = g.op("Softmax", [scores], "sm", axis=-1)
        a = g.squeeze(g.matmul(g.unsqueeze(attn, [1]), V), [1])       # (H, Dh)
        x = g.add(x, g.matmul(g.reshape(a, [d]), wo))
        gu = g.mul(g.matmul(x, w1), rinv(g, x, EPS))
        gate = g.slice(gu, [0], [ff], [0]); up = g.slice(gu, [ff], [2 * ff], [0])
        x = g.add(x, g.matmul(g.mul(silu(g, gate), up), w2))

    ce_w, ce_b = W["ce"]
    ce = g.matmul(x, g.const(ce_w, "ce_w"))
    ce = g.add(g.mul(ce, rinv(g, x, EPS)), g.const(ce_b, "ce_b"))
    s_new = silu(g, g.slice(ce, [0], [hw], [0]))
    e_new = g.squeeze(g.slice(ce, [hw], [hw + 1], [0]), [0])
    k_cache = g.concat(k_outs, 0)                           # (L, H, T+1, Dh)
    v_cache = g.concat(v_outs, 0)
    g.output(z, "z"); g.output(s_new, "s"); g.output(e_new, "eos")
    g.output(k_cache, "k_cache"); g.output(v_cache, "v_cache")

    outs = [helper.make_tensor_value_info("z", TensorProto.FLOAT, [c["latent"]]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [hw]),
            helper.make_tensor_value_info("eos", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("k_cache", TensorProto.FLOAT, [L, H, "T1", Dh]),
            helper.make_tensor_value_info("v_cache", TensorProto.FLOAT, [L, H, "T1", Dh])]
    graph = helper.make_graph(g.nodes, "woof_step", [s_vi, noise_vi, pos_vi, k_vi, v_vi], outs, g.inits)
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


# ------------------------------------------------------------------------------------------------- save / verify
def fp16_weights(model, min_elems=256):
    """Store large float32 initializers as float16 with a Cast-to-fp32 in front, halving weight bytes
    while all compute stays float32. Small tensors (eps, scalars) stay fp32."""
    g = model.graph
    new_inits, casts = [], []
    for init in g.initializer:
        n = int(np.prod(init.dims)) if init.dims else 1
        if init.data_type != TensorProto.FLOAT or n < min_elems:
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


def save_model(model, path, fp16=False):
    if fp16:
        fp16_weights(model)
    model.ir_version = 10  # broad onnxruntime compatibility (opset 17 needs IR >= 8)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def export(weights, outdir, check=True, fp16=False):
    os.makedirs(outdir, exist_ok=True)
    st = load_state(weights)
    W = speech_weights(st)
    write_tokens(os.path.join(outdir, "tokens.txt"))
    print(f"  wrote {os.path.join(outdir, 'tokens.txt')}  ({len(CHARS) + 3} tokens)")
    save_model(build_encoder(W), os.path.join(outdir, "woof_encoder.onnx"), fp16)
    save_model(build_step(W), os.path.join(outdir, "woof_step.onnx"), fp16)
    save_model(build_decoder(st), os.path.join(outdir, "woof_decoder.onnx"), fp16)
    if check:
        verify(weights, outdir, tol=3e-2 if fp16 else 1e-4)


# ------------------------------------------------------------------------------------------------- onnx runtime driver
class WoofONNX:
    """Drives the three exported graphs. Any standard-normal noise source works; pass rng=TorchRandn(seed)
    to reproduce the numpy runtime bit-for-bit."""

    def __init__(self, outdir, voice="juno"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.enc = ort.InferenceSession(os.path.join(outdir, "woof_encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.step = ort.InferenceSession(os.path.join(outdir, "woof_step.onnx"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(outdir, "woof_decoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.voice_id = VOICES.index(voice)
        self.latent = 32
        self.max_len = 1100

    def __call__(self, text, seed=0, temp=0.5, eos=0.9, max_frames=750, rng=None):
        rng = rng if rng is not None else TorchRandn(seed)
        ids = np.array(encode_text(text), np.int64)
        s, e, kc, vc = self.enc.run(None, {"tokens": ids, "voice_id": np.array(self.voice_id, np.int64)})
        p0 = 16 + len(ids) + 1
        max_frames = min(max_frames, self.max_len - p0)
        eos_t = np.log(eos / (1 - eos))
        lat = []
        for f in range(max_frames):
            if f > 3 and float(e) > eos_t:
                break
            noise = rng.randn(self.latent).astype(np.float32) * np.float32(temp)
            z, s, e, kc, vc = self.step.run(None, {"s_in": s, "noise": noise, "pos": np.array(p0 + f, np.int64),
                                                   "k_in": kc, "v_in": vc})
            lat.append(z)
        lat = np.stack(lat).astype(np.float32)
        audio = self.dec.run(None, {"latents": lat})[0]
        return lat, audio.astype(np.float32)


def _assert_close(name, a, b, tol=1e-4):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    err = float(np.abs(a - b).max())
    print(f"  {name:18s} max|d|={err:.3e}")
    assert err < tol, f"{name}: mismatch {err} >= {tol}"


def verify(weights, outdir, text="Please call me back at 415-555-0199 before 5:30 p.m. on Friday.", tol=1e-4):
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
    s_o, e_o, k_o, v_o = enc.run(None, {"tokens": ids.astype(np.int64), "voice_id": np.array(0, np.int64)})
    _assert_close("encoder s", s_ref, s_o, tol)
    _assert_close("encoder eos", e_ref, e_o, tol)
    _assert_close("encoder k_cache", k_ref, k_o, tol)
    _assert_close("encoder v_cache", v_ref, v_o, tol)

    # one autoregressive step fed the reference state
    noise = TorchRandn(0).randn(sp.latent).astype(np.float32) * np.float32(0.5)
    z_ref = noise + sp._head(s_ref, noise)
    xx = sp.lat_in[0] @ z_ref + sp.lat_in[1]
    for i in range(sp.L):
        xx = sp._block1(i, xx, p0)
    sn_ref, en_ref = sp._cond(xx)
    kn_ref = sp.kv[:, 0, :, :p0 + 1, :].copy(); vn_ref = sp.kv[:, 1, :, :p0 + 1, :].copy()
    z_o, sn_o, en_o, kn_o, vn_o = step.run(None, {
        "s_in": s_ref.astype(np.float32), "noise": noise, "pos": np.array(p0, np.int64),
        "k_in": k_ref.astype(np.float32), "v_in": v_ref.astype(np.float32)})
    _assert_close("step z", z_ref, z_o, tol)
    _assert_close("step s", sn_ref, sn_o, tol)
    _assert_close("step eos", en_ref, en_o, tol)
    _assert_close("step k_cache", kn_ref, kn_o, tol)
    _assert_close("step v_cache", vn_ref, vn_o, tol)

    # full free-running ONNX pipeline sanity
    onx = WoofONNX(outdir, voice="juno")
    lat, wav = onx(text, seed=0)
    assert len(lat) > 0 and np.isfinite(wav).all() and len(wav) == len(lat) * 960
    print(f"  full pipeline: {len(lat)} frames -> {len(wav) / SR:.2f}s of finite audio")
    print("  OK: encoder, step and decoder all match the numpy runtime.")


# ------------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "woof_int8.npz"))
    ap.add_argument("--out", default="onnx", help="output directory for tokens.txt and the .onnx graphs")
    ap.add_argument("--no-check", action="store_true", help="skip the numpy-vs-onnx verification")
    ap.add_argument("--fp16", action="store_true", help="store large weights as float16 (halves size; compute stays fp32)")
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

    print(f"Exporting {args.weights} -> {args.out}/" + (" (fp16 weights)" if args.fp16 else ""))
    export(args.weights, args.out, check=not args.no_check, fp16=args.fp16)


if __name__ == "__main__":
    main()