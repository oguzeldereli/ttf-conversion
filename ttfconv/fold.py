"""fold a (QuaRot / SpinQuant style, exact, free at decode): the zero-centred RMSNorm gains (1 + w) folded into the
linears that read them and set to 1, then one shared randomized Hadamard Q on the residual stream: readers W Q^T
(q / k / v, in_proj_qkv / z and the float in_proj_a / b, gate / up, the head), writers Q W (o_proj, out_proj, down_proj),
the embedding E Q^T; the head untied from the embedding for it. The only rotation kept: per-linear rotations need an
online op at decode (+15% to +300% of the decode time in our kernels).
  --ext ttfconv.fold[:q=had,seed=2110]     q: had (randomized Hadamard; d not a power of 2: rand_orth(d / 2^k) kron
                                           Hadamard(2^k)) or orth (random orthogonal)
"""
import math
import torch
import torch.nn as nn
from .spec import parse

READ_IN = ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")
READ_POST = ("gate_proj", "up_proj")
WRITE = ("o_proj", "out_proj", "down_proj")


def hadamard(n, gen):
    """randomized Hadamard H diag(s) / sqrt(n) (Sylvester, n a power of 2)"""
    assert n & (n - 1) == 0, n
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    s = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return H * s[None, :] / math.sqrt(n)


def rand_orth(n, gen):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diagonal(r))[None, :]


def residual_q(d, gen, qtype="had"):
    """the shared residual rotation: randomized Hadamard for d = 2^k; else rand_orth(d / 2^k) kron Hadamard(2^k)
    (27B: 5120 = 5 x 1024)"""
    p = d & -d
    if p == d or qtype != "had":
        return hadamard(d, gen) if qtype == "had" else rand_orth(d, gen)
    return torch.kron(rand_orth(d // p, gen).contiguous(), hadamard(p, gen))


@torch.no_grad()
def fold_float(hf, Qr):
    """the norm gains into a per-layer dict (set to 1), the float residual readers, the embedding and the head rotated"""
    G = {}
    for i, layer in enumerate(hf.model.layers):
        gi = 1 + layer.input_layernorm.weight.float(); gp = 1 + layer.post_attention_layernorm.weight.float()
        G[i] = (gi, gp)
        layer.input_layernorm.weight.zero_(); layer.post_attention_layernorm.weight.zero_()
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            for nm in ("in_proj_a", "in_proj_b"):
                w = getattr(la, nm).weight
                w.copy_(((w.float() * gi[None, :]) @ Qr.T).to(w.dtype))
    gf = 1 + hf.model.norm.weight.float(); hf.model.norm.weight.zero_()
    E, Hd = hf.model.embed_tokens.weight, hf.lm_head.weight
    assert E is not Hd
    for a in range(0, E.shape[0], 32768):
        E[a:a + 32768] = (E[a:a + 32768].float() @ Qr.T).to(E.dtype)
        Hd[a:a + 32768] = ((Hd[a:a + 32768].float() * gf[None, :]) @ Qr.T).to(Hd.dtype)
    return G


def fold_w(leaf, W, gi, gp, Qr):
    if leaf in READ_IN:
        return (W * gi[None, :]) @ Qr.T
    if leaf in READ_POST:
        return (W * gp[None, :]) @ Qr.T
    assert leaf in WRITE, leaf
    return Qr @ W


class Fold:
    def __init__(self, spec):
        o = parse(spec, {"q": "had", "seed": "2110"})
        self.qtype, self.seed = o["q"], int(o["seed"])
        assert self.qtype in ("had", "orth"), self.qtype
        self.Qr = self.G = None

    def build(self, model, swapped, make_linear, log):
        """the head gets its own copy (rotated with the final norm); exists from the build, so a resume loads it"""
        hf = model.hf
        if hf.lm_head.weight is hf.model.embed_tokens.weight:
            hf.lm_head.weight = nn.Parameter(hf.lm_head.weight.detach().clone(), requires_grad=False)
        return swapped

    def begin(self, model, files, dev, log):
        hf = model.hf
        d = hf.model.embed_tokens.weight.shape[1]
        self.Qr = residual_q(d, torch.Generator().manual_seed(self.seed), self.qtype).to(dev)
        self.G = fold_float(hf, self.Qr)
        log(f"ttfconv.fold: norm gains folded, residual stream rotated by a shared {self.qtype} {d} x {d} (embedding, "
            f"head untied)")

    def weight(self, name, W, li):
        return fold_w(name.rsplit(".", 1)[1], W, *self.G[li], self.Qr)


def ttf_ext(spec):
    return Fold(spec)
