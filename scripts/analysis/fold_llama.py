"""fold for a Llama HF model (3 Nov 2026, test 2b of the 110M late-spawn study: rotate the residual stream before the conversion):
ttf-conversion's ttfconv/fold.py for the Llama layout (plain RMSNorm gains g, not Qwen's zero-centred 1 + w). Exact in
fp: the gains folded into the linears that read them and set to 1, then one shared rotation Q of the residual stream:
readers W diag(g) Q^T (q / k / v with input_layernorm's g, gate / up with post_attention_layernorm's), writers Q W
(o_proj, down_proj), the embedding E Q^T, the head H diag(g_final) Q^T (untied from the embedding: the final gain and
the rotation cannot both live in a tied matrix). The model then has an untied head (+V d floats, both stay fp in the
conversion): --q none folds and unties without rotating, the control for that.
  python scripts/analysis/fold_llama.py SRC_HF OUT_HF [--q had|orth|none] [--seed 2110]
    had: randomized Hadamard (d not a power of 2: rand_orth(d / 2^k) kron Hadamard(2^k); 110M: 768 = 3 x 256)
The check: logits of the folded model vs the source, fp32 (ttf.hf_load).
"""
import argparse, json, math, os, shutil, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--q", choices=["had", "orth", "none"], default="had"); ap.add_argument("--seed", type=int, default=2110)
ap.add_argument("--check-tokens", type=int, default=256)
a = ap.parse_args()


def hadamard(n, gen):        # (ttfconv/fold.py's, in float64: the fp check is exact to ~1e-10)
    assert n & (n - 1) == 0, n
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    s = torch.randint(0, 2, (n,), generator=gen).double() * 2 - 1
    return H * s[None, :] / math.sqrt(n)


def rand_orth(n, gen):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))[None, :]


def residual_q(d, gen, qtype):
    if qtype == "none":
        return torch.eye(d, dtype=torch.float64)
    p = d & -d
    if p == d or qtype != "had":
        return hadamard(d, gen) if qtype == "had" else rand_orth(d, gen)
    return torch.kron(rand_orth(d // p, gen).contiguous(), hadamard(p, gen))


import ttf.hf_load as h
cfg = json.load(open(os.path.join(a.src, "config.json")))
sd = h.load_hf_tensors(a.src)
d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
Q = residual_q(d, torch.Generator().manual_seed(a.seed), a.q).double()
assert torch.allclose(Q @ Q.T, torch.eye(d, dtype=torch.float64), atol=1e-10)
out = {}
E = sd["model.embed_tokens.weight"].double()
H = sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).double()
gf = sd["model.norm.weight"].double()
out["model.embed_tokens.weight"] = E @ Q.T
out["lm_head.weight"] = (H * gf[None, :]) @ Q.T
out["model.norm.weight"] = torch.ones(d)
for i in range(L):
    p = f"model.layers.{i}."
    gi, gp = sd[p + "input_layernorm.weight"].double(), sd[p + "post_attention_layernorm.weight"].double()
    for nm in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        out[p + nm + ".weight"] = (sd[p + nm + ".weight"].double() * gi[None, :]) @ Q.T
    for nm in ("mlp.gate_proj", "mlp.up_proj"):
        out[p + nm + ".weight"] = (sd[p + nm + ".weight"].double() * gp[None, :]) @ Q.T
    for nm in ("self_attn.o_proj", "mlp.down_proj"):
        out[p + nm + ".weight"] = Q @ sd[p + nm + ".weight"].double()
    out[p + "input_layernorm.weight"] = torch.ones(d); out[p + "post_attention_layernorm.weight"] = torch.ones(d)
left = sorted(set(sd) - set(out) - {"lm_head.weight"})
assert not [k for k in left if k.endswith(".weight") and "rotary" not in k], left     # every weight handled
for k in left:
    out[k] = sd[k]
os.makedirs(a.out, exist_ok=True)
from safetensors.torch import save_file
save_file({k: v.float().contiguous() for k, v in out.items()}, os.path.join(a.out, "model.safetensors"))
for f in os.listdir(a.src):
    if not f.endswith((".safetensors", ".bin")) and os.path.isfile(os.path.join(a.src, f)):
        shutil.copy(os.path.join(a.src, f), a.out)
cfg["tie_word_embeddings"] = False
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
json.dump({"q": a.q, "seed": a.seed, "src": a.src}, open(os.path.join(a.out, "fold.json"), "w"))

# check: the folded model's logits against the source's (fp32, CPU, the trainer's model)
import torch.nn as nn
from ttf.model import BitTransformer


def build(path):
    c = h.model_config_from_hf(path)
    m = BitTransformer(c, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False))
    h.load_hf_fp(m, path, log=lambda *_: None)
    return m.eval()


x = torch.randint(0, cfg["vocab_size"], (1, a.check_tokens), generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    l0, _ = build(a.src)(x); l1, _ = build(a.out)(x)
rel = float((l1 - l0).norm() / l0.norm()); top1 = float((l1.argmax(-1) == l0.argmax(-1)).float().mean())
print(f"{a.out}: fold a ({a.q}, seed {a.seed}), d {d}, {L} layers; logits vs source rel {rel:.2e}, top-1 agree {top1:.4f}")
assert rel < 1e-4, rel
