"""Prepare an fp model for the ternary conversion (tests 3a / 3b of the 110M late-spawn study,
3 Nov 2026): continue the fp model's own training (its CE on its data, true fp forward and gradients)
while moving its weights toward their ternary rounding Q = s clamp(round(W / s), -1, 1), s the per-row absmean of W, so
that the conversion throws less away; stop before the fp loss has risen more than --rise (fixed windows), export HF.
  --mode prox    3a, rounding-aware fine-tune: after every optimizer step W <- W - lr lam(t) (W - Q) (a proximal pull,
                 lam ramped 0 -> --lam1 over --ramp steps)
  --mode qnoise  3b, QuantNoise (Fan et al. 2020): every forward replaces a random fraction p(t) of each linear's weights
                 by their rounding, straight-through (W + mask (Q - W).detach()), p ramped 0 -> --p1 over --ramp steps
Only the 7 linears per block move toward Q (embedding, head and norms train freely, as the conversion keeps them fp).
Logged every --eval-every steps: fp val, val of the RTN-rounded net (W -> Q in every linear: a cheap proxy of the
conversion's start), mean relative distance |W - Q| / |W|. The 19 Oct lattice transport (prox_fp: KD to
itself + a growing ALS-rounding pull) gave -0.005 after a 1k conversion; this one uses the run's own CE and a rise budget.
  python scripts/analysis/prep_fp.py SRC_HF OUT_HF --mode prox|qnoise [--lr 1.5e-4] [--steps 3000] [--ramp 1000]
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttf.hf_load as h
from ttf.model import BitTransformer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--mode", choices=["prox", "qnoise"], required=True)
ap.add_argument("--steps", type=int, default=3000, help="at most")
ap.add_argument("--ramp", type=int, default=1000, help="steps of the lam / p ramp")
ap.add_argument("--lam1", type=float, default=1.0); ap.add_argument("--p1", type=float, default=0.5)
ap.add_argument("--rise", type=float, default=0.01, help="stop before the fixed-window fp val rises more than this")
ap.add_argument("--lr", type=float, default=1.5e-4, help="AdamW, constant after 50 warmup steps (scratch-110m min-lr)")
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--micro", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--eval-every", type=int, default=50); ap.add_argument("--val-windows", type=int, default=16)
ap.add_argument("--train", default="data/wiki32k_train.bin"); ap.add_argument("--val", default="data/wiki32k_val.bin")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(a.seed)
LIN = [("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_up"), ("ffn", "w_down")]
t0 = time.time()
log = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)


@torch.no_grad()
def rounding(W):
    """Q = s clamp(round(W / s), -1, 1), s the per-row absmean (BitNet b1.58's rule, per output row)"""
    s = W.abs().mean(1, keepdim=True).clamp_min(1e-8)
    return s * (W / s).round().clamp(-1, 1)


class QNLinear(nn.Linear):
    """nn.Linear whose training forward uses W + mask (Q - W).detach(), mask ~ Bernoulli(p) per weight and forward"""
    p = 0.0

    def forward(self, x):
        W = self.weight
        if self.training and self.p > 0:
            with torch.no_grad():
                d = rounding(W) - W
                d.mul_(torch.rand_like(W) < self.p)
            W = W + d
        return F.linear(x, W)


def build(path):
    c = h.model_config_from_hf(path)
    mk = (lambda i, o: QNLinear(i, o, bias=False)) if a.mode == "qnoise" else (lambda i, o: nn.Linear(i, o, bias=False))
    m = BitTransformer(c, grad_checkpoint=False, make_linear=mk).to(dev)
    h.load_hf_fp(m, path, log=lambda *_: None)
    return m, c


m, cfg = build(a.src)
lins = [getattr(getattr(blk, sub), nm) for blk in m.layers for sub, nm in LIN]
tr = np.memmap(a.train, dtype=np.uint16, mode="r"); val = np.memmap(a.val, dtype=np.uint16, mode="r")
rng = np.random.default_rng(1000 + a.seed)
VAL = [torch.tensor(val[i * 2048:i * 2048 + 2049].astype(np.int64), device=dev)[None] for i in range(a.val_windows)]
amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else (lambda: torch.autocast("cpu", enabled=False))


@torch.no_grad()
def val_loss(rounded=False):
    keep = None
    if rounded:
        keep = [l.weight.detach().clone() for l in lins]
        for l in lins:
            l.weight.copy_(rounding(l.weight))
    m.eval()
    with amp():
        v = float(np.mean([float(m(x[:, :-1], x[:, 1:])[1]) for x in VAL]))
    m.train()
    if keep is not None:
        for l, w in zip(lins, keep):
            l.weight.copy_(w)
    return v


@torch.no_grad()
def rel_dist():
    return float(np.mean([float((l.weight - rounding(l.weight)).norm() / l.weight.norm()) for l in lins]))


opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
v0 = val_loss(); r0 = {"step": 0, "val": v0, "val_rtn": val_loss(True), "dist": rel_dist(), "ramp": 0.0}
log(f"{a.mode}: start " + json.dumps(r0)); hist = [r0]
best = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}; best_step = 0
for step in range(1, a.steps + 1):
    ramp = min(step / a.ramp, 1.0)
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1.0, step / 50)
    if a.mode == "qnoise":
        for l in lins:
            l.p = a.p1 * ramp
    for _ in range(a.accum):
        x = torch.stack([torch.tensor(tr[s:s + 2049].astype(np.int64), device=dev)
                         for s in rng.integers(0, len(tr) - 2050, a.micro)])
        with amp():
            _, loss = m(x[:, :-1], x[:, 1:])
        (loss / a.accum).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if a.mode == "prox":
        with torch.no_grad():
            k = opt.param_groups[0]["lr"] * a.lam1 * ramp
            for l in lins:
                l.weight.sub_(k * (l.weight - rounding(l.weight)))
    if step % a.eval_every == 0 or step == a.steps:
        v = val_loss()
        r = {"step": step, "val": v, "val_rtn": val_loss(True), "dist": rel_dist(), "ramp": ramp, "train": loss.item()}
        hist.append(r); log(json.dumps({k_: (round(v_, 5) if isinstance(v_, float) else v_) for k_, v_ in r.items()}))
        if v - v0 > a.rise:
            log(f"stop: val {v:.4f} rose {v - v0:.4f} > {a.rise}; exporting step {best_step}")
            break
        best = {k_: v_.detach().cpu().clone() for k_, v_ in m.state_dict().items()}; best_step = step
m.load_state_dict(best)


def permute(w, n_heads):     # q / k back to HF's rotate_half layout (prox_fp.py's export)
    out, k = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, k).transpose(1, 2).reshape(out, k)


sd = h.load_hf_tensors(a.src)
ms = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
sd["model.embed_tokens.weight"] = ms["tok_emb.weight"]; sd["model.norm.weight"] = ms["norm.weight"]
if "lm_head.weight" in ms:
    sd["lm_head.weight"] = ms["lm_head.weight"]
for i in range(cfg.n_layers):
    p = f"model.layers.{i}."
    sd[p + "input_layernorm.weight"] = ms[f"layers.{i}.attn_norm.weight"]
    sd[p + "post_attention_layernorm.weight"] = ms[f"layers.{i}.ffn_norm.weight"]
    for name_hf, (sub, name) in h._LIN.items():
        W = ms[f"layers.{i}.{sub}.{name}.weight"]
        if name == "wq":
            W = permute(W, cfg.n_heads)
        elif name == "wk":
            W = permute(W, cfg.n_kv_heads)
        sd[p + name_hf + ".weight"] = W
os.makedirs(a.out, exist_ok=True)
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(a.out, "model.safetensors"))
for f in os.listdir(a.src):
    if not f.endswith((".safetensors", ".bin")) and os.path.isfile(os.path.join(a.src, f)):
        shutil.copy(os.path.join(a.src, f), a.out)
json.dump({"args": vars(a), "hist": hist, "exported_step": best_step}, open(os.path.join(a.out, "prep_fp.json"), "w"), indent=1)
a.mode = "prox"     # (the reload below: plain nn.Linear)
m, _ = build(a.out)
lins = [getattr(getattr(blk, sub), nm) for blk in m.layers for sub, nm in LIN]
log(f"export {a.out} (step {best_step}): reloaded val {val_loss():.4f}, rounded {val_loss(True):.4f} "
    f"(start {v0:.4f} / {r0['val_rtn']:.4f})")
