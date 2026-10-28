"""Block-local reorganisation pre-pass (T8): each decoder block of a converted network is trained ALONE, the
others frozen, on a local quadratic model of the global distillation loss at its output:
    L_l(theta_l) = mean_t [ g_t . dh_t + 0.5 dh_t^T G dh_t ],   dh = block_l(x_t; theta_l) - block_l(x_t; theta_l^0)
with x_t the converted network's own stream into block l (pass 1, fixed), g_t = d KL(fp || converted) / d h_t at the
block's output (per token, from one backward of the distillation loss), and G = E[d d^T] at the same point under labels
sampled from the converted model (Gauss-Newton). The block's trits move by TTF (the trainer's ts_layer, fresh
accumulators), its floats (norm gains, linear-attention floats, biases, row / column scales) by AdamW. The swap test
(Q3) found the recovery's block updates additive, so the blocks are trained independently from the same pass-1 point
and assembled. Cost per local step: one block on one window, against the whole network on a 16-window batch.
  python -m ttfconv.tools.blocklocal --model M --state S/state.pt --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1 \
      --data data/broadqw_train.bin --steps 100 --out DIR
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from ttfconv.tools.kfac import sample_labels
from ttfconv.tools.swap import build, sdict, prepare, load


def windows_np(path, n, seq, seed):
    d = np.memmap(path, dtype=np.uint32, mode="r")
    st = np.random.default_rng(seed).integers(0, len(d) - seq - 1, n)
    return [np.asarray(d[s:s + seq + 1]) for s in st]


@torch.no_grad()
def score(m, path, n, dev, teacher=None):
    d = np.memmap(path, dtype=np.uint32, mode="r"); ce = kl = nt = 0.0
    for i in range(n):
        t = torch.from_numpy(np.asarray(d[i * 2048:i * 2048 + 2049], dtype=np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(t[None, :-1])[0][0].float()
            tl = teacher(t[None, :-1]).logits[0].float() if teacher is not None else None
        ce += float(F.cross_entropy(lg, t[1:], reduction="sum")); nt += t.numel() - 1
        if tl is not None:
            for c in range(0, lg.shape[0], 256):              # (chunked: 2048 x 248k fp32 copies do not fit)
                lp, lq = F.log_softmax(tl[c:c + 256], -1), F.log_softmax(lg[c:c + 256], -1)
                kl += float((lp.exp() * (lp - lq)).sum())
        del lg, tl
    return ce / nt, (kl / nt if teacher is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--ext", action="append", default=[]); ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="data/wikiqw_val.bin"); ap.add_argument("--nval", type=int, default=8)
    ap.add_argument("--ncal", type=int, default=16); ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--theta", type=float, default=16.0); ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--blocks", default=""); ap.add_argument("--no-trits", action="store_true")
    ap.add_argument("--no-floats", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--damp", type=float, default=1.0, help="Levenberg damping of G: G + damp * mean(diag G) I")
    ap.add_argument("--warmup", type=int, default=50, help="local steps over which the TTF scale and the float lr ramp up")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"; t0 = time.time()
    from transformers import AutoModelForCausalLM
    from ttf.args import parse
    from ttf.layers import KernelTernaryLinear
    from ttf.update import ts_layer
    from ttfconv.stream import layer_inputs
    # the trainer's TTF settings (convert-qwen recipe, theta / rank as given): ts_cfg as ttf/train.py builds it
    import sys
    argv0 = sys.argv; sys.argv = ["x", "--recipe", os.path.join(os.path.dirname(__file__), "..", "..", "recipes", "qwen-0.8b.toml"), "--ttf-theta", str(a.theta), "--ttf-rank", str(a.rank)]
    targs = parse(); sys.argv = argv0
    cfg = dict(r=targs.ttf_rank, lr_ratio=1.0, theta=targs.ttf_theta, tau=targs.ttf_tau, b2=targs.ttf_b2,
               spend=targs.ttf_spend, store_a="", offload=False, theta_sched=None, progress=0.5, factored=targs.factored_k, step=10 ** 6,
               start=0, vdiag=False, micro=1)
    sd = sdict(a.state); m = build(a.model, a.ext); prepare(m, sd); load(m, sd)
    for p in m.parameters():
        p.requires_grad_(False)
    teacher = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    base = score(m, a.val, a.nval, dev, teacher)
    print(f"pass 1: wiki CE {base[0]:.4f}, KL to fp {base[1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    wins = windows_np(a.data, a.ncal, a.seq, 4321)
    layers = m.hf.model.layers; L = len(layers)
    H0, KW = layer_inputs(m, [w[:-1] for w in wins], 1, dev)
    X = [[] for _ in range(L)]; Y = [[] for _ in range(L)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for h in H0:
            for l in range(L):
                X[l].append(h.cpu())
                o = layers[l](h, **KW[l]); h = o[0] if isinstance(o, tuple) else o
                Y[l].append(h.cpu())
    # per-token KD gradient g at every block output, and the Gauss-Newton G (sampled labels)
    gsave = [[] for _ in range(L)]; G = [torch.zeros(X[0][0].shape[-1], X[0][0].shape[-1], device=dev) for _ in range(L)]
    mode = {"k": "g"}; cnt = {"n": 0}

    def mkhook(l):
        def fh(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if o.requires_grad:
                def gh(gr):
                    g2 = gr.reshape(-1, gr.shape[-1]).float()
                    if mode["k"] == "g":
                        gsave[l].append(g2.to(torch.bfloat16).cpu())
                    else:
                        G[l].addmm_(g2.T, g2)
                o.register_hook(gh)
        return fh
    hooks = [layers[l].register_forward_hook(mkhook(l)) for l in range(L)]
    eh = m.hf.model.embed_tokens.register_forward_hook(lambda mm, i, o: o.requires_grad_())
    gen = torch.Generator(device=dev).manual_seed(7)
    for w in wins:
        t = torch.from_numpy(w[:-1].astype(np.int64)).to(dev)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tl = F.log_softmax(teacher(t).logits[0].float(), -1)
        for k in ("g", "G"):
            mode["k"] = k
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(t)[0][0].float()
            if k == "g":
                loss = sum((tl[c:c + 256].exp() * (tl[c:c + 256] - F.log_softmax(lg[c:c + 256], -1))).sum()
                           for c in range(0, lg.shape[0], 256))
            else:
                loss = F.cross_entropy(lg, sample_labels(lg.detach(), gen), reduction="sum")
            loss.backward(); del lg, loss
        cnt["n"] += t.shape[1]
    for h in hooks:
        h.remove()
    eh.remove()
    for l in range(L):
        G[l] /= cnt["n"]                     # (gsave[l]: one entry per window, in window order)
    print(f"stream, g and G for {L} blocks from {cnt['n']} tokens ({time.time() - t0:.0f} s)", flush=True)
    # local training, block by block, from the same pass-1 point
    blocks = range(L) if not a.blocks else [int(b) for b in a.blocks.split(",")]
    rec = []
    for l in blocks:
        layer = layers[l]; state = {}
        terns = [mod for mod in layer.modules() if isinstance(mod, KernelTernaryLinear)]
        for i_, mod in enumerate(terns):
            mod._ts_idx = 100000 + 100 * l + i_
            mod.on_grad = (lambda lay, g, _s=state: ts_layer(lay, g, _s, cfg))
            mod.capture = not a.no_trits
        fl = [p for n_, p in layer.named_parameters() if p.is_floating_point() and not n_.endswith("frozen_scale")]
        for p in fl:
            p.requires_grad_(not a.no_floats)
        opt = torch.optim.AdamW([p for p in fl if p.requires_grad], lr=a.lr, weight_decay=0.0) if not a.no_floats else None
        layer.train(); losses = []
        Gl = G[l] + a.damp * torch.diagonal(G[l]).mean() * torch.eye(G[l].shape[0], device=dev)   # trust region
        for s in range(a.steps):
            w_ = min(1.0, (s + 1) / max(1, a.warmup))         # warm-up, as the trainer's (TTF scale = lr / peak)
            cfg["lr_ratio"] = w_
            if opt is not None:
                for g_ in opt.param_groups:
                    g_["lr"] = a.lr * w_
            j = s % len(X[l])
            x = X[l][j].to(dev); y0 = Y[l][j].to(dev).float(); g = gsave[l][j].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = layer(x, **KW[l]); h = (o[0] if isinstance(o, tuple) else o).float()
            dh = (h - y0).reshape(-1, h.shape[-1])
            lo = ((g * dh).sum() + 0.5 * ((dh @ Gl) * dh).sum()) / dh.shape[0]
            lo.backward()
            if opt is not None:
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(lo))
        for mod in terns:
            mod.capture = False
        for p in fl:
            p.requires_grad_(False)
        layer.eval()
        v_ = state.get("_ts", {}).get("_nf", 0); nf = int(v_.item()) if torch.is_tensor(v_) else int(v_)
        rec.append({"block": l, "local_first": float(np.mean(losses[:5])), "local_last": float(np.mean(losses[-5:])),
                    "fires": nf})
        print(f"block {l}: local loss {rec[-1]['local_first']:+.4f} -> {rec[-1]['local_last']:+.4f}, fires {nf} "
              f"({time.time() - t0:.0f} s)", flush=True)
        del state
        torch.cuda.empty_cache()
    for mod in m.modules():
        if hasattr(mod, "_invalidate"):
            mod._invalidate()
    after = score(m, a.val, a.nval, dev, teacher)
    print(f"after the block-local pass: wiki CE {after[0]:.4f} (pass 1 {base[0]:.4f}), KL to fp {after[1]:.4f} "
          f"(pass 1 {base[1]:.4f})", flush=True)
    json.dump({"args": vars(a), "pass1": base, "after": after, "blocks": rec}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    from ttfconv.finish import unique_sd
    torch.save({"model": {k: v.cpu() for k, v in unique_sd(m).items() if not v.is_meta}}, os.path.join(a.out, "state.pt"))
    print("BLOCKLOCAL_DONE", flush=True)


if __name__ == "__main__":
    main()
