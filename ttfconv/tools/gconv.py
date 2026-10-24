"""G of a CONVERTED network (T7, P7b: the two-pass conversion): for every adapted linear of a ttfconv state, the
output-gradient second moment G = E[d d^T] at the converted point (labels sampled from the converted model: its Fisher,
the Gauss-Newton curvature of KL(fp || converted) near the fp function), saved like kfac.py --dump-g, so that a second
conversion pass can fit its adapters with --ext ttfconv.adapters:...,gmetric=DIR.
  python -m ttfconv.tools.gconv --model M --state S/state.pt --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1 \
      --data data/broadqw_train.bin --dump-g DIR
"""
import argparse, os, time
import torch
import torch.nn.functional as F
from ttfconv.tools.kfac import windows, sample_labels
from ttfconv.tools.swap import build, sdict, prepare, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--ext", action="append", default=[]); ap.add_argument("--data", required=True)
    ap.add_argument("--ncal", type=int, default=16); ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dump-g", required=True); ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args()
    from ttfconv.adapters import QTad
    sd = sdict(a.state); m = build(a.model, a.ext); prepare(m, sd); load(m, sd)
    m.train()                                  # (the layers' backward needs the training path; no update is run)
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():                    # no TTF update during these backwards
        if hasattr(mod, "capture"):
            mod.capture = False
    names = {n[3:] if n.startswith("hf.") else n: mod for n, mod in m.named_modules() if isinstance(mod, QTad)}
    os.makedirs(a.dump_g, exist_ok=True)
    emb = m.hf.model.embed_tokens
    eh = emb.register_forward_hook(lambda mm, i, o: o.requires_grad_())
    wins = windows(a.data, a.ncal, a.seq, 4321); t0 = time.time(); keys = list(names); ntok = 0
    for c0 in range(0, len(keys), a.chunk):            # G accumulators for a.chunk linears per pass (all 150: ~5 GB)
        part = keys[c0:c0 + a.chunk]
        G = {n: torch.zeros(names[n].out_features, names[n].out_features, device="cuda") for n in part}
        hooks = []
        for n in part:
            def fh(module, inp, out, n=n):
                if out.requires_grad:
                    def gh(g, n=n):
                        g2 = g.reshape(-1, g.shape[-1]).float(); G[n].addmm_(g2.T, g2)
                    out.register_hook(gh)
            hooks.append(names[n].register_forward_hook(fh))
        gen = torch.Generator(device="cuda").manual_seed(99); ntok = 0
        for w in wins:
            t = w.cuda()[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(t)[0][0]
            y = sample_labels(lg.detach(), gen)
            F.cross_entropy(lg.float(), y, reduction="sum").backward()
            ntok += t.shape[1]
            del lg, y
        for h in hooks:
            h.remove()
        for n, g in G.items():
            torch.save((g / ntok).float().cpu(), os.path.join(a.dump_g, n + ".pt"))
        del G
        torch.cuda.empty_cache()
    eh.remove()
    open(os.path.join(a.dump_g, "done"), "w").close()
    G = names; cnt = {"n": ntok}
    print(f"gconv: {len(G)} G matrices from {cnt['n']} tokens in {time.time() - t0:.0f} s -> {a.dump_g}", flush=True)


if __name__ == "__main__":
    main()
