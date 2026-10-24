"""Swap asymmetry between a conversion start S and the same run trained, T (the swap test: is reorganisation
per-layer, or layers compensating for each other?). Same structure for both (fold a + adapters). Scores on fixed
windows:
  S, T                         the two networks
  T with block l from S        how much T relies on its own block l (T <- S(l))
  S with block l from T        whether T's block l helps on its own, in S's context (S <- T(l))
  T with S's floats / S with T's floats (norm gains, linear-attention floats, biases outside the ternary layers)
Co-adaptation shows as both directions hurting, or S <- T(l) giving far less than the full S -> T gain divided over the
blocks.
  python -m ttfconv.tools.swap --model M --s S/state.pt --t ckpt.pt --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=0
      --val data/wikiqw_val.bin --nval 4 --out swap.json
"""
import argparse, json, re
import numpy as np
import torch
import torch.nn.functional as F


def build(model_path, exts):
    from ttf import ext as X
    from ttf.qwen import build_qwen
    from ttf.layers import KernelTernaryLinear
    X.load(exts)
    mk = lambda i, o: KernelTernaryLinear(i, o, 8, beta=True, int8=True, int8_dx=False)
    m, sw = build_qwen(model_path, "kernel", grad_checkpoint=False, device="cuda", make_linear=mk)
    for f in X.having("build"):
        sw = f(m, sw, mk, print)
    from ttf.layers import enable_rc_scales
    enable_rc_scales(m)                      # (the trainer's --rc-scale, on in every conversion recipe)
    return m.cuda().eval()


def sdict(p):
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    return ck.get("model", ck)


@torch.no_grad()
def prepare(m, sd):
    from ttf.layers import KernelTernaryLinear
    for n, mod in m.named_modules():
        if isinstance(mod, KernelTernaryLinear) and f"{n}.in_mean" in sd and getattr(mod, "in_mean", None) is None:
            mod.enable_centering(sd[f"{n}.in_mean"].cuda(), sd[f"{n}.out_bias"].cuda())


@torch.no_grad()
def load(m, sd, keys=None):
    sub = sd if keys is None else {k: sd[k] for k in keys}
    missing, unexpected = m.load_state_dict(sub, strict=False)
    for mod in m.modules():
        if hasattr(mod, "_invalidate"):
            mod._invalidate()
    return [k for k in unexpected]


@torch.no_grad()
def score(m, wins):
    ce = n = 0.0
    for t in wins:
        t = t.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(t[None, :-1])[0][0].float()
        ce += float(F.cross_entropy(lg, t[1:], reduction="sum")); n += t.numel() - 1
    return ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s", required=True); ap.add_argument("--t", required=True)
    ap.add_argument("--ext", action="append", default=[]); ap.add_argument("--val", required=True)
    ap.add_argument("--nval", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=[], help="NAME=REGEX: instead of the per-block loop, T with the "
                    "keys matching REGEX from S and S with them from T (e.g. trits=\\.lin\\.wpacked$: revert a run's moves)")
    a = ap.parse_args()
    d = np.memmap(a.val, dtype=np.uint32, mode="r")
    wins = [torch.from_numpy(np.asarray(d[i * 2048:i * 2048 + 2049], dtype=np.int64)) for i in range(a.nval)]
    S, T = sdict(a.s), sdict(a.t)
    m = build(a.model, a.ext); prepare(m, S)
    keys = [k for k in S if k in T and S[k].shape == T[k].shape and not k.endswith("frozen_scale")]
    blk = lambda k: (re.search(r"model\.layers\.(\d+)\.", k) or [None, None])[1]
    tern = lambda k: bool(re.search(r"\.(lin|ad\.\d|lin2)\.|_proj\.(wpacked|row_scale|col_scale|out_bias|in_mean)$", k))
    nl = 1 + max(int(blk(k)) for k in keys if blk(k) is not None)
    floats = [k for k in keys if not tern(k) and "embed_tokens" not in k and "lm_head" not in k]
    res = {}
    load(m, S); res["S"] = score(m, wins)
    load(m, T); res["T"] = score(m, wins)
    print(f"S {res['S']:.4f}  T {res['T']:.4f}", flush=True)
    load(m, S, floats); res["T_floats_from_S"] = score(m, wins); load(m, T, floats)
    load(m, S); load(m, T, floats); res["S_floats_from_T"] = score(m, wins)
    if a.only:
        for spec in a.only:
            nm, rx = spec.split("=", 1); ks = [k for k in keys if re.search(rx, k)]
            load(m, T); load(m, S, ks); ts = score(m, wins)
            load(m, S); load(m, T, ks); st = score(m, wins)
            res[nm] = {"keys": len(ks), "T_from_S": ts - res["T"], "S_from_T": st - res["S"]}
            print(f"{nm} ({len(ks)} tensors): T<-S {ts - res['T']:+.4f}  S<-T {st - res['S']:+.4f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("SWAP_DONE", flush=True)
        return
    rows = []
    for l in range(nl):
        kl = [k for k in keys if blk(k) == str(l)]
        load(m, T); load(m, S, kl); ts = score(m, wins)
        load(m, S); load(m, T, kl); st = score(m, wins)
        rows.append({"block": l, "T_from_S": ts - res["T"], "S_from_T": st - res["S"]})
        print(f"block {l}: T<-S {ts - res['T']:+.4f}  S<-T {st - res['S']:+.4f}", flush=True)
    res["blocks"] = rows
    res["sum_S_from_T"] = sum(r["S_from_T"] for r in rows); res["sum_T_from_S"] = sum(r["T_from_S"] for r in rows)
    print(f"floats: T<-S {res['T_floats_from_S'] - res['T']:+.4f}, S<-T {res['S_floats_from_T'] - res['S']:+.4f}; "
          f"sum over blocks: S<-T {res['sum_S_from_T']:+.4f} (S->T total {res['T'] - res['S']:+.4f}), "
          f"T<-S {res['sum_T_from_S']:+.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("SWAP_DONE", flush=True)


if __name__ == "__main__":
    main()
