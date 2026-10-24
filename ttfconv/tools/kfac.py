"""K-FAC probe of a converted Qwen (which layers' rounding costs what; reorganisation; adapter metric), training-free.

On the fp model (optionally in the fold-a basis, which keeps its function), per decoder linear:
  H = E[x x^T] (centred), G = E[d d^T] with d = dl/dy under labels sampled from the model (the Fisher for KL(fp || .))
From them, for E = centred-GPTQ rounding error of that linear alone:
  e_H     the H-only relative output error tr(E H E^T) / tr(W H W^T)  (what the conversion logs and greedy used)
  kfac    0.5 tr(G E H E^T)   predicted KL per token of rounding this linear alone (T1, eq. 1)
  benign  kfac / its expectation for an isotropic random E of the same Frobenius norm (1 = random)
  eff ranks of G and H (participation ratio / dim)
  P6a: the second-order damage a rank-r fp correction removes, fitted in the H metric (as the adapters are) vs in the
       two-sided G (x) H metric
Measured: for a spread of linears, the KL per token of the fp model with only that linear rounded, on held-out windows
(P1: kfac should rank them, e_H should not). With --ckpt NAME=PATH (states of the same fold-a conversion, trained or
not): per linear, D = effective trained weight (trits + adapters + T2) - fp weight, its kfac, benign ratio and size
(P4a: reorganisation moves along flat directions iff benign << 1 and falls with training).
  python -m ttfconv.tools.kfac --model M --data data/broadqw_train.bin --val data/wikiqw_val.bin --fold a \
      --ckpt s500=store/cpufix/ckpt_tad.pt --out kfac.json
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj",
          "down_proj")


def linears(hf):
    out = {}
    for n, mod in hf.model.layers.named_modules():
        if isinstance(mod, nn.Linear) and n.rsplit(".", 1)[-1] in LEAVES:
            out["model.layers." + n] = mod
    return out


@torch.no_grad()
def fold_model(hf, dev, seed=2110):
    """fold a on the fp HF model in place (function kept): gains into the readers, shared randomized Hadamard"""
    from ttfconv.fold import residual_q, fold_float, fold_w
    if hf.lm_head.weight is hf.model.embed_tokens.weight:
        hf.lm_head.weight = nn.Parameter(hf.lm_head.weight.detach().clone(), requires_grad=False)
    d = hf.model.embed_tokens.weight.shape[1]
    Qr = residual_q(d, torch.Generator().manual_seed(seed)).to(dev)
    G = fold_float(hf, Qr)
    for n, lin in linears(hf).items():
        li = int(n.split(".")[2])
        lin.weight.copy_(fold_w(n.rsplit(".", 1)[1], lin.weight.float(), *G[li], Qr).to(lin.weight.dtype))


def windows(path, n, seq, seed):
    d = np.memmap(path, dtype=np.uint32, mode="r")
    st = np.random.default_rng(seed).integers(0, len(d) - seq - 1, n)
    return [torch.from_numpy(np.asarray(d[s:s + seq], dtype=np.int64)) for s in st]


def sample_labels(logits, gen):
    """y ~ softmax(logits) per position (Gumbel-max, chunked)"""
    out = []
    for a in range(0, logits.shape[0], 256):
        lg = logits[a:a + 256].float()
        u = torch.rand(lg.shape, device=lg.device, generator=gen).clamp_(1e-20, 1.0)
        out.append((lg - torch.log(-torch.log(u))).argmax(-1))
    return torch.cat(out)


def stats_pass(hf, lins, wins, dev, gen):
    """H (sum x x^T, sum x, count) and G (sum d d^T) for the given linears, fp model, sampled labels"""
    St = {n: dict(H=torch.zeros(l.in_features, l.in_features, device=dev), s=torch.zeros(l.in_features, device=dev),
                  G=torch.zeros(l.out_features, l.out_features, device=dev), n=0) for n, l in lins.items()}
    hooks = []
    for n, l in lins.items():
        def fh(mod, inp, out, n=n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float(); s = St[n]
            s["H"].addmm_(x.T, x); s["s"] += x.sum(0); s["n"] += x.shape[0]
            if out.requires_grad:
                def gh(g, n=n):
                    g2 = g.reshape(-1, g.shape[-1]).float(); St[n]["G"].addmm_(g2.T, g2)   # (returns None: grad kept)
                out.register_hook(gh)
        hooks.append(l.register_forward_hook(fh))
    eh = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_())
    for w in wins:
        t = w.to(dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = hf(t).logits[0]
        y = sample_labels(lg.detach(), gen)
        loss = F.cross_entropy(lg.float(), y, reduction="sum")
        loss.backward()
        del lg, loss
    for h in hooks:
        h.remove()
    eh.remove()
    return St


def prank(M):
    ev = torch.linalg.eigvalsh(M.double()).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-300) / M.shape[0])


def kfac_val(G, E, H):
    return 0.5 * float(((G @ E @ H) * E).sum())


def rand_expect(G, E, H):
    N, K = E.shape
    return 0.5 * float(E.pow(2).sum()) * float(torch.trace(G)) * float(torch.trace(H)) / (N * K)


@torch.no_grad()
def lowrank_capture(R, G, H, r):
    """second-order damage of R removed by the best rank-r fp correction in the H metric vs the G (x) H metric"""
    N, K = R.shape
    eh = 1e-3 * float(torch.diagonal(H).mean()); eg = 1e-3 * float(torch.diagonal(G).mean())   # (fit metric damped only)
    Lh = torch.linalg.cholesky(H.double() + eh * torch.eye(K, device=R.device, dtype=torch.float64))
    Lg = torch.linalg.cholesky(G.double() + eg * torch.eye(N, device=R.device, dtype=torch.float64))
    Rd = R.double(); base = kfac_val(G.double(), Rd, H.double())
    U, S, Vh = torch.linalg.svd(Rd @ Lh, full_matrices=False)
    R1 = torch.linalg.solve_triangular(Lh, (U[:, :r] * S[:r]) @ Vh[:r], upper=False, left=False)
    U, S, Vh = torch.linalg.svd(Lg.T @ Rd @ Lh, full_matrices=False)
    M2 = (U[:, :r] * S[:r]) @ Vh[:r]
    R2 = torch.linalg.solve_triangular(Lg.T, M2, upper=True, left=True)
    R2 = torch.linalg.solve_triangular(Lh, R2, upper=False, left=False)
    k1 = kfac_val(G.double(), Rd - R1, H.double()); k2 = kfac_val(G.double(), Rd - R2, H.double())
    eH = lambda X: float(((X @ H.double()) * X).sum())
    return {"kfac_left_H": k1 / base, "kfac_left_GH": k2 / base, "eH_left_H": eH(Rd - R1) / eH(Rd),
            "eH_left_GH": eH(Rd - R2) / eH(Rd)}


@torch.no_grad()
def eff_weight(sd, pre, dev):
    """effective weight of a converted linear (trits x scales x beta, + adapter up @ down, + T2) from a state dict"""
    from ttf.kernel import unpack_rows, trit_beta

    def tern(p):
        wp = sd[p + ".wpacked"].to(dev); K = sd[p + ".col_scale"].numel()
        T = unpack_rows(wp, K).float(); b = float(trit_beta(wp, K))
        return b * sd[p + ".row_scale"].to(dev).float()[:, None] * T * sd[p + ".col_scale"].to(dev).float()[None, :]
    base = pre + ".lin" if pre + ".lin.wpacked" in sd else pre
    W = tern(base)
    if pre + ".ad.0.wpacked" in sd:
        W = W + tern(pre + ".ad.1") @ tern(pre + ".ad.0")
    if pre + ".lin2.wpacked" in sd:
        W = W + tern(pre + ".lin2")
    return W


@torch.no_grad()
def kl_pass(hf, wins, dev, ref):
    """mean KL(fp || model) per token over wins; ref: list of fp log-prob tensors computed on the fly when None"""
    tot = n = 0.0
    for i, w in enumerate(wins):
        t = w.to(dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lq = hf(t).logits[0]
        lp = ref[i].to(dev)
        for a in range(0, lq.shape[0], 256):
            q = F.log_softmax(lq[a:a + 256].float(), -1); p = F.log_softmax(lp[a:a + 256].float(), -1)
            tot += float((p.exp() * (p - q)).sum()); n += q.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--fold", default="a", choices=["a", "none"]); ap.add_argument("--ncal", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--nval", type=int, default=4); ap.add_argument("--swap-every", type=int, default=4)
    ap.add_argument("--rank", type=float, default=0.2); ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--layers", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--dump-g", default="", help="DIR: save each linear's G (fp32, N x N) as DIR/NAME.pt (for "
                    "ttfconv.adapters gmetric=DIR); with --stats-only nothing else is computed")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from ttf.hf_init import ternarize, ternarize_gptq
    dev = "cuda"; t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    cal = windows(a.data, a.ncal, a.seq, 4321); val = windows(a.val, a.nval, a.seq, 777)
    if a.fold == "a":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            l0 = float(F.cross_entropy(hf(val[0].to(dev)[None]).logits[0, :-1].float(), val[0].to(dev)[1:]))
        fold_model(hf, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            l1 = float(F.cross_entropy(hf(val[0].to(dev)[None]).logits[0, :-1].float(), val[0].to(dev)[1:]))
        print(f"fold a: loss on a val window {l0:.4f} -> {l1:.4f} (function kept)", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = [hf(w.to(dev)[None]).logits[0].to(torch.bfloat16).cpu() for w in val]
    cks = {}
    for spec in a.ckpt:
        nm, p = spec.split("=", 1)
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        sd = ck.get("model", ck); cks[nm] = {k[3:] if k.startswith("hf.") else k: v for k, v in sd.items()}
    L = linears(hf); names = list(L)
    nl = len(hf.model.layers); lay = range(nl) if not a.layers else [int(x) for x in a.layers.split(",")]
    gen = torch.Generator(device=dev).manual_seed(99)
    rows = []
    for c0 in range(0, len(lay), a.chunk):
        ch = list(lay)[c0:c0 + a.chunk]
        lins = {n: L[n] for n in names if int(n.split(".")[2]) in ch}
        St = stats_pass(hf, lins, cal, dev, gen)
        for n, l in lins.items():
            s = St.pop(n); cnt = s["n"]
            mu = s["s"] / cnt; Hc = s["H"] / cnt - torch.outer(mu, mu); G = s["G"] / cnt
            if a.dump_g:
                os.makedirs(a.dump_g, exist_ok=True)
                torch.save(G.float().cpu(), os.path.join(a.dump_g, n + ".pt"))
            if a.stats_only:
                rows.append({"name": n}); continue
            W = l.weight.float()
            T, aa, cc = ternarize(W, "als"); T, aa, cc = ternarize_gptq(W, Hc, aa, cc)
            E = aa[:, None] * cc[None, :] * T.float() - W
            row = {"name": n, "layer": int(n.split(".")[2]), "leaf": n.rsplit(".", 1)[1], "N": W.shape[0], "K": W.shape[1],
                   "e_H": float(((E @ Hc) * E).sum() / ((W @ Hc) * W).sum()), "kfac": kfac_val(G, E, Hc),
                   "rel_w": float(E.norm() / W.norm()), "trG": float(torch.trace(G)), "trH": float(torch.trace(Hc)),
                   "prG": prank(G), "prH": prank(Hc)}
            row["benign"] = row["kfac"] / rand_expect(G, E, Hc)
            r = max(8, int(round(a.rank * min(W.shape) / 8)) * 8)
            row.update({f"p6a_{k}": v for k, v in lowrank_capture(-E, G, Hc, r).items()}); row["p6a_rank"] = r
            for nm, sd in cks.items():
                D = eff_weight(sd, n, dev) - W
                row[f"ck_{nm}"] = {"kfac": kfac_val(G, D, Hc), "benign": kfac_val(G, D, Hc) / rand_expect(G, D, Hc),
                                   "rel_w": float(D.norm() / W.norm()),
                                   "e_H": float(((D @ Hc) * D).sum() / ((W @ Hc) * W).sum())}
            if len(rows) % a.swap_every == 0:                  # measured: only this linear rounded
                old = l.weight.detach().clone()
                with torch.no_grad():
                    l.weight.copy_((W + E).to(l.weight.dtype))
                row["kl_meas"] = kl_pass(hf, val, dev, ref)
                with torch.no_grad():
                    l.weight.copy_(old)
                del old
            rows.append(row)
            del W, E, G, Hc, T
        torch.cuda.empty_cache()
        print(f"layers {ch} done ({len(rows)} linears, {time.time() - t0:.0f} s)", flush=True)
        if a.stats_only:
            continue
        json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=0)
    m = [r for r in rows if "kl_meas" in r]
    if len(m) > 3:
        from scipy.stats import spearmanr
        km = [r["kl_meas"] for r in m]
        res = {k: float(spearmanr(km, [r[k] for r in m]).correlation) for k in ("kfac", "e_H")}
        res["e_H_x_trG"] = float(spearmanr(km, [r["e_H"] * r["trG"] for r in m]).correlation)
        res["kfac_over_meas_median"] = float(np.median([r["kfac"] / max(r["kl_meas"], 1e-12) for r in m]))
        print("P1 spearman vs measured KL: " + json.dumps(res), flush=True)
        json.dump({"args": vars(a), "rows": rows, "p1": res}, open(a.out, "w"), indent=0)
    print("KFAC_DONE", flush=True)


if __name__ == "__main__":
    main()
