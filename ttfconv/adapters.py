"""ternary low-rank adapters (and an optional second ternary layer) on a converted Qwen3.5 / 3.8. Every decoder linear
that ttf-core makes ternary becomes y = T (x - mu) + b + U D (x - mu), D [r, K] and U [N, r] ternary layers with their
own row / column scales, all trained by TTF. With t2 > 0 the early FFN linears (gate, up, down in layer order, until
that share of the ternary weights) also get a second full ternary layer: y = T1 (x - mu) + b + T2 (x - mu) + U D (x - mu).
Init, in the streaming conversion, once the layer's own trits are final (after GPTQ and any coordinate descent): T2 by
GPTQ to R = W - a c T (same centred second moment H); the adapter to what is left: the rank-r SVD of R in the output
metric H, D by GPTQ under H, U by least squares and GPTQ under D H D^T; with alt = n, n rounds of alternating refit
(D by least squares given the quantised U, then U again), the best kept.
State-dict keys: .lin (the main ternary layer), .ad.0 (down), .ad.1 (up), .lin2 (T2).
Needs the centred streaming conversion (--convert gptqc --convert-stream, the default): the fit runs in its hooks.
  --ext ttfconv.adapters:rank=0.2,alt=1[,t2=0.229][,shrink=q_proj+in_proj_qkv+in_proj_z@0.05,boost=0-7][,ttf_rank=64][,svd=rand]
                        [,lin_rank=down_proj+out_proj+o_proj@1024]
prop=1: calibration stream through the adapted layers (default 0, the wip behaviour).
rank: an int, or a fraction of min(K, N) (rounded to a multiple of 8, at least 8). alt must be given (0 = the fit before
23 Oct, 1 = recommended), so comparisons cannot change their fit silently.
"""
import torch
import torch.nn as nn
from .spec import parse


def rank_of(K, N, r):
    r = float(r)
    return max(8, int(round(r * min(K, N) / 8)) * 8) if r < 1 else int(r)


def abytes(K, N, r):
    """packed adapter bytes: down [r, K] and up [N, r] trits at 5 per byte, row / column scales as fp16"""
    return r * -(-K // 5) + N * -(-r // 5) + 2 * (2 * r + K + N)


def plan_ranks(shapes, rank, shrink="", boost="", log=print):
    """{name: (K, N)} -> {name: r}. shrink "types@f": those linear types at rank fraction f; boost "lo-hi": every other
    adapter in those layers raised so the total packed adapter bytes match the uniform rank"""
    base = {n: rank_of(K, N, rank) for n, (K, N) in shapes.items()}
    if not shrink:
        return base
    types, f = shrink.split("@"); types = types.split("+"); lo, hi = (int(x) for x in boost.split("-"))
    li = lambda n: int(n.split("layers.")[1].split(".")[0])
    small = {n for n in shapes if n.rsplit(".", 1)[1] in types}
    big = {n for n in shapes if n not in small and lo <= li(n) <= hi}
    target = sum(abytes(K, N, base[n]) for n, (K, N) in shapes.items())

    def plan(g):
        return {n: rank_of(K, N, f) if n in small else (rank_of(K, N, g) if n in big else base[n])
                for n, (K, N) in shapes.items()}
    a, b = float(rank), 1.0
    for _ in range(40):                       # bisection on the boosted fraction
        g = (a + b) / 2; tot = sum(abytes(K, N, r) for (K, N), r in zip(shapes.values(), plan(g).values()))
        a, b = (g, b) if tot < target else (a, g)
    rk = plan(a); tot = sum(abytes(K, N, r) for (K, N), r in zip(shapes.values(), rk.values()))
    log(f"ttfconv.adapters re-allocated: {sorted(types)} at {f}, layers {lo}-{hi} at {a:.3f} (others {rank}): "
        f"{tot / 1e6:.2f} MB vs {target / 1e6:.2f} MB uniform ({tot / target - 1:+.2%})")
    return rk


class QTad(nn.Module):
    """y = lin(x) [+ lin2(x)] + up(down(x))"""
    def __init__(self, lin, down, up, lin2=None):
        super().__init__()
        self.lin, self.ad = lin, nn.ModuleList([down, up])
        if lin2 is not None:
            self.lin2 = lin2
        self.in_features, self.out_features = lin.K, lin.N

    def forward(self, x):    # (the sums go into each next layer's output as acc: one pass each with --fused-layer)
        y = self.lin(x)
        if getattr(self, "lin2", None) is not None:
            y = self.lin2(x, acc=y)
        return self.ad[1](self.ad[0](x), acc=y)


def set_tern(l, T, a, c):
    from ttf.kernel import pack_rows, trit_beta
    l.wpacked.copy_(pack_rows(T))
    beta = float(trit_beta(l.wpacked, l.K)) if l.use_beta else 1.0
    l.row_scale.copy_(a / beta); l.col_scale.copy_(c)


def top_svd(A, r, mode="full", over=None, niter=8):
    """the top r singular triplets of A: the full SVD (mode full) or a randomized one (rand: Halko et al., r + over
    columns, niter power iterations; torch.svd_lowrank)"""
    if mode == "full":
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :r], S[:r], Vh[:r]
    q = min(r + (over if over is not None else max(16, r // 2)), *A.shape)
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
    return U[:, :r], S[:r], V[:, :r].T


@torch.no_grad()
def fit_tadapt(ad, R, H, alt, ternarize, gptq, G=None, svd="full"):
    """down [r, K], up [N, r] ternary with R ~ up down in the output metric H (= L L^T); returns the relative error left
    and the per-round errors. G (N x N, optional): D's subspace from the SVD of L_G^T R L_H instead of R L_H, and the
    alternating refit of D weighted by G (the up factor's least squares does not depend on G)"""
    r = ad[0].N; dev = R.device; K = R.shape[1]
    Hd = H + 1e-6 * torch.diagonal(H).mean() * torch.eye(K, device=dev)
    Lc = torch.linalg.cholesky(Hd)
    if G is None:
        U, S, Vh = top_svd(R @ Lc, r, svd)
    else:
        Gs = 0.5 * (G + G.T)                       # robust whitening: eigenvalues floored at 1e-3 of their mean
        ev, Q = torch.linalg.eigh(Gs.double())
        ev = ev.clamp_min(1e-3 * float(ev.clamp_min(0).mean()))
        Gd = (Q * ev) @ Q.T; Gd = Gd.float()
        LgT = (ev.sqrt()[:, None] * Q.T).float()   # L_G^T with L_G L_G^T = Gd
        U, S, Vh = top_svd(LgT @ R @ Lc, r, svd)
    Dt = torch.linalg.solve_triangular(Lc, S[:r].sqrt()[:, None] * Vh[:r], upper=False, left=False)   # (.) L^-1
    T, a, c = ternarize(Dt, "als"); T, a, c = gptq(Dt, H, a, c); set_tern(ad[0], T, a, c); tac_d = (T, a, c)
    Dq = a[:, None] * c[None, :] * T.float()
    M = Dq @ H @ Dq.T; M += 1e-6 * torch.diagonal(M).mean() * torch.eye(r, device=dev)
    Ut = R @ H @ Dq.T @ torch.linalg.inv(M)
    T, a, c = ternarize(Ut, "als"); T, a, c = gptq(Ut, M, a, c); set_tern(ad[1], T, a, c)
    E = R - (a[:, None] * c[None, :] * T.float()) @ Dq
    nR = ((R @ H) * R).sum()
    err = float(((E @ H) * E).sum() / nR)
    if alt <= 0:
        return err, [err]
    # alternating refit: down by least squares given the quantized up (the H-metric optimum (Uq^T Uq)^-1 Uq^T R does
    # not depend on H), GPTQ under H; then up again as above. The best round is kept.
    errs = [err]; best = (err, tac_d, (T, a, c)); Uq = a[:, None] * c[None, :] * T.float()
    for _ in range(alt):
        UG = Uq.T if G is None else Uq.T @ Gd
        GU = UG @ Uq; GU += 1e-6 * torch.diagonal(GU).mean() * torch.eye(r, device=dev)
        Dt = torch.linalg.solve(GU, UG @ R)
        Td, ad_, cd = ternarize(Dt, "als"); Td, ad_, cd = gptq(Dt, H, ad_, cd)
        Dq = ad_[:, None] * cd[None, :] * Td.float()
        M = Dq @ H @ Dq.T; M += 1e-6 * torch.diagonal(M).mean() * torch.eye(r, device=dev)
        Ut = R @ H @ Dq.T @ torch.linalg.inv(M)
        Tu, au, cu = ternarize(Ut, "als"); Tu, au, cu = gptq(Ut, M, au, cu)
        Uq = au[:, None] * cu[None, :] * Tu.float()
        E = R - Uq @ Dq; e_ = float(((E @ H) * E).sum() / nR); errs.append(e_)
        if e_ < best[0]:
            best = (e_, (Td, ad_, cd), (Tu, au, cu))
    set_tern(ad[0], *best[1]); set_tern(ad[1], *best[2])
    return best[0], errs


class Adapters:
    def __init__(self, spec):
        o = parse(spec, {"rank": "0.2", "t2": "0", "shrink": "", "boost": "", "prop": "0", "gmetric": "", "t2sel": "",
                         "ttf_rank": "", "svd": "full", "lin_rank": ""}, required=("alt",))
        # svd=rand: the adapter subspace from a randomized SVD (r + r/2 columns, 8 power iterations: 99.9% of the full
        # SVD's top-r energy, the same fit error to +-0.001 on 0.8B layers; the full SVD was 47% of the 27B conversion)
        self.svd = o["svd"]
        assert self.svd in ("full", "rand"), self.svd
        # ttf_rank=R: the adapter factors' own TTF accumulator rank (the layers' ts_rank; needs ttf-ternary 359a098 or
        # later); empty: --ttf-rank, as the main layers
        self.ttf_rank = int(o["ttf_rank"]) if o["ttf_rank"] else None
        # lin_rank=TYPES@R: the main ternary layer's (and its T2's) TTF accumulator rank R for those linear types (25 Oct
        # audit: down_proj / out_proj moved no trits at r 256 or 512); empty: --ttf-rank, as every other layer
        self.lin_rank = None
        if o["lin_rank"]:
            t_, r_ = o["lin_rank"].split("@"); self.lin_rank = (set(t_.split("+")), int(r_))
        # t2sel=FILE: the T2 linears chosen in the order of FILE (one name per line, e.g. by measured damage per byte,
        # tools/kfac.py) instead of the early FFNs, until the t2 share 
        self.t2sel = o["t2sel"]
        # gmetric=DIR: the output-gradient second moments G (tools/kfac.py --dump-g, same basis as the conversion): the
        # adapter's subspace from the two-sided SVD of L_G^T R L_H and the refit of D in the G-weighted metric; empty: the H-only fit (before 24 Oct)
        self.gdir = o["gmetric"]
        self.rank, self.alt, self.t2 = o["rank"], int(o["alt"]), float(o["t2"])
        # prop=1: the conversion's calibration stream goes through the adapted layer (adapters, T2) instead of the bare
        # ternary layer, so later layers are rounded for the inputs they will see (T6, P6c; before 24 Oct: 0)
        self.propagate_wrapped = o["prop"] == "1"
        self.shrink, self.boost = o["shrink"], o["boost"]
        self.errs, self.t2errs, self.alt_log = [], [], []

    def build(self, model, swapped, make_linear, log):
        dev = next(p for p in model.parameters() if not p.is_meta).device
        tot = ext = 0
        rk = plan_ranks({n: (swapped[n].K, swapped[n].N) for n in swapped}, self.rank, self.shrink, self.boost, log)
        layers = model.hf.model.layers
        for name in list(swapped):
            lin = swapped[name]; r = rk[name]; tot += lin.N * lin.K; ext += r * (lin.K + lin.N)
            q = QTad(lin, make_linear(lin.K, r).to(dev), make_linear(r, lin.N).to(dev))
            if self.ttf_rank:
                q.ad[0].ts_rank = q.ad[1].ts_rank = self.ttf_rank
            par, cn = name[len("model.layers."):].rsplit(".", 1)
            setattr(layers.get_submodule(par), cn, q)
            swapped[name] = q
        log(f"ttfconv.adapters (rank {self.rank}, alt {self.alt}): {ext / tot:+.1%} trits over one per weight "
            f"({ext / 1e6:.1f} M extra)" + (f"; adapter TTF rank {self.ttf_rank}" if self.ttf_rank else ""))
        if self.t2 > 0:     # the early FFNs (gate, up, down) in layer order until the share of the ternary weights
            acc = 0; sel = []
            order = ([l.strip() for l in open(self.t2sel) if l.strip() in swapped] if self.t2sel else
                     sorted((n for n in swapped if ".mlp." in n), key=lambda n: (int(n.split(".")[2]), n)))
            for n in order:
                if acc >= self.t2 * tot:
                    break
                q = swapped[n]; acc += q.lin.N * q.lin.K; sel.append(n)
                q.lin2 = make_linear(q.lin.K, q.lin.N).to(dev)
            log(f"ttfconv.adapters t2 ({'order ' + self.t2sel if self.t2sel else 'early FFNs'}): second ternary layer on {len(sel)} linears (layers "
                f"{min(int(n.split('.')[2]) for n in sel)}-{max(int(n.split('.')[2]) for n in sel)}), {acc / tot:.1%} "
                f"of the ternary weights")
        if self.lin_rank:   # (after T2: its second layer takes the same rank; set on resume too, so the state shapes match)
            types, r_ = self.lin_rank; nl = 0
            for n, q in swapped.items():
                if n.rsplit(".", 1)[1] in types:
                    for l in (q.lin, getattr(q, "lin2", None)):
                        if l is not None:
                            l.ts_rank = r_; nl += 1
            log(f"ttfconv.adapters lin_rank: TTF rank {r_} on {nl} main / T2 layers ({'+'.join(sorted(types))})")
        return swapped

    def fitted(self, name, q, W, Hc, mu, T, a, c):
        if not isinstance(q, QTad):
            return
        from ttf.hf_init import ternarize, ternarize_gptq
        R = W - a[:, None] * c[None, :] * T.float()
        G = None
        if self.gdir:
            import os
            G = torch.load(os.path.join(self.gdir, name + ".pt"), map_location=R.device).float()
        l2 = getattr(q, "lin2", None)
        if l2 is not None:
            T2, a2, c2 = ternarize(R, "als"); T2, a2, c2 = ternarize_gptq(R, Hc, a2, c2); set_tern(l2, T2, a2, c2)
            R2 = R - a2[:, None] * c2[None, :] * T2.float()
            e0 = float(((R @ Hc) * R).sum()); e2 = float(((R2 @ Hc) * R2).sum())
            eA, h = fit_tadapt(q.ad, R2, Hc, self.alt, ternarize, ternarize_gptq, G, self.svd)
            self.t2errs.append((e2 / e0, eA * e2 / e0)); self.errs.append(eA * e2 / e0)
        else:
            eA, h = fit_tadapt(q.ad, R, Hc, self.alt, ternarize, ternarize_gptq, G, self.svd)
            self.errs.append(eA)
        self.alt_log.append(h)
        q.ad[0].enable_centering(mu, torch.zeros(q.ad[0].N, device=mu.device))
        if l2 is not None:
            l2.enable_centering(mu, torch.zeros(l2.N, device=mu.device))

    def end(self, model, log):
        n_ad = sum(isinstance(q, QTad) for q in model.modules())
        if n_ad and not self.errs:
            raise SystemExit("ttfconv.adapters: no adapter was fitted: the fit runs only in the centred streaming "
                             "conversion (--convert gptqc, --convert-stream)")
        if self.errs:
            log(f"ttfconv.adapters: {len(self.errs)} fitted, output-space error of what the ternary layer left, after "
                f"the adapter {sum(self.errs) / len(self.errs):.4f} (relative)")
        if self.t2errs:
            log(f"ttfconv.adapters t2: {len(self.t2errs)} second layers: after T2 "
                f"{sum(e for e, _ in self.t2errs) / len(self.t2errs):.4f}, after T2 + adapter "
                f"{sum(e for _, e in self.t2errs) / len(self.t2errs):.4f} (relative)")


def ttf_ext(spec):
    return Adapters(spec)
