"""ternary low-rank adapters on a converted Qwen3.5 / 3.8. Every decoder linear that ttf-core makes ternary becomes
y = T (x - mu) + b + U D (x - mu), D [r, K] and U [N, r] ternary layers with their own row / column scales, all trained
by TTF. Init, in the streaming conversion, once the layer's own trits are final (after GPTQ and any coordinate descent):
the adapter to what rounding left, R = W - a c T: the rank-r SVD of R in the output metric H (the centred second
moment), D by GPTQ under H, U by least squares and GPTQ under D H D^T; with alt = n, n rounds of alternating refit
(D by least squares given the quantised U, then U again), the best kept.
State-dict keys: .lin (the main ternary layer), .ad.0 (down), .ad.1 (up).
Needs the centred streaming conversion (--convert gptqc --convert-stream, the default): the fit runs in its hooks.
  --ext ttfconv.adapters:rank=0.2,alt=1[,ttf_rank=64][,svd=rand][,gmetric=DIR]
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


class QTad(nn.Module):
    """y = lin(x) + up(down(x))"""
    def __init__(self, lin, down, up):
        super().__init__()
        self.lin, self.ad = lin, nn.ModuleList([down, up])
        self.in_features, self.out_features = lin.K, lin.N

    def forward(self, x):    # (the sum goes into the up layer's output as acc: one pass with --fused-layer)
        y = self.lin(x)
        if getattr(self, "ad_off", False):    # (ttfconv.adapter_ctl mode=off)
            return y
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
        o = parse(spec, {"rank": "0.2", "prop": "0", "gmetric": "", "ttf_rank": "", "svd": "full"}, required=("alt",))
        # svd=rand: the adapter subspace from a randomized SVD (r + r/2 columns, 8 power iterations: 99.9% of the full
        # SVD's top-r energy, the same fit error to +-0.001 on 0.8B layers; the full SVD was 47% of the 27B conversion)
        self.svd = o["svd"]
        assert self.svd in ("full", "rand"), self.svd
        # ttf_rank=R: the adapter factors' own TTF accumulator rank (the layers' ts_rank; needs ttf-ternary 359a098 or
        # later); empty: --ttf-rank, as the main layers
        self.ttf_rank = int(o["ttf_rank"]) if o["ttf_rank"] else None
        # gmetric=DIR: the output-gradient second moments G (tools/kfac.py --dump-g, same basis as the conversion): the
        # adapter's subspace from the two-sided SVD of L_G^T R L_H and the refit of D in the G-weighted metric; empty: the H-only fit (before 24 Oct)
        self.gdir = o["gmetric"]
        self.rank, self.alt = o["rank"], int(o["alt"])
        # prop=1: the conversion's calibration stream goes through the adapted layer (its adapter) instead of the bare
        # ternary layer, so later layers are rounded for the inputs they will see (T6, P6c; before 24 Oct: 0)
        self.propagate_wrapped = o["prop"] == "1"
        self.errs, self.alt_log = [], []

    def build(self, model, swapped, make_linear, log):
        dev = next(p for p in model.parameters() if not p.is_meta).device
        tot = ext = 0
        layers = model.hf.model.layers
        for name in list(swapped):
            lin = swapped[name]; r = rank_of(lin.K, lin.N, self.rank); tot += lin.N * lin.K; ext += r * (lin.K + lin.N)
            q = QTad(lin, make_linear(lin.K, r).to(dev), make_linear(r, lin.N).to(dev))
            if self.ttf_rank:
                q.ad[0].ts_rank = q.ad[1].ts_rank = self.ttf_rank
            par, cn = name[len("model.layers."):].rsplit(".", 1)
            setattr(layers.get_submodule(par), cn, q)
            swapped[name] = q
        log(f"ttfconv.adapters (rank {self.rank}, alt {self.alt}): {ext / tot:+.1%} trits over one per weight "
            f"({ext / 1e6:.1f} M extra)" + (f"; adapter TTF rank {self.ttf_rank}" if self.ttf_rank else ""))
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
        eA, h = fit_tadapt(q.ad, R, Hc, self.alt, ternarize, ternarize_gptq, G, self.svd)
        self.errs.append(eA); self.alt_log.append(h)
        q.ad[0].enable_centering(mu, torch.zeros(q.ad[0].N, device=mu.device))

    def end(self, model, log):
        n_ad = sum(isinstance(q, QTad) for q in model.modules())
        if n_ad and not self.errs:
            raise SystemExit("ttfconv.adapters: no adapter was fitted: the fit runs only in the centred streaming "
                             "conversion (--convert gptqc, --convert-stream)")
        if self.errs:
            log(f"ttfconv.adapters: {len(self.errs)} fitted, output-space error of what the ternary layer left, after "
                f"the adapter {sum(self.errs) / len(self.errs):.4f} (relative)")


def ttf_ext(spec):
    return Adapters(spec)
