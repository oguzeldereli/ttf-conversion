"""Qwen3.8-Flash-Next -> ternary, layer by layer on one GPU.

(A residual-stream rotation does not fold exactly here: the gated residual and the attention / DeltaNet output gates
act elementwise on the streams, so the conversion uses what is exact: centring, row / column scales, rank-16 ternary
adapters.)

Every ternary matrix W [N, K] (the 512 routed experts' gate+up [1280, 2560] and down [2560, 640]; the dense projections
attention q / k / v / o, DeltaNet in_proj_qkv / in_proj_z / out_proj, shared expert gate / up / down as one "expert")
becomes, in the TernaryExperts form (ttf.q38.moe),
    y = r * (beta T + TA diag(cb) TB) (c * (x - mu)) + W mu,     beta = beta_nnz(T), sa = 1
with T by ternary GPTQ (ALS row / column scales, centred input covariance, act-order), mu the input mean, and the
adapter (TA [N, 16], TB [16, K] trits, per-rank cb) fitted to the rounding residual in the layer's output metric: the
top-16 subspace of R L (L L^T = Hc), TB by GPTQ under c Hc c with a per-rank scale, TA by GPTQ under the rank-space
covariance with its row scale held at 1 (the adapter shares the main row / column scales). The routed experts' inputs
are weighted by their router weight (the expert's output enters the layer output times it).

Sequential: layer l's statistics come from the calibration windows run through the converted layers 0..l-1 (fake-
quantised: the float weight W^ = r (beta T + adapter) c with the exact bias (W - W^) mu); within a layer every matrix
sees the float layer's inputs. Output per layer: OUT/L{i}.pt = {module name: TernaryExperts state_dict}, OUT/L{i}.json
(errors); OUT/h.pt the next layer's calibration inputs (resume restarts at the first missing layer).
  python -m ttfconv.q38.convert HF_DIR DATA.bin OUT [--windows 256] [--seq 2048] [--mb 4] [--rank 16] [--layers N]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttf.q38 import model as M
from ttf.q38 import store as IO

DENSE = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv",
         "linear_attn.in_proj_z", "linear_attn.out_proj", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
         "mlp.shared_expert.down_proj")


# ---- batched (over experts) ternary rounding: ttfconv.gptq's ternarize("als") / ternarize_gptq with a leading E dim
@torch.no_grad()
def row_grid_b(W, grid=torch.linspace(0.3, 2.0, 35)):
    m = W.abs().mean(-1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full_like(m, float("inf")); best_s = m.clone()
    ww = (W * W).sum(-1, keepdim=True)
    for g in grid.tolist():
        T = (W / (g * m)).round_().clamp_(-1, 1)
        nt = (T * T).sum(-1, keepdim=True).clamp_min(1)
        a = (W * T).sum(-1, keepdim=True) / nt
        err = ww - a * a * nt
        better = err < best_err
        best_err = torch.where(better, err, best_err); best_s = torch.where(better, g * m, best_s)
        del T
    T = (W / best_s).round_().clamp_(-1, 1)
    a = ((W * T).sum(-1, keepdim=True) / (T * T).sum(-1, keepdim=True).clamp_min(1)).clamp_min(1e-12)
    return a.squeeze(-1), T


@torch.no_grad()
def als_b(W, iters=4):
    """W [E, N, K] -> (T, a [E, N], c [E, K]), W ~ a_i c_j T_ij"""
    c = torch.ones(W.shape[0], W.shape[2], device=W.device)
    a, T = row_grid_b(W)
    for _ in range(iters):
        num = (W * T * a[..., None]).sum(-2); den = (T * T * (a * a)[..., None]).sum(-2)
        c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-3)
        c = c / c.mean(-1, keepdim=True)
        a, T = row_grid_b(W / c[:, None, :])
    return T, a, c


@torch.no_grad()
def _chol_upper_inv(Hp):
    """GPTQ's upper Cholesky factor of Hp^-1, per expert, with more damping where Hp is not positive definite"""
    K = Hp.shape[-1]
    eye = torch.eye(K, device=Hp.device)
    for extra in (0.0, 0.05, 0.3, 1.0, 10.0):
        Hd = Hp + extra * torch.diagonal(Hp, dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye if extra else Hp
        L, info = torch.linalg.cholesky_ex(Hd)
        if int(info.max()) == 0:
            Hinv = torch.cholesky_inverse(L); del L
            U, info2 = torch.linalg.cholesky_ex(Hinv, upper=True)
            if int(info2.max()) == 0:
                return U, Hd
    raise RuntimeError("GPTQ: Hessian not positive definite even at damping 10")


@torch.no_grad()
def gptq_b(W, H, a, c, block=128, damp=0.01, refits=2, fit_a=True):
    """ternary GPTQ, batched over E: W [E, N, K] fp32, H [E, K, K] (centred), a [E, N], c [E, K] (ALS) -> (T int8, a)"""
    E, N, K = W.shape
    Wc = W / c[:, None, :]
    Hc = H * c[:, :, None] * c[:, None, :]
    dg = torch.diagonal(Hc, dim1=-2, dim2=-1)
    dead = dg <= 0
    Hc = Hc + torch.diag_embed(torch.where(dead, 1.0 - dg, torch.zeros_like(dg)))
    Wc = Wc.masked_fill(dead[:, None, :], 0.0)
    dg = torch.diagonal(Hc, dim1=-2, dim2=-1)
    Hc = Hc + torch.diag_embed(damp * dg.mean(-1, keepdim=True).expand(-1, K))
    perm = torch.argsort(torch.diagonal(Hc, dim1=-2, dim2=-1), dim=-1, descending=True)
    inv = torch.argsort(perm, dim=-1)
    Wp = Wc.gather(2, perm[:, None, :].expand(E, N, K))
    Hp = Hc.gather(1, perm[:, :, None].expand(E, K, K)).gather(2, perm[:, None, :].expand(E, K, K))
    del Hc, Wc
    U, Hp = _chol_upper_inv(Hp)
    for _ in range(refits):
        Q = Wp.clone()
        T = torch.zeros(E, N, K, device=W.device)
        ai = a.clone()
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            Wb = Q[:, :, i1:i2].clone(); Ub = U[:, i1:i2, i1:i2]; Eb = torch.zeros_like(Wb)
            for j in range(i2 - i1):
                w = Wb[:, :, j]
                t = (w / ai).round().clamp(-1, 1)
                T[:, :, i1 + j] = t
                e = (w - ai * t) / Ub[:, j, j][:, None]
                Wb[:, :, j:] -= e[:, :, None] * Ub[:, j, j:][:, None, :]
                Eb[:, :, j] = e
            Q[:, :, i2:] -= Eb @ U[:, i1:i2, i2:]
        del Q
        if fit_a:
            TH = T @ Hp
            a = ((Wp @ Hp) * T).sum(-1) / (TH * T).sum(-1).clamp_min(1e-30)
            a = torch.where(a > 0, a, torch.full_like(a, 1e-3))
            del TH
    return T.gather(2, inv[:, None, :].expand(E, N, K)).to(torch.int8), a


@torch.no_grad()
def rsvd_b(A, r, over=16, iters=4, seed=0):
    """top-r SVD of A [E, N, K] (randomized, power iterations)"""
    E, N, K = A.shape
    g = torch.Generator(device=A.device).manual_seed(seed)
    q = min(r + over, N, K)
    Y = A @ torch.randn(E, K, q, device=A.device, generator=g)
    Qm, _ = torch.linalg.qr(Y)
    for _ in range(iters):
        Z, _ = torch.linalg.qr(A.transpose(1, 2) @ Qm)
        Qm, _ = torch.linalg.qr(A @ Z)
    Ub, S, Vh = torch.linalg.svd(Qm.transpose(1, 2) @ A, full_matrices=False)
    return (Qm @ Ub)[:, :, :r], S[:, :r], Vh[:, :r]


def beta_nnz(T):
    """[E] the core's beta per expert: (nnz / N)^-1/2 clamped at 1 (ttf.q38.moe._beta_nnz)"""
    return (T.ne(0).sum((1, 2)).float() / T.shape[1]).clamp_min(1.0).rsqrt()


@torch.no_grad()
def convert_group(W, Hc, mu, ra, damp):
    """W [E, N, K] fp32, Hc [E, K, K] centred covariance, mu [E, K] -> (state dict pieces, W^ [E, N, K], errors)"""
    E, N, K = W.shape
    T, a, c = als_b(W)
    T, a = gptq_b(W, Hc, a, c, damp=damp)
    Tf = T.float()
    beta = beta_nnz(T)
    r = a / beta[:, None]
    R = W - a[:, :, None] * c[:, None, :] * Tf
    nW = ((W @ Hc) * W).sum((1, 2)).clamp_min(1e-30)
    e_main = ((R @ Hc) * R).sum((1, 2)) / nW
    out = {"T": T, "beta": beta, "row": r, "col": c, "mu": mu}
    What = a[:, :, None] * c[:, None, :] * Tf
    e_ad = e_main
    if ra:
        rr = min(ra, N, K)
        eye = torch.eye(K, device=W.device)
        Hd = Hc + (1e-4 * torch.diagonal(Hc, dim1=-2, dim2=-1).mean(-1)[:, None, None] + 1e-12) * eye
        Lc, info = torch.linalg.cholesky_ex(Hd)
        if int(info.max()):
            Hd = Hc + 0.05 * torch.diagonal(Hc, dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye + 1e-12 * eye
            Lc = torch.linalg.cholesky(Hd)
        U_, S, Vh = rsvd_b(R @ Lc, rr)
        Dt = torch.linalg.solve_triangular(Lc, S.sqrt()[:, :, None] * Vh, upper=False, left=False)   # [E, r, K] x-space
        del Lc, U_
        Dp = Dt / c[:, None, :]                                              # in x' = c (x - mu)
        Hp = Hc * c[:, :, None] * c[:, None, :]                              # covariance of x'
        bta, TB = row_grid_b(Dp)
        TB, bta = gptq_b(Dp, Hp, bta, torch.ones(E, K, device=W.device), damp=damp)
        Dq = bta[:, :, None] * TB.float()                                    # [E, r, K]
        Gs = R / r[:, :, None] / c[:, None, :]                               # the residual in the scaled space
        Mm = Dq @ Hp @ Dq.transpose(1, 2)
        Mm = Mm + 1e-6 * torch.diagonal(Mm, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(rr, device=W.device) + 1e-30
        Ut = torch.linalg.solve(Mm, (Gs @ Hp @ Dq.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)   # [E, N, r]
        al, TAt = row_grid_b(Ut.transpose(1, 2))                             # per-rank scale alpha [E, r]
        TA, _ = gptq_b(Ut, Mm, torch.ones(E, N, device=W.device), al, damp=damp, fit_a=False)
        cb = al * bta
        Ad = (TA.float() * cb[:, None, :]) @ TB.float()                      # [E, N, K] scaled space
        Ahat = r[:, :, None] * Ad * c[:, None, :]
        Ra = R - Ahat
        e_ad = ((Ra @ Hc) * Ra).sum((1, 2)) / nW
        What = What + Ahat
        out.update(TA=TA, TB=TB, cb=cb, sa=torch.ones(E, device=W.device))
    out["bias"] = torch.einsum("enk,ek->en", W, mu)
    return out, What, {"main": e_main, "adapter": e_ad}


def to_state(o, pack3):
    """TernaryExperts state_dict from convert_group's pieces (CPU)"""
    sd = {"wpacked": pack3(o["T"]).cpu(), "beta": o["beta"].float().cpu(), "row_scale": o["row"].float().cpu(),
          "col_scale": o["col"].float().cpu(), "in_mean": o["mu"].float().cpu(), "out_bias": o["bias"].float().cpu()}
    if "TA" in o:
        sd.update(ta_packed=pack3(o["TA"]).cpu(), tb_packed=pack3(o["TB"]).cpu(), ad_cb=o["cb"].float().cpu(),
                  ad_sa=o["sa"].float().cpu())
    return sd


class Stats:
    """per-expert weighted input moments: H = sum (w x)(w x)^T, s = sum w^2 x, n = sum w^2, cnt = slots"""

    def __init__(self, E, K, dev, chunk=None):
        self.E, self.K, self.dev = E, K, dev
        self.chunk = chunk or E
        self.H = [torch.zeros(min(self.chunk, E - a), K, K, device=dev) for a in range(0, E, self.chunk)]
        self.s = torch.zeros(E, K, device=dev); self.n = torch.zeros(E, device=dev)
        self.cnt = torch.zeros(E, dtype=torch.long, device=dev)

    def add(self, x, offs, w=None):
        """x [T, K] sorted by expert, offs [E + 1] (list), w [T] router weights or None"""
        xf = x.float()
        if w is not None:
            xw = xf * w.float()[:, None]
            w2 = w.float().square()
        else:
            xw, w2 = xf, torch.ones(x.shape[0], device=x.device)
        for e in range(self.E):
            a, b = offs[e], offs[e + 1]
            if b > a:
                Hc = self.H[e // self.chunk]
                Hc[e % self.chunk].addmm_(xw[a:b].T, xw[a:b])
                self.s[e] += (w2[a:b, None] * xf[a:b]).sum(0)
                self.n[e] += w2[a:b].sum()
                self.cnt[e] += b - a

    def centred(self, e0, e1, min_cnt=64):
        """(Hc [e1 - e0, K, K], mu [e1 - e0, K], too_few [e1 - e0]); experts with fewer than min_cnt slots get Hc = I,
        mu = 0 (plain ALS rounding)"""
        assert e0 % self.chunk == 0 and (e1 - e0) <= self.chunk
        H = self.H[e0 // self.chunk][: e1 - e0]
        n = self.n[e0:e1].clamp_min(1e-30)
        mu = self.s[e0:e1] / n[:, None]
        Hc = H / n[:, None, None] - mu[:, :, None] * mu[:, None, :]
        few = self.cnt[e0:e1] < min_cnt
        if few.any():
            eye = torch.eye(self.K, device=self.dev)
            Hc[few] = eye
            mu[few] = 0
        return Hc, mu, few


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=256); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0); ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--chunk", type=int, default=32, help="experts per GPTQ batch")
    a = ap.parse_args()
    from ttf.q38.moe import pack3
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    cfg = M.text_config(a.hf)
    nl = a.layers or cfg.num_hidden_layers
    m = M.build(cfg, layers=nl, device="meta")
    st = IO.Store(a.hf, cfg, layers=nl, log=log)
    st.fill_globals(m, dev)
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(a.seed)
    starts = rng.integers(0, len(data) - a.seq - 1, a.windows)
    ids = torch.from_numpy(np.stack([data[s:s + a.seq].astype(np.int64) for s in starts]))
    ids_d = ids.to(dev)
    pe = m.rope(a.seq, dev)
    D = cfg.hc_count * cfg.hidden_size
    first = next((i for i in range(nl) if not os.path.exists(os.path.join(a.out, f"L{i}.pt"))), nl)
    if first == nl:
        log("all layers converted"); return
    if first == 0:
        h = torch.empty(a.windows, a.seq, D, dtype=torch.bfloat16).pin_memory()
        for b0 in range(0, a.windows, a.mb):
            h[b0:b0 + a.mb] = m.embed(ids[b0:b0 + a.mb]).to(torch.bfloat16)
    else:
        h = torch.load(os.path.join(a.out, "h.pt")).pin_memory()
        assert int(open(os.path.join(a.out, "h.layer")).read()) == first
    log(f"start at layer {first}: {a.windows} windows x {a.seq}")
    E, I, d = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    for li in range(first, nl):
        t0 = time.time()
        L = m.layers[li]
        st.fill(L, li, dev, experts=False)
        gu, dn = st.expert_weights(li, dev)
        M.set_param(L, "mlp.gu.weight", gu); M.set_param(L, "mlp.dn.weight", dn)
        dense = [n for n in DENSE if _has(L, n)]
        S = {n: Stats(1, _get(L, n).in_features, dev) for n in dense}
        Sgu, Sdn = Stats(E, d, dev, a.chunk), Stats(E, I, dev)
        hooks = [_get(L, n).register_forward_pre_hook(
            (lambda nm: lambda mod, inp: S[nm].add(inp[0].reshape(-1, inp[0].shape[-1]), [0, inp[0].numel() // inp[0].shape[-1]]))(n))
            for n in dense]
        box = {}

        def cap(xs, offsets, order, w):
            box["w"] = w.reshape(-1)[order]; box["o"] = offsets.tolist()
            Sgu.add(xs, box["o"], box["w"])

        def cap_dn(x, offsets):
            Sdn.add(x, box["o"], box["w"])
        L.mlp.capture, L.mlp.capture_dn = cap, cap_dn
        with torch.no_grad():
            for b0 in range(0, a.windows, a.mb):
                L(h[b0:b0 + a.mb].to(dev, non_blocking=True), pe, ids_d[b0:b0 + a.mb])
        for hk in hooks:
            hk.remove()
        L.mlp.capture = L.mlp.capture_dn = None
        t1 = time.time()
        sd, errs = {}, {}
        # routed experts: gate+up (by chunks, each chunk's covariance freed once used), then down
        M.set_param(L, "mlp.gu.weight", torch.empty(0, device="meta")); del gu
        Wgu_h, Wdn_h = None, None
        What_gu = torch.empty(E, 2 * I, d, dtype=torch.bfloat16, device=dev)
        bias_gu = torch.empty(E, 2 * I, device=dev)
        parts, eg = [], {"main": [], "adapter": []}
        Sl = st.layers[li]
        for e0 in range(0, E, a.chunk):
            e1 = min(e0 + a.chunk, E)
            W = IO.Store._deq(Sl["gu"][e0:e1], Sl.get("sgu")[e0:e1] if "sgu" in Sl else None, dev, torch.float32)
            Hc, mu, few = Sgu.centred(e0, e1)
            Sgu.H[e0 // a.chunk] = None
            damp = torch.where(Sgu.cnt[e0:e1] < 2 * d, 0.05, 0.01)
            o, Wh, er = convert_group(W, Hc, mu, a.rank, float(damp.max()))
            What_gu[e0:e1] = Wh.to(torch.bfloat16)
            bias_gu[e0:e1] = torch.einsum("enk,ek->en", W - Wh, mu)
            parts.append(o); eg["main"].append(er["main"]); eg["adapter"].append(er["adapter"])
            del W, Hc, Wh
        sd["mlp.gu"] = _cat_state(parts, pack3)
        errs["mlp.gu"] = _err_summary(eg, Sgu.cnt)
        W = dn.float(); M.set_param(L, "mlp.dn.weight", torch.empty(0, device="meta")); del dn
        Hc, mu, few = Sdn.centred(0, E)
        o, Wh, er = convert_group(W, Hc, mu, a.rank, 0.01)
        sd["mlp.dn"] = to_state(o, pack3)
        errs["mlp.dn"] = _err_summary({"main": [er["main"]], "adapter": [er["adapter"]]}, Sdn.cnt)
        What_dn = Wh.to(torch.bfloat16); bias_dn = torch.einsum("enk,ek->en", W - Wh, mu)
        del W, Hc, Wh, Sgu, Sdn
        for n in dense:
            lin = _get(L, n)
            W = lin.weight.float()[None]
            Hc, mu, _ = S[n].centred(0, 1)
            o, Wh, er = convert_group(W, Hc, mu, a.rank, 0.01)
            sd[n] = to_state(o, pack3)
            errs[n] = {"main": float(er["main"][0]), "adapter": float(er["adapter"][0])}
            b = torch.einsum("enk,ek->en", W - Wh, mu)[0]
            lin.weight = nn.Parameter(Wh[0].to(lin.weight.dtype), requires_grad=False)
            lin.bias = nn.Parameter(b.to(lin.weight.dtype), requires_grad=False)
            del S[n]
        t2 = time.time()
        # propagate the calibration windows through the converted (fake-quantised) layer
        M.set_param(L, "mlp.gu.weight", What_gu); M.set_param(L, "mlp.dn.weight", What_dn)
        L.mlp.gu.bias, L.mlp.dn.bias = bias_gu.to(torch.bfloat16), bias_dn.to(torch.bfloat16)
        with torch.no_grad():
            for b0 in range(0, a.windows, a.mb):
                h[b0:b0 + a.mb] = L(h[b0:b0 + a.mb].to(dev, non_blocking=True), pe, ids_d[b0:b0 + a.mb]).to("cpu")
        L.mlp.gu.bias = L.mlp.dn.bias = None
        for n in dense:
            _get(L, n).bias = None
        st.unfill(L)
        del What_gu, What_dn
        torch.cuda.empty_cache()
        torch.save(sd, os.path.join(a.out, f"L{li}.pt.tmp")); os.replace(os.path.join(a.out, f"L{li}.pt.tmp"),
                                                                          os.path.join(a.out, f"L{li}.pt"))
        json.dump(errs, open(os.path.join(a.out, f"L{li}.json"), "w"), indent=1)
        if li + 1 < nl:
            torch.save(h, os.path.join(a.out, "h.pt.tmp")); os.replace(os.path.join(a.out, "h.pt.tmp"), os.path.join(a.out, "h.pt"))
            open(os.path.join(a.out, "h.layer"), "w").write(str(li + 1))
        rms = float(h[:4].float().pow(2).mean().sqrt())
        log(f"layer {li} ({L.kind}): stats {t1 - t0:.0f}s, rounding {t2 - t1:.0f}s, total {time.time() - t0:.0f}s; "
            f"rel out err gu {errs['mlp.gu']['main']:.4f}->{errs['mlp.gu']['adapter']:.4f} "
            f"dn {errs['mlp.dn']['main']:.4f}->{errs['mlp.dn']['adapter']:.4f} "
            + " ".join(f"{n.split('.')[-1]} {v['main']:.3f}->{v['adapter']:.3f}" for n, v in errs.items() if n in DENSE)
            + f"; h rms {rms:.3f}; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()
    open(os.path.join(a.out, "done"), "w").close()
    log("conversion done")


def _has(L, n):
    try:
        L.get_submodule(n); return True
    except AttributeError:
        return False


def _get(L, n):
    return L.get_submodule(n)


def _cat_state(parts, pack3):
    o = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return to_state(o, pack3)


def _err_summary(eg, cnt):
    em, ea = torch.cat(eg["main"]), torch.cat(eg["adapter"])
    w = cnt.float() / cnt.float().sum()                       # usage-weighted mean over experts
    return {"main": float((em * w).sum()), "adapter": float((ea * w).sum()), "main_median": float(em.median()),
            "few_experts": int((cnt < 64).sum())}


if __name__ == "__main__":
    main()
