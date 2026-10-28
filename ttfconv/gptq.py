"""Rounding a float weight onto the ternary grid W ~ a_i c_j T_ij: ALS (row / column scales by alternating least
squares), GPTQ (Frantar et al. 2022) onto the ternary grid under the layer's input second moment, and the rank-R fit of
the rounding error for the float adapters. Moved here from ttf-core (ttf/hf_init.py, d5f6256) on 28 Oct 2026 when the
conversion left the core; the code is unchanged except mag_svd_fit, which returns its factors instead of parking them
on the layer."""
import torch


@torch.no_grad()
def _row_grid(W, grid=torch.linspace(0.3, 2.0, 35)):
    """per-row scale a_i and trits T minimising |W_i - a_i T_i|^2: for each candidate threshold-scale s = g mean|W_i|,
    T = round(clip(W / s)) and the best magnitude a = <W, T> / <T, T>; keep the best per row"""
    m = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full((W.shape[0], 1), float("inf"), device=W.device)
    best_s = m.clone()
    for g in grid.tolist():
        T = (W / (g * m)).round_().clamp_(-1, 1)
        nt = (T * T).sum(1, keepdim=True).clamp_min(1)
        a = (W * T).sum(1, keepdim=True) / nt
        err = (W * W).sum(1, keepdim=True) - a * a * nt          # |W - a T|^2 at the optimal a
        better = err < best_err
        best_err = torch.where(better, err, best_err); best_s = torch.where(better, g * m, best_s)
    T = (W / best_s).round_().clamp_(-1, 1)
    a = ((W * T).sum(1, keepdim=True) / (T * T).sum(1, keepdim=True).clamp_min(1)).clamp_min(1e-12)
    return a.squeeze(1), T


@torch.no_grad()
def ternarize(W, method, iters=4):
    """W [N, K] -> (T int8, row a [N], col c [K]) with W ~ a_i c_j T_ij"""
    N, K = W.shape
    c = torch.ones(K, device=W.device)
    if method == "absmean":
        g = W.abs().mean().clamp_min(1e-12)
        T = (W / g).round_().clamp_(-1, 1)
        return T.to(torch.int8), torch.full((N,), float(g), device=W.device), c
    a, T = _row_grid(W)
    if method == "als":
        for _ in range(iters):
            num = (W * T * a[:, None]).sum(0); den = (T * T * (a * a)[:, None]).sum(0)
            c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-3)
            c = c / c.mean()
            a, T = _row_grid(W / c[None, :])
    return T.to(torch.int8), a, c


def _rel_err(W, T, a, c):
    return float((W - a[:, None] * c[None, :] * T.float()).norm() / W.norm())


def _out_err(W, T, a, c, H, mu=None):
    """output-space relative error tr(E H E^T) / tr(W H W^T), E = W - a c T, H = X^T X of the layer's inputs; mu: a
    centred layer (gptqc: x - mu goes through the trits, W mu is added back exactly), whose error is tr(E Hc E^T),
    Hc = H - mu mu^T (the raw H counted the mean's error, which the layer does not make: ALS > 1 at 27B layer 0)"""
    E = W - a[:, None] * c[None, :] * T.float()
    Hn = H if mu is None else H - torch.outer(mu, mu)
    return float(((E @ Hn) * E).sum() / ((W @ H) * W).sum())


@torch.no_grad()
def ternarize_gptq(W, H, a, c, block=128, damp=0.01, act_order=True, refits=2):
    """GPTQ (Frantar et al. 2022) onto the ternary grid {-a_i, 0, a_i} c_j: columns are rounded one at a time and each
    one's rounding error is pushed onto the not-yet-rounded columns through the inverse input second moment, so the
    trits minimise the layer's output error |X (W - a c T)^T| instead of |W - a c T|. The column scale c comes from
    ALS (folded into the inputs), the row magnitude a is refitted in output space after each pass."""
    N, K = W.shape
    Wc = W / c[None, :]                                  # fold c into the input: X' = X c, H' = c H c
    Hc = H.to(W.device) * c[:, None] * c[None, :]
    dead = torch.diagonal(Hc) <= 0
    Hc[dead, dead] = 1.0
    Wc[:, dead] = 0.0
    Hc += damp * torch.diagonal(Hc).mean() * torch.eye(K, device=W.device)
    perm = torch.argsort(torch.diagonal(Hc), descending=True) if act_order else torch.arange(K, device=W.device)
    inv = torch.argsort(perm)
    Wp, Hp = Wc[:, perm], Hc[perm][:, perm]
    L = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)          # GPTQ's upper factor of H^-1
    del L, Hinv
    for _ in range(refits):
        Q = Wp.clone()
        T = torch.zeros(N, K, device=W.device)
        ai = a[:, None]
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            Wb = Q[:, i1:i2].clone(); Ub = U[i1:i2, i1:i2]; E = torch.zeros_like(Wb)
            for j in range(i2 - i1):
                w = Wb[:, j]
                t = (w / ai[:, 0]).round().clamp(-1, 1)
                T[:, i1 + j] = t
                e = (w - ai[:, 0] * t) / Ub[j, j]
                Wb[:, j:] -= e[:, None] * Ub[j, j:][None, :]
                E[:, j] = e
            Q[:, i2:] -= E @ U[i1:i2, i2:]
        # output-space least-squares magnitude per row: a_i = <W_i H, T_i> / <T_i H, T_i>
        TH = T @ Hp
        a = ((Wp @ Hp) * T).sum(1) / (TH * T).sum(1).clamp_min(1e-30)
        a = torch.where(a > 0, a, torch.full_like(a, 1e-3))
        del TH, Q
    return T[:, inv].to(torch.int8), a, c


@torch.no_grad()
def mag_svd_fit(W, T, a, c, beta, Hc, r):
    """--adapter-init svd R: the rounding error, moved into the layer's float adapter instead of dropped. The layer computes
    y = (a / beta) (beta T + A B^T) x' with x' = c o (x - mu), so the adapter that restores W exactly would be
    G = diag(beta / a) (W - a c T) diag(1 / c); of G, the rank-R part that matters most for the output is kept: the
    best rank-R fit in the input covariance's norm, |(G - A B^T) L|_F with L L^T = H' (H' the covariance of x'; Hc the
    covariance of the layer's input as GPTQ rounded against: centred for gptqc), = truncated SVD of G L, then
    B^T = V_R^T L^-1. Returns ((A, B) on the host, the share of the output error |G L|_F^2 it removes). The K x K
    factors are float64; G and G L (N x K) float32, built in place, and their top R by a randomized SVD (R + 16 columns, 8 power iterations): a full float64 SVD of G L took ~4 x N K x 8 bytes
    beside the 27B streaming conversion's 21.5 GiB and ran out of a 24 GB card (27 Oct)"""
    cd = c.double().clamp_min(1e-12)
    Hp = cd[:, None] * Hc.double() * cd[None, :]
    Hp.diagonal().add_(1e-4 * Hp.diagonal().mean() + 1e-30)
    L, info = torch.linalg.cholesky_ex(Hp)
    if int(info) == 0:
        Linv_t = None    # B = L^-T V_R: (x' @ B) = V_R^T L^-1 x'
    else:                # numerically singular covariance: the symmetric square root, eigenvalues clamped
        lam, Q = torch.linalg.eigh(Hp)
        lam = lam.clamp_min(1e-6 * float(lam.max()))
        L, Linv_t = (Q * lam.sqrt()) @ Q.T, (Q / lam.sqrt()) @ Q.T
        del lam, Q
    del Hp
    G = T.float().mul_(c.float()[None, :]).mul_(-a.float()[:, None]).add_(W.float())   # W - a c T
    G.div_(c.float().clamp_min(1e-12)[None, :]).mul_((beta / a.float().clamp_min(1e-12))[:, None])
    M = G @ L.float(); del G
    tot = float(torch.linalg.vector_norm(M)) ** 2
    with torch.random.fork_rng(devices=[M.device] if M.is_cuda else []):   # its own draws: the global RNG (stochastic
        torch.manual_seed(0)                                               # rounding, later options) is left as it was
        U, S, V = torch.svd_lowrank(M, q=min(r + 16, *M.shape), niter=8)
    del M
    Vr = V[:, :r].double()
    B = torch.linalg.solve_triangular(L.T, Vr, upper=True) if Linv_t is None else Linv_t @ Vr
    A = U[:, :r] * S[:r]
    # (the factors wait on the host until the adapters exist: 400 layers' factors on the GPU, ~0.6 GB at 27B rank 16,
    # took the streaming conversion out of a 24 GB card at layer 33, 27 Oct)
    return (A.float().cpu(), B.float().cpu()), float((S[:r].double() ** 2).sum() / max(tot, 1e-300))
