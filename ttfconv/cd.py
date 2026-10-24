"""coordinate descent on the trits after GPTQ (QuantEase-style): per column, every row tries the
other two trits and keeps the one that lowers the output error tr(R H R^T); row scales refitted in output space between
sweeps; at most `sweeps` sweeps, stops when a sweep gains < tol relative.
  --ext ttfconv.cd[:sweeps=8,tol=1e-4]
"""
import numpy as np
import torch
from .spec import parse


@torch.no_grad()
def cd_refine(W, H, T, a, c, sweeps=8, tol=1e-4):
    """coordinate descent on the trits of W ~ a_i T_ij c_j under the output error tr(R H R^T) (R = W/c - a T in the
    column-scaled space, H scaled by c); returns T (int8), a, the relative error per sweep"""
    Wc = W / c[None, :]
    Hc = H * c[:, None] * c[None, :]
    d = torch.diagonal(Hc).clone()
    T = T.float().clone()
    R = Wc - a[:, None] * T
    RH = R @ Hc
    E0 = float((RH * R).sum()); WH = Wc @ Hc; Enorm = float((WH * Wc).sum()); hist = [E0 / Enorm]
    cand = torch.tensor([-1.0, 0.0, 1.0], device=W.device)
    for sw in range(sweeps):
        for j in range(W.shape[1]):
            t0 = T[:, j]
            dl = a[:, None] * (cand[None, :] - t0[:, None])                  # [N, 3]
            dE = -2 * dl * RH[:, j:j + 1] + dl * dl * d[j]
            k = dE.argmin(1)
            dlt = dl.gather(1, k[:, None])[:, 0]
            ch = dlt != 0
            if ch.any():
                T[:, j] = cand[k]
                R[:, j] -= dlt
                RH -= dlt[:, None] * Hc[j][None, :]
        TH = T @ Hc                                                          # row scales refitted in output space
        an = (WH * T).sum(1) / (TH * T).sum(1).clamp_min(1e-30)
        a = torch.where(an > 0, an, a)
        R = Wc - a[:, None] * T; RH = R @ Hc
        hist.append(float((RH * R).sum()) / Enorm)
        if hist[-2] - hist[-1] < tol * hist[-2]:
            break
    return T.to(torch.int8), a, hist


class CD:
    def __init__(self, spec):
        o = parse(spec, {"sweeps": "8", "tol": "1e-4"})
        self.sweeps, self.tol = int(o["sweeps"]), float(o["tol"])
        self.hist = []

    def refine(self, name, W, Hc, T, a, c):
        T, a, h = cd_refine(W, Hc, T, a, c, self.sweeps, self.tol)
        self.hist.append(h)
        return T, a, c

    def end(self, model, log):
        if self.hist:
            log(f"ttfconv.cd: mean relative output error {np.mean([h[0] for h in self.hist]):.4f} after GPTQ -> "
                f"{np.mean([h[-1] for h in self.hist]):.4f}, {np.mean([len(h) - 1 for h in self.hist]):.1f} sweeps on average")


def ttf_ext(spec):
    return CD(spec)
