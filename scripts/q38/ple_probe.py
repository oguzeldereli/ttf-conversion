"""ple_probe.py: how compressible Qwen3.8-Flash-Next's PLE n-gram table
is, measured in the coordinates the model reads it through. Read-only, CPU.

The table [rows, 160] per head (16 heads, ~20 M rows each, FP8 + scale) enters the model only through e -> key_proj(e)
(10240 wide, then a grouped RMS norm: only its direction per stream matters) and value_proj(e) (2560 wide, linear).
So a compression x -> x^ of a head's rows costs ||W_h (x - x^)||, W_h = that head's 160 columns of key_proj / value_proj.

Per head, on rows sampled two ways:
  "text":    the rows real text hits (the model's own hashing on --nwin windows of --data), weighted by use;
  "uniform": rows drawn uniformly (every row equal; rare rows dominate a hashed table).
Reports, for rank r in --ranks:
  raw       share of sum ||x||^2 kept by the top-r PCA of the rows;
  value     share of sum ||W_v,h x||^2 kept by the best rank-r in the value metric (PCA of W_v,h-whitened rows);
  key       the same in the key metric;
  int4      relative error of 4-bit (group 32 absmax, symmetric) on the full rows, in each metric;
  r+int4    rank r in the value metric, then the row factors in int4: total relative error in the value / key metric.
  size_gb   the whole table at that setting (rows x r x 0.5 B + group scales + bases).
  python scripts/q38/ple_probe.py --model DIR [--data TOKENS.bin --u32] [--nwin 64] [--uniform 200000] [--out ple_probe.json]
  python scripts/q38/ple_probe.py --fake        (logic check on a random low-rank table)"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def q4(X, g=32):
    """symmetric int4 with one absmax scale per group of g columns; returns the dequantised X"""
    n, d = X.shape
    pad = (-d) % g
    Xp = torch.nn.functional.pad(X, (0, pad)).view(n, -1, g)
    s = Xp.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.0
    return ((Xp / s).round().clamp(-8, 7) * s).view(n, -1)[:, :d]


def metric_basis(W):
    """W [out, 160] -> (P, Pinv): x -> P x is the whitened coordinate in which ||W x|| = ||P x||"""
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    P = torch.diag(S) @ Vh                                  # [160, 160] (out >= 160)
    return P, torch.linalg.pinv(P)


def head_report(X, w, Wv, Wk, ranks):
    """X [n, 160] rows (float64), w [n] weights; Wv / Wk [out, 160]"""
    w = w / w.sum()
    res = {}
    Pv, Pvi = metric_basis(Wv)
    Pk, _ = metric_basis(Wk)

    def err(D, P):          # weighted relative error sum w ||P d||^2 / sum w ||P x||^2
        return float((w[:, None] * (D @ P.T) ** 2).sum() / (w[:, None] * (X @ P.T) ** 2).sum())
    I = torch.eye(X.shape[1], dtype=X.dtype)
    C = (X * w[:, None]).T @ X
    ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
    Xv = X @ Pv.T
    Cv = (Xv * w[:, None]).T @ Xv
    evv, Vv = torch.linalg.eigh(Cv); evv, Vv = evv.flip(0).clamp_min(0), Vv.flip(1)
    Xk = X @ Pk.T
    evk = torch.linalg.eigvalsh((Xk * w[:, None]).T @ Xk).flip(0).clamp_min(0)
    Xq = q4(X.float()).double()
    res["int4"] = {"raw": err(X - Xq, I), "value": err(X - Xq, Pv), "key": err(X - Xq, Pk)}
    res["ranks"] = {}
    for r in ranks:
        B = Vv[:, :r]                                       # value-metric basis (whitened coords)
        A = Xv @ B                                          # [n, r] row factors
        Xr = (A @ B.T) @ Pvi.T                              # back to table coordinates
        Aq = q4(A.float()).double()
        Xrq = (Aq @ B.T) @ Pvi.T
        res["ranks"][r] = {"raw_kept": float(ev[:r].sum() / ev.sum()), "value_kept": float(evv[:r].sum() / evv.sum()),
                           "key_kept": float(evk[:r].sum() / evk.sum()),
                           "rank_err_value": err(X - Xr, Pv), "rank_err_key": err(X - Xr, Pk),
                           "rank_int4_err_value": err(X - Xrq, Pv), "rank_int4_err_key": err(X - Xrq, Pk)}
    return res


def size_gb(rows, dim, heads, r=None, bits=4, g=32):
    if r is None:
        return rows * dim * bits / 8 / 1e9 + rows * math.ceil(dim / g) * 2 / 1e9
    return rows * r * bits / 8 / 1e9 + rows * math.ceil(r / g) * 2 / 1e9 + heads * (r * dim + dim * dim) * 4 / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--data", default=""); ap.add_argument("--u32", action="store_true")
    ap.add_argument("--layer", type=int, default=-1); ap.add_argument("--nwin", type=int, default=64)
    ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--uniform", type=int, default=200000)
    ap.add_argument("--ranks", default="8,16,32,48,64,96"); ap.add_argument("--out", default="ple_probe.json")
    ap.add_argument("--fake", action="store_true"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ranks = [int(x) for x in a.ranks.split(",")]
    g = torch.Generator().manual_seed(0)
    if a.fake:      # a table of rank 24 + small noise; value_proj / key_proj random: rank 32 should keep ~all
        heads, dim, rows_h = 4, 160, 20000
        Z = torch.randn(heads * rows_h, 24, generator=g) @ torch.randn(24, dim, generator=g) + 0.01 * torch.randn(heads * rows_h, dim, generator=g)
        get = lambda idx: Z[idx]
        Wv = torch.randn(2560 // 16 * heads, dim * heads, generator=g); Wk = torch.randn(4 * Wv.shape[0], dim * heads, generator=g)
        offs, sizes = [h * rows_h for h in range(heads)], [rows_h] * heads
        text = {h: (torch.randint(0, rows_h, (50000,), generator=g) ** 1).clamp_max(rows_h - 1) for h in range(heads)}
        total = heads * rows_h
    else:
        from ttf.q38 import model as QM
        cfg = QM.text_config(a.model)
        # cfg.ple_layer_ids is 1-based ([2]); the table / key_proj / value_proj sit at layers.{id - 1}
        lid = (a.layer + 1) if a.layer >= 0 else cfg.ple_layer_ids[0]
        li = lid - 1
        ng = QM.NGramEmbedding(cfg, cfg.ple_layer_ids.index(lid))
        heads, dim = ng.heads, ng.dim
        tab, scale = QM.ngram_table_memmap(a.model, li)
        rd = QM.Reader(a.model)
        p = f"{QM.PFX}layers.{li}.ple."
        Wk, Wv = rd.weight(p + "key_proj").float(), rd.weight(p + "value_proj").float()
        offs, sizes, total = ng.offs.tolist(), ng.sizes.tolist(), ng.total

        def get(idx):
            x = tab[idx].float()
            if scale is not None:
                s = scale.float()
                x = x * (s.reshape(()) if s.numel() == 1 else s[idx][:, None] if s.numel() >= total else s.reshape(-1))
            return x
        text = {}
        if a.data:
            d = np.memmap(a.data, dtype=np.uint32 if a.u32 else np.uint16, mode="r")
            st = np.random.default_rng(1).integers(0, len(d) - a.seq - 1, a.nwin)
            ids = torch.from_numpy(np.stack([d[s:s + a.seq].astype(np.int64) for s in st]))
            idx = ng.indices(ids).reshape(-1, heads)                      # [tokens, heads] global rows
            text = {h: idx[:, h] - offs[h] for h in range(heads)}
    out = {"heads": heads, "dim": dim, "rows": total, "ranks": ranks, "per_head": {}, "time": time.ctime()}
    t0 = time.time()
    for h in range(heads):
        Wv_h, Wk_h = Wv[:, h * dim:(h + 1) * dim].double(), Wk[:, h * dim:(h + 1) * dim].double()
        rep = {}
        for kind in ("text", "uniform"):
            if kind == "text":
                if h not in text:
                    continue
                r_, cnt = torch.unique(text[h], return_counts=True)
                rows, w = r_, cnt.double()
                rep["text_distinct_rows"] = int(len(r_)); rep["text_hits"] = int(cnt.sum())
            else:
                rows = torch.randint(0, sizes[h], (min(a.uniform, sizes[h]),), generator=g)
                w = torch.ones(len(rows), dtype=torch.float64)
            X = get(rows + offs[h]).double()
            rep[kind] = head_report(X, w, Wv_h, Wk_h, ranks)
        out["per_head"][h] = rep
        print(f"head {h}: {time.time() - t0:.0f} s; text int4 value err "
              f"{rep.get('text', rep['uniform'])['int4']['value']:.2e}; rank 32 value kept "
              f"{rep.get('text', rep['uniform'])['ranks'][32]['value_kept']:.3f}", flush=True)
    # summary: means over heads, text-weighted where available
    def mean(f):
        v = [f(rep.get("text", rep["uniform"])) for rep in out["per_head"].values()]
        return sum(v) / len(v)
    out["summary"] = {"fp8_gb": round(total * dim / 1e9, 1), "int4_gb": round(size_gb(total, dim, heads), 1),
                      "int4_err_value": mean(lambda r: r["int4"]["value"]), "int4_err_key": mean(lambda r: r["int4"]["key"]),
                      "ranks": {r: {"size_gb": round(size_gb(total, dim, heads, r), 2),
                                    "value_kept": mean(lambda x: x["ranks"][r]["value_kept"]),
                                    "key_kept": mean(lambda x: x["ranks"][r]["key_kept"]),
                                    "raw_kept": mean(lambda x: x["ranks"][r]["raw_kept"]),
                                    "rank_int4_err_value": mean(lambda x: x["ranks"][r]["rank_int4_err_value"]),
                                    "rank_int4_err_key": mean(lambda x: x["ranks"][r]["rank_int4_err_key"])} for r in ranks}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
