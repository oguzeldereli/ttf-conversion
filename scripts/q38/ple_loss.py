"""ple_loss.py: the validation loss of a Q38 student (or its
conversion start) with its PLE n-gram table replaced by compressed versions, nothing else changed.

Run exactly as python -m ttf.q38.train with the run's own flags (model, conv, kd cache, data, --resume + --out for a checkpoint),
plus --ple-variants; instead of training it evaluates the val windows once per variant. The compressed tables are
views: each row is read from the original table and compressed on the fly (projected on the rank-r basis, int4), the
same numbers a stored compressed table would give, so nothing over the 51 GB table is precomputed.
Variants (comma list):
  orig            the FP8 table (the reference)
  int4            4-bit, group 32 absmax, symmetric
  rR / rRq        rank R per head in the value_proj metric (bases from rows real training text hits), / + int4 row factors
  krR / krRq      the same in the key_proj metric
  off             the module's table zeroed (the PLE output is then 0)
  intB[gG]        B-bit symmetric group quant (group G, default 32; int2 = {-1,0,1} x absmax, packs at 2 bits)
  foldK           hash folded K-fold: head row j -> j mod ceil(size/K) (a code change at load), the merged row the
                  hit-count-weighted mean of its K source rows (counts from --ple-count-windows training windows,
                  + 0.1 prior); table K x smaller
  pruneN          rows hit < N times in the count windows zeroed (shipping only the kept rows + a row index)
  hK              keep the first K of each n-gram order's heads, zero the rest
  A+intB          any of the above (orig/foldK/pruneN/hK) then B-bit quant, e.g. fold2+int4
  python scripts/q38/ple_loss.py <ttf.q38.train flags> --ple-variants orig,int4,r64q,r32q,r16q,off [--ple-basis-windows 64] \
      [--ple-out ple_loss.json]
(The basis fit reads ~--ple-basis-windows x 2048 x 16 table rows from the training stream, not the val windows.)"""
import json, math, os, re, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttf.q38 import train as QT
from ttf.q38 import model as QM
from ple_probe import metric_basis


def _pop(flag, default, cast=str):
    if flag in sys.argv:
        i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]
        return cast(v)
    return default


VARIANTS = _pop("--ple-variants", "orig,int4,r64q,r32q,r16q,off").split(",")
NBW = _pop("--ple-basis-windows", 64, int)
OUT = _pop("--ple-out", "ple_loss.json")
NCW = _pop("--ple-count-windows", 1024, int)


def qb(X, bits=4, g=32, kind="int"):
    """B-bit with one absmax scale per group of g columns; dequantised. kind "int": symmetric with zero (2^(B-1) - 1
    levels each side; int2 = -1, 0, 1); kind "u": mid-rise, all 2^B levels used, no zero (u2 = +-0.5, +-1.5)"""
    n, d = X.shape
    pad = (-d) % g
    Xp = torch.nn.functional.pad(X, (0, pad)).view(n, -1, g)
    if kind == "u":
        h = 2 ** (bits - 1)                                  # levels (k + 0.5) s, k = -h .. h - 1
        s = Xp.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (h - 0.5)
        Q = ((Xp / s - 0.5).round().clamp(-h, h - 1) + 0.5) * s
    else:
        qm = max(2 ** (bits - 1) - 1, 1)
        s = Xp.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qm
        Q = (Xp / s).round().clamp(-qm, qm) * s
    return Q.view(n, -1)[:, :d].contiguous()


def parse_q(op):
    """"intB[gG]" / "uB[gG]" -> (bits, group, kind)"""
    m = re.fullmatch(r"(int|u)(\d)(?:g(\d+))?", op)
    return (int(m.group(2)), int(m.group(3) or 32), m.group(1)) if m else None


def kmeans(X, w, K, iters=15, seed=0, chunk=None):
    """weighted k-means of X [n, D] (float32) -> centroids [K, D]"""
    chunk = chunk or max(1024, (1 << 28) // (4 * K))
    g = torch.Generator().manual_seed(seed)
    C = X[torch.multinomial(w, K, replacement=len(X) < K, generator=g)].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(X[i:i + chunk], C).argmin(1) for i in range(0, len(X), chunk)])
        S = torch.zeros_like(C).index_add_(0, a, X * w[:, None]); W = torch.zeros(K).index_add_(0, a, w)
        ok = W > 0
        C[ok] = S[ok] / W[ok, None]
    return C


def fit_pq(L, view, args, D, K, nmax=None):
    """per head: (P, Pinv, codebooks [160 / D, K, D]) - product quantisation in the value_proj metric, fitted on the
    use-weighted rows real training text hits (--ple-basis-windows windows, at most nmax distinct rows per head)"""
    ng = L.ple.ple_embedding
    W = L.ple.value_proj.weight.detach().float().cpu()
    d = np.memmap(args.data, dtype=np.uint32 if args.data_u32 else np.uint16, mode="r")
    st = np.random.default_rng(5).integers(0, len(d) - args.seq - 1, NBW)
    ids = torch.from_numpy(np.stack([d[s:s + args.seq].astype(np.int64) for s in st]))
    idx = ng.indices(ids.to(ng.mult.device)).reshape(-1, ng.heads).cpu()
    out = {}
    nmax = nmax or (100000 if K <= 256 else 40000)
    for h in range(ng.heads):
        r_, cnt = torch.unique(idx[:, h], return_counts=True)
        if len(r_) > nmax:
            keep = torch.randperm(len(r_), generator=torch.Generator().manual_seed(h))[:nmax]
            r_, cnt = r_[keep], cnt[keep]
        P, Pi = metric_basis(W[:, h * ng.dim:(h + 1) * ng.dim])
        Xm = (view.raw(r_).double() @ P.T).float()
        w = cnt.float()
        cb = torch.stack([kmeans(Xm[:, j * D:(j + 1) * D].contiguous(), w, K, seed=h * 1000 + j)
                          for j in range(ng.dim // D)])
        out[h] = (P, Pi, cb)
    return out


def row_counts(ng, args, nwin, seed=7, vi=None, log=None):
    """hits per global table row over nwin training windows (int32 [total]); with vi (val lookups) and a log list:
    appends (windows, rows hit, share of val lookups on rows not hit yet) at 256, 1024, 4096, ... windows"""
    cnt = torch.zeros(ng.total, dtype=torch.int32)
    d = np.memmap(args.data, dtype=np.uint32 if args.data_u32 else np.uint16, mode="r")
    st = np.random.default_rng(seed).integers(0, len(d) - args.seq - 1, nwin)
    for c in range(0, nwin, 32):
        ids = torch.from_numpy(np.stack([d[s:s + args.seq].astype(np.int64) for s in st[c:c + 32]]))
        idx = ng.indices(ids.to(ng.mult.device)).reshape(-1).cpu()
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
        nw = c + 32
        if log is not None and vi is not None and (nw >= nwin or (nw >= 256 and (nw & (nw - 1)) == 0 and (nw.bit_length() - 1) % 2 == 0)):
            log.append({"windows": nw, "rows_hit": int((cnt > 0).sum()), "val_unseen": float((cnt[vi] == 0).float().mean())})
            print(f"coverage: {log[-1]}", flush=True)
    return cnt


class View:
    """table rows -> compressed rows (float32, scale applied). mode: the structural op (orig / off / foldK / pruneN / hK
    / r / rq), q: (bits, group) quant applied after it, or None"""

    def __init__(self, tab, scale, offs, total, mode, bases=None, q=None, sizes=None, counts=None, hpn=8):
        self.tab, self.scale, self.mode, self.bases, self.q = tab, scale, mode, bases, q
        self.offs = torch.tensor(list(offs) + [total])
        self.sizes, self.counts, self.hpn = sizes, counts, hpn
        self.keep_n = 0                                      # pruneN+X: rows hit < N zeroed after X
        self._s = None

    def raw(self, rows):
        if torch.is_tensor(self.tab):                       # (a resident table may sit on the GPU)
            x = self.tab[rows.to(self.tab.device)].float().cpu()
        else:
            x = self.tab[rows].float()
        if self.scale is not None:
            if self._s is None:
                self._s = self.scale.float().cpu()
            s = self._s
            x = x * (s.reshape(()) if s.numel() == 1 else s[rows][:, None] if s.numel() >= int(self.offs[-1]) else s.reshape(-1))
        return x

    def struct(self, rows):
        m = self.mode
        if m == "orig":
            return self.raw(rows)
        if m == "off":
            return torch.zeros_like(self.raw(rows))
        h = torch.searchsorted(self.offs, rows, right=True) - 1
        if m.startswith("fold"):
            K = int(m[4:])
            j = rows - self.offs[h]
            size = self.sizes[h]
            M = (size + K - 1) // K
            mm = j % M
            acc, ws = None, torch.zeros(len(rows))
            for t in range(K):
                jj = mm + t * M
                ok = jj < size
                rr = torch.minimum(jj, size - 1) + self.offs[h]
                w = (self.counts[rr].float() + 0.1) * ok
                x = self.raw(rr) * w[:, None]
                acc = x if acc is None else acc + x
                ws += w
            return acc / ws[:, None]
        if m.startswith("prune"):
            N = int(m[5:])
            return self.raw(rows) * (self.counts[rows] >= N).float()[:, None]
        if m.startswith("h") and m[1:].isdigit():
            K = int(m[1:])
            return self.raw(rows) * ((h % self.hpn) < K).float()[:, None]
        if m == "pq":
            x = self.raw(rows)
            out = torch.empty_like(x)
            for hh in torch.unique(h).tolist():
                sel = h == hh
                P, Pi, cb = self.bases[hh]
                Xm = (x[sel].double() @ P.T).float()
                nS, K, D = cb.shape
                R = torch.empty_like(Xm)
                for j in range(nS):
                    sub = Xm[:, j * D:(j + 1) * D]
                    R[:, j * D:(j + 1) * D] = cb[j][torch.cdist(sub, cb[j]).argmin(1)]
                out[sel] = (R.double() @ Pi.T).float()
            return out
        x = self.raw(rows)                                       # r / rq: rank-R per head
        out = torch.empty_like(x)
        for hh in torch.unique(h).tolist():
            sel = h == hh
            P, Pi, B = self.bases[hh]
            A = x[sel].double() @ P.T @ B
            if m == "rq":
                A = qb(A.float(), 4).double()
            out[sel] = (A @ B.T @ Pi.T).float()
        return out

    def __getitem__(self, rows):
        rows = rows.cpu().long()
        x = self.struct(rows)
        if self.keep_n:
            x = x * (self.counts[rows] >= self.keep_n).float()[:, None]
        if self.q is not None:
            x = qb(x, *self.q)
        return x


def est_gb(v, ng, counts):
    """stored size of the whole table for variant v (values + fp16 group scales; prune: + 4 B row index)"""
    ops = v.split("+")
    pn = 0
    if len(ops) > 1 and ops[0].startswith("prune") and not parse_q(ops[1]):
        pn, ops = int(ops[0][5:]), ops[1:]
    st, q = ops[0], (parse_q(ops[-1]) if len(ops) > 1 else None)
    if parse_q(st):
        st, q = "orig", parse_q(st)
    rows, dim, extra = ng.total, ng.dim, 0.0
    if pn and counts is not None:                            # kept rows + a bitmap with rank blocks (~1.25 bit/row)
        rows = int((counts >= pn).sum()); extra = ng.total * 1.25 / 8
    if st == "off":
        return 0.0
    if st.startswith("fold"):
        rows = sum(-(-int(s) // int(st[4:])) for s in ng.sizes.tolist())
    elif st.startswith("prune") and counts is not None:
        rows = int((counts >= int(st[5:])).sum()); extra = ng.total * 1.25 / 8
    elif st.startswith("h") and st[1:].isdigit():
        rows = sum(int(s) for k, s in enumerate(ng.sizes.tolist()) if k % ng.hpn < int(st[1:]))
    elif st.startswith("pq"):
        D, K = (int(x) for x in st[2:].split("."))
        return round((rows * (dim // D) * math.log2(K) / 8 + extra) / 1e9, 2)
    elif st.startswith("r") or st.startswith("kr"):
        return None
    per = 1.0 if q is None else q[0] / 8 + 2 / q[1]
    return round((rows * dim * per + extra) / 1e9, 2)


def fit_bases(L, view, args, rank, metric):
    """per head: (P, Pinv, B [160, rank]) - the rank-r PCA of the rows real training text hits, use-weighted, in the
    value (or key) projection's metric"""
    ng = L.ple.ple_embedding
    W = (L.ple.value_proj.weight if metric == "v" else L.ple.key_proj.weight).detach().float().cpu()
    d = np.memmap(args.data, dtype=np.uint32 if args.data_u32 else np.uint16, mode="r")
    st = np.random.default_rng(5).integers(0, len(d) - args.seq - 1, NBW)
    ids = torch.from_numpy(np.stack([d[s:s + args.seq].astype(np.int64) for s in st]))
    idx = ng.indices(ids.to(ng.mult.device)).reshape(-1, ng.heads).cpu()
    bases = {}
    for h in range(ng.heads):
        r_, cnt = torch.unique(idx[:, h], return_counts=True)
        X = view.raw(r_).double()
        w = cnt.double() / cnt.sum()
        P, Pi = metric_basis(W[:, h * ng.dim:(h + 1) * ng.dim])
        Xm = X @ P.T
        ev, V = torch.linalg.eigh((Xm * w[:, None]).T @ Xm)
        bases[h + 0] = (P, Pi, V.flip(1)[:, :rank].contiguous())
    # (rows are global: head h's rows sit at offs[h] ..; View maps a row to its head)
    return bases


def my_run(tr, args, batches, val_batches):
    Ls = [(i, L) for i, L in enumerate(tr.m.layers) if L.ple is not None]
    assert Ls, "no PLE layer in this model / stage"
    i, L = Ls[0]
    ng = L.ple.ple_embedding
    tab, scale = ng.table, ng.scale
    base = View(tab, scale, ng.offs.tolist(), ng.total, "orig")
    vb = val_batches()
    res = {"step": tr.step, "variants": {}, "time": time.ctime()}
    cache = {}
    counts = None
    if any(v.split("+")[0].startswith(("fold", "prune")) for v in VARIANTS) and NCW > 0:
        t0 = time.time()
        vi = torch.cat([ng.indices(b["ids"].to(ng.mult.device)).reshape(-1).cpu() for b in vb])
        res["coverage"] = []
        counts = row_counts(ng, args, NCW, vi=vi, log=res["coverage"])
        res["count_windows"] = NCW
        res["rows_hit"] = int((counts > 0).sum()); res["rows_total"] = int(ng.total)
        res["val_lookups_on_unseen_rows"] = float((counts[vi] == 0).float().mean())
        print(f"row counts: {res['rows_hit']} / {ng.total} rows hit in {NCW} windows ({time.time() - t0:.0f} s); "
              f"val lookups on unseen rows {res.get('val_lookups_on_unseen_rows')}", flush=True)
    sizes = ng.sizes.cpu().clone()
    for v in VARIANTS:
        t0 = time.time()
        ops = v.split("+")
        pn = 0
        if len(ops) > 1 and ops[0].startswith("prune") and not parse_q(ops[1]):
            pn, ops = int(ops[0][5:]), ops[1:]
        st, q = ops[0], (parse_q(ops[-1]) if len(ops) > 1 else None)
        if parse_q(st):
            st, q = "orig", parse_q(st)
        if st == "int4":
            st, q = "orig", (4, 32, "int")
        bases = None
        mode = st
        if st.startswith("pq"):
            D, K = (int(x) for x in st[2:].split("."))
            if ("pq", D, K) not in cache:
                t1 = time.time()
                cache[("pq", D, K)] = fit_pq(L, base, args, D, K)
                print(f"pq{D}.{K} fitted ({time.time() - t1:.0f} s)", flush=True)
            bases = cache[("pq", D, K)]
            mode = "pq"
        elif st.startswith("r") or st.startswith("kr"):
            metric = "k" if st.startswith("k") else "v"
            rank = int(st.lstrip("kr").rstrip("q"))
            if (metric, rank) not in cache:
                cache[(metric, rank)] = fit_bases(L, base, args, rank, metric)
            bases = cache[(metric, rank)]
            mode = "rq" if st.endswith("q") else "r"
        vw = View(tab, scale, ng.offs.tolist(), ng.total, mode, bases, q, sizes, counts, ng.hpn)
        vw.keep_n = pn
        ng.table, ng.scale = vw, None
        val = tr.eval_loss(vb)
        res["variants"][v] = {"val": val, "s": round(time.time() - t0, 1), "gb": est_gb(v, ng, counts)}
        print(f"PLE variant {v}: val {val:.4f}, ~{res['variants'][v]['gb']} GB ({time.time() - t0:.0f} s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    ng.table, ng.scale = tab, scale
    o = res["variants"].get("orig", {}).get("val")
    if o is not None:
        for v, r in res["variants"].items():
            r["delta"] = round(r["val"] - o, 5)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    QT.run = my_run
    QT.main()
