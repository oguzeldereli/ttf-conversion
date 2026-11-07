"""ple_pq_build.py: the Q38 PLE n-gram table as prune1 + pq4.256 (ple_loss.py, 7.95 GB; zero-shot loss +0.0073 against
the FP8 table at step 300 of a Flash-Next run), stored for ttf.q38.ple_pq.PQTable, which the trainer reads instead of
the 51.2 GB FP8 table.

Stored form (DIR):
  codes.bin   uint8 [total, 160 / D]: per table row, the codebook index of each D-dim sub-vector of P_h x (P_h: head h's
              value_proj metric basis, ple_probe.metric_basis); rows not kept are 0
  keep.bin    uint8 [total]: 1 where the row was hit >= N times in the count windows (prune N), else the row reads as 0
  meta.pt     {"D", "K", "N", "count_windows", "offs", "total", "Pi" [heads, 160, 160], "cb" [heads, 160 / D, K, D]}
A row decodes as keep * Pi_h (cb_h[j, code_j])_j: the same numbers as ple_loss.View("pq") with keep_n = N (float32 on
the GPU instead of float64 on the host).

Build (no model load: the table, its scale, the hashing and value_proj of the PLE layer only; the fits are ple_loss's
own functions with its seeds, so the table is the one measured there):
  python scripts/q38/ple_pq_build.py --model SLIM --data mix5c_train.bin --data-u32 --out DIR \
      [--D 4 --K 256 --prune 1 --count-windows 65536 --basis-windows 64]
Use: python -m ttf.q38.train ... --ple-pq DIR"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # (ple_loss, beside this file)
from ttf.q38.ple_pq import PQTable


def build(a):
    from ttf.q38 import model as QM
    sys.argv = [sys.argv[0], "--ple-basis-windows", str(a.basis_windows), "--ple-count-windows", str(a.count_windows)]
    import ple_loss as PL                                        # (its fits: row_counts seed 7, fit_pq seed 5 / h*1000+j)
    dev = "cuda"
    cfg = QM.text_config(a.model)
    rd = QM.Reader(a.model)
    pl = cfg.ple_layer_ids or []
    assert len(pl) == 1, f"one PLE layer expected, got {pl}"
    i = pl[0] - 1
    ng = QM.NGramEmbedding(cfg, 0).to(dev)                      # (the hashing buffers on the GPU, as with --resident)
    ng.table, ng.scale = QM.ngram_table_memmap(a.model, i)
    Wv = rd.weight(f"{QM.PFX}layers.{i}.ple.value_proj").float().cpu()
    L = argparse.Namespace(ple=argparse.Namespace(ple_embedding=ng, value_proj=argparse.Namespace(weight=Wv)))
    args = argparse.Namespace(data=a.data, data_u32=a.data_u32, seq=a.seq)
    view = PL.View(ng.table, ng.scale, ng.offs.tolist(), ng.total, "orig")
    t0 = time.time()
    counts = PL.row_counts(ng, args, a.count_windows)
    keep = (counts >= a.prune)
    print(f"row counts: {int(keep.sum())} / {ng.total} rows kept (hit >= {a.prune} in {a.count_windows} windows, "
          f"{time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    fit = PL.fit_pq(L, view, args, a.D, a.K)
    print(f"pq{a.D}.{a.K} fitted ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    nS = ng.dim // a.D
    codes = np.memmap(os.path.join(a.out, "codes.bin.tmp"), mode="w+", dtype=np.uint8, shape=(ng.total, nS))
    offs = ng.offs.tolist() + [ng.total]
    t0 = time.time(); done = 0
    for h in range(ng.heads):
        P, Pi, cb = fit[h]
        Pg, cbg = P.T.to(dev), cb.to(dev)                         # (float64 basis, as View; codebooks float32)
        rows = (keep[offs[h]:offs[h + 1]].nonzero().squeeze(1) + offs[h])
        for c0 in range(0, len(rows), a.chunk):
            r = rows[c0:c0 + a.chunk]
            Xm = (view.raw(r).double().to(dev) @ Pg).float()
            cc = torch.stack([torch.cdist(Xm[:, j * a.D:(j + 1) * a.D], cbg[j]).argmin(1) for j in range(nS)], 1)
            codes[r.numpy()] = cc.to(torch.uint8).cpu().numpy()
            done += len(r)
        print(f"head {h}: {len(rows)} rows encoded ({done} total, {time.time() - t0:.0f} s)", flush=True)
    codes.flush(); del codes
    os.replace(os.path.join(a.out, "codes.bin.tmp"), os.path.join(a.out, "codes.bin"))
    keep.to(torch.uint8).numpy().tofile(os.path.join(a.out, "keep.bin"))
    torch.save({"D": a.D, "K": a.K, "N": a.prune, "count_windows": a.count_windows, "basis_windows": a.basis_windows,
                "offs": ng.offs.tolist(), "total": ng.total,
                "Pi": torch.stack([fit[h][1] for h in range(ng.heads)]).float(),
                "cb": torch.stack([fit[h][2] for h in range(ng.heads)]).float()}, os.path.join(a.out, "meta.pt"))
    gb = (int(keep.sum()) * nS * math.log2(a.K) / 8 + ng.total * 1.25 / 8) / 1e9
    print(f"stored: {a.out} (shipping size ~{gb:.2f} GB: kept rows' codes + a ranked bitmap)", flush=True)
    # check against ple_loss's own view on random rows (kept and pruned)
    pv = PL.View(ng.table, ng.scale, ng.offs.tolist(), ng.total, "pq", fit, None, ng.sizes, counts, ng.hpn)
    pv.keep_n = a.prune
    T = PQTable(a.out, dev)
    g = torch.Generator().manual_seed(1)
    r = torch.randint(0, ng.total, (20000,), generator=g)
    ref, got = pv[r], T[r].cpu()
    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
    bad = float(((got - ref).norm(dim=1) > 1e-3 * ref.norm(dim=1).clamp_min(1e-12)).float().mean())
    print(f"check vs ple_loss.View(pq{a.D}.{a.K}, prune{a.prune}): rel err {rel:.2e}, rows off {bad:.4f} "
          f"(argmin ties under float32 vs float64)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--data-u32", action="store_true"); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=4); ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--prune", type=int, default=1); ap.add_argument("--count-windows", type=int, default=65536)
    ap.add_argument("--basis-windows", type=int, default=64); ap.add_argument("--chunk", type=int, default=1 << 21)
    build(ap.parse_args())
