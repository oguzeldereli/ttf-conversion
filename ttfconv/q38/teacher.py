"""Teacher targets for the Qwen3.8-Flash-Next TTF run: the float (FP8 release) model's top-64 next-token
log-probabilities plus 64 tail tokens sampled with replacement from the rest of its distribution, at every position of
the training windows.

Same files as ttf-core scripts/quant/teacher_topk.py --k 64 --tail 64 (OUT/kd_00000.npz ...: starts [B] int64, ids
[B, seq, 64] int32, lp fp16, sids [B, seq, 64] int32, slp fp16; OUT/done), window starts uniform with seed 1234, read in
order by the trainer. Layer-major: a shard's B windows stay on the GPU while the layers come and go (FP8 experts
dequantised on the GPU). Also logs the teacher's own next-token loss per shard (a check of the model code).
  python -m ttfconv.q38.teacher HF_DIR DATA.bin OUT [--windows 16384] [--batch 256] [--mb 8] [--check N]
--check N: N windows only, no files: the loss with and without the PLE (n-gram) branch, then exit.
"""
import argparse, os, sys, time
import numpy as np
import torch

from ttf.q38 import model as M
from ttf.q38 import store as IO


@torch.no_grad()
def run_layers(m, st, H, ids, pe, mb, dev, ple=True):
    for li, L in enumerate(m.layers):
        st.fill(L, li, dev)
        keep = L.ple
        if not ple:
            L.ple = None
        for b0 in range(0, H.shape[0], mb):
            H[b0:b0 + mb] = L(H[b0:b0 + mb], pe, ids[b0:b0 + mb])
        L.ple = keep
        st.unfill(L)
    return H


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=16384); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mb", type=int, default=8); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--k", type=int, default=64); ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--first", default="", help="comma list of shards to compute before the others (the trainer's val "
                    "shard)")
    ap.add_argument("--table-dir", default="", help="read the PLE tables from here (the slim copy) and close HF_DIR's "
                    "files after loading, so HF_DIR can be deleted while this runs")
    a = ap.parse_args()
    dev = "cuda"
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
    cfg = M.text_config(a.hf)
    nl = a.layers or cfg.num_hidden_layers
    m = M.build(cfg, layers=nl, device="meta")
    st = IO.Store(a.hf, cfg, layers=nl, log=log, table_path=a.table_dir or None)
    st.fill_globals(m, dev)
    if a.table_dir:
        st.close(); log(f"checkpoint files closed; PLE tables from {a.table_dir}")
    data = np.memmap(a.data, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(a.seed)
    starts_all = rng.integers(0, len(data) - a.seq - 1, a.windows)
    pe = m.rope(a.seq, dev)
    head = m.lm_head.weight

    def windows(st_):
        w = np.stack([data[s:s + a.seq + 1].astype(np.int64) for s in st_])
        return torch.from_numpy(w[:, :-1]).to(dev), torch.from_numpy(w[:, 1:]).to(dev)

    if a.check:
        x, y = windows(starts_all[:a.check])
        for ple in (True, False):
            t0 = time.time()
            H = m.embed(x.cpu()).to(dev, torch.bfloat16)
            H = run_layers(m, st, H, x, pe, a.mb, dev, ple)
            ce = []
            for b in range(H.shape[0]):
                lg = (m.head_in(H[b:b + 1])[0] @ head.T).float()
                ce.append(torch.nn.functional.cross_entropy(lg, y[b]).item())
            log(f"check: {a.check} windows, PLE {'on' if ple else 'off'}: next-token loss {np.mean(ce):.4f} "
                f"(per window {[round(c, 3) for c in ce]}), {time.time() - t0:.0f}s")
        return
    os.makedirs(a.out, exist_ok=True)
    nsh = len(starts_all) // a.batch
    first = [int(x) for x in a.first.split(",") if x]
    for sh in first + [s_ for s_ in range(nsh) if s_ not in first]:
        fn = os.path.join(a.out, f"kd_{sh:05d}.npz")
        if os.path.exists(fn):
            continue
        t1 = time.time()
        stt = starts_all[sh * a.batch:(sh + 1) * a.batch]
        x, y = windows(stt)
        H = torch.empty(len(stt), a.seq, cfg.hc_count * cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        for b0 in range(0, len(stt), a.mb):
            H[b0:b0 + a.mb] = m.embed(x[b0:b0 + a.mb].cpu()).to(dev, torch.bfloat16)
        H = run_layers(m, st, H, x, pe, a.mb, dev)
        t2 = time.time()
        top_i = np.empty((len(stt), a.seq, a.k), dtype=np.int32); top_l = np.empty((len(stt), a.seq, a.k), dtype=np.float16)
        s_i = np.empty((len(stt), a.seq, a.tail), dtype=np.int32); s_l = np.empty((len(stt), a.seq, a.tail), dtype=np.float16)
        gen = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + sh)
        ce = 0.0
        for b in range(len(stt)):
            hb = m.head_in(H[b:b + 1])[0]
            for c0 in range(0, a.seq, 512):
                lp = torch.log_softmax((hb[c0:c0 + 512] @ head.T).float(), -1)
                ce -= float(lp.gather(1, y[b, c0:c0 + 512, None]).sum())
                v, i = lp.topk(a.k, -1)
                top_i[b, c0:c0 + 512] = i.int().cpu().numpy(); top_l[b, c0:c0 + 512] = v.half().cpu().numpy()
                pt = lp.exp().scatter_(1, i, 0.0)
                pt[pt.sum(-1) <= 0] = 1.0
                j = torch.multinomial(pt, a.tail, replacement=True, generator=gen)
                s_i[b, c0:c0 + 512] = j.int().cpu().numpy(); s_l[b, c0:c0 + 512] = lp.gather(1, j).half().cpu().numpy()
                del pt, j, lp
        del H
        tmp = fn + ".tmp.npz"
        np.savez(tmp, starts=stt, ids=top_i, lp=top_l, sids=s_i, slp=s_l)
        os.replace(tmp, fn)
        cov = float(np.exp(top_l.astype(np.float32)).sum(-1).mean())
        log(f"shard {sh + 1}/{nsh}: {len(stt)} windows, layers {t2 - t1:.0f}s, head {time.time() - t2:.0f}s; "
            f"teacher loss {ce / (len(stt) * a.seq):.4f}, top-{a.k} mass {cov:.3f}")
    if all(os.path.exists(os.path.join(a.out, f"kd_{s:05d}.npz")) for s in range(nsh)):
        open(os.path.join(a.out, "done"), "w").close()
        log("teacher cache done")


if __name__ == "__main__":
    main()
