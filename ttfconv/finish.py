"""training-free result right after the conversion : the packed size from the tensors, the
embedding / head as 4-bit groups of 64 (the release pack), fixed windows scored, the state saved, exit.
  --ext ttfconv.finish:out=DIR[,eval=wiki@FILE@N@STRIDE+chat@FILE@N@STRIDE][,exit=1][,q4=1]
eval files are uint32 token files, N windows of 2049 tokens every STRIDE tokens (2048 = the wiki val file, 2049 =
eval27's exported windows). Must come last among the --ext plug-ins (it runs in the conversion's end hook).
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from .spec import parse


@torch.no_grad()
def fqb(W, bits, g=64):
    """bits-bit asymmetric per group of g along each row (as pack_ttf.py's --emb_bits / --group): dequantized copy"""
    L = 2 ** bits - 1
    Wf = W.float().reshape(W.shape[0], -1, g); lo = Wf.amin(-1, keepdim=True); hi = Wf.amax(-1, keepdim=True)
    sc = ((hi - lo) / L).clamp_min(1e-12); q = ((Wf - lo) / sc).round().clamp(0, L)
    return (q * sc + lo).reshape(W.shape).to(W.dtype)


def fq4(W, g=64):
    """4-bit groups of 64: the release pack's embedding / head"""
    return fqb(W, 4, g)


def sizes(m):
    """bytes by group: trits (5 per byte), embedding / head 4-bit g64, other floats fp16"""
    hf = m.hf; by = {}
    emb, head = hf.model.embed_tokens.weight, hf.lm_head.weight
    for k, v in m.state_dict().items():
        if v.is_meta:
            continue
        if k.endswith("wpacked"):
            g, b = ("trits2" if ".lin2." in k else "adapter trits" if ".ad." in k else "trits"), v.numel()
        elif k.endswith(("embed_tokens.weight", "lm_head.weight")):
            if k.endswith("lm_head.weight") and head is emb:
                continue
            g, b = "embed/head (4-bit g64)", v.numel() * (4 + 32 / 64) / 8
        elif k.endswith(("frozen_scale",)) or not v.is_floating_point():
            continue
        else:
            g, b = "other floats (fp16)", v.numel() * 2
        by[g] = by.get(g, 0) + b
    return by


@torch.no_grad()
def score(m, f, n, st, dev):
    d = np.memmap(f, dtype=np.uint32, mode="r"); ce = nt = 0
    for i in range(n):
        t = torch.from_numpy(np.asarray(d[i * st:i * st + 2049], dtype=np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(t[None, :-1])[0][0].float()
        ce += float(F.cross_entropy(lg, t[1:], reduction="sum")); nt += t.numel() - 1
    return ce / nt


class Finish:
    def __init__(self, spec):
        o = parse(spec, {"eval": "", "exit": "1", "q4": "1"}, required=("out",))
        self.out, self.exit, self.q4 = o["out"], o["exit"] == "1", o["q4"] == "1"
        self.evals = [e.split("@") for e in o["eval"].split("+") if e]

    @torch.no_grad()
    def end(self, m, log):
        os.makedirs(self.out, exist_ok=True)
        dev = next(p for p in m.parameters() if not p.is_meta).device
        by = sizes(m); tot = sum(by.values())
        log("ttfconv.finish size: " + ", ".join(f"{g} {b / 1e9:.3f} GB" for g, b in by.items()) + f"; total {tot / 1e9:.3f} GB")
        hf = m.hf; emb, head = hf.model.embed_tokens.weight, hf.lm_head.weight
        if self.q4:                  # q4=0: the embedding / head kept in bf16 (a training start, not the release pack)
            emb.copy_(fq4(emb))
            if head is not emb:
                head.copy_(fq4(head))
        for l in m.modules():
            if hasattr(l, "_invalidate"):
                l._invalidate()
        m.eval(); res = {"size_GB": tot / 1e9, "size_parts_GB": {g: b / 1e9 for g, b in by.items()}}
        for name, f, n, st in self.evals:
            res[name] = score(m, f, int(n), int(st), dev)
            log(f"ttfconv.finish eval {name}: {res[name]:.4f} ({n} windows)")
        json.dump(res, open(os.path.join(self.out, "result.json"), "w"), indent=1)
        torch.save({"model": {k: v.cpu() for k, v in m.state_dict().items() if not v.is_meta}},
                   os.path.join(self.out, "state.pt"))
        log("TTFCONV_FINISHED")
        if self.exit:
            os._exit(0)


def ttf_ext(spec):
    return Finish(spec)
