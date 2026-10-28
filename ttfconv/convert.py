"""Convert a full-precision Qwen3.5 / 3.8 checkpoint into a ternary network, standalone (no ttf.train):

  python -m ttfconv.convert --model M --data CAL.bin --out OUT [--ext ttfconv.fold] [--ext ttfconv.adapters:...]
         [--svd-rank 16] [--fine 27] [--method gptqc] [--calib 64] [--seq 2048] [--eval wiki@FILE@N@STRIDE]

The model is built by ttf-core (ttf.qwen.build_qwen, KernelTernaryLinear with row / column scales), converted layer by
layer from the safetensors (ttfconv.stream), and saved as OUT/ckpt.pt in ttf-core's checkpoint form ({"model": state
dict, "step": -1, ...}): a TTF training run starts from it with --init-ckpt (passing the same structural --ext plug-ins
and --adapter-rank R for the float adapters). --fine: the layers carry the fine residual (ttfconv.fine); such a
checkpoint is for ttfconv's own tools (ttfconv.model.load_converted), not for ttf.train. --eval: fixed windows scored
after the conversion (ttfconv.finish.score) into OUT/result.json.

--ext takes the plug-in specs of ttfconv.fold / cd / adapters / embq; their conversion hooks run here, in the order
given."""
import argparse
import importlib
import json
import os
import time

import torch


def load_plugins(specs):
    out = []
    for s in specs or ():
        mod, _, spec = s.partition(":")
        out.append(importlib.import_module(mod).ttf_ext(spec))
    return out


def build_model(arch, dev, plugins=(), act_bits=8, beta=True, own=None, log=print):
    """ttf-core's Qwen in kernel mode with row / column scales, the plug-ins' structure (build) applied"""
    from ttf.qwen import build_qwen
    from ttf.layers import KernelTernaryLinear, enable_rc_scales
    mk = lambda i, o: KernelTernaryLinear(i, o, act_bits, beta=beta)
    model, sw = build_qwen(arch, "kernel", grad_checkpoint=False, device=dev, make_linear=mk, own=own)
    for p in plugins:
        if hasattr(p, "build"):
            sw = p.build(model, sw, mk, log)
    model = model.to(dev)
    enable_rc_scales(model)
    return model, sw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the full-precision HF directory")
    ap.add_argument("--data", required=True, help="calibration tokens (uint32 file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", action="append", default=[], help="ttfconv plug-in MODULE[:SPEC] (repeatable, in order)")
    ap.add_argument("--method", default="gptqc", choices=["gptqc", "gptq", "als"])
    ap.add_argument("--calib", type=int, default=64, help="calibration windows")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mb", type=int, default=4, help="calibration micro-batch")
    ap.add_argument("--svd-rank", type=int, default=0, help="float adapters: the rank-R fit of the rounding error")
    ap.add_argument("--fine", type=int, default=0, help="fine residual grid R (27: |F| <= 100); 0: none")
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--eval", default="", help="NAME@FILE@N@STRIDE[+...] fixed windows scored after the conversion")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    from ttf.layers import KernelTernaryLinear
    from .stream import convert_streaming, base
    plugins = load_plugins(a.ext)
    dev = torch.device(a.device)
    model, sw = build_model(a.model, dev, plugins, a.act_bits, log=log)
    if a.fine:
        from .fine import enable_fine
        for n, q in sw.items():
            enable_fine(base(q), a.fine)
        log(f"fine residual: grid {a.fine} under {len(sw)} ternary linears")
    fits, st = convert_streaming(model, a.model, sw, a.data, plugins, a.method, a.calib, a.seq, a.mb, dev,
                                 svd_rank=a.svd_rank, fine=bool(a.fine), log=log)
    if a.svd_rank:              # the float adapters (ttf-core's enable_lowrank_mag 'add'), set to the fit
        nf = 0
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear):
                m.enable_lowrank_mag("add", a.svd_rank)
        for n, q in sw.items():
            l = base(q)
            if n in fits:
                A, B = fits[n]
                with torch.no_grad():
                    l.mag_A.copy_(A.to(l.mag_A.device)); l.mag_B.copy_(B.to(l.mag_B.device))
                nf += 1
        log(f"float adapters rank {a.svd_rank}: {nf} set to their fit")
    if a.fine:
        from .fine import extra_bits
        eb = extra_bits(model)
        st["fine_extra_GB"] = eb / 8e9
        log(f"fine residual: extra digits {eb / 8e9:.3f} GB entropy-coded")
    from ttf.qwen import model_config
    ck = os.path.join(a.out, "ckpt.pt")
    torch.save({"model": {k: v for k, v in model.state_dict().items() if not v.is_meta}, "step": -1,
                "cfg": model_config(a.model), "mode": "kernel", "beta": True,
                "args": {"ttfconv_convert": vars(a)}, "ttfconv": {"fine": a.fine, "svd_rank": a.svd_rank,
                                                                  "ext": a.ext, "method": a.method}}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)
    res = {"stats": st, "seconds": time.time() - t0, "peak_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}
    log(f"saved {ck} ({os.path.getsize(ck) / 1e9:.2f} GB) after {res['seconds']:.0f} s, peak {res['peak_GiB']:.1f} GiB")
    if a.eval:
        from .finish import score
        model.eval()
        for e in a.eval.split("+"):
            name, f, n, stride = e.split("@")
            res[name] = score(model, f, int(n), int(stride), dev)
            log(f"eval {name}: {res[name]:.4f} ({n} windows)")
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
