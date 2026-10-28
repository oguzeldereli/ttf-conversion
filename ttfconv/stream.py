"""The sequential, layer-by-layer conversion of a Qwen3.5 / 3.8 model built by ttf-core (ttf.qwen.build_qwen, kernel
mode) for models that do not fit as a whole in full precision (27B). Moved out of ttf-core (ttf/qwen.py,
convert_qwen_streaming, d5f6256 + the --fine residual of squeeze/2026-10-27) on 28 Oct 2026; the computation is the
same, the plug-ins are passed in instead of read from ttf.ext and the options are arguments.

Plug-ins (ttfconv.fold / cd / adapters / embq objects) may define, all optional:
  begin(model, files, dev, log)           after the float parts are loaded, before the calibration inputs
  weight(name, W, layer_idx) -> W         the fp weight (float32) that is rounded (e.g. the fold)
  refine(name, W, Hc, T, a, c) -> (T, a, c)   after GPTQ (e.g. coordinate descent)
  fitted(name, wrapper, W, Hc, mu, T, a, c)   after the layer's own trits, scales and centring are written
  end(model, log)                         after the last layer
  propagate_wrapped (attribute)           the calibration stream through the whole wrapper (else the bare layer)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from ttf.kernel import pack_rows, trit_beta
from .gptq import ternarize, ternarize_gptq, _rel_err, _out_err, mag_svd_fit


def base(q):
    """a swapped module's ternary layer: the module itself, or a wrapper's .lin"""
    while not hasattr(q, "wpacked") and hasattr(q, "lin"):
        q = q.lin
    return q


def having(plugins, name):
    return [getattr(p, name) for p in plugins if hasattr(p, name)]


class _StopFwd(Exception):
    pass


@torch.no_grad()
def layer_inputs(m, windows, mb, dev):
    """the first decoder layer's inputs for the windows (micro-batches of mb) and the keyword arguments transformers
    passes to EACH layer (rotary embeddings; the full-attention and the recurrent masks differ by layer type), taken
    from one full forward of the first micro-batch; the other micro-batches stop at layer 0"""
    layers = m.hf.model.layers
    KW, H, hooks = [None] * len(layers), [], []

    def mk(i):
        def pre(mod, args, kwargs):
            if KW[i] is None:
                KW[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            if i == 0:
                H.append((args[0] if args else kwargs["hidden_states"]).detach())
                if all(k is not None for k in KW):
                    raise _StopFwd
        return pre
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(mk(i), with_kwargs=True))
    for b in range(0, len(windows), mb):
        x = torch.tensor(np.stack([w.astype(np.int64) for w in windows[b:b + mb]]), device=dev)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m.hf.model(input_ids=x, use_cache=False)
        except _StopFwd:
            pass
    for h in hooks:
        h.remove()
    return H, KW


def safetensor_index(path):
    """{tensor name (normalised to 'model.'): (name in the file, file)} for lazy per-layer loading"""
    import glob
    idx = os.path.join(path, "model.safetensors.index.json")
    files = {}
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        for k, f in wm.items():
            files[k.replace("model.language_model.", "model.")] = (k, os.path.join(path, f))
    else:
        from safetensors import safe_open
        for f in glob.glob(os.path.join(path, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    files[k.replace("model.language_model.", "model.")] = (k, f)
    return files


def get_tensor(files, name, dev):
    from safetensors import safe_open
    k, f = files[name]
    with safe_open(f, "pt", device="cpu") as h:
        return h.get_tensor(k).to(dev)


@torch.no_grad()
def convert_streaming(m, path, swapped, data_path, plugins=(), method="gptqc", n_seq=64, seq=2048, mb=4, dev="cuda",
                      svd_rank=0, fine=False, log=print):
    """for each decoder layer in turn: its full-precision linears put in place (read lazily from the safetensors), the
    calibration activations (inputs as produced by the already converted layers below) run through it to collect each
    linear's input second moment, the linears rounded (GPTQ or ALS) into the ternary layers, and the activations go on
    through the converted layer. Peak: the model's ternary form + one layer in full precision + the calibration
    activations (n_seq x seq x d bf16).
    svd_rank R: the rank-R float adapter fit of the rounding error (returned as {name: (A, B)}; the caller makes the
    adapters). fine: the layers are FineTernaryLinear (ttfconv.fine.enable_fine) and get the GPTQ fine residual of
    what the trits and that adapter fit left. Returns ({name: (A, B)}, stats)."""
    files = safetensor_index(path)
    hf = m.hf
    wrapped = {n: q for n, q in swapped.items() if base(q) is not q}   # wrappers: their ternary layer is rounded and
    swapped = {n: base(q) for n, q in swapped.items()}                 # carries the stream; the wrapper goes back at the end
    sw = {k + ".weight" for k in swapped}
    for k, p in hf.named_parameters():                  # the float parts
        src = k if k in files else ("model.embed_tokens.weight" if k == "lm_head.weight" else None)
        if src in files and k not in sw:
            p.copy_(get_tensor(files, src, dev).to(p.dtype))
    hf.model.embed_tokens.weight.requires_grad_(False); hf.lm_head.weight.requires_grad_(False)
    for f_ in having(plugins, "begin"):
        f_(m, files, dev, log)
    prop_w = any(getattr(p, "propagate_wrapped", False) for p in plugins)
    data = np.memmap(data_path, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(4321)
    starts = rng.integers(0, len(data) - seq - 1, n_seq)
    layers = hf.model.layers
    H_in, KW = layer_inputs(m, [data[s:s + seq] for s in starts], mb, dev)
    log(f"streaming conversion: {len(layers)} layers, {n_seq} x {seq} calibration tokens in {len(H_in)} micro-batches")
    errs, oerrs, kept, fine_errs, fits = [], [], [], [], {}
    for li, layer in enumerate(layers):
        pre_ = f"model.layers.{li}."
        mine = {n: l for n, l in swapped.items() if n.startswith(pre_)}
        fp = {}                                         # this layer's full-precision linears, in place of the ternary ones
        for n, lin in mine.items():
            W = get_tensor(files, n + ".weight", dev)
            for f_ in having(plugins, "weight"):        # e.g. a residual-stream fold (rounded to bf16 with the rest)
                W = f_(n, W.float(), li)
            f = nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev, dtype=torch.bfloat16); f.weight.copy_(W)
            parent = layer.get_submodule(n[len(pre_):].rsplit(".", 1)[0]); setattr(parent, n.rsplit(".", 1)[1], f)
            fp[n] = (f, parent)
        Hs = {n: torch.zeros(f.in_features, f.in_features, device=dev) for n, (f, _) in fp.items()}
        Ss = {n: torch.zeros(f.in_features, device=dev) for n, (f, _) in fp.items()}
        cnt = {n: 0 for n in fp}
        hooks = []
        for n, (f, _) in fp.items():
            def hook(mod_, inp, n=n):
                xx = inp[0].reshape(-1, inp[0].shape[-1]).float(); Hs[n].addmm_(xx.T, xx); Ss[n] += xx.sum(0)
                cnt[n] += xx.shape[0]
            hooks.append(f.register_forward_pre_hook(hook))
        kw = KW[li]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for h in H_in:
                layer(h, **kw)
        for hh in hooks:
            hh.remove()
        for n, (f, parent) in fp.items():               # round into the ternary layer, put it back
            lin = mine[n]
            W = f.weight.float(); Hl = Hs.pop(n) / max(cnt[n], 1); mu = Ss.pop(n) / max(cnt[n], 1)
            T, a, c = ternarize(W, "als")
            mu_e = mu if method == "gptqc" else None    # (centred: the layer's error, not the raw input's)
            e0 = _out_err(W, T, a, c, Hl, mu_e)
            Hc_ = Hl - torch.outer(mu, mu) if method == "gptqc" else Hl
            if method == "gptq":
                T, a, c = ternarize_gptq(W, Hl, a, c)
            elif method == "gptqc":                     # centred: GPTQ on the input covariance, the layer takes the
                T, a, c = ternarize_gptq(W, Hc_, a, c)  # input mean out before its activation quantisation, W mu as a bias
                for f_ in having(plugins, "refine"):    # e.g. coordinate descent on the trits
                    T, a, c = f_(n, W, Hc_, T, a, c)
                lin.enable_centering(mu, W @ mu)
            oerrs.append((e0, _out_err(W, T, a, c, Hl, mu_e))); errs.append(_rel_err(W, T, a, c))
            lin.wpacked.copy_(pack_rows(T))
            beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
            if method == "gptqc":
                for f_ in having(plugins, "fitted"):    # e.g. adapters fitted to what the trits left (the layer written)
                    f_(n, wrapped.get(n, lin), W, Hc_, mu, T, a, c)
            fit = None
            if svd_rank and method in ("gptq", "gptqc"):          # the float adapter: the rounding error's best rank-R part
                fit, sh = mag_svd_fit(W, T, a, c, beta, Hc_, svd_rank)
                fits[n] = fit; kept.append(sh)
            if fine:                                    # what the trits and the adapter left, onto the fine grid
                from .fine import gptq_fine
                E_ = W - a[:, None] * c[None, :] * T.float()
                if fit is not None:
                    E_ -= (a / beta)[:, None] * (fit[0].to(dev) @ fit[1].to(dev).T) * c[None, :]
                F_, fe_ = gptq_fine(E_, Hc_, a, c, lin.fine_R)
                lin.wfine.copy_(F_); lin._after_fine()
                fine_errs.append(fe_)
                del E_, F_
            # the calibration stream goes on through the bare ternary layer, unless a plug-in asks for the whole wrapper
            # (propagate_wrapped: the layers above are then rounded for the inputs they will see)
            setattr(parent, n.rsplit(".", 1)[1], wrapped[n] if n in wrapped and prop_w else lin)
            del W, Hl, T, Hc_
        del fp, Hs, Ss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # on through the converted layer
            H_in = [(lambda o: o[0] if isinstance(o, tuple) else o)(layer(h, **kw)) for h in H_in]
        torch.cuda.empty_cache()
        if li % 8 == 0 or li == len(layers) - 1:
            e0, e1 = zip(*oerrs[-len(mine):])
            per = " ".join(f"{n.rsplit('.', 2)[-2]}.{n.rsplit('.', 1)[-1]} {e:.3f}" for n, e in zip(mine, e1))
            log(f"  layer {li}: {len(mine)} linears, output-space error {sum(e1) / len(e1):.4f} "
                f"(ALS {sum(e0) / len(e0):.4f}), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; {per}")
    for n, q in wrapped.items():                         # the wrappers back in the tree
        par, cn = n[len("model.layers."):].rsplit(".", 1)
        setattr(hf.model.layers.get_submodule(par), cn, q)
    for f_ in having(plugins, "end"):
        f_(m, log)
    e0, e1 = zip(*oerrs)
    st = {"weight_err": sum(errs) / len(errs), "out_err": sum(e1) / len(e1), "out_err_als": sum(e0) / len(e0)}
    if fine_errs:
        st["fine_left"] = sum(fine_errs) / len(fine_errs)
        log(f"fine: output-space error left after the fine residual, relative to what the trits (and adapters) left: "
            f"mean {st['fine_left']:.4f} (max {max(fine_errs):.4f}) over {len(fine_errs)} linears")
    if kept:
        st["svd_kept"] = sum(kept) / len(kept)
        log(f"svd adapters rank {svd_rank}: they take {st['svd_kept']:.3f} of the output-space rounding error "
            f"(min {min(kept):.3f}, max {max(kept):.3f}) over {len(kept)} linears")
    log(f"conversion of {path}: {method} (streaming), {len(errs)} ternary linears, weight error "
        f"{st['weight_err']:.4f}, output-space error {st['out_err']:.4f} (ALS {st['out_err_als']:.4f})")
    return fits, st
