"""Converting a Llama-style HF checkpoint (the 110M models) into ttf-core's BitTransformer: the calibration second
moments of the fp model, then every linear rounded (ALS, GPTQ or centred GPTQ) into its KernelTernaryLinear. Moved out
of ttf-core (ttf/hf_init.py calib_hessians / init_from_hf, 0ef62dd) on 28 Oct 2026; ttf-core keeps the fp-only loader
(ttf.hf_load.load_hf_fp). The computation is unchanged; the float adapters' fit is returned instead of parked on the
layers."""
import numpy as np
import torch
import torch.nn as nn

from .gptq import ternarize, ternarize_gptq, _rel_err, _out_err, mag_svd_fit


@torch.no_grad()
def calib_hessians(path, data_path, n_seq=64, seq=2048, dev="cuda", dtype_u32=False, log=print):
    """H = X^T X / tokens of every linear's input in the full-precision model on n_seq random windows of the data
    (q / k / v share one, gate / up another): {(block, name): H [K, K] fp32 on the CPU}, plus the input means"""
    from ttf.model import BitTransformer
    from ttf.hf_load import model_config_from_hf, load_hf_fp
    mc = model_config_from_hf(path, max_seq_len=seq)
    m = BitTransformer(mc, grad_checkpoint=False, make_linear=lambda i, o: nn.Linear(i, o, bias=False)).to(dev)
    load_hf_fp(m, path, log=lambda *_: None)
    m.eval()
    H, hooks, cnt, MU = {}, [], {}, {}
    own = {"wq": "wq", "wk": "wq", "wv": "wq", "wo": "wo", "w_gate": "w_gate", "w_up": "w_gate", "w_down": "w_down"}
    for i, blk in enumerate(m.layers):
        for sub, name in (("attn", "wq"), ("attn", "wo"), ("ffn", "w_gate"), ("ffn", "w_down")):
            lin = getattr(getattr(blk, sub), name)
            key = (i, name)
            H[key] = torch.zeros(lin.in_features, lin.in_features, device=dev)
            MU[key] = torch.zeros(lin.in_features, device=dev)
            cnt[key] = 0

            def hook(mod, inp, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key].addmm_(x.T, x); MU[key] += x.sum(0); cnt[key] += x.shape[0]
            hooks.append(lin.register_forward_pre_hook(hook))
    data = np.memmap(data_path, dtype=np.uint32 if dtype_u32 else np.uint16, mode="r")
    rng = np.random.default_rng(4321)                   # not the training stream's windows
    for s in rng.integers(0, len(data) - seq - 1, n_seq):
        x = torch.tensor(data[s:s + seq + 1].astype(np.int64), device=dev)[None]
        m(x[:, :-1], x[:, 1:])
    for h in hooks:
        h.remove()
    out = {}
    for (i, name), h in H.items():
        h /= max(cnt[(i, name)], 1)
        mu = MU[(i, name)] / max(cnt[(i, name)], 1)
        for n2, o in own.items():
            if o == name:
                out[(i, n2)] = h.cpu()
                out[("mu", i, n2)] = mu.cpu()       # the input mean (gptqc: centred conversion)
    del m, H
    torch.cuda.empty_cache()
    log(f"calibration: input second moments of {len(out)} linears from {n_seq} x {seq} tokens of {data_path}")
    return out


@torch.no_grad()
def convert_llama(model, path, method="als", hess=None, svd_rank=0, log=print):
    """the checkpoint into model (ttf-core BitTransformer built with model_config_from_hf, kernel mode, row / column
    scales on): float parts copied, every linear rounded. method gptq / gptqc needs hess = calib_hessians(...).
    Returns {(block, name): (A, B)}: the rank-svd_rank float adapter fits (gptq / gptqc only)."""
    from ttf.hf_load import load_hf_tensors, _unpermute, _LIN
    from ttf.layers import KernelTernaryLinear
    from ttf.kernel import pack_rows, trit_beta
    sd = load_hf_tensors(path)
    c_ = model.c
    dev = next(model.parameters()).device
    model.tok_emb.weight.copy_(sd["model.embed_tokens.weight"])
    model.norm.weight.copy_(sd["model.norm.weight"])
    if model.lm_head is not None:
        model.lm_head.weight.copy_(sd.get("lm_head.weight", sd["model.embed_tokens.weight"]))
    errs, kept, oerrs, fits = [], [], [], {}
    for i, blk in enumerate(model.layers):
        p = f"model.layers.{i}."
        blk.attn_norm.weight.copy_(sd[p + "input_layernorm.weight"])
        blk.ffn_norm.weight.copy_(sd[p + "post_attention_layernorm.weight"])
        for hf, (sub, name) in _LIN.items():
            W = sd.pop(p + hf + ".weight").to(dev)
            if name == "wq":
                W = _unpermute(W, c_.n_heads)
            elif name == "wk":
                W = _unpermute(W, c_.n_kv_heads)
            lin = getattr(getattr(blk, sub), name)
            assert isinstance(lin, KernelTernaryLinear) and lin.row_scale is not None, \
                "convert_llama: kernel-mode layers with row / column scales"
            T, a, c = ternarize(W, "als" if method in ("gptq", "gptqc") else method)
            Hc = None
            if hess is not None:
                Hl = hess[(i, name)].to(dev)
                mu = hess[("mu", i, name)].to(dev) if method == "gptqc" else None
                e0 = _out_err(W, T, a, c, Hl, mu)
                Hc = Hl - torch.outer(mu, mu) if method == "gptqc" else Hl
                if method == "gptq":
                    T, a, c = ternarize_gptq(W, Hl, a, c)
                elif method == "gptqc":     # centred: GPTQ on the input covariance; the layer subtracts the input mean
                    T, a, c = ternarize_gptq(W, Hc, a, c)                     # before its activation quantisation
                    lin.enable_centering(mu, W @ mu)                          # and adds W mu back as a bias
                oerrs.append((e0, _out_err(W, T, a, c, Hl, mu)))
                del Hl
            errs.append(_rel_err(W, T, a, c))
            lin.wpacked.copy_(pack_rows(T))
            beta = float(trit_beta(lin.wpacked, lin.K)) if lin.use_beta else 1.0
            if svd_rank and hess is not None and method in ("gptq", "gptqc"):   # (the covariance in float64, as before)
                Hm = hess[(i, name)].to(dev).double()
                if method == "gptqc":
                    mm = hess[("mu", i, name)].to(dev).double(); Hm = Hm - torch.outer(mm, mm)
                fits[(i, name)], sh = mag_svd_fit(W, T, a, c, beta, Hm, svd_rank); kept.append(sh)
                del Hm
            lin.row_scale.copy_(a / beta); lin.col_scale.copy_(c)
            del W, Hc
    if errs:
        log(f"conversion of {path}: {method}, relative error |W - a c T| / |W| mean {sum(errs) / len(errs):.4f} "
            f"(min {min(errs):.4f}, max {max(errs):.4f}) over {len(errs)} layers")
    if oerrs:
        e0, e1 = zip(*oerrs)
        log(f"  output-space relative error mean {sum(e1) / len(e1):.4f} (als rounding {sum(e0) / len(e0):.4f}) "
            f"over {len(e1)} layers")
    if kept:
        log(f"  svd adapters rank {svd_rank}: they take {sum(kept) / len(kept):.3f} of the output-space rounding error "
            f"(min {min(kept):.3f}, max {max(kept):.3f}) over {len(kept)} layers")
    return fits
