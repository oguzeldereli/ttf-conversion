"""The ternary Parakeet TDT student: every nn.Linear and pointwise (kernel 1) nn.Conv1d of the teacher becomes a TTF
KernelTernaryLinear (trits times row / column scales, int8 activations), converted from the teacher's weights by GPTQ
onto the ternary grid under each layer's centred input second moment (as ttfconv's gptqc: y = T(x - mu) + W mu + b).
What stays float: the decoder embedding (5.2M), its 2-layer LSTM (6.6M), the depthwise convs (0.2M), the subsampling
Conv2d's (0.1M), the norms and the scales (about 2% of the 627M parameters).
"""
import time

import torch
import torch.nn as nn

from ttf.kernel import pack_rows, trit_beta
from ttf.layers import KernelTernaryLinear

from ttfconv.gptq import _out_err, ternarize, ternarize_gptq


class TLin(nn.Module):
    """a KernelTernaryLinear in place of an nn.Linear ([..., K] inputs) or a kernel-1 nn.Conv1d ([B, K, T]); the input
    goes to the packed kernels in bf16"""
    def __init__(self, K, N, conv, dev):
        super().__init__()
        self.conv = conv
        self.core = KernelTernaryLinear(K, N, 8, beta=True).to(dev)
        self.core.enable_rc_scale()
        self.core.enable_centering(torch.zeros(K, device=dev), torch.zeros(N, device=dev))

    def forward(self, x):
        if self.conv:
            return self.core(x.transpose(1, 2).to(torch.bfloat16)).transpose(1, 2)
        return self.core(x.to(torch.bfloat16))


def targets(model):
    """(name, module, conv) of the layers that go ternary"""
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out.append((n, m, False))
        elif isinstance(m, nn.Conv1d) and m.kernel_size == (1,) and m.groups == 1:
            out.append((n, m, True))
    return out


def _set(model, name, new):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, new)


def weight_2d(m):
    return m.weight.detach().float().reshape(m.weight.shape[0], -1)          # (Conv1d k=1: [N, K, 1] -> [N, K])


def skeleton(model, dev):
    """replace the target layers by untrained TLin's (to load a converted state into)"""
    for n, m, conv in targets(model):
        W = weight_2d(m)
        _set(model, n, TLin(W.shape[1], W.shape[0], conv, dev))
    return model


def tern_layers(model):
    return [m.core for m in model.modules() if isinstance(m, TLin)]


@torch.no_grad()
def calibrate(model, batches, run):
    """input second moments of the target layers: H = sum x x^T, s = sum x, n rows, over run(model, batch) for each
    batch (padding frames excluded by running one utterance per batch)"""
    stats, hooks = {}, []
    for n, m, conv in targets(model):
        K = weight_2d(m).shape[1]
        stats[n] = [torch.zeros(K, K, device=m.weight.device), torch.zeros(K, device=m.weight.device, dtype=torch.float64), 0]

        def hook(mod, inp, _n=n, _conv=conv):
            x = inp[0].detach()
            x = (x.transpose(1, 2) if _conv else x).reshape(-1, x.shape[1] if _conv else x.shape[-1]).float()
            st = stats[_n]
            st[0] += x.T @ x; st[1] += x.sum(0).double(); st[2] += x.shape[0]
        hooks.append(m.register_forward_hook(lambda mod, inp, out, _h=hook: _h(mod, inp)))
    for b in batches:
        run(model, b)
    for h in hooks:
        h.remove()
    return stats


@torch.no_grad()
def convert(model, stats, dev, method="gptq", center=True, log=print):
    """round every target layer (in place: model's layers become TLin's); returns {name: output-space rel. error}"""
    errs, t0 = {}, time.time()
    for i, (n, m, conv) in enumerate(targets(model)):
        W = weight_2d(m).to(dev)
        N, K = W.shape
        b = m.bias.detach().float().to(dev) if m.bias is not None else torch.zeros(N, device=dev)
        H, s, cnt = stats[n]
        mu = (s / max(cnt, 1)).float().to(dev) if center else torch.zeros(K, device=dev)
        H2 = (H / max(cnt, 1)).float().to(dev)
        Hc = H2 - torch.outer(mu, mu)
        T, a, c = ternarize(W, "als")
        if method == "gptq" and cnt > 0:
            T, a, c = ternarize_gptq(W, Hc, a, c)
        errs[n] = _out_err(W, T, a, c, H2, mu) if cnt > 0 else float("nan")
        new = TLin(K, N, conv, dev)
        l = new.core
        l.wpacked.copy_(pack_rows(T))
        beta = float(trit_beta(l.wpacked, l.K))
        l.row_scale.copy_(a / beta); l.col_scale.copy_(c)
        l.in_mean.copy_(mu); l.out_bias.copy_(W @ mu + b)
        _set(model, n, new)
        del W, H2, Hc
        if i % 20 == 0:
            log(f"  {i} {n} [{N}x{K}] err {errs[n]:.4f} ({time.time() - t0:.0f}s)")
    return errs
