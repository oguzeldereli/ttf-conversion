"""Extra digits under each ternary layer: W = r_n c_k beta (T + F / R) (+ its float adapter).

T are the trits of ttf-core's KernelTernaryLinear; F is an int8 fine residual on a grid R times finer than one trit
step. R T + F fits int8 (R 27: |F| <= 100), so the forward stays one int8 GEMM. F written in balanced ternary is the
layer's extra digits: plane j (j = 0 the finest) holds the trits d_j with F = sum_j 3^j d_j, at a step of 3^j / R
trits. Each plane is entropy-coded in the file, so its size is N K (h(p_j) + p_j) bits for a nonzero share p_j (h the
binary entropy; the sign of a nonzero is one bit).

FineTernaryLinear is a KernelTernaryLinear whose forward runs the R T + F GEMM (enable_fine turns a converted layer
into one in place). Its backward gives the input gradient through T + F / R and, with layer.capture, the weight
gradient in layer.gw (dL/dW in the layer's units, beta folded in); nothing in it updates the weights.
Moved out of ttf-core (squeeze/2026-10-27, never merged) on 28 Oct 2026.
"""
from __future__ import annotations
import math

import torch

from ttf.layers import KernelTernaryLinear
from ttf.kernel import act_quant_i8, act_quant_i8_fast, unpack_fast, i8_epilogue, _mm_f32, _slice2_kernel, \
    _dx2_epi_kernel

PLANES = 5          # balanced-ternary planes of F: |F| <= 121
DX_MODE = ["int8x2"]    # the input gradient's GEMM: int8x2 (gy in two int8 slices), int8 (one), bf16


def fmax(R):
    """the largest |F| for grid R: R T + F within int8 and F within PLANES balanced digits"""
    return min(127 - R, (3 ** PLANES - 1) // 2)


def digits(F):
    """int8 F [..] -> int8 [PLANES, ..] of balanced-ternary digits, F = sum_j 3^j d_j (j = 0 the finest)"""
    f = F.to(torch.int16)
    out = []
    for _ in range(PLANES):
        d = torch.remainder(f + 1, 3) - 1
        out.append(d.to(torch.int8)); f = (f - d) // 3
    return torch.stack(out)


def plane_bits(nnz, n):
    """entropy-coded size in bits of a plane of n trits with nnz nonzero: n (h(p) + p)"""
    if nnz <= 0 or n <= 0:
        return 0.0
    p = min(nnz / n, 1.0)
    h = 0.0 if p >= 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return n * (h + p)


def layer_planes(l):
    """per plane the nonzero count of the layer's F (int64 list)"""
    return [int(x) for x in (digits(l.wfine) != 0).flatten(1).sum(1).tolist()]


def extra_bits(model):
    """the extra digits' entropy-coded size over all fine layers, in bits"""
    tot = 0.0
    for m in model.modules():
        if getattr(m, "wfine", None) is not None:
            n = m.N * m.K
            tot += sum(plane_bits(z, n) for z in layer_planes(m))
    return tot


def w8_gemm_i8(xq, xs, w8, beta=1.0):
    """y = (xq @ w8^T) xs beta for an int8 weight in hand (R T + F, any int8 values)"""
    M_, K = xq.shape
    N_ = w8.shape[0]
    if M_ > 16 and K % 16 == 0 and N_ % 8 == 0:
        y32 = torch._int_mm(xq.contiguous(), w8.t())
        if y32.is_contiguous():
            return i8_epilogue(y32, xs, beta)
        return (y32.float() * xs.float()[:, None] * beta).to(torch.bfloat16)
    return (_mm_f32(xq.to(torch.bfloat16), w8.to(torch.bfloat16).t()) * xs.float()[:, None] * beta).to(torch.bfloat16)


def w8_gemm_dx(gy, w8, two=True, mode="int8x2"):
    """grad_x = gy @ w8, gy [M, N] -> bf16 [M, K]: mode int8x2 / int8 with gy in int8 slices (|q| 127 x |w| 127 x N
    stays far inside int32), else bf16 (w8 is exact in bf16)"""
    import triton
    M, N = gy.shape
    K = w8.shape[1]
    if mode not in ("int8x2", "int8") or not (gy.is_cuda and M > 16 and N % 8 == 0 and K % 8 == 0):
        return torch.mm(gy.contiguous().to(torch.bfloat16), w8.to(torch.bfloat16))
    g = gy.contiguous()
    q1 = torch.empty(M, N, dtype=torch.int8, device=g.device); q2 = torch.empty_like(q1)
    sc = torch.empty(M, dtype=torch.float32, device=g.device)
    _slice2_kernel[(M,)](g, q1, q2, sc, N, g.stride(0), q1.stride(0), BLOCK=1024)
    try:
        y1 = torch._int_mm(q1, w8); y2 = torch._int_mm(q2, w8) if two else y1
    except RuntimeError:          # (a cuBLASLt that takes only a column-major second operand)
        wt = w8.t().contiguous().t()
        y1 = torch._int_mm(q1, wt); y2 = torch._int_mm(q2, wt) if two else y1
    out = torch.empty(M, K, dtype=torch.bfloat16, device=g.device)
    _dx2_epi_kernel[(triton.cdiv(M, 32), triton.cdiv(K, 128))](y1, y2, sc, out, M, K, TWO=two, BM=32, BK=128)
    return out


class _FineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        beta = layer._cached_beta(wpacked) if layer.use_beta else None
        xq, xs = (act_quant_i8_fast if x.is_cuda else act_quant_i8)(x.reshape(-1, layer.K), layer.act_bits)
        y = w8_gemm_i8(xq, xs, layer.w8(), (beta if beta is not None else 1.0) / layer.fine_R)
        ctx.save_for_backward(xq, xs)
        ctx.layer, ctx.beta, ctx.xshape = layer, beta, x.shape
        return y.view(*x.shape[:-1], layer.N).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        layer, beta = ctx.layer, ctx.beta
        gyf = gy.reshape(-1, layer.N).contiguous()
        if beta is not None:
            gyf = gyf * beta.to(gyf.dtype)
        xq, xs = ctx.saved_tensors
        gx = (w8_gemm_dx(gyf, layer.w8(), two=DX_MODE[0] == "int8x2", mode=DX_MODE[0]) / layer.fine_R).view(ctx.xshape)
        if layer.capture:          # the weight gradient for the caller (dL/dW of T + F / R, beta folded in)
            xf = xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)
            gw = (gyf.transpose(0, 1) @ xf).float()
            layer.gw = gw if layer.gw is None else layer.gw + gw
        return gx.to(gy.dtype), None, None


class FineTernaryLinear(KernelTernaryLinear):
    """a KernelTernaryLinear with the fine residual: made from a converted layer by enable_fine (never built directly)"""

    def w8(self, t=False):
        """R T + F as int8 [N, K] ([K, N] view for t), kept until the trits or F change"""
        k = (self._wkey(self.wpacked), getattr(self, "_nfire", 0), getattr(self, "_nfine", 0))
        c = self.__dict__.setdefault("_w8c", {})
        if c.get("f", (None,))[0] != k:
            c.pop("f", None)
            w = unpack_fast(self.wpacked, self.K)
            c["f"] = (k, (w.to(torch.int16) * self.fine_R + self.wfine.to(torch.int16)).to(torch.int8))
        w = c["f"][1]
        return w.t() if t else w

    def _after_fine(self):
        """after F changed in place: the cached R T + F is stale"""
        self._nfine = getattr(self, "_nfine", 0) + 1

    def forward(self, x, acc=None):
        y = self._fine_forward(x)
        return y if acc is None else acc + y

    def _fine_forward(self, x):
        if getattr(self, "in_mean", None) is not None:      # centred input (before the activation quantisation)
            x = x - self.in_mean.to(x.dtype)
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        y = _FineFn.apply(x, self.wpacked, self)
        if getattr(self, "mag_A", None) is not None:
            y = y + (x @ self.mag_B.to(x.dtype)) @ self.mag_A.to(y.dtype).T
        if self.row_scale is not None:
            y = y * self.row_scale.to(y.dtype)
        if getattr(self, "out_bias", None) is not None:
            y = y + self.out_bias.to(y.dtype)
        return y


def enable_fine(lin, R):
    """turn a KernelTernaryLinear into a FineTernaryLinear in place, F = 0 (the same output until F is written)"""
    assert type(lin) is KernelTernaryLinear, type(lin)
    lin.__class__ = FineTernaryLinear
    lin.fine_R = int(R)
    lin.register_buffer("wfine", torch.zeros(lin.N, lin.K, dtype=torch.int8, device=lin.wpacked.device))
    lin.__dict__.pop("_w8c", None)
    return lin


@torch.no_grad()
def gptq_fine(E, H, a, c, R, block=128, damp=0.01, act_order=True):
    """GPTQ of the residual E [N, K] (what the trits and the adapter left, in weight units) onto the fine grid
    a_n c_k F / R, |F| <= fmax(R): columns rounded one at a time, each one's error pushed onto the later columns through
    the inverse input second moment H (centred, as the trits were rounded). Returns F (int8) and the output-space
    relative error of what is left over that of E"""
    N, K = E.shape
    Wc = E / c[None, :]
    Hc = H.to(E.device).float() * c[:, None] * c[None, :]
    dead = torch.diagonal(Hc) <= 0
    Hc[dead, dead] = 1.0
    Wc[:, dead] = 0.0
    Hc += damp * torch.diagonal(Hc).mean() * torch.eye(K, device=E.device)
    perm = torch.argsort(torch.diagonal(Hc), descending=True) if act_order else torch.arange(K, device=E.device)
    inv = torch.argsort(perm)
    Wp, Hp = Wc[:, perm].clone(), Hc[perm][:, perm]
    L = torch.linalg.cholesky(Hp)
    U = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    del L
    u = (a / R)[:, None]
    fm = fmax(R)
    Fq = torch.zeros(N, K, device=E.device)
    Q = Wp.clone()
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        Wb = Q[:, i1:i2].clone(); Ub = U[i1:i2, i1:i2]; Eb = torch.zeros_like(Wb)
        for j in range(i2 - i1):
            w = Wb[:, j]
            f = (w / u[:, 0]).round().clamp(-fm, fm)
            Fq[:, i1 + j] = f
            e = (w - u[:, 0] * f) / Ub[j, j]
            Wb[:, j:] -= e[:, None] * Ub[j, j:][None, :]
            Eb[:, j] = e
        Q[:, i2:] -= Eb @ U[i1:i2, i2:]
    Rm = Wp - u * Fq
    e1 = float(((Rm @ Hp) * Rm).sum() / ((Wp @ Hp) * Wp).sum().clamp_min(1e-30))
    return Fq[:, inv].to(torch.int8), e1
