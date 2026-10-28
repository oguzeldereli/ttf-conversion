"""ttfconv.fine (the fine residual) and ttfconv.gptq: the digits, the fine GPTQ, the float-adapter fit on the CPU; the
fine layer's forward and input gradient against a dense float reference on the GPU."""
import math

import pytest
import torch

from ttfconv import fine as Fi


def test_digits_roundtrip():
    F = torch.arange(-121, 122, dtype=torch.int16).to(torch.int8)
    D = Fi.digits(F)
    assert D.shape == (Fi.PLANES, 243)
    assert set(D.unique().tolist()) <= {-1, 0, 1}
    back = sum((3 ** j) * D[j].to(torch.int64) for j in range(Fi.PLANES))
    assert torch.equal(back, F.to(torch.int64))
    assert Fi.fmax(27) == 100 and Fi.fmax(81) == 46


def test_plane_bits():
    n, p = 1000, 0.1
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert Fi.plane_bits(0, n) == 0.0
    assert abs(Fi.plane_bits(100, n) - n * (h + p)) < 1e-9


def test_gptq_fine_reduces_error():
    torch.manual_seed(0)
    N, K = 32, 64
    X = torch.randn(4096, K) @ torch.randn(K, K) * 0.3
    H = X.T @ X / X.shape[0]
    E = torch.randn(N, K) * 0.05
    a = torch.full((N,), 0.1); c = torch.ones(K)
    F, e1 = Fi.gptq_fine(E, H, a, c, 27)
    assert F.dtype == torch.int8 and int(F.abs().max()) <= Fi.fmax(27)
    left = E - a[:, None] * F.float() / 27
    e_ref = float(((left @ H) * left).sum() / ((E @ H) * E).sum())
    assert abs(e1 - e_ref) < 1e-3 and e1 < 0.02           # step 0.1 / 27: the residual nearly gone


def test_mag_svd_fit_returns_factors():
    from ttfconv.gptq import ternarize, mag_svd_fit
    torch.manual_seed(1)
    N, K, r = 48, 64, 8
    W = torch.randn(N, K)
    X = torch.randn(2048, K)
    H = X.T @ X / X.shape[0]
    T, a, c = ternarize(W, "als")
    (A, B), share = mag_svd_fit(W, T, a, c, 1.0, H, r)
    assert A.shape == (N, r) and B.shape == (K, r) and A.device.type == "cpu"
    assert 0 < share < 1
    G = (W - a[:, None] * c[None, :] * T.float()) / c[None, :] / a[:, None]
    Hp = c[:, None] * H * c[None, :]
    e0 = float(((G @ Hp) * G).sum()); R = G - A @ B.T; e1 = float(((R @ Hp) * R).sum())
    assert abs((1 - e1 / e0) - share) < 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("dx", ["bf16", "int8x2"])
def test_fine_layer_matches_dense(dx):
    from ttf.layers import KernelTernaryLinear
    from ttf import kernel as Kn
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    dev = "cuda"
    torch.manual_seed(1)
    K, N, R = 256, 128, 27
    l = KernelTernaryLinear(K, N, 8, beta=True).to(dev); l.enable_rc_scale()
    with torch.no_grad():
        l.row_scale.uniform_(0.5, 1.5); l.col_scale.uniform_(0.5, 1.5)
    x = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
    y_plain = l(x)
    Fi.enable_fine(l, R)
    assert torch.equal(l(x), y_plain)                       # F = 0: the plain layer's output
    l.wfine.copy_(torch.randint(-Fi.fmax(R), Fi.fmax(R) + 1, (N, K), dtype=torch.int8, device=dev)); l._after_fine()
    T = Kn.unpack_rows(l.wpacked, K).float()
    beta = float(l._cached_beta(l.wpacked))
    W = l.row_scale.detach()[:, None] * beta * (T + l.wfine.float() / R) * l.col_scale.detach()[None, :]
    old = Fi.DX_MODE[0]; Fi.DX_MODE[0] = dx
    try:
        xr = x.clone().requires_grad_(True)
        y = l(xr)
        gy = torch.randn_like(y)
        l.capture = True
        y.backward(gy)
    finally:
        Fi.DX_MODE[0] = old; l.capture = False
    xc = x * l.col_scale.detach().to(torch.bfloat16)[None, :]          # (the layer rounds its scales to bf16)
    xq, xs = Kn.act_quant_i8(xc)
    yref = (l.row_scale.detach().to(torch.bfloat16).float()[None, :]
            * ((xq.float() * xs[:, None]) @ (beta * (T + l.wfine.float() / R)).T))
    assert (y.float() - yref).abs().max() / yref.abs().max() < 1e-2
    gxref = gy.float() @ W
    assert (xr.grad.float() - gxref).abs().max() / gxref.abs().max() < 2e-2
    assert l.gw is not None and l.gw.shape == (N, K)
