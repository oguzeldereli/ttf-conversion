"""CPU checks of the conversion maths (no GPU, no model download): the fold is exact, the spec parser, rank / byte
accounting, and the adapter fit on a small random layer (it must remove a positive share of the residual's
H-weighted energy, and the alternating refit must not make it worse)."""
import pytest
import torch

from ttfconv.spec import parse
from ttfconv.fold import hadamard, rand_orth, residual_q, fold_w
from ttfconv.adapters import rank_of, abytes, plan_ranks


def test_spec_parse():
    o = parse("rank=0.3,alt=1,eval=wiki@f@4@2048", {"rank": "0.2", "eval": ""}, required=("alt",))
    assert o == {"rank": "0.3", "alt": "1", "eval": "wiki@f@4@2048"}
    with pytest.raises(SystemExit):
        parse("bogus=1", {"rank": "0.2"})
    with pytest.raises(SystemExit):
        parse("rank=0.2", {"rank": "0.2"}, required=("alt",))


@pytest.mark.parametrize("d", [64, 96, 320])
def test_residual_rotation_orthogonal(d):
    g = torch.Generator().manual_seed(0)
    Q = residual_q(d, g)
    assert torch.allclose(Q @ Q.T, torch.eye(d), atol=1e-5)
    H = hadamard(64, torch.Generator().manual_seed(1))
    assert torch.allclose(H @ H.T, torch.eye(64), atol=1e-5)
    R = rand_orth(7, torch.Generator().manual_seed(2))
    assert torch.allclose(R @ R.T, torch.eye(7), atol=1e-5)


def test_fold_is_exact():
    """reader: W (g * x) == fold_w(W) (Q x); writer: Q (W x) == fold_w(W) x (the residual stream rotated by Q)"""
    torch.manual_seed(0)
    d, n = 64, 48
    Q = residual_q(d, torch.Generator().manual_seed(3))
    gi, gp = 1 + 0.1 * torch.randn(d), 1 + 0.1 * torch.randn(d)
    x = torch.randn(5, d)
    W = torch.randn(n, d)
    for leaf, g in (("q_proj", gi), ("gate_proj", gp)):
        assert torch.allclose((x * g) @ W.T, (x @ Q.T) @ fold_w(leaf, W, gi, gp, Q).T, atol=1e-4)
    Ww = torch.randn(d, n); h = torch.randn(5, n)
    assert torch.allclose((h @ Ww.T) @ Q.T, h @ fold_w("down_proj", Ww, gi, gp, Q).T, atol=1e-4)


def test_ranks_and_bytes():
    assert rank_of(1024, 3584, 0.2) == 208            # 0.2 * 1024, a multiple of 8
    assert rank_of(64, 64, 0.01) == 8                 # at least 8
    assert rank_of(1024, 1024, 96) == 96              # an absolute rank
    assert abytes(1024, 3584, 208) == 208 * 205 + 3584 * 42 + 2 * (2 * 208 + 1024 + 3584)
    shapes = {"model.layers.0.self_attn.q_proj": (1024, 2048), "model.layers.5.mlp.up_proj": (1024, 3584)}
    assert plan_ranks(shapes, "0.2") == {k: rank_of(K, N, "0.2") for k, (K, N) in shapes.items()}


def test_adapter_fit_removes_energy():
    from ttf.hf_init import ternarize, ternarize_gptq
    from ttf.layers import KernelTernaryLinear
    from ttfconv.adapters import fit_tadapt
    torch.manual_seed(0)
    K, N, r = 96, 80, 16
    X = torch.randn(512, K) @ torch.randn(K, K) / K ** 0.5
    H = X.T @ X / X.shape[0]
    W = torch.randn(N, K) / K ** 0.5
    T, a, c = ternarize(W, "als"); T, a, c = ternarize_gptq(W, H, a, c)
    R = W - a[:, None] * c[None, :] * T.float()
    errs = []
    for alt in (0, 1):
        ad = torch.nn.ModuleList([KernelTernaryLinear(K, r, beta=True), KernelTernaryLinear(r, N, beta=True)])
        for l in ad:
            l.enable_rc_scale()
        e, hist = fit_tadapt(ad, R, H, alt, ternarize, ternarize_gptq)
        assert 0.0 < e < 1.0, e                      # a share of the residual's H-energy is left
        errs.append(e)
    assert errs[1] <= errs[0] + 1e-6                 # the alternating refit keeps the best round
