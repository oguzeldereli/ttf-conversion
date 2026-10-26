"""CPU checks of the conversion maths (no GPU, no model download): the fold is exact, the spec parser, rank / byte
accounting, and the adapter fit on a small random layer (it must remove a positive share of the residual's
H-weighted energy, and the alternating refit must not make it worse)."""
import pytest
import torch

from ttfconv.spec import parse
from ttfconv.fold import hadamard, rand_orth, residual_q, fold_w
from ttfconv.adapters import rank_of


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


def test_fqb_levels_and_fq4():
    """fqb: at most 2^bits values per group, the error shrinks with the bits; fq4 is fqb at 4 bits"""
    from ttfconv.finish import fqb, fq4
    torch.manual_seed(0)
    W = torch.randn(32, 256)
    errs = []
    for b in (2, 3, 4):
        Q = fqb(W, b)
        assert max(len(torch.unique(r)) for r in Q.reshape(-1, 64)) <= 2 ** b
        errs.append(float((Q - W).norm() / W.norm()))
    assert errs[0] > errs[1] > errs[2]
    assert torch.equal(fq4(W), fqb(W, 4))


def test_embq_and_adapter_ttf_rank_specs():
    from ttfconv.embq import ttf_ext as embq
    from ttfconv.adapters import ttf_ext as ad
    e = embq("emb=2,head=3"); assert (e.eb, e.hb, e.g) == (2, 3, 64)
    assert ad("rank=0.2,alt=1").ttf_rank is None and ad("rank=0.2,alt=1,ttf_rank=64").ttf_rank == 64


def test_top_svd_rand_matches_full():
    """svd=rand: the randomized top-r subspace keeps the full SVD's top-r energy (flat-ish spectrum, as a rounding
    residual's)"""
    from ttfconv.adapters import top_svd
    torch.manual_seed(0)
    A = torch.randn(300, 200) + torch.randn(300, 8) @ torch.randn(8, 200)
    r = 40
    _, S0, _ = top_svd(A, r, "full"); _, _, Vh = top_svd(A, r, "rand")
    assert Vh.shape == (r, 200)
    assert float(((A @ Vh.T) ** 2).sum()) / float((S0 ** 2).sum()) > 0.995


def test_unique_sd_drops_aliases():
    from ttfconv.finish import unique_sd
    m = torch.nn.Module(); m.hf = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)); m.layers = m.hf
    sd = unique_sd(m)
    assert len(m.state_dict()) == 8 and sorted(sd) == ["hf.0.bias", "hf.0.weight", "hf.1.bias", "hf.1.weight"]


def test_adapter_build():
    """build wraps every swapped linear as lin + up(down) at the planned rank; ttf_rank goes to the factors only; the
    removed options (t2, shrink, lin_rank: 25 Oct) are refused"""
    import types
    import torch.nn as nn
    from ttfconv.adapters import ttf_ext as ad

    class Lin(nn.Module):
        def __init__(self, K, N):
            super().__init__(); self.K, self.N = K, N; self.w = nn.Parameter(torch.zeros(1))

    layer = nn.Module(); layer.mlp = nn.Module()
    layer.mlp.down_proj, layer.mlp.gate_proj = Lin(96, 64), Lin(64, 96)
    hf = nn.Module(); hf.model = nn.Module(); hf.model.layers = nn.ModuleList([layer])
    model = nn.Module(); model.hf = hf
    sw = {"model.layers.0.mlp.down_proj": layer.mlp.down_proj, "model.layers.0.mlp.gate_proj": layer.mlp.gate_proj}
    out = ad("rank=8,alt=1,ttf_rank=4").build(model, sw, lambda K, N: Lin(K, N), lambda m: None)
    d, g = out["model.layers.0.mlp.down_proj"], out["model.layers.0.mlp.gate_proj"]
    assert (d.ad[0].K, d.ad[0].N, d.ad[1].K, d.ad[1].N) == (96, 8, 8, 64) and g.ad[0].N == 8
    assert d.ad[0].ts_rank == d.ad[1].ts_rank == 4 and not hasattr(d.lin, "ts_rank")
    assert layer.mlp.down_proj is d and not hasattr(d, "lin2")
    for bad in ("t2=0.229", "shrink=q_proj@0.05", "lin_rank=down_proj@1024"):
        with pytest.raises(SystemExit):
            ad("rank=0.2,alt=1," + bad)


def test_adapter_ctl():
    """adapter_ctl: frozen marks both factors ts_frozen with no trainable parameters, the main layer untouched; off also
    takes the adapters out of the forward; the plug-in is not structural (no build: the checkpoint's --ext list holds)"""
    import torch.nn as nn
    from ttfconv.adapters import QTad
    from ttfconv.adapter_ctl import ttf_ext as ctl

    class Lin(nn.Module):
        def __init__(self, K, N):
            super().__init__(); self.K, self.N = K, N; self.w = nn.Parameter(torch.randn(N, K))

        def forward(self, x, acc=None):
            y = x @ self.w.T
            return y if acc is None else acc + y

    for mode in ("frozen", "off"):
        torch.manual_seed(0)
        q = QTad(Lin(16, 12), Lin(16, 4), Lin(4, 12)); m = nn.Sequential(q)
        x = torch.randn(3, 16)
        y0 = q(x)
        c = ctl(f"mode={mode}")
        assert not hasattr(c, "build")
        c.setup(m, lambda s: None)
        assert all(l.ts_frozen and not l.w.requires_grad for l in q.ad) and q.lin.w.requires_grad
        assert not hasattr(q.lin, "ts_frozen")
        y = q(x)
        assert torch.equal(y, y0 if mode == "frozen" else q.lin(x))
    with pytest.raises(AssertionError):
        ctl("mode=fast")
