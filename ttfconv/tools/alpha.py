""" why does the K-FAC second-order damage under-predict the measured KL of rounding one linear by
5-25x? For a few linears, with E = centred-GPTQ error (fold-a basis), on held-out windows:
  measured KL(fp || fp with W + alpha E) for alpha in --alphas      quadratic regime iff KL ~ alpha^2
  K-FAC      0.5 tr(G E H E^T) alpha^2           (G, H from calibration windows, delta and x assumed independent)
  GN-token   0.5 mean_t (d_t^T E x_t)^2 alpha^2   (Gauss-Newton per token on the held-out windows, labels sampled; no
                                                  independence assumption, cross-token terms still dropped)
  python -m ttfconv.tools.alpha --model M --data data/broadqw_train.bin --val data/wikiqw_val.bin --out alpha.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from ttfconv.tools.kfac import linears, fold_model, windows, sample_labels, stats_pass, kfac_val, kl_pass

PICK = ["model.layers.0.linear_attn.out_proj", "model.layers.3.self_attn.o_proj", "model.layers.3.mlp.down_proj",
        "model.layers.8.mlp.gate_proj", "model.layers.11.self_attn.q_proj", "model.layers.12.linear_attn.in_proj_qkv",
        "model.layers.18.mlp.up_proj", "model.layers.23.mlp.down_proj"]


def gn_token(hf, lin, E, wins, dev, gen):
    """0.5 mean over tokens of (d_t . E x_t)^2, d = dl/dy under sampled labels"""
    acc = {"s": 0.0, "n": 0}; xs = {}

    def fh(mod, inp, out):
        xs["x"] = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if out.requires_grad:
            def gh(g):
                y = xs["x"] @ E.T                                          # E x_t
                s = (g.reshape(-1, g.shape[-1]).float() * y).sum(-1)
                acc["s"] += float((s * s).sum()); acc["n"] += s.numel()
            out.register_hook(gh)
    h = lin.register_forward_hook(fh)
    eh = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_())
    for w in wins:
        t = w.to(dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = hf(t).logits[0]
        y = sample_labels(lg.detach(), gen)
        F.cross_entropy(lg.float(), y, reduction="sum").backward()
    h.remove(); eh.remove()
    return 0.5 * acc["s"] / max(acc["n"], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--ncal", type=int, default=16); ap.add_argument("--nval", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--alphas", default="0.1,0.3,1")
    ap.add_argument("--pick", default=",".join(PICK)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from ttf.hf_init import ternarize, ternarize_gptq
    dev = "cuda"; t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    fold_model(hf, dev)
    cal = windows(a.data, a.ncal, a.seq, 4321); val = windows(a.val, a.nval, a.seq, 777)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = [hf(w.to(dev)[None]).logits[0].to(torch.bfloat16).cpu() for w in val]
    L = linears(hf); gen = torch.Generator(device=dev).manual_seed(5)
    alphas = [float(x) for x in a.alphas.split(",")]; rows = []
    for n in a.pick.split(","):
        l = L[n]; St = stats_pass(hf, {n: l}, cal, dev, gen); s = St[n]
        mu = s["s"] / s["n"]; Hc = s["H"] / s["n"] - torch.outer(mu, mu); G = s["G"] / s["n"]
        W = l.weight.float()
        T, aa, cc = ternarize(W, "als"); T, aa, cc = ternarize_gptq(W, Hc, aa, cc)
        E = aa[:, None] * cc[None, :] * T.float() - W
        row = {"name": n, "kfac1": kfac_val(G, E, Hc), "gn1": gn_token(hf, l, E, val, dev, gen), "kl": {}}
        old = l.weight.detach().clone()
        for al in alphas:
            with torch.no_grad():
                l.weight.copy_((W + al * E).to(l.weight.dtype))
            row["kl"][str(al)] = kl_pass(hf, val, dev, ref)
        with torch.no_grad():
            l.weight.copy_(old)
        k = row["kl"]; row["exp_0.1_to_1"] = float(np.log(k["1.0"] / k["0.1"]) / np.log(10)) if {"0.1", "1.0"} <= set(k) else None
        print(f"{n[13:]}: KL {' '.join(f'{x}:{v:.5f}' for x, v in k.items())}  kfac {row['kfac1']:.5f} gn {row['gn1']:.5f}  "
              f"slope(log KL / log alpha) {row['exp_0.1_to_1']}  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(row); json.dump(rows, open(a.out, "w"), indent=1)
        del St, G, Hc, E, W, T, old
        torch.cuda.empty_cache()
    print("ALPHA_DONE", flush=True)


if __name__ == "__main__":
    main()
