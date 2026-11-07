"""End-to-end check of the Q38 pipeline on a tiny random model written in the FP8 release's layout:
ttf.q38.store.Store + MmapTable == the model with the same (dequantised) weights in memory; ttfconv.q38.teacher --check
and a small cache; ttfconv.q38.convert on 3 layers, its state dicts load into ttf.q38.moe.TernaryExperts and reproduce
the fake-quantised layer.
  python tests/checks/q38_pipeline.py [REAL_CONFIG.json [WORKDIR]]   (default: built-in fields, a temporary WORKDIR)"""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttf.q38 import model as M

real = json.load(open(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1] else {"text_config": {}}
wd = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="q38_pipeline_")
os.makedirs(wd, exist_ok=True)
ck = os.path.join(wd, "ck")
os.makedirs(ck, exist_ok=True)
t = dict(real["text_config"])
t.update(hidden_size=128, hc_count=4, hc_lowrank=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
         head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
         num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
         vocab_size=1000, ngram_vocab_size_base=101, heads_per_ngram=2, ple_embed_dim=128, eos_token_id=999,
         bos_token_id=999, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=2048,
         layer_types=["linear_attention"] * 3 + ["full_attention"], ple_layer_ids=[2], split_ngram_parts=4,
         max_position_embeddings=4096)
for k, v in dict(ngram_size=3, ple_conv_kernel_size=3, hidden_act="silu", rms_norm_eps=1e-6).items():
    t.setdefault(k, v)          # (a real config.json, when given, keeps its own values)
cfgj = {"text_config": t}
json.dump(cfgj, open(os.path.join(ck, "config.json"), "w"))
cfg = M.text_config(cfgj)
torch.manual_seed(0)
m = M.build(cfg, device="cpu")
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("A_log"):
            p.copy_(torch.randn_like(p) * 0.1)
        elif n.endswith("norm.weight") and "linear_attn" in n:
            p.copy_(1 + 0.1 * torch.randn_like(p))
        elif p.dim() >= 2:
            p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.1)
P = M.PFX
sd = {}


def fp8(W):
    N, K = W.shape
    s = W.float().view(N // 128, 128, K // 128, 128).abs().amax((1, 3)).clamp_min(1e-8) / 448
    q = (W.float().view(N // 128, 128, K // 128, 128) / s[:, None, :, None]).view(N, K).to(torch.float8_e4m3fn)
    return q, s, (q.float().view(N // 128, 128, K // 128, 128) * s[:, None, :, None]).view(N, K)


sd["lm_head.weight"] = m.lm_head.weight.data.bfloat16()
sd[P + "embed_tokens.weight"] = m.embed_tokens.weight.data.bfloat16()
for n, p in m.hyper_connection_mixer.named_parameters():
    sd[P + "hyper_connection_mixer." + n] = p.data.bfloat16()
tables = {}
for i, L in enumerate(m.layers):
    p = f"{P}layers.{i}."
    I = cfg.moe_intermediate_size
    for n, q in L.named_parameters():
        if n.startswith("mlp.gu.") or n.startswith("mlp.dn."):
            continue
        sd[p + n] = q.data.bfloat16()
    for e in range(cfg.num_experts):
        for nm, W in (("gate_proj", L.mlp.gu.weight.data[e, :I]), ("up_proj", L.mlp.gu.weight.data[e, I:]),
                      ("down_proj", L.mlp.dn.weight.data[e])):
            q, s, Wd = fp8(W)
            sd[f"{p}mlp.experts.{e}.{nm}.weight"] = q; sd[f"{p}mlp.experts.{e}.{nm}.weight_scale_inv"] = s
            W.copy_(Wd)                      # the reference model uses the dequantised weights
    if L.ple is not None:
        ng = L.ple.ple_embedding
        tab = torch.randn(ng.rows, ng.dim)
        sc = tab.abs().max() / 448
        q8 = (tab / sc).to(torch.float8_e4m3fn)
        ng.table, ng.scale = q8.float() * sc, None
        parts = cfg.split_ngram_parts
        rows = (ng.rows + parts - 1) // parts
        for j in range(parts):
            sd[f"{p}ple.ple_embedding.ngram_embedding.shard_{j}.weight"] = q8[j * rows:(j + 1) * rows].contiguous()
        sd[f"{p}ple.ple_embedding.ngram_embedding.weight_scale"] = sc.reshape(())
        sd[f"{p}ple.ple_embedding.layer_multipliers"] = ng.mult
        sd[f"{p}ple.ple_embedding.ngram_heads_vocab_sizes"] = ng.sizes
        sd[f"{p}ple.ple_embedding.ngram_heads_offsets"] = ng.offs
# two shards + index (as the release)
keys = sorted(sd)
half = len(keys) // 2
wm = {}
for fi, ks in enumerate((keys[:half], keys[half:])):
    fn = f"model-{fi:05d}.safetensors"
    save_file({k: sd[k].contiguous() for k in ks}, os.path.join(ck, fn))
    wm.update({k: fn for k in ks})
json.dump({"weight_map": wm}, open(os.path.join(ck, "model.safetensors.index.json"), "w"))
data = np.random.default_rng(0).integers(0, 999, 200_000).astype(np.uint32)
data[::97] = 999
data.tofile(os.path.join(wd, "data.bin"))
print("tiny checkpoint written", flush=True)

# reference: the in-memory model (bf16, dequantised weights) on the same windows as teacher --check
dev = "cuda"
mr = m.to(torch.bfloat16).to(dev)
for L in mr.layers:
    if L.ple is not None:
        L.ple.ple_embedding.table = L.ple.ple_embedding.table.to(dev)
rng = np.random.default_rng(1234)
st = rng.integers(0, len(data) - 256 - 1, 64)
w = np.stack([data[s:s + 257].astype(np.int64) for s in st[:2]])
x, y = torch.from_numpy(w[:, :-1]).to(dev), torch.from_numpy(w[:, 1:]).to(dev)
with torch.no_grad():
    ce = [torch.nn.functional.cross_entropy(mr(x[b:b + 1])[0].float(), y[b]).item() for b in range(2)]
print("reference loss", np.mean(ce), [round(c, 3) for c in ce], flush=True)
env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
r = subprocess.run([sys.executable, "-m", "ttfconv.q38.teacher", ck, os.path.join(wd, "data.bin"), "-",
                    "--check", "2", "--mb", "1", "--seq", "256"], env=env, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
assert r.returncode == 0
line = [l for l in r.stdout.splitlines() if "PLE on" in l][0]
got = float(line.split("next-token loss ")[1].split()[0])
print("teacher check loss", got, "vs reference", np.mean(ce))
assert abs(got - np.mean(ce)) < 0.02, "Store path differs from the in-memory model"
r = subprocess.run([sys.executable, "-m", "ttfconv.q38.teacher", ck, os.path.join(wd, "data.bin"),
                    os.path.join(wd, "kd"), "--windows", "8", "--batch", "4", "--mb", "2", "--seq", "256"], env=env,
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:], flush=True)
assert r.returncode == 0 and os.path.exists(os.path.join(wd, "kd", "done"))
z = np.load(os.path.join(wd, "kd", "kd_00000.npz"))
assert z["ids"].shape == (4, 256, 64) and z["sids"].shape == (4, 256, 64), {k: z[k].shape for k in z}
# the cached top-1 equals the reference model's argmax
w0 = torch.from_numpy(np.stack([data[s:s + 256].astype(np.int64) for s in z["starts"][:1]])).to(dev)
with torch.no_grad():
    lpr = torch.log_softmax(mr(w0)[0].float(), -1)
gap = (lpr.max(-1).values - lpr.gather(1, torch.from_numpy(z["ids"][0, :, :1]).long().to(dev))[:, 0]).cpu()
print("top-1 agreement with the reference", float((gap == 0).float().mean()), "max logprob gap", float(gap.max()),
      "cached vs reference top-1 logprob", float((torch.from_numpy(z["lp"][0, :, 0]).float() - lpr.max(-1).values.cpu()).abs().max()))
assert float(gap.max()) < 0.15   # bf16 noise across batch shapes
r = subprocess.run([sys.executable, "-m", "ttfconv.q38.convert", ck, os.path.join(wd, "data.bin"),
                    os.path.join(wd, "conv"), "--windows", "16", "--seq", "256", "--mb", "4", "--chunk", "4", "--rank", "8"],
                   env=env, capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-3000:], flush=True)
assert r.returncode == 0 and os.path.exists(os.path.join(wd, "conv", "done"))
# the state dicts load into ttf.q38.moe.TernaryExperts and reproduce W^ (x - mu) + W mu up to the int8 activations
from ttf.q38.moe import TernaryExperts
sdc = torch.load(os.path.join(wd, "conv", "L0.pt"))
for name in ("mlp.gu", "mlp.dn", "linear_attn.in_proj_qkv"):
    s = sdc[name]
    E, N, Kp = s["wpacked"].shape
    K = s["col_scale"].shape[1]
    te = TernaryExperts(E, K, N, centring=True, adapter_rank=s["ad_cb"].shape[1], device=dev)
    te.load_state_dict({k: v.to(dev) for k, v in s.items()}, strict=True)
    te.beta_nnz = True
    W = (mr.layers[0].get_submodule(name).weight.float() if name.startswith("mlp.") else
         mr.layers[0].get_submodule(name).weight.float()[None])
    xx = torch.randn(64 * E, K, device=dev) * 0.5 + s["in_mean"].to(dev).repeat_interleave(64, 0)
    offs = torch.arange(0, 64 * E + 1, 64, device=dev)
    with torch.no_grad():
        yt = te(xx.to(torch.bfloat16), offs).float()
        yf = torch.cat([xx[64 * e:64 * e + 64] @ W[e].T for e in range(E)])
    rel = float((yt - yf).norm() / yf.norm())
    print(f"{name}: TernaryExperts vs float layer rel err {rel:.3f}")
    assert rel < 0.6, rel
print("PIPELINE OK")
