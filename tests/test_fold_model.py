"""The fold end to end on a tiny random Qwen3.5 (fp32; on the GPU when there is one: the DeltaNet kernels are Triton):
logits identical before and after fold_float + fold_w on every linear, and every linear that reads or writes the
residual stream is one the fold knows (a reader it skipped would leave that layer seeing the rotated stream unrotated).
Needs a transformers with qwen3_5 (skipped otherwise)."""
import types
import pytest
import torch
import torch.nn as nn

Q5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from ttfconv.fold import Fold, READ_IN, READ_POST, WRITE, fold_float, fold_w, residual_q

FLOAT_READERS = ("in_proj_a", "in_proj_b")      # rotated by fold_float (kept float by the core)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(d=64, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=d, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_num_key_heads=2, linear_num_value_heads=2, tie_word_embeddings=True,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    m = Q5.Qwen3_5ForCausalLM(cfg).float().eval().to(DEV)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("layernorm.weight") or n == "model.norm.weight":
                p.normal_(0, 0.3)                    # zero-centred gains (1 + w) away from 1, so a missed gain shows
    return m


@pytest.mark.parametrize("d", [64, 80])          # 80: not a power of 2 (rand_orth(5) kron Hadamard(16), as 27B's 5120)
def test_fold_whole_model_exact(d):
    m = tiny(d)
    x = torch.randint(0, 97, (2, 12), device=DEV)
    with torch.no_grad():
        ref = m(x).logits
        f = Fold("")
        f.build(types.SimpleNamespace(hf=m), {}, None, print)
        Qr = residual_q(d, torch.Generator().manual_seed(2110)).to(DEV)
        G = fold_float(m, Qr)
        seen = set()
        for i, layer in enumerate(m.model.layers):
            for n, mod in layer.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                leaf = n.rsplit(".", 1)[-1]
                if leaf in READ_IN + READ_POST + WRITE:
                    mod.weight.copy_(fold_w(leaf, mod.weight.float(), *G[i], Qr))
                    seen.add(leaf)
                else:
                    assert leaf in FLOAT_READERS, f"layer {i}: linear {n} is not handled by the fold"
        out = m(x).logits
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj",
            "down_proj"} <= seen, seen
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), float((out - ref).abs().max())
