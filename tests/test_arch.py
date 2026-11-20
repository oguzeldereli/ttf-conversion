"""ttfconv.arch: the residual-stream fold found from the architecture alone, on tiny random models of several HF
families (CPU, fp32). Exact families must reproduce the logits under a random rotation; post-norm and LayerNorm
families must be reported as having no exact rotation. Qwen3.5's discovered rule must equal the hand-written one in
ttfconv.fold (the readers READ_IN / READ_POST + in_proj_a / b, the writers WRITE, zero-centred gains)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import CONFIG_MAPPING
from ttfconv.arch import hf_class, discover, tiny_model, apply_fold, random_orth, rule_for, decoder_layers
from ttfconv.fold import READ_IN, READ_POST, WRITE


def small(name, **over):
    cfg = hf_class(name, "config")()
    t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    small_ = dict(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512,
                  moe_intermediate_size=128, num_experts=4, num_local_experts=4, num_experts_per_tok=2)
    if getattr(t, "layer_types", None) is None:     # (a per-layer type list fixes the depth: keep it)
        small_["num_hidden_layers"] = 2           # (the default depth / expert count is minutes on a CPU: MoE 48 x 128)
    for k, v in dict(small_, **over).items():
        if hasattr(t, k):
            setattr(t, k, v)
    return cfg


@pytest.mark.parametrize("name", ["llama", "qwen3", "qwen3_5", "qwen3_moe", "mixtral", "gpt_oss", "phi3", "gemma",
                                  "bitnet"])
def test_exact(name):
    if name not in CONFIG_MAPPING:
        pytest.skip(f"transformers has no {name}")
    r = discover(small(name), log=lambda s: None)
    assert r["rotation"], r["reason"]
    assert r["fold_err"] < 1e-4


@pytest.mark.parametrize("name,why", [("gemma2", "post-norm"), ("olmo2", "post-norm"), ("starcoder2", "commute"),
                                      ("cohere", "commute")])
def test_no_rotation(name, why):
    if name not in CONFIG_MAPPING:
        pytest.skip(f"transformers has no {name}")
    r = discover(small(name), log=lambda s: None)
    assert not r["rotation"] and why in r["reason"], r["reason"]


def test_qwen35_rule_matches_hand_fold():
    if "qwen3_5" not in CONFIG_MAPPING:
        pytest.skip("transformers has no qwen3_5")
    r = discover(small("qwen3_5"), log=lambda s: None)
    readers, writers, forms = set(), set(), set()
    for rule in r["kinds"].values():
        for seg in rule:
            forms.add(seg["form"])
            readers |= {p.split(".")[-2] for p, a in seg["in"]}
            writers |= {p.split(".")[-2] for p, a in seg["out"]}
    assert forms == {"1+w"}
    assert readers == set(READ_IN + READ_POST) | {"in_proj_a", "in_proj_b"}, readers
    assert writers == set(WRITE), writers
    assert len(r["kinds"]) == 2                     # linear-attention and full-attention layers


def test_fold_on_other_seed_model():
    """rules found on one random instance fold a differently initialised instance of the same config exactly"""
    cfg = small("qwen3_moe")
    r = discover(cfg, seed=0, log=lambda s: None)
    m = tiny_model(cfg, seed=7)
    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        ref = m(input_ids=x).logits
        apply_fold(m, r, random_orth(r["d"], torch.Generator().manual_seed(3)))
        out = m(input_ids=x).logits
    assert float((out - ref).abs().max() / ref.abs().max()) < 1e-4
    assert all(rule_for(r, l) is not None for l in decoder_layers(m)[1])
