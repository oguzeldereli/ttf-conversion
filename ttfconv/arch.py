"""Architecture discovery for any Hugging Face causal LM: where the decoder layers, the embedding, the final norm and the
head are, which norms sit on the residual stream, which parameter axes carry the residual stream, and whether the
residual-stream rotation of fold a (ttfconv.fold) is exact for the architecture.

Nothing here is per model or per parameter name. The rules are found on a tiny random instance of the same config
(a few decoder layers, small vocabulary and MLP width, random norm gains and biases) from one traced forward pass:
  - a residual norm: a norm module in a decoder layer whose input is the residual stream at that point;
  - a branch: a direct child of the layer whose input is a residual norm's output and whose output (alone, or summed
    with a second branch for parallel blocks) is the change of the stream up to the next residual point;
  - the branch's input axes: the parameter axes of size d that contract with the branch input. Found as the set of
    axes that, scaled by a diagonal D, absorb an input scaled by D^-1 (output unchanged);
  - the branch's output axes: the further axes of size d that must turn with a rotation Q so that
    branch'(x Q^T) = branch(x) Q^T;
  - the gain form of each norm: w or (1 + w).
Then the whole fold (gains into the input axes, Q on every residual axis, the embedding, the head) is applied to the
tiny model with a random orthogonal Q and the logits compared with the original. Only a fold that reproduces them
(fp32, rel 1e-4) is reported as exact. Otherwise the rules say rotation=False and why, and the converter leaves the
model unrotated.

Post-norm blocks (a norm between a branch and the add, e.g. Gemma 2 / 3, OLMo 2) and LayerNorm models have no exact
residual rotation: the gain after the branch, or the mean subtraction, does not commute with Q. They convert unrotated.

The rules are keyed by the layer's structural signature (the sorted names of its parameters), so hybrid models
(Qwen3.5: linear-attention and full-attention layers) get one rule per layer kind.

  python -m ttfconv.arch MODEL_DIR_OR_HF_ID [--json OUT]
"""
import argparse
import copy
import itertools
import re
import json
import sys

import torch
import torch.nn as nn

NORM_HINT = ("norm", "ln")
MAX_SEARCH = 12         # the axis-subset searches are exhaustive up to this many candidate axes


# ---------------------------------------------------------------------------------------------------------------------
# finding the parts

def decoder_layers(model):
    """(dotted path, ModuleList) of the decoder layers: the ModuleList whose length is num_hidden_layers and whose items
    hold the most parameters"""
    cfg = getattr(model, "config", None)
    t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    n = getattr(t, "num_hidden_layers", None)
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and (n is None or len(mod) == n):
            size = sum(p.numel() for p in mod.parameters())
            if best is None or size > best[2]:
                best = (name, mod, size)
    if best is None:
        raise ValueError("no decoder-layer ModuleList found")
    return best[0], best[1]


def is_norm(mod):
    w = getattr(mod, "weight", None)
    cls = type(mod).__name__.lower()
    return (isinstance(w, torch.Tensor) and w.dim() == 1 and not isinstance(mod, (nn.Linear, nn.Embedding))
            and any(h in cls for h in NORM_HINT))


def signature(layer):
    return "|".join(sorted(f"{n}:{tuple(p.shape)}" for n, p in layer.named_parameters()))


def module_path(model, target):
    for n, m in model.named_modules():
        if m is target:
            return n
    return None


def cpu_reference_kernels():
    """on the CPU, make HF use its torch reference paths for ops that also have Triton / CUDA kernels (fla's gated
    delta rule, causal-conv1d, mamba-ssm): transformers wraps the torch function and calls the package's kernel when
    the package imports, and those kernels fail on CPU tensors. Every such wrapper in the loaded modeling modules is
    replaced by the torch function it wraps."""
    for mn, mod in list(sys.modules.items()):
        if not (mn.startswith("transformers.models.") and mn.rsplit(".", 1)[-1].startswith("modeling_")):
            continue
        for n, f in list(vars(mod).items()):
            w = getattr(f, "__wrapped__", None)
            code = getattr(f, "__code__", None)
            if w is not None and code is not None and code.co_filename.endswith("hub_kernels.py"):
                setattr(mod, n, w)


# ---------------------------------------------------------------------------------------------------------------------
# a tiny instance of the config

def tiny_config(cfg, n_layers=None, vocab=512, inter=256):
    """a copy of cfg with few layers (at least one of each layer type), a small vocabulary and MLP width; hidden size and
    heads unchanged (their relations are architecture-specific)"""
    c = copy.deepcopy(cfg)
    t = c.get_text_config() if hasattr(c, "get_text_config") else c
    types = getattr(t, "layer_types", None)
    if n_layers is None:
        if types:
            first = {}
            for i, ty in enumerate(types):
                first.setdefault(ty, i)
            n_layers = max(first.values()) + 1
        else:
            n_layers = 2
    n_layers = max(2, n_layers)
    for k in ("first_k_dense_replace",):          # keep both dense and MoE layers in the tiny model
        if isinstance(getattr(t, k, None), int) and t.num_hidden_layers > t.first_k_dense_replace:
            setattr(t, k, min(getattr(t, k), 1))
    if types:
        try:
            t.layer_types = list(types[:n_layers])
        except AttributeError:
            pass
    t.num_hidden_layers = n_layers
    for k in ("intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size", "ffn_hidden_size"):
        if isinstance(getattr(t, k, None), int):
            setattr(t, k, min(getattr(t, k), inter))
    for k in ("num_experts", "num_local_experts", "n_routed_experts"):
        if isinstance(getattr(t, k, None), int):
            setattr(t, k, min(getattr(t, k), 4))
    if isinstance(getattr(t, "num_experts_per_tok", None), int):
        t.num_experts_per_tok = min(t.num_experts_per_tok, 2)
    for k in ("n_group", "topk_group"):
        if isinstance(getattr(t, k, None), int):
            setattr(t, k, 1)
    if isinstance(getattr(t, "vocab_size", None), int):
        t.vocab_size = min(t.vocab_size, vocab)
        for k in ("pad_token_id", "bos_token_id", "eos_token_id"):
            v = getattr(t, k, None)
            if isinstance(v, int) and v >= t.vocab_size:
                setattr(t, k, t.vocab_size - 1)
            elif isinstance(v, list):
                setattr(t, k, [min(x, t.vocab_size - 1) for x in v])
    return c


def hf_class(model_type, kind="model"):
    """transformers' own config / causal-LM class for a model_type. Other packages register classes over the same
    model_type in the Auto mappings (fla registers its own 'bitnet' when imported), so the names table is used."""
    import importlib
    from transformers.models.auto import configuration_auto as ca, modeling_auto as ma
    if kind == "config":
        name = ca.CONFIG_MAPPING_NAMES[model_type]
    else:
        name = ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
        name = name[0] if isinstance(name, (list, tuple)) else name
    mod = importlib.import_module(f"transformers.models.{ca.model_type_to_module_name(model_type)}")
    return getattr(mod, name)


def model_class(cfg):
    """(transformers' causal-LM class, the config it takes) for cfg; a multimodal config gives its text decoder. The
    class's modeling module is imported, so cpu_reference_kernels can reach it."""
    from transformers.models.auto import modeling_auto as ma
    t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    for c in ((t, cfg) if t is not cfg else (cfg,)):
        if c.model_type in ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            return hf_class(c.model_type), t
    raise ValueError(f"no transformers causal-LM class for {cfg.model_type}")


def tiny_model(cfg, seed=0, **kw):
    cls, c = model_class(tiny_config(cfg, **kw))
    cpu_reference_kernels()
    torch.manual_seed(seed)
    c._attn_implementation = "eager"
    m = cls._from_config(c, dtype=torch.float32)
    with torch.no_grad():                   # norm gains and zero-initialised parameters (biases) away from their init,
        for mod in m.modules():             # so a fold that misses them shows in the check
            if is_norm(mod):
                mod.weight.normal_(0.3 if mod.weight.abs().mean() < 0.5 else 1.0, 0.2)
                if getattr(mod, "bias", None) is not None:
                    mod.bias.normal_(0, 0.1)
        for n, p in m.named_parameters():
            if p.dim() >= 1 and not p.any():
                p.normal_(0, 0.02)
    return m.eval()


# ---------------------------------------------------------------------------------------------------------------------
# watching one forward

def _close(a, b, rtol=1e-4):
    return a.shape == b.shape and float((a - b).abs().max()) <= rtol * max(float(b.abs().max()), 1e-6)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _same(a, b):
    """the same activation: one tensor, or a view of the same storage with the same element count"""
    return a is b or (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.numel() == b.numel()
                      and a.data_ptr() == b.data_ptr())


@torch.no_grad()
def gain_form(norm, d):
    """'w' (y = w * n(x)), '1+w' (y = (1 + w) * n(x)) or None (neither)"""
    x = torch.randn(2, 3, d)
    w0 = norm.weight.detach().clone()
    try:
        norm.weight.zero_()
        y0 = _first(norm(x))
        norm.weight.fill_(-1.0)
        ym = _first(norm(x))
    finally:
        norm.weight.copy_(w0)
    b = getattr(norm, "bias", None)
    off = b if b is not None else 0
    if float((y0 - off).abs().max()) < 1e-6:
        return "w"
    if float((ym - off).abs().max()) < 1e-6:
        return "1+w"
    return None


def random_orth(d, gen):
    q, r = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))[None, :]).float()


@torch.no_grad()
def rot_equivariant(norm, d, gen):
    """n(x Q^T) == n(x) Q^T with the gain set to identity (RMSNorm yes; LayerNorm's mean subtraction no)"""
    form = gain_form(norm, d)
    if form is None or getattr(norm, "bias", None) is not None:
        return False
    Q = random_orth(d, gen)
    w0 = norm.weight.detach().clone()
    try:
        norm.weight.fill_(1.0 if form == "w" else 0.0)
        x = torch.randn(4, d)
        return _close(_first(norm(x @ Q.T)), _first(norm(x)) @ Q.T, 1e-4)
    finally:
        norm.weight.copy_(w0)


@torch.no_grad()
def trace(model, ids):
    """one forward with hooks: per decoder layer, its input / output and, in execution order, every residual-candidate
    norm's (input, output) and every direct child's (args, kwargs, output); plus the head input"""
    lpath, layers = decoder_layers(model)
    rec = {"layers": [dict(inp=None, out=None, ev=[]) for _ in layers]}
    hooks = []
    for i, layer in enumerate(layers):
        L = rec["layers"][i]
        for n, m in layer.named_children():
            def f(mod, args, kwargs, out, n=n, L=L):
                L["ev"].append(("norm" if is_norm(mod) else "child", n, args, kwargs, _first(out)))
            hooks.append(m.register_forward_hook(f, with_kwargs=True))

        def lf(mod, args, kwargs, out, L=L):
            L["inp"] = args[0] if args else kwargs["hidden_states"]
            L["out"] = _first(out)
        hooks.append(layer.register_forward_hook(lf, with_kwargs=True))
    head = model.get_output_embeddings()
    hooks.append(head.register_forward_hook(lambda m, a, o: rec.__setitem__("head_in", a[0])))
    try:
        rec["logits"] = model(input_ids=ids, use_cache=False).logits
    finally:
        for h in hooks:
            h.remove()
    return rec


def _arg0(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("hidden_states", next(iter(kwargs.values())) if kwargs else None)


def segments(L, d):
    """the residual segments of one traced layer: [(norm name, [branch names])], or (None, reason)"""
    r = L["inp"]
    pend_norms, out, used = [], [], set()
    ev = L["ev"]

    def match(delta, upto):
        """the branches (children seen before index upto, reading one of the pending norms) whose outputs sum to delta"""
        cands = []
        for k, (kind, n, a, kw, o) in enumerate(ev[:upto]):
            if kind != "child" or n in used or not isinstance(o, torch.Tensor) or o.shape[-1] != d:
                continue
            x = _arg0(a, kw)
            src = next((nn_ for nn_, no in pend_norms if _same(x, no)), None)
            if src is not None:
                cands.append((n, src, o))
        if float(delta.abs().max()) <= 1e-6 * max(float(r.abs().max()), 1e-6):
            return []
        for c in cands:
            if _close(c[2], delta):
                return [c]
        for c1, c2 in itertools.combinations(cands, 2):
            if _close(c1[2] + c2[2], delta):
                return [c1, c2]
        return None

    for k, (kind, n, a, kw, o) in enumerate(ev):
        if kind != "norm":
            continue
        x = _arg0(a, kw)
        if x is None or x.shape[-1] != d or not isinstance(o, torch.Tensor) or o.shape[-1] != d:
            continue
        hit = match(x - r, k)
        if hit is None:
            continue                    # a norm on something else (or a post-norm: caught at the layer output)
        for n_, src, _ in hit:
            used.add(n_)
            out.append((src, n_))
        r = x
        pend_norms.append((n, o))
    hit = match(L["out"] - r, len(ev))
    if hit is None:
        return None, "the layer output is not the stream plus branch outputs (a post-norm, or another non-linear write)"
    for n_, src, _ in hit:
        out.append((src, n_))
    segs = {}
    for src, br in out:
        segs.setdefault(src, []).append(br)
    return [(s, segs[s]) for s, _ in pend_norms if s in segs], None


def _axes(mod, d):
    """[(param name, axis)] of every axis of size d of the module's parameters"""
    return [(pn, ax) for pn, p in mod.named_parameters() for ax, s in enumerate(p.shape) if s == d]


def _turn(t, ax, M):
    """t with axis ax multiplied by M (t' [.., j, ..] = sum_k M[j, k] t [.., k, ..])"""
    return torch.movedim(torch.tensordot(M, torch.movedim(t, ax, 0), dims=([1], [0])), 0, ax)


def _scale(t, ax, s):
    shape = [1] * t.dim()
    shape[ax] = -1
    return t * s.view(shape)


@torch.no_grad()
def _branch_call(br, args, kwargs, x):
    if args:
        return _first(br(x, *args[1:], **kwargs))
    kw = dict(kwargs)
    kw["hidden_states"] = x
    return _first(br(**kw))


def _tied(cand):
    """cand grouped so siblings (names equal up to their numeric indices: experts.0.w1 / experts.7.w1, same axis) are
    one choice; the subset searches are exponential in the number of choices (per-expert MoE modules: 3E + 1 axes)"""
    groups = {}
    for pn, ax in cand:
        groups.setdefault((re.sub(r"\.\d+(?=\.)", ".*", pn), ax), []).append((pn, ax))
    return list(groups.values())


@torch.no_grad()
def branch_axes(br, args, kwargs, d, gen):
    """(input axes, output axes) of one branch, or None when no assignment makes it rotation-equivariant"""
    cand = _axes(br, d)
    tied = _tied(cand)
    if len(tied) < len(cand):                              # siblings first; the untied search only if that fails
        r = _branch_axes(br, args, kwargs, d, gen, tied)
        if r is not None:
            return r
    return _branch_axes(br, args, kwargs, d, gen, [[c] for c in cand])


@torch.no_grad()
def _branch_axes(br, args, kwargs, d, gen, cand):
    x = _arg0(args, kwargs)
    y = _branch_call(br, args, kwargs, x)
    flat = lambda groups: [c for g in groups for c in g]
    params = dict(br.named_parameters())
    orig = {pn: p.detach().clone() for pn, p in params.items()}

    def restore():
        for pn, p in params.items():
            p.copy_(orig[pn])

    s = torch.rand(d, generator=gen) + 0.5
    ins = None
    for k in range(0, min(len(cand), MAX_SEARCH) + 1):     # smallest set of input axes absorbing an input scaling
        for sub in itertools.combinations(cand, k):
            sub = flat(sub)
            if len({pn for pn, _ in sub}) != len(sub):
                continue                                   # one input axis per parameter
            for pn, ax in sub:
                params[pn].copy_(_scale(orig[pn], ax, s))
            ok = _close(_branch_call(br, args, kwargs, x / s), y, 1e-4)
            restore()
            if ok:
                ins = list(sub)
                break
        if ins is not None:
            break
    if ins is None:
        return None
    rest = [g for g in cand if g[0] not in ins]
    Q = random_orth(d, gen)
    for k in range(0, min(len(rest), MAX_SEARCH) + 1):     # the smallest set of further axes to turn with Q
        for sub in itertools.combinations(rest, k):
            sub = flat(sub)
            for pn, ax in ins + list(sub):
                params[pn].copy_(_turn(params[pn], ax, Q))
            ok = _close(_branch_call(br, args, kwargs, x @ Q.T), y @ Q.T, 1e-4)
            restore()
            if ok:
                return ins, list(sub)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# the rules and their check

@torch.no_grad()
def discover(cfg, seed=0, check=True, log=print):
    model = tiny_model(cfg, seed)
    emb = model.get_input_embeddings()
    d = emb.weight.shape[1]
    lpath, layers = decoder_layers(model)
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, emb.weight.shape[0], (1, 12), generator=gen)
    rec = trace(model, ids)
    out = {"layers_path": lpath, "embed": module_path(model, emb), "head": module_path(model, model.get_output_embeddings()),
           "d": d, "kinds": {}, "rotation": True, "reason": ""}

    def fail(why):
        if out["rotation"]:
            out["rotation"], out["reason"] = False, why

    for i, layer in enumerate(layers):
        sig = signature(layer)
        if sig in out["kinds"]:
            continue
        L = rec["layers"][i]
        segs, why = segments(L, d)
        if segs is None:
            fail(f"layer {i}: {why}")
            out["kinds"][sig] = None
            continue
        rule = []
        for nname, brs in segs:
            nm = layer.get_submodule(nname)
            form = gain_form(nm, d)
            if form is None or not rot_equivariant(nm, d, gen):
                fail(f"layer {i}: norm {nname} ({type(nm).__name__}) does not commute with a rotation")
            seg = {"norm": nname, "form": form, "in": [], "out": []}
            for b in brs:
                ev = next(e for e in L["ev"] if e[0] == "child" and e[1] == b)
                ax = branch_axes(layer.get_submodule(b), ev[2], ev[3], d, gen)
                if ax is None:
                    fail(f"layer {i}: branch {b} has no rotation-equivariant axis assignment")
                    continue
                seg["in"] += [[f"{b}.{pn}", a] for pn, a in ax[0]]
                seg["out"] += [[f"{b}.{pn}", a] for pn, a in ax[1]]
            rule.append(seg)
        out["kinds"][sig] = rule
        # a residual parameter outside every branch (not seen) would break the fold: the end-to-end check catches it
    fn = final_norm(model, rec, lpath)
    out["final_norm"] = fn
    if fn is None:
        fail("no final norm feeding the head")
    else:
        out["final_form"] = gain_form(model.get_submodule(fn), d)
        if not rot_equivariant(model.get_submodule(fn), d, gen):
            fail(f"final norm {fn} does not commute with a rotation")
    out["tied"] = model.get_output_embeddings().weight is emb.weight
    if out["rotation"] and check:
        err = check_fold(model, out, ids, rec["logits"], gen)
        out["fold_err"] = err
        if not err < 1e-4:
            fail(f"the fold does not reproduce the logits (rel {err:.2e})")
    if not out["rotation"]:
        log(f"arch: no exact residual rotation: {out['reason']}")
    return out


def final_norm(model, rec, lpath):
    """the norm outside the layers whose output, on the last layer's output, is the head's input"""
    hi, last = rec["head_in"], rec["layers"][-1]["out"]
    for n, m in model.named_modules():
        if is_norm(m) and not n.startswith(lpath + ".") and last.shape[-1] == m.weight.shape[0]:
            if _close(_first(m(last)), hi):
                return n
    return None


def gain(nm, form):
    w = nm.weight.float()
    return w if form == "w" else 1 + w


def set_identity(nm, form):
    nm.weight.fill_(1.0 if form == "w" else 0.0)


def rule_for(rules, layer):
    return rules["kinds"].get(signature(layer))


@torch.no_grad()
def fold_layer(layer, rule, Q, weight_fn=None):
    """one decoder layer in place: each residual norm's gain into its branches' input axes (the norm set to identity),
    then Q on every input and output axis. weight_fn(param name, tensor) -> tensor, if given, receives each changed
    parameter instead of it being written (streaming conversions)"""
    for seg in rule:
        nm = layer.get_submodule(seg["norm"])
        g = gain(nm, seg["form"])
        todo = {}
        for pn, ax in seg["in"]:
            todo.setdefault(pn, []).append(("in", ax))
        for pn, ax in seg["out"]:
            todo.setdefault(pn, []).append(("out", ax))
        for pn, ops in todo.items():
            p = layer.get_parameter(pn)
            t = p.float()
            for kind, ax in ops:
                if kind == "in":
                    t = _scale(t, ax, g)
                t = _turn(t, ax, Q)
            p.copy_(t.to(p.dtype))
        set_identity(nm, seg["form"])


@torch.no_grad()
def apply_fold(model, rules, Q):
    """the fold on a float model in place: every layer (fold_layer), the embedding E Q^T, the head untied and
    (H * g_final) Q^T, the final norm set to identity"""
    emb = model.get_submodule(rules["embed"])
    head = model.get_submodule(rules["head"])
    if head.weight is emb.weight:
        head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    for layer in model.get_submodule(rules["layers_path"]):
        fold_layer(layer, rule_for(rules, layer), Q)
    emb.weight.copy_((emb.weight.float() @ Q.T).to(emb.weight.dtype))
    fn = model.get_submodule(rules["final_norm"])
    g = gain(fn, rules["final_form"])
    head.weight.copy_(((head.weight.float() * g[None, :]) @ Q.T).to(head.weight.dtype))
    set_identity(fn, rules["final_form"])


@torch.no_grad()
def check_fold(model, rules, ids, ref, gen):
    m2 = copy.deepcopy(model)
    apply_fold(m2, rules, random_orth(rules["d"], gen))
    y = m2(input_ids=ids, use_cache=False).logits
    return float((y - ref).abs().max() / ref.abs().max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    from transformers import AutoConfig
    r = discover(AutoConfig.from_pretrained(a.model))
    s = json.dumps(r, indent=1)
    print(s)
    if a.json:
        open(a.json, "w").write(s)


if __name__ == "__main__":
    main()
