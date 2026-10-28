"""A converted checkpoint (ttfconv.convert's OUT/ckpt.pt, or a ttf-core training checkpoint of a converted model) back
into a model: ttf-core's Qwen built with the checkpoint's structure (the plug-ins' build, centred layers, float
adapters, fine residuals, all read from the checkpoint's keys), then its tensors loaded."""
import torch


def load_converted(arch, ckpt, dev="cuda", ext=None, log=print):
    """arch: the HF directory (config only); ckpt: a path or a loaded blob; ext: the structural plug-in specs (default:
    the ones the checkpoint was converted with). Returns (model in eval mode, the blob's state dict)."""
    from ttf.layers import KernelTernaryLinear
    from .convert import build_model, load_plugins
    from .fine import enable_fine
    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True) if isinstance(ckpt, str) else ckpt
    sd = blob.get("model", blob)
    meta = blob.get("ttfconv", {}) if isinstance(blob, dict) else {}
    if ext is None:
        ext = meta.get("ext") or [s for s in (blob.get("args") or {}).get("ext", []) or []]
    model, sw = build_model(arch, dev, load_plugins(ext), log=log)
    nc = nm = nf = 0
    with torch.no_grad():
        for n, m in model.named_modules():
            if not isinstance(m, KernelTernaryLinear):
                continue
            if f"{n}.in_mean" in sd and getattr(m, "in_mean", None) is None:
                m.enable_centering(sd[f"{n}.in_mean"].to(dev), sd[f"{n}.out_bias"].to(dev)); nc += 1
            if f"{n}.mag_A" in sd and getattr(m, "mag_A", None) is None:
                m.enable_lowrank_mag("add", sd[f"{n}.mag_A"].shape[1]); nm += 1
            if f"{n}.wfine" in sd and getattr(m, "wfine", None) is None:
                enable_fine(m, int(meta.get("fine") or 27)); nf += 1
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("frozen_scale")]
    if missing:
        raise SystemExit(f"load_converted: {len(missing)} model tensors not in the checkpoint, e.g. {missing[:4]}")
    for m in model.modules():
        if hasattr(m, "_invalidate"):
            m._invalidate()
        if hasattr(m, "_after_fine"):
            m._after_fine()
    log(f"load_converted: {nc} centred layers, {nm} float adapters, {nf} fine layers"
        + (f"; {len(unexpected)} checkpoint tensors unused" if unexpected else ""))
    return model.eval(), sd
