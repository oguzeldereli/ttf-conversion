"""A slim copy of the Qwen3.8-Flash-Next checkpoint for the TTF trainer: config.json plus every text
tensor except the routed experts (the conversion replaces them), without MTP and vision; files of ~4 GB, with an index,
so the FP8 original (185 GB) can be deleted once the conversion and the teacher cache are done.
  python scripts/q38/slim.py SRC_DIR DST_DIR"""
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in ("config.json", "generation_config.json"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
keep = sorted(k for k in wm if (k.startswith("model.language_model.") or k == "lm_head.weight") and ".mlp.experts." not in k)
by_file = {}
for k in keep:
    by_file.setdefault(wm[k], []).append(k)
out_map, buf, size, n = {}, {}, 0, 0


def flush():
    global buf, size, n
    if not buf:
        return
    fn = f"slim-{n:05d}.safetensors"
    save_file(buf, os.path.join(dst, fn + ".tmp"))
    os.replace(os.path.join(dst, fn + ".tmp"), os.path.join(dst, fn))
    out_map.update({k: fn for k in buf})
    print(f"{fn}: {len(buf)} tensors, {size / 2**30:.1f} GiB", flush=True)
    buf, size, n = {}, 0, n + 1


for f, ks in sorted(by_file.items()):
    with safe_open(os.path.join(src, f), "pt", device="cpu") as h:
        for k in ks:
            t = h.get_tensor(k).contiguous()
            buf[k] = t; size += t.numel() * t.element_size()
            if size > 4 * 2**30:
                flush()
flush()
json.dump({"metadata": {"source": src}, "weight_map": out_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
print("slim done:", len(out_map), "tensors", flush=True)
