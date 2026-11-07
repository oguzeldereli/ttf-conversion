"""hf_fetch.py: the Q38 "slim" base checkpoint (the language model without the routed experts: the
1199 tensors model.language_model.* + lm_head.weight minus .mlp.experts.*, incl. the 51 GB FP8 PLE table) fetched
straight from the Hugging Face repo by HTTP range reads, so a training machine needs no 185 GB download. Same tensor
names and dtypes as scripts/q38/slim.py's output; one output shard per source shard, plus the index,
config, generation config and tokenizer files.

  python scripts/q38/hf_fetch.py OUT [--repo Qwen/Qwen3.8-Flash-Next-FP8] [--jobs 8]
Resumable: a finished shard is skipped (OUT/<shard> exists and has the expected size)."""
import json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = sys.argv[1]
REPO = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else "Qwen/Qwen3.8-Flash-Next-FP8"
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
FILES = ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt"]


def get(name, lo=None, hi=None):
    """bytes [lo, hi] (inclusive) of a repo file, or the whole file"""
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "q38-fetch"})
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi}")
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if lo is None or len(b) == hi - lo + 1:
                return b
        except Exception as e:
            print(f"  retry {k} {name} {lo}-{hi}: {e}", flush=True)
    raise RuntimeError(f"fetch failed: {name} {lo}-{hi}")


def want(k):
    return (k.startswith("model.language_model.") or k == "lm_head.weight") and ".mlp.experts." not in k


def header(shard):
    n = struct.unpack("<Q", get(shard, 0, 7))[0]
    return n, json.loads(get(shard, 8, 8 + n - 1))


def fetch_shard(shard, names):
    n, h = header(shard)
    base = 8 + n
    keep = sorted(names, key=lambda k: h[k]["data_offsets"][0])
    newh, off = {}, 0
    for k in keep:
        b, e = h[k]["data_offsets"]
        newh[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + e - b]}
        off += e - b
    hb = json.dumps(newh, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    total = 8 + len(hb) + off
    fn = os.path.join(OUT, shard)
    if os.path.exists(fn) and os.path.getsize(fn) == total:
        return shard, 0
    # coalesce source ranges that are adjacent (gap < 4 MiB) into one read, split reads at 256 MiB
    runs = []
    for k in keep:
        b, e = h[k]["data_offsets"]
        if runs and b - runs[-1][1] < (4 << 20) and e - runs[-1][0] <= (256 << 20):
            runs[-1][1] = max(runs[-1][1], e); runs[-1][2].append(k)
        else:
            runs.append([b, e, [k]])
    with open(fn + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b, e, ks in runs:
            if e - b > (256 << 20):          # one big tensor: chunked reads
                for c in range(b, e, 256 << 20):
                    f.write(get(shard, base + c, base + min(e, c + (256 << 20)) - 1))
                continue
            blob = get(shard, base + b, base + e - 1)
            for k in ks:
                kb, ke = h[k]["data_offsets"]
                f.write(blob[kb - b:ke - b])
    assert os.path.getsize(fn + ".part") == total, shard
    os.rename(fn + ".part", fn)
    return shard, off


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in FILES:
        if not os.path.exists(os.path.join(OUT, f)):
            open(os.path.join(OUT, f), "wb").write(get(f))
    wm = json.loads(get("model.safetensors.index.json"))["weight_map"]
    by = {}
    for k, s in wm.items():
        if want(k):
            by.setdefault(s, []).append(k)
    print(f"{sum(len(v) for v in by.values())} tensors in {len(by)} shards", flush=True)
    json.dump({"metadata": {}, "weight_map": {k: s for s, ks in by.items() for k in ks}},
              open(os.path.join(OUT, "model.safetensors.index.json"), "w"))
    got = 0
    with ThreadPoolExecutor(JOBS) as ex:
        for s, b in ex.map(lambda kv: fetch_shard(*kv), sorted(by.items())):
            got += b
            print(f"{s}: {b / 2**30:.2f} GiB (total {got / 2**30:.1f})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
