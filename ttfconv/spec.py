"""--ext spec strings: "key=value,key=value" (a value may itself contain '=' or ':'; lists inside a value use '+')"""


def parse(spec, defaults, required=()):
    out = dict(defaults)
    for kv in filter(None, (spec or "").split(",")):
        k, _, v = kv.partition("=")
        if k not in defaults and k not in required:
            raise SystemExit(f"--ext spec: unknown key {k!r} (known: {sorted(set(defaults) | set(required))})")
        out[k] = v
    miss = [k for k in required if k not in out]
    if miss:
        raise SystemExit(f"--ext spec: {miss} must be set explicitly")
    return out
