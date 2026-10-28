"""what the ternary adapters (ttfconv.adapters) do in training; not structural (the checkpoint's --ext list need not
name it), so the same converted start loads in every arm.
  --ext ttfconv.adapter_ctl:mode=frozen   the adapters stay at their conversion fit: no TTF moves of their factors'
                                          trits, their scales and centring not trained (they still shape the forward
                                          and pass the gradient through)
  --ext ttfconv.adapter_ctl:mode=off      the adapters out of the forward (y = the main layer's output alone), frozen
  --ext ttfconv.adapter_ctl:mode=trits    the adapters' trits fixed, their row / column scales train (nothing else of
                                          theirs: float adapters on the factors stay at zero)
  --ext ttfconv.adapter_ctl:mode=train    the adapters train as without this plug-in (for main=frozen alone)
  main=frozen (with either mode)          the main layers' trits fixed as well (no TTF state or moves); their row /
                                          column scales and float adapters (--adapter-rank) still train: recovery that
                                          approximates the fp model with the conversion's trits (27B release, 28 Oct)
For the question whether the adapters take over what the main trits would otherwise reorganise (26 Oct): the arms
against the trained adapters from one saved start."""
from .spec import parse


class AdapterCtl:
    def __init__(self, spec):
        o = parse(spec, {"main": ""}, required=("mode",))
        assert o["mode"] in ("frozen", "off", "trits", "train"), \
            f"ttfconv.adapter_ctl: mode frozen, off, trits or train, not {o['mode']}"
        assert o["main"] in ("", "frozen"), f"ttfconv.adapter_ctl: main frozen or unset, not {o['main']}"
        self.mode, self.main = o["mode"], o["main"]

    def setup(self, model, log):
        from .adapters import QTad
        n = p_ = 0
        for q in model.modules():
            if not isinstance(q, QTad):
                continue
            q.ad_off = self.mode == "off"
            n += 1
            if self.mode == "train":
                continue
            for l in q.ad:
                l.ts_frozen = True
                for pn, p in l.named_parameters():
                    if self.mode == "trits" and pn in ("row_scale", "col_scale"):
                        continue
                    p.requires_grad_(False); p.grad = None; p_ += p.numel()
        assert n, "ttfconv.adapter_ctl: no adapters (--ext ttfconv.adapters first)"
        if self.main == "frozen":
            from ttf.layers import KernelTernaryLinear
            ad_ = {id(l) for q in model.modules() if isinstance(q, QTad) for l in q.ad}
            nm = 0
            for l in model.modules():
                if isinstance(l, KernelTernaryLinear) and id(l) not in ad_:
                    l.ts_frozen = True; nm += 1
            log(f"ttfconv.adapter_ctl: {nm} main ternary layers' trits frozen (their scales and float adapters train)")
        what = {"off": "out of the forward", "frozen": "frozen", "trits": "with fixed trits (row / column scales train)",
                "train": "training"}[self.mode]
        log(f"ttfconv.adapter_ctl: {n} adapters {what}" + (f" (their trits fixed, {p_ / 1e6:.2f} M float parameters not "
                                                            f"trained)" if self.mode != "train" else ""))


def ttf_ext(spec):
    return AdapterCtl(spec)
