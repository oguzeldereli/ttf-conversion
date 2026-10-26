"""what the ternary adapters (ttfconv.adapters) do in training; not structural (the checkpoint's --ext list need not
name it), so the same converted start loads in every arm.
  --ext ttfconv.adapter_ctl:mode=frozen   the adapters stay at their conversion fit: no TTF moves of their factors'
                                          trits, their scales and centring not trained (they still shape the forward
                                          and pass the gradient through)
  --ext ttfconv.adapter_ctl:mode=off      the adapters out of the forward (y = the main layer's output alone), frozen
For the question whether the adapters take over what the main trits would otherwise reorganise (26 Oct): the arms
against the trained adapters from one saved start."""
from .spec import parse


class AdapterCtl:
    def __init__(self, spec):
        o = parse(spec, {}, required=("mode",))
        assert o["mode"] in ("frozen", "off"), f"ttfconv.adapter_ctl: mode frozen or off, not {o['mode']}"
        self.mode = o["mode"]

    def setup(self, model, log):
        from .adapters import QTad
        n = p_ = 0
        for q in model.modules():
            if not isinstance(q, QTad):
                continue
            q.ad_off = self.mode == "off"
            for l in q.ad:
                l.ts_frozen = True
                for p in l.parameters():
                    p.requires_grad_(False); p.grad = None; p_ += p.numel()
            n += 1
        assert n, "ttfconv.adapter_ctl: no adapters (--ext ttfconv.adapters first)"
        log(f"ttfconv.adapter_ctl: {n} adapters {'out of the forward' if self.mode == 'off' else 'frozen'} "
            f"(their trits fixed, {p_ / 1e6:.2f} M float parameters not trained)")


def ttf_ext(spec):
    return AdapterCtl(spec)
