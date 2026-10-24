"""the frozen embedding and head at fewer bits, before training (so the body trains around their error): each row in
asymmetric groups of g, dequantized in place at the end of the conversion. The 27B training-free probe (22 Oct):
embedding 2-bit + head 3-bit saves 0.48 GB for +0.050 wiki against 4-bit both.
  --ext ttfconv.embq:emb=2,head=3[,g=64]
After ttfconv.fold (which unties the head); with a tied head the embedding's bits apply to both. Before ttfconv.finish
(its q4=1 would requantize them to 4 bits: use q4=0 with this plug-in).
"""
import torch
from .finish import fqb
from .spec import parse


class EmbQ:
    def __init__(self, spec):
        o = parse(spec, {"g": "64"}, required=("emb", "head"))
        self.eb, self.hb, self.g = int(o["emb"]), int(o["head"]), int(o["g"])

    @torch.no_grad()
    def end(self, model, log):
        hf = model.hf; emb, head = hf.model.embed_tokens.weight, hf.lm_head.weight
        e0 = emb.detach().float().clone()
        emb.copy_(fqb(emb, self.eb, self.g))
        msg = f"embedding {self.eb}-bit (rel err {float((emb.float() - e0).norm() / e0.norm()):.4f})"
        if head is not emb:
            h0 = head.detach().float().clone()
            head.copy_(fqb(head, self.hb, self.g))
            msg += f", head {self.hb}-bit (rel err {float((head.float() - h0).norm() / h0.norm()):.4f})"
        else:
            msg += " (head tied: the embedding's bits)"
        log(f"ttfconv.embq: {msg}, groups of {self.g}")


def ttf_ext(spec):
    return EmbQ(spec)
