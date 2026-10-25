"""ttfconv: converting full-precision models (Qwen3.5 / 3.8) into ternary networks that TTF keeps training.

Built on ttf-core through its plug-in interface (ttf/ext.py, --ext MODULE:SPEC), in the order given on the command line:

  ttfconv.fold:q=had          the norm gains folded and one randomized Hadamard on the residual stream ("fold a")
  ttfconv.cd:sweeps=8         coordinate descent on the trits after GPTQ
  ttfconv.adapters:rank=0.2,alt=1  ternary low-rank adapters
  ttfconv.finish:out=DIR      training-free size / eval / state right after the conversion, then exit

  python -m ttf.train --recipe convert-qwen --init-hf $M --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1 ...
"""
