"""ttfconv: converting full-precision models (Qwen3.5 / 3.8) into ternary networks, standalone.

The conversion is its own code base: it builds models with ttf-core (the Qwen wrapper, the packed ternary layers and
kernels) and writes ttf-core checkpoints, but ttf-core holds none of it.

  python -m ttfconv.convert --model M --data CAL.bin --out OUT --ext ttfconv.fold [--ext ttfconv.adapters:rank=0.2,alt=1]
      [--svd-rank 16] [--fine 27]

  ttfconv.stream     the layer-by-layer streaming conversion       ttfconv.gptq   ALS / GPTQ rounding, the adapter fit
  ttfconv.fine       the fine residual (extra digits)              ttfconv.model  load_converted (checkpoint -> model)
  plug-ins (--ext)   ttfconv.fold, ttfconv.cd, ttfconv.adapters, ttfconv.embq (their build() is also read by ttf-core's
                     trainer, so a training run rebuilds the structure)
"""
