# Qwen3.8-27B

## Conversion (one 24–48 GB GPU plus host memory)

`scripts/convert27.sh` runs the streaming conversion with the plug-ins and exits with `ttfconv.finish` (a state file
and fixed-window scores). The structure:
- the main ternary layer T1;
- r0.2 ternary adapters with one alternating refit, `prop=1`, fold a and a randomized SVD (`svd=rand`).

No second ternary layer (T2; removed from the code on 25 Oct): at 0.8B the same bytes spent as adapter rank did better
(rank 0.4 without T2: −0.029 wiki training-free against T2 on 0.229 of the weights + rank 0.2), it costs +37% time per
27B block, and in training it moved in lockstep with its main layer.

```bash
CORE=path/to/ttf-ternary M=models/qw27 DATA=data/calib.bin OUT=runs/q27_conv Q4=0 \
  EVAL="wiki@data/wiki_eval.bin@16@2049" bash scripts/convert27.sh
```

`Q4=0` keeps the embedding and head in bf16, for a training start. `Q4=1` (the default) makes the 4-bit release pack.

## Recovery training

`recipes/qwen-27b.toml` is the proposed recipe:
- centred GPTQ start, θ16;
- accumulator rank 1024 with the state in host memory;
- KD from the online teacher where it fits (96 GB cards); otherwise from a cached teacher top-k (ttf-core
  `scripts/quant/teacher_topk.py`). At 0.8B the cache cost +0.014 wiki at k64 and +0.037 at k32 after 1k steps;
- 1 × 16 micro-batches of 2048 tokens;
- `--head-fused`.

```bash
python -m ttf.train --recipe recipes/qwen-27b.toml --init-hf $M --teacher $M --data DATA.bin --steps 2000 \
  --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1,prop=1,svd=rand --out-dir runs/q27
```

The run converts first (streaming, layer by layer), then trains. Its checkpoints resume with `--resume` and the same
`--ext` list. `ttfconv.finish`'s `state.pt` is for scoring and packing, not a training checkpoint.

Measured so far (L40S unless stated):
- the conversion structure trains at a 21.7 GiB peak (2 × 8, r256, state on the host);
- 83 s per step at r256; rank 1024 costs about 29% more time (RTX 6000 Ada);
- θ16 with rank 1024 was the best of the 27B threshold / rank probes (30 steps).

The full recipe has not been trained end to end at 27B yet.

## What the 0.8B results suggest for 27B

| change | 0.8B gain after 1k steps | cost at 27B |
|---|---|---|
| `prop=1` | −0.027 wiki | none |
| `gmetric` (fp G) | −0.008 more | one backward pass of the fp model per calibration window; a streamed tool for 27B is not written yet |
| T2 placed by measured damage (T2 removed 25 Oct) | −0.07 training-free | one single-linear KL per linear (~400 streamed forwards) |
