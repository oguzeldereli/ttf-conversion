# Qwen3.8-27B

## Conversion (one 24–48 GB GPU plus host memory)

`scripts/convert27.sh` runs `python -m ttfconv.convert`: the streaming conversion with the plug-ins, saved as
`OUT/ckpt.pt` (a ttf-core checkpoint at step -1) with fixed-window scores in `OUT/result.json`. The structure:
- the main ternary layer T1;
- r0.2 ternary adapters with one alternating refit, `prop=1`, fold a and a randomized SVD (`svd=rand`).

No second ternary layer (T2; removed from the code on 25 Oct): at 0.8B the same bytes spent as adapter rank did better
(rank 0.4 without T2: −0.029 wiki training-free against T2 on 0.229 of the weights + rank 0.2), it costs +37% time per
27B block, and in training it moved in lockstep with its main layer.

```bash
CORE=path/to/ttf-ternary M=models/qw27 DATA=data/calib.bin OUT=runs/q27_conv \
  EVAL="wiki@data/wiki_eval.bin@16@2049" bash scripts/convert27.sh
```

The embedding and head stay in bf16 (a training start); `EMBQ=E,H` quantizes them to E / H bits in groups of 64.
`SVD=16 FINE=27` is the other structure: rank-16 float adapters and the fine residual (extra digits) instead of the
ternary adapters, the start of the squeeze (27 Oct: wiki16 1.9491 with a bf16 embedding / head, +0.006 over Q8).

## Recovery training

`recipes/qwen-27b.toml` holds the settings of the 27B recovery run started on 26 Oct 2026 (2 × RTX PRO 6000, 96 GB):
- centred GPTQ start, θ16;
- accumulator rank 1536 (30% of the width), the TTF state in bf16 on the GPU;
- the embedding and head as 2- / 3-bit integer codes (`intk-emb = "2:3"`): their group minima and steps train; the
  codes' own bar, θ / `emb-lr-mult` = 320, is not reached (no code moved in 1000 steps at 0.8B);
- KD 0.75 from a cached teacher sample (ttf-core `scripts/quant/teacher_topk.py`; `kd-cache-mode = "sample"`). Where the
  bf16 teacher (~54 GB) fits beside the student, `--teacher M` instead: at 0.8B a cached top-k cost +0.014 wiki at k64
  and +0.037 at k32 after 1k steps;
- 1 × 16 micro-batches of 2048 tokens, one pipeline stage per GPU (`--pipe`, under torchrun).

Training starts from a converted checkpoint (ttf-core does not convert):

```bash
CORE=path/to/ttf-ternary M=models/qw27 DATA=data/calib.bin OUT=runs/q27_start bash scripts/convert27.sh
torchrun --nproc_per_node 2 -m ttf.train --recipe recipes/qwen-27b.toml --init-hf $M --init-ckpt runs/q27_start/ckpt.pt \
  --resume --data DATA.bin --pipe --kd-cache KD_DIR --steps 7750 \
  --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1,prop=1,svd=rand --out-dir runs/q27
```

On 24–48 GB cards use `--ttf-store fp12s --ttf-offload async` (the state in host memory) and a lower `--ttf-rank`.

A run's checkpoints resume with `--resume` and the same `--ext` list.

Measured so far:
- one RTX PRO 6000 (10 steps each): rank 1024 / 1536 / 2048 at 47.5 / 54.3 / 59.8 s per step, peak 66.2 / 77.9 /
  92.8 GiB;
- two RTX PRO 6000 as a two-stage pipeline at rank 1536: 27.0 s per step, peaks 72.8 / 66.5 GiB;
- L40S: the conversion structure trains at a 21.7 GiB peak (2 × 8, r256, state on the host); 83 s per step at r256.

The run's results are not in yet.

## What the 0.8B results suggest for 27B

| change | 0.8B gain after 1k steps | cost at 27B |
|---|---|---|
| `prop=1` | −0.027 wiki | none |
| `gmetric` (fp G) | −0.008 more | one backward pass of the fp model per calibration window; a streamed tool for 27B is not written yet |
| T2 placed by measured damage (T2 removed 25 Oct) | −0.07 training-free | one single-linear KL per linear (~400 streamed forwards) |
