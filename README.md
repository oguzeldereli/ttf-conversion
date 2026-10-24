# ttf-conversion

Convert a full-precision Qwen3.5 / Qwen3.8 model into a **ternary** network (weights in {−1, 0, +1} times row and column
scales), then keep training it with [TTF](https://github.com/oguzeldereli/ttf-ternary). TTF trains the trits directly:
there are no full-precision master weights, during conversion or afterwards, so a converted model can keep
improving on a consumer GPU.

This repository holds the conversion side as plug-ins for ttf-core's trainer (`--ext MODULE:SPEC`):

| plug-in | what it does |
|---|---|
| `ttfconv.fold[:q=had,seed=2110]` | folds the RMSNorm gains into the linears that read them and rotates the residual stream by one randomized Hadamard (exact, free at decode); the head is untied from the embedding |
| `ttfconv.cd[:sweeps=8,tol=1e-4]` | coordinate descent on the trits after GPTQ |
| `ttfconv.adapters:rank=R,alt=N[,t2=S][,prop=1][,gmetric=DIR][,t2sel=FILE]` | **ternary low-rank adapters**: `y = T(x−μ) + b + U·D(x−μ)`, with D and U ternary, each with its own scales, trained by TTF. Optionally a second ternary layer (T2) on a share S of the weights. |
| `ttfconv.finish:out=DIR[,eval=...][,q4=1]` | right after the conversion: the packed size, the embedding/head as 4-bit groups (`q4=1`), scores on fixed windows, `DIR/state.pt`; then exit |

The adapters' options:
- `rank`: an integer, or a fraction of min(K, N).
- `alt`: rounds of alternating refit of D and U (1 is recommended).
- `prop=1`: the calibration stream goes through each adapted layer, so later layers are rounded for the inputs they
  will see.
- `gmetric=DIR`: fit the adapter subspace in the output-sensitivity metric G⊗H instead of H alone (G from
  `ttfconv.tools.kfac --dump-g`).
- `t2sel=FILE`: the order in which T2 is placed (e.g. by measured damage).

The adapters are fitted inside the centred streaming conversion (`--convert gptqc`, the default with
`--convert-stream`); the recipes here set it.

## Install

```bash
pip install "ttf-ternary[qwen] @ git+https://github.com/oguzeldereli/ttf-ternary"
pip install -e .                 # this repository (package ttfconv)
```

You need a CUDA GPU for conversion and training (the ternary kernels are Triton). The tests in `tests/` run on the CPU
(`pytest`).

## Quickstart: Qwen3.5-0.8B

The data files are uint32 token streams in Qwen's vocabulary (ttf-core: `scripts/quant/retok.py`).

```bash
M=models/Qwen3.5-0.8B-Base
# 1. (optional) the output-sensitivity metric for the adapter fit: ~2 min, ~9 GB GPU, ~5 GB of files
python -m ttfconv.tools.kfac --model $M --data data/train.bin --val data/val.bin --stats-only --dump-g runs/g08 \
    --ncal 16 --seq 1024 --chunk 1 --out runs/g08.json

# 2. convert, then 1000 steps of recovery with distillation from the fp model (RTX 4070: ~16.5 s per step)
python -m ttf.train --recipe recipes/qwen-0.8b.toml --init-hf $M --teacher $M --data data/train.bin \
    --ext ttfconv.fold --ext ttfconv.cd:sweeps=1 --ext ttfconv.adapters:rank=0.2,alt=1,prop=1,gmetric=runs/g08 \
    --out-dir runs/q08
```

Conversion only, then score and save (no training):

```bash
python -m ttf.train --recipe recipes/qwen-0.8b.toml --init-hf $M --data data/train.bin --kd-weight 0 --teacher "" \
    --ext ttfconv.fold --ext ttfconv.adapters:rank=0.2,alt=1,prop=1 \
    --ext "ttfconv.finish:out=runs/q08_conv,eval=wiki@data/val.bin@16@2048" --out-dir runs/q08_conv_run
```

To resume, pass the same `--ext` list without `finish`; the structure is rebuilt and the conversion skipped.

## Results (Qwen3.5-0.8B, wiki / chat / mix held-out loss)

Full precision: 2.5141 / 1.3378 / 1.0396.

Training-free conversion (fold + one CD sweep + r0.2 adapters, 4-bit embedding/head):

| conversion | wiki | chat | mix |
|---|---|---|---|
| calibration through the bare ternary layers | 3.7614 | 2.7198 | 2.0707 |
| + `prop=1` | 3.6648 | 2.6195 | 1.9602 |
| + `gmetric` (fp model's G) | 3.6101 | 2.6044 | 1.9351 |
| + T2 on 0.229 of the weights, placed by measured damage (`t2sel`) | 3.2707 | 2.1921 | 1.6726 |

After 1000 recovery steps (θ16, accumulator rank 256, KD 0.75, 32k tokens per step):

| start | wiki | chat | mix |
|---|---|---|---|
| fold + adapters (alt 0) | 3.0037 | 1.6914 | 1.3733 |
| fold + cd1 + adapters alt 1 | 2.9918 | 1.6816 | 1.3580 |
| + `prop=1` | 2.9650 | 1.6583 | 1.3398 |
| + `gmetric` | **2.9572** | **1.6542** | **1.3368** |

## Qwen3.8-27B

See [docs/CONVERT_27B.md](docs/CONVERT_27B.md). The conversion streams one decoder layer at a time from the safetensors.
Recovery training of the adapter structure fits a 24 GB card: we measured a 21.7 GiB peak with the TTF state in host
memory.

## Tools (`ttfconv/tools`)

| tool | what it measures |
|---|---|
| `kfac` | per linear: the Kronecker second-order damage ½tr(G E H Eᵀ) of its rounding, the measured KL of rounding it alone, how much of the damage a rank-r fit removes in H vs G⊗H; `--dump-g` writes G for `gmetric` |
| `kfac_report` | a summary of a `kfac` result |
| `alpha` | KL of scaled rounding errors αE (is the error in the quadratic regime?) and a per-token Gauss–Newton estimate |
| `swap` | swaps blocks or tensor groups between two states of one model (e.g. two steps of a run) and scores each |
| `gconv` | G of a converted network (for a second conversion pass) |
| `blocklocal` | an experimental block-by-block pre-pass on a local quadratic model; it diverges as written (kept as a negative result) |

What these found on 0.8B:
- A layer's local rounding error does not predict its damage to the model's output; output sensitivity does.
- During recovery the blocks' updates are nearly additive, and each block's trits move together with that block's
  own scales and norms.

## License

PolyForm Noncommercial 1.0.0, the same as ttf-core (see LICENSE).
