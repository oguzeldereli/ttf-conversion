# ttf-conversion

Convert a full-precision Qwen3.5 / Qwen3.8 model into a **ternary** network: weights in {−1, 0, +1} times row and
column scales, optionally with low-rank adapters and a fine residual of extra digits. The conversion is its own code
base. It builds its models with [ttf-core](https://github.com/oguzeldereli/ttf-ternary) (the Qwen wrapper, the packed
ternary layers and kernels) and writes checkpoints that ttf-core's trainer can start from (`--init-ckpt`), but none of
the conversion lives in ttf-core.

Entry point:

```bash
python -m ttfconv.convert --model M --data CAL.bin --out OUT [--ext PLUGIN[:SPEC] ...] [--svd-rank R] [--fine 27] \
    [--method gptqc] [--calib 64] [--eval wiki@FILE@N@STRIDE]
```

It streams the model one decoder layer at a time from the safetensors: a layer's linears are calibrated on the inputs
the already-converted layers below produce, then rounded (centred GPTQ by default). The output is `OUT/ckpt.pt` in
ttf-core's checkpoint form, plus `OUT/result.json`.

| module | what it does |
|---|---|
| `ttfconv.convert` | the command line above: build, convert, save, score |
| `ttfconv.stream` | the layer-by-layer streaming conversion |
| `ttfconv.gptq` | ALS rounding, GPTQ onto the ternary grid, the rank-R fit of the rounding error (float adapters) |
| `ttfconv.fine` | the fine residual: `W = β(T + F/R)`, `F` int8 whose balanced-ternary digits are the extra digits; its layer (`FineTernaryLinear`), GEMMs and GPTQ fit |
| `ttfconv.model` | `load_converted(arch, ckpt)`: a converted checkpoint back into a model (structure read from its keys) |
| `ttfconv.finish` | sizes, 4-bit groups, fixed-window scoring |

Plug-ins (`--ext`, in the order given; the same specs work for ttf-core's trainer, which uses only their structure):

| plug-in | what it does |
|---|---|
| `ttfconv.fold[:q=had,seed=2110]` | folds the RMSNorm gains into the linears that read them and rotates the residual stream by one randomized Hadamard (exact, free at decode); the head is untied from the embedding |
| `ttfconv.cd[:sweeps=8,tol=1e-4]` | coordinate descent on the trits after GPTQ |
| `ttfconv.adapters:rank=R,alt=N[,prop=1][,gmetric=DIR][,svd=rand][,ttf_rank=R]` | **ternary low-rank adapters**: `y = T(x−μ) + b + U·D(x−μ)`, with D and U ternary, each with its own scales |
| `ttfconv.embq:emb=E,head=H[,g=64]` | the embedding / head at E / H bits in groups of g at the end of the conversion |
| `ttfconv.adapter_ctl:mode=frozen\|off` | training only (ttf-core): the adapters kept at their conversion fit, or taken out of the forward |
| `ttfconv.finish:out=DIR[,eval=...][,q4=1]` | (legacy, ttf-core's trainer before the split) size / eval / state right after the conversion |

The adapters' options:
- `rank`: an integer, or a fraction of min(K, N).
- `alt`: rounds of alternating refit of D and U (1 is recommended).
- `prop=1`: the calibration stream goes through each adapted layer, so later layers are rounded for the inputs they
  will see.
- `gmetric=DIR`: fit the adapter subspace in the output-sensitivity metric G⊗H instead of H alone (G from
  `ttfconv.tools.kfac --dump-g`).
- `svd=rand`: the adapter subspace by randomized SVD (the 27B default; the full SVD's fit to ±0.001).
- `ttf_rank=R`: the adapter factors' own TTF accumulator rank (when ttf-core trains them).

The adapters are fitted inside the centred streaming conversion (`--method gptqc`, the default).

## Install

```bash
pip install "ttf-ternary[qwen] @ git+https://github.com/oguzeldereli/ttf-ternary"
pip install -e .                 # this repository (package ttfconv)
```

You need a CUDA GPU for conversion (the ternary kernels are Triton). `pytest` runs the tests (`-m "not gpu"` for the
CPU ones).

## Quickstart: Qwen3.5-0.8B

The data files are uint32 token streams in Qwen's vocabulary (ttf-core: `scripts/quant/retok.py`).

```bash
M=models/Qwen3.5-0.8B-Base
# convert (fold + rank-16 float adapters from the rounding error's SVD), score 16 wiki windows
python -m ttfconv.convert --model $M --data data/cal.bin --out runs/q08_conv --ext ttfconv.fold --svd-rank 16 \
    --eval wiki@data/val.bin@16@2048
# optional: recovery training with ttf-core from the converted checkpoint (same structural plug-ins, --adapter-rank
# for the float adapters)
python -m ttf.train --recipe recipes/qwen-0.8b.toml --arch-hf $M --init-ckpt runs/q08_conv/ckpt.pt --teacher $M \
    --data data/train.bin --ext ttfconv.fold --adapter-rank 16 --out-dir runs/q08
```

A Llama-style checkpoint (the 110M models) converts with the same command (`ttfconv.llama`, not streamed; uint16
data: `--data-dtype u16`); it trains with `ttf.train --arch-hf $M --init-ckpt OUT/ckpt.pt`.

`scripts/analysis` and `scripts/quant` hold the conversion studies that left ttf-core on 28 Oct 2026; they find
ttf-core through `$TTF_CORE` (default `../ttf-core`). `scripts/legacy` keeps three Qwen checks that need ttf-core
0ef62dd (see its README).

A checkpoint converted with `--fine` carries the extra digits; it loads with `ttfconv.model.load_converted`, not in
ttf-core's trainer.

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
`recipes/qwen-27b.toml` is the recovery run on two 96 GB cards (rank 1536, 27.0 s per step as a two-stage pipeline);
with the TTF state in host memory the structure also trains on a 24 GB card (21.7 GiB peak measured at rank 256).

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
