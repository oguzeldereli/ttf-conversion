# ttf-conversion

Convert a full-precision Qwen3.5 / Qwen3.8 model into a **ternary** network: weights in {−1, 0, +1} times row and
column scales, optionally with low-rank adapters and a fine residual of extra digits. The conversion is its own code
base. It builds its models with [ttf-core](https://github.com/oguzeldereli/ttf-ternary) (the Qwen wrapper, the packed
ternary layers and kernels) and writes checkpoints that ttf-core's trainer can start from (`--init-ckpt`), but none of
the conversion lives in ttf-core. It needs ttf-core v0.3.0 or later (v0.2.0 is the release without the conversion: `--arch-hf`, `ttf.hf_load`; v0.3.0 adds `ttf.q38`, which `ttfconv.q38` builds on).

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

## Released models

Made with this conversion and TTF recovery training; GGUF files that [ttf-home](https://github.com/oguzeldereli/ttf-home)
runs:

| model | file | GSM8K (first 200) | HumanEval |
|---|---|---|---|
| [ttf-Qwen3.8-27B-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF) | 8.2 GB | 0.895 (full precision 0.895) | 0.841 (0.933) |
| [ttf-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF) | 40.4 GB | 0.905 (FP8 0.905) | 0.848 (0.927) |

## Install

Linux with an NVIDIA GPU (the ternary kernels are Triton), Python 3.11+, PyTorch 2.13+. The data scripts live in
ttf-ternary, so clone both:

```bash
git clone https://github.com/oguzeldereli/ttf-ternary
git clone https://github.com/oguzeldereli/ttf-conversion
pip install -U pip "setuptools>=68"   # editable installs need setuptools 64+ (PEP 660)
pip install -e "ttf-ternary[qwen,data]" -e ttf-conversion
```

If pip says the build backend "is missing the 'build_editable' hook", it built without isolation (`--no-build-isolation`,
`PIP_NO_BUILD_ISOLATION` or a pip config) on a setuptools older than 64: `pip install -U "setuptools>=68"` and rerun.

`pytest -q -m "not gpu"` in `ttf-conversion` runs the CPU tests; `pytest -q` runs them all on the GPU machine.

## Quickstart: Qwen3.5-0.8B on one 16 GB GPU

The data files are uint32 token streams in Qwen's vocabulary. Below, 20M tokens of English Wikipedia are tokenized
with Llama's tokenizer (ttf-ternary's `prep_wiki.py`) and re-tokenized for Qwen.

```bash
hf download Qwen/Qwen3.5-0.8B-Base --local-dir models/Qwen3.5-0.8B-Base
M=models/Qwen3.5-0.8B-Base
python ttf-ternary/scripts/data/prep_wiki.py 2e7 --tokenizer llama --val_tokens 1000000 --out_prefix data/wiki
python ttf-ternary/scripts/quant/retok.py data/wiki_train.bin data/qw_train.bin --target $M
python ttf-ternary/scripts/quant/retok.py data/wiki_val.bin data/qw_val.bin --target $M

# convert: norm gains folded + a Hadamard rotation of the residual stream, centred ternary GPTQ, one coordinate-descent
# sweep, ternary low-rank adapters of rank 0.2; 16 Wikipedia windows scored at the end (OUT/result.json)
python -m ttfconv.convert --model $M --data data/qw_train.bin --out runs/q08_conv \
    --ext ttfconv.fold --ext ttfconv.cd:sweeps=1 --ext "ttfconv.adapters:rank=0.2,alt=1,prop=1" \
    --eval wiki@data/qw_val.bin@16@2048

# recovery: 1000 TTF steps (32k tokens each) with distillation from the original model, the same plug-ins
python -m ttf.train --recipe ttf-conversion/recipes/qwen-0.8b.toml --arch-hf $M --init-ckpt runs/q08_conv/ckpt.pt \
    --teacher $M --data data/qw_train.bin --val data/qw_val.bin --batch-size 2 --accum 8 \
    --ext ttfconv.fold --ext ttfconv.cd:sweeps=1 --ext "ttfconv.adapters:rank=0.2,alt=1,prop=1" --out-dir runs/q08
```

`--batch-size 2 --accum 8` is the fastest shape on a 16 GB card. On a 12 GB card, keep the recipe's `1 x 16`. [Results](#results-qwen35-08b-wiki--chat--mix-held-out-loss)
has this recipe on our larger training mix. With 20M tokens, 1000 steps (32M tokens) read some windows twice. Export the result to GGUF and chat with it in
[ttf-home](https://github.com/oguzeldereli/ttf-home) (`ttf export runs/q08/ckpt.pt q08.gguf --base $M`).

A conversion with rank-16 float adapters fitted to the rounding error instead (`--svd-rank 16`, no adapters plug-in)
trains with `--adapter-rank 16`.

A Llama-style checkpoint (the 110M models) converts with the same command (`ttfconv.llama`, not streamed; uint16
data: `--data-dtype u16`); it trains with `ttf.train --arch-hf $M --init-ckpt OUT/ckpt.pt`.

`scripts/analysis` and `scripts/quant` hold the conversion studies (they find ttf-ternary through `$TTF_CORE`,
default `../ttf-core`) and the fp-side tools: `prep_fp.py` (move the fp model toward its
rounding before converting, proximal or QuantNoise, under a loss-rise budget) and `fold_llama.py` (the gain fold and
residual rotation for the Llama layout). `scripts/legacy` keeps three Qwen checks that need ttf-core
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

## Qwen3.8-Flash-Next (MoE)

`ttfconv.q38.convert` turns every routed expert (512 per layer) and dense projection into trits, layer by layer on one
GPU: ternary GPTQ on the centred input covariance (act-order), ALS row / column scales, the input mean folded into a
bias, a rank-16 ternary adapter. `ttfconv.q38.teacher` writes the KD cache (top 64 + 64 sampled tail per position).
The result trains with ttf-core's `ttf.q38.train` (layer-major, model and TTF state in host memory); the whole
pipeline and the recipe are in ttf-core's docs/Q38.md. `scripts/q38`: the slim base checkpoint (`slim.py` from a
download, `hf_fetch.py` by HTTP range reads), the PLE n-gram table's compressibility (`ple_probe.py`), the loss of
compressed tables (`ple_loss.py`) and the stored prune1 + pq4.256 table (`ple_pq_build.py`, 7.95 GB instead of
51.2 GB). Check: `tests/checks/q38_pipeline.py` (GPU, needs ttf-core's `ttf.q38` and transformers >= 5.19).

## Examples

`examples/asr`: a ternary Parakeet TDT 0.6B speech recogniser by the same centred ternary GPTQ (`ttfconv.gptq`) and
KD, WER on the Open ASR leaderboard test sets.

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
