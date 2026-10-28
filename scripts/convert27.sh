#!/bin/bash
# Qwen3.8-27B conversion (python -m ttfconv.convert): fold a, centred GPTQ, streamed layer by layer from the safetensors
# on one 24-96 GB GPU. Two structures:
#   default        r0.2 ternary adapters (ttfconv.adapters) with the training-free fixes that need no extra pass:
#                    prop=1   the calibration stream through the adapted layers (0.8B: -0.10 nats on all three sets)
#                    alt=1    one alternating refit round of the adapters
#                    svd=rand the adapters' subspace by randomized SVD (same fit error to +-0.001 on 0.8B layers)
#   SVD=R          instead: rank-R float adapters from the rounding error's SVD (no ternary adapters)
#   FINE=27        the fine residual of what the trits and the adapters left (the extra digits; ttfconv.fine)
# EMBQ=2,3: embedding 2-bit / head 3-bit (ttfconv.embq) at the end of the conversion.
# Output: OUT/ckpt.pt (a ttf-core checkpoint, step -1: a training run starts from it with --init-ckpt) and
# OUT/result.json (sizes, errors, EVAL scores).
#   CORE=... CONV=... M=models/qw27 DATA=data/mix5b_cal.bin OUT=runs/q27_conv EVAL="wiki@data/wikiq27_eval27.bin@16@2049" \
#     [ALT=1] [PROP=1] [ADR=0.2] [CD=0] [SVDA=rand] [SVD=R] [FINE=27] [EMBQ=2,3] [ADTTF=R] [CALIB=64] bash scripts/convert27.sh
set -eu
CORE=${CORE:?}; CONV=${CONV:-$(cd "$(dirname "$0")/.." && pwd)}; M=${M:?}; DATA=${DATA:?}; OUT=${OUT:?}
PY=${PY:-python}; ALT=${ALT:-1}; PROP=${PROP:-1}; ADR=${ADR:-0.2}; CD=${CD:-0}
SVDA=${SVDA:-rand}; SVD=${SVD:-0}; FINE=${FINE:-0}; EMBQ=${EMBQ:-}; ADTTF=${ADTTF:-}; EVAL=${EVAL:-}; CALIB=${CALIB:-64}
EXT=(--ext ttfconv.fold)
[ "$CD" != 0 ] && EXT+=(--ext ttfconv.cd:sweeps=$CD)
if [ "$SVD" = 0 ]; then
  AD=rank=$ADR,alt=$ALT,prop=$PROP,svd=$SVDA; [ -n "$ADTTF" ] && AD=$AD,ttf_rank=$ADTTF
  EXT+=(--ext ttfconv.adapters:$AD)
fi
[ -n "$EMBQ" ] && EXT+=(--ext ttfconv.embq:emb=${EMBQ%,*},head=${EMBQ#*,})
X=(--svd-rank $SVD --fine $FINE --calib $CALIB)
[ -n "$EVAL" ] && X+=(--eval "$EVAL")
mkdir -p $OUT
env PYTHONPATH=$CORE:$CONV $PY -m ttfconv.convert --model $M --data $DATA --out $OUT "${EXT[@]}" "${X[@]}" 2>&1 | tee $OUT/convert.log
