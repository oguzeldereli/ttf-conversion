#!/bin/bash
# Qwen3.8-27B conversion with the ttfconv plug-ins: fold a, T2 on a 0.229 share, r0.2 ternary adapters, centred GPTQ
# (recipes/qwen-27b.toml: convert = gptqc, the adapters' fit needs it), streamed layer by layer from the safetensors on
# one 24-48 GB GPU, with the training-free fixes that need no extra pass at 27B:
#   prop=1   the calibration stream through the adapted layers (0.8B: -0.10 nats on all three sets)
#   alt=1    one alternating refit round of the adapters (0.8B: -0.024 training-free, 3 rounds; -0.008 after 1k)
#   svd=rand the adapters' subspace by randomized SVD (the full SVD was 47% of the 24 Oct L40S conversion, 4.7 s per
#            call; same fit error to +-0.001 on 0.8B layers). SVD=full: the old fit
# EMBQ=2,3: embedding 2-bit / head 3-bit (ttfconv.embq) before training; use with Q4=0.
# Not included (cost at 27B, see docs/CONVERT_27B.md): gmetric (needs G from a backward of the 27B fp model, streamed),
# T2 by measured damage (needs a single-linear KL per linear: ~400 forwards of 27B).
# Ends in ttfconv.finish: the packed size, the 4-bit embedding / head (q4=1: the release pack; q4=0 for a training
# start), the fixed windows scored, OUT/state.pt.
#   CORE=... CONV=... M=models/qw27 DATA=data/mix5b_cal.bin OUT=runs/q27_conv_v3 EVAL="wiki@data/wikiq27_eval27.bin@16@2049" \
#     [Q4=0] [ALT=1] [PROP=1] [T2=0.229] [RANK=0.2] [CD=0] [SVD=rand] [EMBQ=2,3] [ADTTF=R] bash scripts/convert27.sh
set -eu
CORE=${CORE:?}; CONV=${CONV:-$(cd "$(dirname "$0")/.." && pwd)}; M=${M:?}; DATA=${DATA:?}; OUT=${OUT:?}
PY=${PY:-python}; ALT=${ALT:-1}; PROP=${PROP:-1}; T2=${T2:-0.229}; RANK=${RANK:-0.2}; CD=${CD:-0}; Q4=${Q4:-1}
SVD=${SVD:-rand}; EMBQ=${EMBQ:-}; ADTTF=${ADTTF:-}
EVAL=${EVAL:-}
EXT=(--ext ttfconv.fold)
[ "$CD" != 0 ] && EXT+=(--ext ttfconv.cd:sweeps=$CD)
AD=rank=$RANK,alt=$ALT,t2=$T2,prop=$PROP,svd=$SVD; [ -n "$ADTTF" ] && AD=$AD,ttf_rank=$ADTTF
EXT+=(--ext ttfconv.adapters:$AD)
[ -n "$EMBQ" ] && EXT+=(--ext ttfconv.embq:emb=${EMBQ%,*},head=${EMBQ#*,})
FIN="ttfconv.finish:out=$OUT,q4=$Q4"; [ -n "$EVAL" ] && FIN="$FIN,eval=$EVAL"
EXT+=(--ext "$FIN")
mkdir -p $OUT
cd $CORE
env PYTHONPATH=$CORE:$CONV $PY -m ttf.train --recipe $CONV/recipes/qwen-27b.toml --init-hf $M --data $DATA --kd-weight 0 \
  --teacher "" --ttf-theta 1000000 --steps 1000 --mem-cap-gib ${CAP:-22.4} "${EXT[@]}" --out-dir ${OUT}_run 2>&1 | tee $OUT/convert.log
