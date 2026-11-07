"""WER of the fp teacher (or a saved student) on the Open ASR leaderboard test sets (common.EVAL_SETS):
  python examples/asr/eval_teacher.py [--limit N] [--sets a,b] [--out FILE]"""
import argparse
import json
import time

import torch

from common import EVAL_SETS, Normalizer, decode_audio, eval_rows, load_teacher, transcribe, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="utterances per set (0: all downloaded)")
    ap.add_argument("--sets", default=",".join(EVAL_SETS))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    proc, model = load_teacher(dev)
    norm = Normalizer()
    res = {}
    for s in a.sets.split(","):
        rows = eval_rows(s, a.limit)
        audios = [decode_audio(r[1]) for r in rows]
        t0 = time.time()
        hyps = transcribe(proc, model, audios, dev, a.bs)
        w = wer([r[2] for r in rows], hyps, norm)
        hours = sum(len(x) for x in audios) / 16000 / 3600
        res[s] = {"wer": w, "n": len(rows), "hours": round(hours, 3), "secs": round(time.time() - t0, 1)}
        print(f"{s:24s} n {len(rows):5d} {hours:6.2f} h  WER {100 * w:6.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        print("   e.g. ref:", rows[0][2][:90], "\n        hyp:", hyps[0][:90], flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
