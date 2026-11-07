"""Ternary Parakeet TDT by conversion + distillation: the student (student.py: every linear / pointwise conv
ternary, converted by GPTQ from nvidia/parakeet-tdt-0.6b-v3) trained by TTF against the fp teacher on unlabelled
English audio (fetch_data.py). The teacher decodes each batch greedily; the loss is the KL divergence of the
student's joint distributions (tokens and durations) from the teacher's at every step of the teacher's path (frame t,
u tokens emitted), plus w_feat times the encoder output's relative squared error. Floats (scales, centring biases,
norms, the LSTM decoder, embedding, depthwise / subsampling convs) by AdamW. Resumable: OUT/ckpt.pt.

  python examples/asr/kd.py --out runs/asr_kd1 [--conv runs/asr_conv/conv.pt] [--steps 3000] ...
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import EVAL_SETS, Normalizer, batches, decode_audio, encode, eval_rows, features, greedy, load_teacher, \
    store, stt, wer
from student import calibrate, convert, skeleton, tern_layers
from ttf.layers import DX_MODE, FACTORED, FUSED_LAYER
from ttf.schedule import theta_at
from ttf.update import ts_layer, ts_step

TRACK = ("librispeech/test.other", "ami", "earnings22", "voxpopuli")


def sources():
    root = store("train")
    out = {}
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        fs = sorted(os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d)) if f.endswith(".parquet"))
        if fs and not d.startswith("."):
            out[d] = (fs, 1.0)                          # (each source equally often)
    return out


def remap(st, f):
    """the TTF state with every layer id key (an int, or in a tuple) mapped by f"""
    def k2(k):
        if isinstance(k, tuple) and k[:1] != ("L",):
            return tuple(k2(x) for x in k)
        return f(k)
    return {k2(k): (remap(v, f) if isinstance(v, dict) else v) for k, v in st.items()}


def evaluate(proc, model, dev, sets, n, norm, cache):
    res = {}
    for s in sets:
        if (s, n) not in cache:
            rows = eval_rows(s, n)
            cache[(s, n)] = ([decode_audio(r[1]) for r in rows], [r[2] for r in rows])
        au, refs = cache[(s, n)]
        res[s] = round(100 * wer(refs, stt(proc, model, au, dev), norm), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conv", default="", help="converted student state (made and saved here when missing)")
    ap.add_argument("--method", default="gptq", choices=("gptq", "als"))
    ap.add_argument("--calib", type=int, default=128, help="utterances for the conversion's input statistics")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-lr", type=float, default=1.5e-4)
    ap.add_argument("--float-lr-mult", type=float, default=0.1, help="for the teacher's float matrices (LSTM, convs, embedding)")
    ap.add_argument("--theta", type=float, default=16.0, help="the base threshold (ttf.train --ttf-theta)")
    ap.add_argument("--theta-start", type=float, default=64.0, help="theta at step 0, down to --theta over the warmup")
    ap.add_argument("--theta-end", type=float, default=80.0, help="theta at the last step (start = end = theta: constant)")
    ap.add_argument("--tau", type=float, default=1000.0)
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--sec", type=float, default=300.0, help="seconds of audio per step")
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--save-secs", type=float, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute the encoder layers in the backward (memory)")
    ap.add_argument("--final-n", type=int, default=0, help="utterances per set in the final score (0: all downloaded)")
    ap.add_argument("--freeze-trits", action="store_true", help="no TTF update (the floats alone)")
    ap.add_argument("--freeze-floats", action="store_true", help="no AdamW step (the trits alone)")
    ap.add_argument("--bench", type=int, default=0, help="N steps with the time split, no eval or save, then exit")
    a = ap.parse_args()
    dry = os.environ.get("DRY", "0") == "1"
    if dry:
        a.steps, a.calib, a.eval_n, a.eval_every = 2, 4, 8, 1
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    say = lambda *x: print(*x, flush=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    DX_MODE[0] = "int8"; FUSED_LAYER[0] = True
    proc, teacher = load_teacher(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    norm = Normalizer()
    src = sources()
    say("sources:", {k: len(v[0]) for k, v in src.items()})
    data = batches(src, a.seed, a.sec, a.workers)

    from transformers import ParakeetForTDT
    tpath = os.environ.get("ASR_TEACHER", store("..", "..", "models", "parakeet-tdt-0.6b-v3"))
    tpath = tpath if os.path.isdir(tpath) else "nvidia/parakeet-tdt-0.6b-v3"
    ck = os.path.join(a.out, "ckpt.pt")
    if a.conv and os.path.exists(a.conv) or os.path.exists(ck):
        student = skeleton(ParakeetForTDT.from_pretrained(tpath, dtype=torch.float32).to(dev), dev)
        if not os.path.exists(ck):
            student.load_state_dict(torch.load(a.conv, map_location=dev)["model"])
            say("loaded the converted student", a.conv)
    else:
        student = ParakeetForTDT.from_pretrained(tpath, dtype=torch.float32).to(dev).eval()
        calib = [next(data)[0] for _ in range(a.calib)]           # (one utterance per batch: no padding rows)

        def run(m, w):
            x, mk = features(proc, [w], dev)
            enc, lens = encode(m, x, mk)
            greedy(m, enc, lens)
        t0 = time.time()
        stats = calibrate(student, calib, run)
        say(f"calibrated on {len(calib)} utterances ({time.time() - t0:.0f}s)")
        errs = convert(student, stats, dev, a.method, log=say)
        del stats, calib
        torch.cuda.empty_cache()
        e = [v for v in errs.values() if v == v]
        say(f"converted {len(errs)} layers ({a.method}): output-space error mean {sum(e) / len(e):.4f}, "
            f"max {max(e):.4f} ({max(errs, key=lambda k: errs[k] if errs[k] == errs[k] else -1)}) ({time.time() - t0:.0f}s)")
        if a.conv and not dry:
            os.makedirs(os.path.dirname(a.conv) or ".", exist_ok=True)
            torch.save({"model": student.state_dict(), "errs": errs, "method": a.method}, a.conv + ".tmp")
            os.replace(a.conv + ".tmp", a.conv)
    student.eval()                                   # (no dropout / layerdrop; BatchNorm keeps the teacher's statistics)
    student.decoder.lstm.train()                     # (cudnn's LSTM backward needs it; the LSTM has no dropout)
    if a.grad_ckpt:
        student.encoder.gradient_checkpointing_enable({"use_reentrant": False})
        student.encoder.layers.train()               # (HF checkpoints only in training mode; the encoder itself stays in
        for m in student.encoder.layers.modules():   # eval: no layerdrop; the layers' dropout rates to 0, BatchNorm eval)
            for k in ("activation_dropout", "attention_dropout", "dropout"):
                if isinstance(getattr(m, k, None), float):
                    setattr(m, k, 0.0)
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
    tern = tern_layers(student)
    n_t = sum(m.N * m.K for m in tern)
    n_all = sum(p.numel() for p in student.parameters()) + n_t
    say(f"student: {len(tern)} ternary layers, {n_t / 1e6:.1f}M trits; float parameters "
        f"{sum(p.numel() for p in student.parameters()) / 1e6:.2f}M ({100 * (1 - n_t / n_all):.1f}%)")

    small = {id(p) for m in tern for p in (m.row_scale, m.col_scale, m.out_bias)}
    small |= {id(p) for n, p in student.named_parameters() if p.dim() == 1}
    fl = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in fl if id(p) in small], "mult": 1.0},
                             {"params": [p for p in fl if id(p) not in small], "mult": a.float_lr_mult}],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    lr_state = {}
    cfg = dict(r=a.rank, lr_ratio=1.0, theta=a.theta, tau=a.tau, b2=0.95, spend=1.0, store_a="", offload=False,
               theta_sched=None, progress=0.0, factored=256, band=0.0, band_lo=0.5, hspend=0.0)
    for i, m in enumerate(tern):
        m._ts_idx = i
        if not a.freeze_trits:
            m.on_grad = lambda layer, g, _c=cfg: ts_layer(layer, g, lr_state, _c)
    step = 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        student.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); step = c["step"]
        lr_state.update(remap(c["ttf"], lambda k: id(tern[k[1]]) if isinstance(k, tuple) and k[:1] == ("L",) else k))
        say(f"resumed at step {step} ({len(lr_state)} TTF state entries)")

    def save():
        if dry:
            return
        idx = {id(m): ("L", i) for i, m in enumerate(tern)}
        st = remap(lr_state, lambda k: idx.get(k, k) if isinstance(k, int) else k)
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "ttf": st, "step": step}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    cache = {}
    if a.bench:
        a.steps, a.eval_every, a.save_secs = a.bench, 10 ** 9, 1e18
    if step == 0 and not a.bench:
        r = evaluate(proc, student, dev, TRACK[:1] if dry else TRACK, a.eval_n, norm, cache)
        say("converted student, WER", r); log.write(json.dumps({"step": 0, "wer": r}) + "\n"); log.flush()
        if not dry and not os.path.exists(os.path.join(a.out, "teacher_track.json")):
            rt = evaluate(proc, teacher, dev, TRACK, a.eval_n, norm, cache)
            say("teacher, WER", rt); json.dump(rt, open(os.path.join(a.out, "teacher_track.json"), "w"))
    lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else \
        a.min_lr + (a.lr - a.min_lr) * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
    V = teacher.config.vocab_size
    blank = teacher.config.blank_token_id
    t0, t_save, sec_seen = time.time(), time.time(), 0.0
    while step < a.steps:
        wavs = next(data)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr * g["mult"]
        # the core theta schedule in rescale form, as ttf.train (before 7 Nov: the step (lr / peak)^0.75)
        cfg["lr_ratio"] = a.theta / theta_at(step / a.steps, a.theta, a.theta_start, a.theta_end, a.warmup / a.steps)
        cfg["progress"] = step / a.steps
        cfg["step"], cfg["start"], cfg["micro"], cfg["vdiag"] = step, 0, 1, False
        FACTORED[0] = True
        for m in tern:
            m.capture = not a.freeze_trits
        torch.cuda.synchronize(); tA = time.time()
        x, mk = features(proc, wavs, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tenc, lens = encode(teacher, x.to(torch.bfloat16), mk)
            toks, (bi, ti, ui, tz) = greedy(teacher, tenc, lens, record=True)
        U = max(1, max(len(t) for t in toks))
        lab = torch.full((len(toks), U + 1), blank, dtype=torch.long)
        for b, tk in enumerate(toks):
            lab[b, 1:1 + len(tk)] = torch.tensor(tk, dtype=torch.long)
        lab = lab.to(dev)
        torch.cuda.synchronize(); tB = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            senc, _ = encode(student, x, mk)
            sdec = student.decoder(lab)
            sz = student.joint(decoder_hidden_states=sdec[bi, ui], encoder_hidden_states=senc[bi, ti]).float()
        tz = tz.float()
        kl_tok = (F.softmax(tz[:, :V], -1) * (F.log_softmax(tz[:, :V], -1) - F.log_softmax(sz[:, :V], -1))).sum(-1).mean()
        kl_dur = (F.softmax(tz[:, V:], -1) * (F.log_softmax(tz[:, V:], -1) - F.log_softmax(sz[:, V:], -1))).sum(-1).mean()
        valid = (torch.arange(senc.shape[1], device=dev)[None, :] < lens[:, None]).unsqueeze(-1)
        te = tenc.float()
        feat = (((senc.float() - te) ** 2) * valid).sum() / ((te ** 2) * valid).sum().clamp_min(1e-12)
        loss = kl_tok + kl_dur + a.w_feat * feat
        loss.backward()
        info = ts_step(student, lr_state, cfg) if not a.freeze_trits else {}
        if a.bench:
            torch.cuda.synchronize()
            say(f"bench step {step}: {sum(len(w) for w in wavs) / 16000:.0f} s audio, bs {len(wavs)}, teacher {tB - tA:.2f} s, "
                f"student fwd+bwd+ttf {time.time() - tB:.2f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
        torch.nn.utils.clip_grad_norm_(fl, 1.0)
        if not a.freeze_floats:
            opt.step()
        step += 1
        sec_seen += sum(len(w) for w in wavs) / 16000
        if step % 10 == 0 or dry:
            rec = {"step": step, "loss": loss.item(), "kl_tok": kl_tok.item(), "kl_dur": kl_dur.item(), "feat": feat.item(),
                   "lr": lr, "bs": len(wavs), "path": int(bi.numel()), "hours": round(sec_seen / 3600, 2),
                   "s": round(time.time() - t0, 1), **{k: v for k, v in info.items() if isinstance(v, (int, float))}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 50 == 0 or dry:
                say(f"step {step} loss {loss.item():.4f} (tok {kl_tok.item():.4f} dur {kl_dur.item():.4f} feat {feat.item():.4f}) "
                    f"lr {lr:.2e} {sec_seen / 3600:.1f} h {(time.time() - t0) / step:.2f} s/step "
                    f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
        if a.bench:
            continue
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(proc, student, dev, TRACK[:1] if dry else TRACK, a.eval_n, norm, cache)
            say(f"step {step} WER", r); log.write(json.dumps({"step": step, "wer": r}) + "\n"); log.flush()
        if time.time() - t_save > a.save_secs:
            save(); t_save = time.time()
    if a.bench:
        os._exit(0)
    save()
    sets = EVAL_SETS[:1] if dry else EVAL_SETS
    r = evaluate(proc, student, dev, sets, 16 if dry else a.final_n, norm, {})
    say("FINAL student WER (all downloaded test shards)", r)
    json.dump(r, open(os.path.join(a.out, "final_wer.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(a.out, "teacher_final.json")) and not dry:
        rt = evaluate(proc, teacher, dev, sets, a.final_n, norm, {})
        say("teacher WER (same shards)", rt); json.dump(rt, open(os.path.join(a.out, "teacher_final.json"), "w"), indent=1)
    if dry:
        open(os.path.join(a.out, "DRY_OK"), "w").close()
    os._exit(0)                                  # (the DataLoader workers)


if __name__ == "__main__":
    main()
