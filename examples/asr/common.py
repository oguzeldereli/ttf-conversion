"""Speech-to-text with TTF: data, the teacher, decoding and WER shared by the scripts in examples/asr.

The teacher is nvidia/parakeet-tdt-0.6b-v3 (transformers ParakeetForTDT, CC-BY-4.0). Audio comes as parquet shards
(an `audio` column of {bytes, path}, 16 kHz, decoded with soundfile); WER is scored as the Open ASR leaderboard scores
it: Whisper's English normaliser on both sides, word edit distance summed over a set.
"""
import io
import json
import os

import numpy as np
import torch

SR = 16000
EVAL_SETS = ("librispeech/test.clean", "librispeech/test.other", "ami", "earnings22", "gigaspeech", "spgispeech",
             "voxpopuli")


def store(*p):
    return os.path.join(os.environ.get("ASR_STORE", os.path.expandvars("$S/data/asr")), *p)


def decode_audio(a):
    """a parquet audio cell {bytes, path} -> float32 mono at 16 kHz"""
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:                                   # (linear resampling: only for the few non-16 kHz sources)
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def eval_rows(name, limit=0):
    """an Open ASR leaderboard test set (its shards under store('oal', name...)): [(id, audio, text)]"""
    import glob
    import pyarrow.parquet as pq
    d, _, split = name.partition("/")
    pat = store("oal", d, (split + "-*.parquet") if split else "test-*.parquet")
    rows = []
    for f in sorted(glob.glob(pat)):
        t = pq.read_table(f, columns=["id", "audio", "text"]).to_pylist()
        rows += [(r["id"], r["audio"], r["text"]) for r in t]
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


class Normalizer:
    def __init__(self):
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        self.n = EnglishTextNormalizer(json.load(open(store("whisper_normalizer.json"))))

    def __call__(self, s):
        return self.n(s)


def word_errors(ref, hyp):
    """(edits, reference words) by word-level Levenshtein distance"""
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def wer(refs, hyps, norm):
    e = n = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        if not r.strip():                          # (the leaderboard drops empty references)
            continue
        a, b = word_errors(r, h)
        e += a; n += b
    return e / max(1, n)


def load_teacher(dev, dtype=torch.bfloat16):
    from transformers import AutoProcessor, ParakeetForTDT
    p = os.environ.get("ASR_TEACHER", store("..", "..", "models", "parakeet-tdt-0.6b-v3"))
    if not os.path.isdir(p):
        p = "nvidia/parakeet-tdt-0.6b-v3"
    proc = AutoProcessor.from_pretrained(p)
    model = ParakeetForTDT.from_pretrained(p, dtype=dtype).to(dev).eval()
    return proc, model


def features(proc, audios, dev):
    f = proc.feature_extractor(audios, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
    return f["input_features"].to(dev), f["attention_mask"].to(dev)


@torch.no_grad()
def transcribe(proc, model, audios, dev, bs=32):
    """greedy TDT decoding (model.generate) of a list of waveforms, longest first in batches"""
    order = sorted(range(len(audios)), key=lambda i: -len(audios[i]))
    out = [None] * len(audios)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        x, m = features(proc, [audios[i] for i in idx], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(input_features=x.to(model.dtype), attention_mask=m)
        seq = g.sequences if hasattr(g, "sequences") else g
        txt = proc.batch_decode(seq, skip_special_tokens=True)
        for i, t in zip(idx, txt):
            out[i] = t
    return out


def encode(model, x, m):
    """encoder outputs as the joint sees them (encoder_projector applied) [B, T, 640] and their valid lengths"""
    o = model.get_audio_features(input_features=x, attention_mask=m)
    return o.pooler_output, o.attention_mask.sum(-1)


@torch.no_grad()
def greedy(model, enc, lens, record=False, max_sym=10):
    """batched greedy TDT decoding (model.generate's rule: token and duration by argmax, a blank advances at least one
    frame) on encoder outputs enc [B, T, D] with valid lengths lens [B]. Returns the token lists and, with record,
    the path: (b, t, u, logits) at every decoding step, u the number of tokens emitted before it. After max_sym
    consecutive tokens at one frame the frame advances (a bound for a broken student; the teacher never reaches it)."""
    cfg = model.config
    V, blank, dev = cfg.vocab_size, cfg.blank_token_id, enc.device
    durs = torch.tensor(cfg.durations, device=dev)
    B, T, _ = enc.shape
    dec, ar = model.decoder, torch.arange(B, device=dev)
    k = torch.full((B,), blank, dtype=torch.long, device=dev)
    o, (h, c) = dec.lstm(dec.embedding(k[:, None]))
    g = dec.decoder_projector(o[:, 0])
    t = torch.zeros(B, dtype=torch.long, device=dev); u = torch.zeros_like(t); sym = torch.zeros_like(t)
    lens = lens.to(dev)
    emits, rec = [], []
    for it in range(T * (max_sym + 1) + 1):
        act = t < lens
        if it % 8 == 0 and not bool(act.any()):
            break
        z = model.joint(decoder_hidden_states=g, encoder_hidden_states=enc[ar, t.clamp(max=T - 1)])
        k = z[:, :V].argmax(-1); d = durs[z[:, V:].argmax(-1)]
        isb = k == blank
        d = torch.where(isb & (d == 0), torch.ones_like(d), d)
        nb = act & ~isb
        sym = torch.where(nb & (d == 0), sym + 1, torch.zeros_like(sym))
        d = torch.where(sym >= max_sym, torch.ones_like(d), d); sym = torch.where(sym >= max_sym, 0, sym)
        if record:
            rec.append((act, t.clone(), u.clone(), z))
        emits.append((nb, k))
        o2, (h2, c2) = dec.lstm(dec.embedding(k[:, None]), (h, c))
        w = nb[None, :, None]
        h, c = torch.where(w, h2, h), torch.where(w, c2, c)
        g = torch.where(nb[:, None], dec.decoder_projector(o2[:, 0]), g)
        u = u + nb.long(); t = t + torch.where(act, d, torch.zeros_like(d))
    NB = torch.stack([e[0] for e in emits], 1).cpu(); K_ = torch.stack([e[1] for e in emits], 1).cpu()
    toks = [K_[b][NB[b]].tolist() for b in range(B)]
    if not record:
        return toks, None
    A = torch.stack([r[0] for r in rec], 1)                       # [B, S]
    bi, si = A.nonzero(as_tuple=True)
    ti = torch.stack([r[1] for r in rec], 1)[bi, si]; ui = torch.stack([r[2] for r in rec], 1)[bi, si]
    Z = torch.stack([r[3] for r in rec], 1)[bi, si]
    return toks, (bi, ti, ui, Z)


def stt(proc, model, audios, dev, bs=32):
    """greedy() transcripts of a list of waveforms (batches longest first)"""
    order = sorted(range(len(audios)), key=lambda i: -len(audios[i]))
    out = [None] * len(audios)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        x, m = features(proc, [audios[i] for i in idx], dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc, lens = encode(model, x, m)
            toks, _ = greedy(model, enc, lens)
        for i, tk in zip(idx, toks):
            out[i] = proc.decode(tk, skip_special_tokens=True)       # (the processor: no CTC merging of repeats)
    return out


class Stream(torch.utils.data.IterableDataset):
    """endless random waveforms from parquet shards (an `audio` column): a source by weight, a shard and a row group
    at random, its rows shuffled; clips over max_s are cut to a random max_s window, under min_s dropped"""
    def __init__(self, sources, seed, max_s=20.0, min_s=1.0):
        self.sources, self.seed, self.max_n, self.min_n = sources, seed, int(max_s * SR), int(min_s * SR)

    def __iter__(self):
        import random
        import pyarrow.parquet as pq
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        names, files, wts = zip(*[(k, v[0], v[1]) for k, v in self.sources.items()])
        while True:
            f = rng.choice(files[rng.choices(range(len(names)), wts)[0]])
            p = pq.ParquetFile(f)
            rows = p.read_row_group(rng.randrange(p.num_row_groups), columns=["audio"]).column(0)
            order = list(range(len(rows))); rng.shuffle(order)
            for i in order[:64]:                      # (at most 64 rows per row group: less host memory, more mixing)
                a = rows[i].as_py()
                try:
                    x = decode_audio(a)
                except Exception:
                    continue
                if len(x) < self.min_n:
                    continue
                if len(x) > self.max_n:
                    s = rng.randrange(len(x) - self.max_n); x = x[s:s + self.max_n]
                yield torch.from_numpy(np.ascontiguousarray(x))


def batches(sources, seed, sec, workers=4, buf=256, max_bs=64):
    """batches of about sec seconds of audio (lists of waveforms of similar length, padding kept small)"""
    import random
    dl = torch.utils.data.DataLoader(Stream(sources, seed), batch_size=None, num_workers=workers, prefetch_factor=2,
                                     persistent_workers=True)
    it, rng = iter(dl), random.Random(seed)
    while True:
        pool = sorted((next(it) for _ in range(buf)), key=len)
        out, cur = [], []
        for x in pool:
            if cur and (len(cur) + 1) * len(x) > sec * SR or len(cur) >= max_bs:
                out.append(cur); cur = []
            cur.append(x)
        rng.shuffle(out)                                  # (the last, partial group is dropped)
        yield from out
