"""Download the KD audio (a broad English mix, ~250 h; transcripts unused: the teacher labels it) and the Open ASR
leaderboard test shards into ASR_STORE (default $S/data/asr). Idempotent: files already there are skipped.
  python examples/asr/fetch_data.py [--eval-only] [--eval-all] [--first]      (--first: one training shard, for a dry run)
The teacher goes to ASR_TEACHER (default ASR_STORE/../../models/parakeet-tdt-0.6b-v3)."""
import argparse
import os

from huggingface_hub import hf_hub_download

from common import store

TRAIN = (                     # (repo, files): about 106 + 45 + 30 + 28 + 60 h
    ("openslr/librispeech_asr", [f"all/train.clean.100/{i:04d}.parquet" for i in range(4)]
     + [f"all/train.clean.360/{i:04d}.parquet" for i in range(4)]
     + [f"all/train.other.500/{i:04d}.parquet" for i in range(6)]),
    ("facebook/voxpopuli", [f"en/train-{i:05d}-of-00030.parquet" for i in range(2)]),
    ("MLCommons/peoples_speech", [f"clean/train-{i:05d}-of-00804.parquet" for i in range(0, 804, 54)]),
    ("edinburghcstr/ami", [f"ihm/train-{i:05d}-of-00042.parquet" for i in range(0, 42, 3)]),
    ("parler-tts/mls_eng", [f"data/train-{i:05d}-of-01416.parquet" for i in (0, 700)]),
)
EVAL_SHARDS = {"librispeech": ["test.clean-00000-of-00001", "test.other-00000-of-00001"], "ami": 15, "earnings22": 5,
               "gigaspeech": 19, "spgispeech": 38, "voxpopuli": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-all", action="store_true", help="every test shard (default: shard 0 of each set)")
    ap.add_argument("--first", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/parakeet-tdt-0.6b-v3", allow_patterns=["*.json", "model.safetensors"],
                      local_dir=os.environ.get("ASR_TEACHER", store("..", "..", "models", "parakeet-tdt-0.6b-v3")))
    for d, s in EVAL_SHARDS.items():
        names = s if isinstance(s, list) else [f"test-{i:05d}-of-{s:05d}" for i in range(s if a.eval_all else 1)]
        for n in names:
            hf_hub_download("hf-audio/open-asr-leaderboard", f"{d}/{n}.parquet", repo_type="dataset",
                            local_dir=store("oal"))
    if not os.path.exists(store("whisper_normalizer.json")):
        import shutil
        shutil.copy(hf_hub_download("openai/whisper-large-v3", "normalizer.json"), store("whisper_normalizer.json"))
    if a.eval_only:
        return
    for repo, files in TRAIN[:1] if a.first else TRAIN:
        for f in files[:1] if a.first else files:
            dst = store("train", repo.split("/")[1], f.replace("/", "_"))
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=store("train", ".dl"))
            os.replace(p, dst)
            print("got", dst, round(os.path.getsize(dst) / 1e6), "MB", flush=True)


if __name__ == "__main__":
    main()
