#!/usr/bin/env python3
"""
encode.py - the ONLY GPU step of the v2 experiments.

The Whisper-small encoder in our detector is frozen, so its output for a given
input never changes. We therefore run every clip through it once per
preprocessing condition and save the pooled 768-d vectors. All detector heads
(ablation, seeds, split types, injection tests, reverse direction) are then
trained on these saved vectors by heads.py in seconds.

Outputs (in --out_dir):
  emb/<source>__<condition>.npz   ids, y (1=fake), emb_all, emb_valid, n_valid,
                                  hand (10 hand features), mfcc (26)
  spk/gary_wavlm.npz              speaker embeddings for the group split
  meta.json                       versions, dataset revision, timings

Sources:
  gary       garystafford/deepfake-audio-detection, all 1,866 clips (index order)
  asv_eval   the same 2,000 ASVspoof 2019 LA eval clips as the paper (manifest)
  asv_train  1,000 bona fide + 1,000 spoof from ASVspoof 2019 LA train (seed 42)
  itw        ALLA1N in-the-wild validation set (102 clips), optional
  inj_*      counterfactual injection sets built from 500+500 asv_eval clips

Run on a Kaggle GPU (T4) with "Save & Run All". Re-running skips finished files.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SIL_LEVELS_MS = [0, 50, 100, 200, 400]        # added at BOTH ends
GAIN_PEAK_DBFS = [-1, -6, -12, -20]           # peak normalised to this level
INJ_CONDITIONS = ["raw", "sil", "all"]


# ------------------------------------------------------------------ loaders
def load_gary():
    from datasets import load_dataset, Audio
    ds = load_dataset("garystafford/deepfake-audio-detection", split="train")
    names = ds.features["label"].names
    assert [n.lower() for n in names] == ["real", "fake"], f"unexpected label names {names}"
    ds = ds.cast_column("audio", Audio(decode=False))
    ids, ys, wavs, srs = [], [], [], []
    for i in range(len(ds)):
        row = ds[i]
        a = row["audio"]
        data = a["bytes"] if a.get("bytes") is not None else open(a["path"], "rb").read()
        y, sr, _ = C.decode_bytes(data)
        ids.append(f"gary_{i:04d}"); ys.append(int(row["label"]))   # 1 = fake
        wavs.append(C.to_16k(y, sr)); srs.append(sr)
    print(f"  gary: {len(ids)} clips, fake={sum(ys)}, native sr real={sorted(set(s for s, l in zip(srs, ys) if l == 0))} "
          f"fake={sorted(set(s for s, l in zip(srs, ys) if l == 1))}")
    return ids, np.array(ys), wavs


def read_file(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return C.to_16k(y.mean(axis=1), sr)


def find_asv_root():
    hits = glob.glob("/kaggle/input/**/ASVspoof2019_LA_cm_protocols", recursive=True)
    if not hits:
        raise FileNotFoundError("ASVspoof 2019 LA not found: add the awsaf49/asvpoof-2019-dataset input")
    return os.path.dirname(hits[0])


def load_asv_eval(manifest):
    m = pd.read_csv(manifest)
    root = find_asv_root()
    paths = []
    for p in m["filepath"]:
        if not os.path.exists(p):                      # manifest written in another session
            p = os.path.join(root, "ASVspoof2019_LA_eval", "flac", os.path.basename(p))
        paths.append(p)
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    y_fake = 1 - m["label"].astype(int).values         # manifest: 1 = bona fide
    wavs = [read_file(p) for p in paths]
    print(f"  asv_eval: {len(ids)} clips, fake={int(y_fake.sum())}")
    return ids, y_fake, wavs


def load_asv_train(n_per_class=1000, seed=42):
    root = find_asv_root()
    proto = os.path.join(root, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
    rows = [l.split() for l in open(proto)]
    df = pd.DataFrame({"id": [r[1] for r in rows], "key": [r[-1] for r in rows]})
    parts = []
    for key in ("bonafide", "spoof"):
        sub = df[df.key == key]
        parts.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    df = pd.concat(parts).reset_index(drop=True)
    ids = df["id"].tolist()
    y_fake = (df["key"] == "spoof").astype(int).values
    wavs = [read_file(os.path.join(root, "ASVspoof2019_LA_train", "flac", i + ".flac")) for i in ids]
    print(f"  asv_train: {len(ids)} clips, fake={int(y_fake.sum())}")
    return ids, y_fake, wavs


def load_itw():
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id="ALLA1N/In-the-wild_validation-dataset", repo_type="dataset")
    ids, ys, wavs = [], [], []
    for lab, sub in ((0, "real"), (1, "fake")):
        folder = os.path.join(root, sub)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                import librosa
                y, sr = librosa.load(os.path.join(folder, f), sr=None, mono=True)
                ids.append(f"itw_{sub}_{f}"); ys.append(lab); wavs.append(C.to_16k(y, sr))
    print(f"  itw: {len(ids)} clips, fake={sum(ys)}")
    return ids, np.array(ys), wavs


# ------------------------------------------------------------------ encoder
class Encoder:
    def __init__(self, name="openai/whisper-small", fp16=False):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.enc = WhisperModel.from_pretrained(name).encoder.to(self.dev).eval()
        self.fp16 = fp16 and self.dev == "cuda"
        self.gpu_fbank = True
        print(f"  encoder on {self.dev}, fp16={self.fp16}")

    def features(self, batch):
        if self.gpu_fbank and self.dev == "cuda":
            try:
                return self.fe(batch, sampling_rate=C.SR, return_tensors="pt", device=self.dev).input_features
            except TypeError:
                self.gpu_fbank = False
        return self.fe(batch, sampling_rate=C.SR, return_tensors="pt").input_features

    def __call__(self, waves, n_valid):
        torch = self.torch
        batch = [C.fit_length(w, C.WHISPER_SAMPLES) for w in waves]    # exactly what model.py does
        feats = self.features(batch).to(self.dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16):
            h = self.enc(feats).last_hidden_state.float()               # (B, 1500, 768)
        e_all = h.mean(dim=1)
        nf = [int(min(max(np.ceil(n / C.SAMPLES_PER_FRAME), 1), h.shape[1])) for n in n_valid]
        e_valid = torch.stack([h[i, :k].mean(dim=0) for i, k in enumerate(nf)])
        return e_all.cpu().numpy(), e_valid.cpu().numpy()


def encode_set(enc, name, ids, y, wavs, condition, out_dir, bs, transform=None):
    path = os.path.join(out_dir, "emb", f"{name}__{condition}.npz")
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)}")
        return
    t0 = time.time()
    E_all, E_val, NV, HAND, MF = [], [], [], [], []
    for s in range(0, len(ids), bs):
        ws, nvs = [], []
        for cid, w in zip(ids[s:s + bs], wavs[s:s + bs]):
            if transform is not None:
                w = transform(w)
            pw, nv = C.preprocess(w, condition, cid)
            content = pw[:nv]
            hf = C.core_features(content)
            HAND.append([hf[k] for k in C.CORE7 + C.REL3])
            MF.append(C.mfcc_stats(content))
            ws.append(pw); nvs.append(nv)
        a, v = enc(ws, nvs)
        E_all.append(a); E_val.append(v); NV.extend(nvs)
    np.savez_compressed(path, ids=np.array(ids), y=np.asarray(y, dtype=np.int8),
                        emb_all=np.concatenate(E_all).astype(np.float32),
                        emb_valid=np.concatenate(E_val).astype(np.float32),
                        n_valid=np.array(NV), hand=np.array(HAND, dtype=np.float32),
                        hand_names=np.array(C.CORE7 + C.REL3), mfcc=np.array(MF, dtype=np.float32))
    dt = time.time() - t0
    print(f"  saved {os.path.basename(path)}: {len(ids)} clips in {dt/60:.1f} min ({len(ids)/dt:.1f} clips/s)")


# ------------------------------------------------------------ injections
def add_silence(ms):
    n = int(C.SR * ms / 1000)
    z = np.zeros(n, dtype=np.float32)
    return lambda w: np.concatenate([z, w, z])


def set_peak(dbfs):
    target = 10 ** (dbfs / 20)
    return lambda w: (w * (target / max(float(np.max(np.abs(w))), 1e-9))).astype(np.float32)


# ------------------------------------------------------------ speakers
def speaker_embeddings(ids, wavs, out_dir):
    path = os.path.join(out_dir, "spk", "gary_wavlm.npz")
    if os.path.exists(path):
        print("  [skip] speaker embeddings"); return
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    E = []
    with torch.no_grad():
        for w in wavs:
            x = C.trim(w)[: 10 * C.SR]
            inp = fe([x], sampling_rate=C.SR, return_tensors="pt", padding=True).to(dev)
            E.append(torch.nn.functional.normalize(m(**inp).embeddings, dim=-1).cpu().numpy()[0])
    np.savez_compressed(path, ids=np.array(ids), emb=np.array(E, dtype=np.float32))
    print(f"  saved speaker embeddings {np.array(E).shape}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/kaggle/working/v2_out")
    ap.add_argument("--manifest", default=None, help="asvspoof_eval_manifest.csv (2,000 clips used in the paper)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first N clips per source")
    ap.add_argument("--skip", default="", help="comma list of steps to skip: gary,asv_eval,asv_train,itw,inj,spk")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    for d in ("emb", "spk"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    import torch, transformers, datasets
    meta = {"torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "conditions": C.CONDITIONS, "crop_s": C.CROP_S, "target_rms_dbfs": C.TARGET_RMS_DBFS,
            "sil_levels_ms": SIL_LEVELS_MS, "gain_peak_dbfs": GAIN_PEAK_DBFS}
    try:
        from huggingface_hub import HfApi
        meta["gary_revision"] = HfApi().dataset_info("garystafford/deepfake-audio-detection").sha
        meta["whisper_revision"] = HfApi().model_info("openai/whisper-small").sha
    except Exception as e:
        meta["revision_error"] = str(e)
    T0 = time.time()
    enc = Encoder(fp16=args.fp16)

    def lim(ids, y, w):
        if args.limit:
            k = np.r_[np.where(np.asarray(y) == 0)[0][:args.limit // 2], np.where(np.asarray(y) == 1)[0][:args.limit // 2]]
            return [ids[i] for i in k], np.asarray(y)[k], [w[i] for i in k]
        return ids, y, w

    if "gary" not in skip:
        print("\n== gary"); ids, y, w = lim(*load_gary())
        for c in C.CONDITIONS:
            encode_set(enc, "gary", ids, y, w, c, args.out_dir, args.batch)
        if "spk" not in skip:
            try:
                speaker_embeddings(ids, w, args.out_dir)
            except Exception as e:
                print(f"  !! speaker embeddings failed ({type(e).__name__}: {e}); group split will be missing")
        del w

    if "asv_eval" not in skip or "inj" not in skip:
        print("\n== asv_eval"); ids, y, w = lim(*load_asv_eval(args.manifest))
        if "asv_eval" not in skip:
            for c in C.CONDITIONS + ["native"]:
                encode_set(enc, "asv_eval", ids, y, w, c, args.out_dir, args.batch)
        if "inj" not in skip:
            rng = np.random.default_rng(42)
            pick = np.concatenate([rng.choice(np.where(y == k)[0], min(500, int((y == k).sum())), replace=False)
                                   for k in (0, 1)])
            iid = [ids[i] for i in pick]; iy = y[pick]; iw = [w[i] for i in pick]
            for c in INJ_CONDITIONS:
                for ms in SIL_LEVELS_MS:
                    encode_set(enc, f"inj_sil{ms}", iid, iy, iw, c, args.out_dir, args.batch, add_silence(ms))
                for g in GAIN_PEAK_DBFS:
                    encode_set(enc, f"inj_peak{abs(g)}", iid, iy, iw, c, args.out_dir, args.batch, set_peak(g))
        del w

    if "asv_train" not in skip:
        print("\n== asv_train"); ids, y, w = lim(*load_asv_train())
        for c in C.CONDITIONS:
            encode_set(enc, "asv_train", ids, y, w, c, args.out_dir, args.batch)
        del w

    if "itw" not in skip:
        try:
            print("\n== itw"); ids, y, w = lim(*load_itw())
            for c in C.CONDITIONS:
                encode_set(enc, "itw", ids, y, w, c, args.out_dir, args.batch)
        except Exception as e:
            print(f"  itw skipped: {type(e).__name__}: {e}")

    meta["total_minutes"] = round((time.time() - T0) / 60, 1)
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)
    print(f"\nDONE in {meta['total_minutes']} min. Files in {args.out_dir}")


if __name__ == "__main__":
    main()
