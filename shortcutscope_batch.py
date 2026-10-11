#!/usr/bin/env python3
"""
ShortcutScope batch mode: run the acoustic-shortcut audit on many datasets.

For each dataset in a JSON list it:
  1. streams a class-balanced random sample (default 500 real + 500 fake clips),
  2. works out which label means FAKE (explicit in the JSON, or from the label
     names, e.g. "fake", "spoof", "bonafide", "real") and PRINTS the mapping so
     you can verify it,
  3. resamples to 16 kHz and extracts the 7 ShortcutScope features,
  4. reports Cohen's d (fake minus real), per-feature AUC, and the features-only
     logistic-regression AUC (stratified 80/20 split, seed 42),
  5. writes results after every dataset, so a crash never loses earlier work.

Usage (Kaggle / Colab, CPU is enough):
    python shortcutscope_batch.py --datasets datasets.json --per_class 500 \
        --out_dir /kaggle/working/audit_results

Requirements: pip install librosa numpy scikit-learn datasets soundfile huggingface_hub
"""

import argparse
import io
import json
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shortcutscope import FEATURE_NAMES, extract_features, cohens_d, interpret_d, run_shortcut_classifier

TARGET_SR = 16000
LABEL_CANDIDATES = ["label", "labels", "real_or_fake", "is_fake", "class", "target"]
AUDIO_CANDIDATES = ["audio", "wav", "speech", "file"]


# ---------------------------------------------------------------- labels
def classify_label_name(name):
    """Return 'fake', 'real' or None for a label name/value."""
    s = str(name).strip().lower()
    if re.search(r"bona|genuine|real|human|original|authentic", s) or s in {"r", "true_speech"}:
        if not re.search(r"fake|spoof", s):
            return "real"
    if re.search(r"fake|spoof|synth|generated|deepfake|clone|tts|convert|vocod", s) or re.fullmatch(r"f\d+", s):
        return "fake"
    return None


def label_names(features, label_col):
    try:
        return list(features[label_col].names)
    except Exception:
        return None


# ---------------------------------------------------------------- audio
def decode_audio(x):
    """Turn whatever the datasets library gives us into (mono float32 array, sr)."""
    import soundfile as sf

    if isinstance(x, dict):
        if x.get("array") is not None:
            y, sr = np.asarray(x["array"], dtype=np.float32), int(x["sampling_rate"])
        else:
            data = x.get("bytes")
            if data is None and x.get("path"):
                import fsspec
                with fsspec.open(x["path"], "rb") as f:
                    data = f.read()
            try:
                y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
            except Exception:
                import librosa
                y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    elif hasattr(x, "get_all_samples"):  # torchcodec AudioDecoder (datasets >= 4)
        s = x.get_all_samples()
        y, sr = s.data.mean(0).numpy().astype(np.float32), int(s.sample_rate)
    else:
        raise TypeError(f"Unrecognised audio object: {type(x)}")
    if y.ndim > 1:
        y = y.mean(axis=-1)
    return y, sr


def to_16k(y, sr):
    if sr == TARGET_SR:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)


# ---------------------------------------------------------------- loaders
def pick_split(ds_id, config):
    from datasets import get_dataset_split_names
    splits = get_dataset_split_names(ds_id, config)
    for pref in ("train", "test", "validation", "eval"):
        if pref in splits:
            return pref, splits
    return splits[0], splits


def stream_hf(entry, per_class, max_scan, seed):
    from datasets import load_dataset, Audio

    ds_id, config = entry["id"], entry.get("config")
    split = entry.get("split")
    if not split:
        split, all_splits = pick_split(ds_id, config)
        print(f"  splits available: {all_splits}; using '{split}'")
    ds = load_dataset(ds_id, config, split=split, streaming=True)
    feats = ds.features

    audio_col = entry.get("audio_col") or next((c for c in AUDIO_CANDIDATES if feats and c in feats), "audio")
    label_col = entry.get("label_col") or next((c for c in LABEL_CANDIDATES if feats and c in feats), "label")
    try:
        ds = ds.cast_column(audio_col, Audio(decode=False))
    except Exception:
        pass
    names = label_names(feats, label_col) if feats else None
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    fake_values = entry.get("fake_values")
    real_values = entry.get("real_values")

    def side(v):
        if fake_values is not None:
            return "fake" if v in fake_values else ("real" if (real_values is None or v in real_values) else None)
        name = names[v] if (names is not None and isinstance(v, (int, np.integer))) else v
        return classify_label_name(name)

    out = {"real": [], "fake": []}
    sr_seen = {"real": [], "fake": []}
    mapping, skipped, scanned = {}, 0, 0
    for row in ds:
        scanned += 1
        v = row[label_col]
        s = side(v)
        key = names[v] if (names is not None and isinstance(v, (int, np.integer))) else v
        mapping[str(v) + (f" ({key})" if key != v else "")] = s
        if s is None:
            skipped += 1
        elif len(out[s]) < per_class:
            try:
                y, sr = decode_audio(row[audio_col])
                if len(y) == 0:
                    continue
                sr_seen[s].append(sr)
                out[s].append(extract_features(to_16k(y, sr), TARGET_SR))
            except Exception as e:
                skipped += 1
                if skipped < 4:
                    print(f"  skip clip: {e}")
        if all(len(out[k]) >= per_class for k in out) or scanned >= max_scan:
            break
    return out["real"], out["fake"], mapping, sr_seen, scanned, skipped, split


def load_folders(entry, per_class, seed):
    import librosa
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=entry["id"], repo_type="dataset")
    rng = np.random.default_rng(seed)
    out, sr_seen = {}, {}
    for side, sub in (("real", entry["real_dir"]), ("fake", entry["fake_dir"])):
        folder = os.path.join(root, sub)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")))
        files = list(rng.permutation(files))[:per_class]
        out[side], sr_seen[side] = [], []
        for f in files:
            y, sr = librosa.load(os.path.join(folder, f), sr=None, mono=True)
            sr_seen[side].append(sr)
            out[side].append(extract_features(to_16k(y, sr), TARGET_SR))
    mapping = {f"{entry['real_dir']}/": "real", f"{entry['fake_dir']}/": "fake"}
    return out["real"], out["fake"], mapping, sr_seen, len(out["real"]) + len(out["fake"]), 0, "folders"


# ---------------------------------------------------------------- analysis
def per_feature_auc(real, fake, name):
    from sklearn.metrics import roc_auc_score
    x = np.array([f[name] for f in real] + [f[name] for f in fake])
    y = np.array([0] * len(real) + [1] * len(fake))
    if np.all(x == x[0]):
        return 0.5
    return float(roc_auc_score(y, x))  # >0.5 means larger in fake


def summarise(entry, real, fake, mapping, sr_seen, scanned, skipped, split, secs):
    row = {"dataset": entry["id"], "group": entry.get("group", ""), "split": split,
           "n_real": len(real), "n_fake": len(fake), "rows_scanned": scanned, "skipped": skipped,
           "label_mapping": json.dumps(mapping),
           "native_sr_real": json.dumps(sorted(set(sr_seen.get("real", [])))),
           "native_sr_fake": json.dumps(sorted(set(sr_seen.get("fake", [])))),
           "seconds": round(secs, 1)}
    for name in FEATURE_NAMES:
        r = np.array([f[name] for f in real]); k = np.array([f[name] for f in fake])
        row[f"d_{name}"] = round(cohens_d(k, r), 3)
        row[f"auc_{name}"] = round(per_feature_auc(real, fake, name), 4)
    clf = run_shortcut_classifier(real, fake)
    row["lr_auc"] = round(clf.get("auc_roc", float("nan")), 4)
    row["lr_accuracy"] = round(clf.get("accuracy", float("nan")), 4)
    row["max_abs_d"] = max(abs(row[f"d_{n}"]) for n in FEATURE_NAMES)
    row["largest_feature"] = max(FEATURE_NAMES, key=lambda n: abs(row[f"d_{n}"]))
    auc = row["lr_auc"]
    row["verdict"] = "HIGH" if auc > 0.85 else ("MODERATE" if auc > 0.65 else "LOW")
    return row


def main():
    ap = argparse.ArgumentParser(description="Run ShortcutScope on many datasets.")
    ap.add_argument("--datasets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json"))
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--max_scan", type=int, default=40000, help="stop after scanning this many rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="comma-separated dataset ids to run (default: all)")
    ap.add_argument("--out_dir", default="audit_results")
    args = ap.parse_args()

    import pandas as pd
    entries = json.load(open(args.datasets))
    if args.only:
        keep = set(args.only.split(","))
        entries = [e for e in entries if e["id"] in keep]
    os.makedirs(os.path.join(args.out_dir, "per_clip"), exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    rows = pd.read_csv(summary_path).to_dict("records") if os.path.exists(summary_path) else []
    done = {r["dataset"] for r in rows if r.get("n_real", 0) > 0}

    for entry in entries:
        if entry["id"] in done:
            print(f"\n=== {entry['id']}: already in summary.csv, skipping")
            continue
        print(f"\n=== {entry['id']}  ({entry.get('note', '')})")
        t0 = time.time()
        try:
            if entry.get("mode") == "folders":
                res = load_folders(entry, args.per_class, args.seed)
            else:
                res = stream_hf(entry, args.per_class, args.max_scan, args.seed)
            real, fake, mapping, sr_seen, scanned, skipped, split = res
            print(f"  LABEL MAPPING (check this!): {mapping}")
            if len(real) < 20 or len(fake) < 20:
                raise RuntimeError(f"too few clips after mapping: real={len(real)}, fake={len(fake)}")
            row = summarise(entry, real, fake, mapping, sr_seen, scanned, skipped, split, time.time() - t0)
            safe = entry["id"].replace("/", "__")
            pd.DataFrame([dict(f, label="real") for f in real] + [dict(f, label="fake") for f in fake]) \
              .to_csv(os.path.join(args.out_dir, "per_clip", f"{safe}.csv"), index=False)
            print(f"  n={len(real)}+{len(fake)} | LR AUC {row['lr_auc']} ({row['verdict']}) | "
                  f"largest: {row['largest_feature']} d={row['d_' + row['largest_feature']]} | "
                  f"sr real {row['native_sr_real']} fake {row['native_sr_fake']} | {row['seconds']} s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)
            row = {"dataset": entry["id"], "group": entry.get("group", ""), "error": f"{type(e).__name__}: {e}",
                   "n_real": 0, "n_fake": 0}
        rows = [r for r in rows if r["dataset"] != entry["id"]] + [row]
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    df = pd.DataFrame(rows)
    cols = ["dataset", "group", "n_real", "n_fake", "lr_auc", "verdict", "largest_feature", "max_abs_d",
            "d_leading_silence_ms", "d_zcr", "native_sr_real", "native_sr_fake"]
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    if "error" in df.columns:
        failed = df[df["error"].notna()] if df["error"].notna().any() else None
        if failed is not None:
            print("\nFailed datasets:\n" + failed[["dataset", "error"]].to_string(index=False))
    print(f"\nSaved: {summary_path} and per-clip features in {os.path.join(args.out_dir, 'per_clip')}")


if __name__ == "__main__":
    main()
