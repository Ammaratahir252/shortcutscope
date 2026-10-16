#!/usr/bin/env python3
"""
audit.py - multi-dataset ShortcutScope audit used in the paper (CPU only).

Compared with shortcutscope_batch.py it adds:
  - sampling by index where the dataset fits on disk (uniform random sample),
    otherwise streaming with a 10,000-clip buffer and shuffled shard order;
    the method, split and Hugging Face revision hash are recorded;
  - silence features with a threshold relative to the clip's peak, next to
    the absolute 0.01 threshold;
  - features at the NATIVE sampling rate: roll-off, bandwidth, centroid, noise
    floor, dynamic range, DC offset, clipping rate, plus container metadata
    (format, subtype/bit depth, channels);
  - the option to audit every clip of a dataset ("per_class": "all"), used for
    garystafford so the audit and the case study use the same clips.
This script only extracts per-clip features. All statistics (repeated CV,
bootstrap CIs, permutation tests, gradient-boosted trees, the cross-dataset
transfer matrix) are computed afterwards from the CSVs by analyze_audit.py.

Usage (Kaggle CPU notebook, internet on):
  python audit.py --datasets audit_datasets.json --out_dir /kaggle/working/audit_results
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

LABEL_CANDIDATES = ["label", "labels", "real_or_fake", "is_fake", "class", "target"]
AUDIO_CANDIDATES = ["audio", "wav", "speech", "file"]


def classify_label_name(name):
    s = str(name).strip().lower()
    if re.search(r"bona|genuine|real|human|original|authentic", s) or s in {"r", "true_speech"}:
        if not re.search(r"fake|spoof", s):
            return "real"
    if re.search(r"fake|spoof|synth|generated|deepfake|clone|tts|convert|vocod", s) or re.fullmatch(r"f\d+", s):
        return "fake"
    return None


def clip_row(data, idx, side):
    y, sr, info = C.decode_bytes(data)
    if len(y) == 0:
        raise ValueError("empty clip")
    row = {"idx": idx, "label": side, "format": info.get("format"), "subtype": info.get("subtype"),
           "channels": info.get("channels")}
    row.update(C.native_features(y, sr))
    row.update(C.core_features(C.to_16k(y, sr)))
    return row


def audio_bytes(a):
    if isinstance(a, dict):
        if a.get("bytes") is not None:
            return a["bytes"]
        if a.get("path"):
            import fsspec
            with fsspec.open(a["path"], "rb") as f:
                return f.read()
    raise TypeError(f"cannot read audio field of type {type(a)}")


def revision(ds_id):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(ds_id).sha
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def pick_split(ds_id, config):
    from datasets import get_dataset_split_names
    splits = get_dataset_split_names(ds_id, config)
    for pref in ("train", "test", "validation", "eval"):
        if pref in splits:
            return pref, splits
    return splits[0], splits


def size_gb(ds_id, config):
    try:
        from datasets import load_dataset_builder
        info = load_dataset_builder(ds_id, config).info
        b = info.download_size or info.dataset_size
        return b / 1e9 if b else None
    except Exception:
        return None


def side_fn(entry, names):
    fv, rv = entry.get("fake_values"), entry.get("real_values")

    def side(v):
        if fv is not None:
            return "fake" if v in fv else ("real" if (rv is None or v in rv) else None)
        name = names[v] if (names is not None and isinstance(v, (int, np.integer))) else v
        return classify_label_name(name)
    return side


def audit_hf(entry, per_class, seed, max_gb, max_scan):
    from datasets import Audio, load_dataset
    ds_id, config = entry["id"], entry.get("config")
    split = entry.get("split") or pick_split(ds_id, config)[0]
    gb = size_gb(ds_id, config)
    meta = {"split": split, "size_gb": gb}
    by_index = gb is not None and gb <= max_gb
    ds = load_dataset(ds_id, config, split=split, streaming=not by_index)
    feats = ds.features
    audio_col = entry.get("audio_col") or next((c for c in AUDIO_CANDIDATES if feats and c in feats), "audio")
    label_col = entry.get("label_col") or next((c for c in LABEL_CANDIDATES if feats and c in feats), "label")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    try:
        names = list(feats[label_col].names)
    except Exception:
        names = None
    side = side_fn(entry, names)
    rows, mapping, skipped = [], {}, 0
    want = {"real": per_class, "fake": per_class}

    if by_index:
        labels = ds[label_col]
        sides = [side(v) for v in labels]
        for v, s in zip(labels, sides):
            key = names[v] if (names is not None and isinstance(v, (int, np.integer))) else v
            mapping[str(v) + (f" ({key})" if key != v else "")] = s
        rng = np.random.default_rng(seed)
        chosen = []
        for s in ("real", "fake"):
            pool = np.array([i for i, x in enumerate(sides) if x == s])
            n = len(pool) if per_class is None else min(per_class, len(pool))
            chosen += sorted(rng.choice(pool, n, replace=False).tolist()) if n else []
        for i in chosen:
            try:
                rows.append(clip_row(audio_bytes(ds[int(i)][audio_col]), int(i), sides[i]))
            except Exception as e:
                skipped += 1
                if skipped < 4:
                    print(f"  skip {i}: {e}")
        meta.update({"sampling": "index", "rows_total": len(labels)})
    else:
        ds = ds.shuffle(seed=seed, buffer_size=10000)
        have = {"real": 0, "fake": 0}; scanned = 0
        for row in ds:
            v = row[label_col]; s = side(v); scanned += 1
            key = names[v] if (names is not None and isinstance(v, (int, np.integer))) else v
            mapping[str(v) + (f" ({key})" if key != v else "")] = s
            if s is not None and (want[s] is None or have[s] < want[s]):
                try:
                    rows.append(clip_row(audio_bytes(row[audio_col]), scanned - 1, s)); have[s] += 1
                except Exception as e:
                    skipped += 1
                    if skipped < 4:
                        print(f"  skip: {e}")
            if all(want[k] is not None and have[k] >= want[k] for k in have) or scanned >= max_scan:
                break
        meta.update({"sampling": "stream_buffer10000_shuffled_shards", "rows_scanned": scanned,
                     "n_shards": getattr(ds, "n_shards", None)})
    meta.update({"label_mapping": mapping, "skipped": skipped, "label_col": label_col})
    return rows, meta


def audit_folders(entry, per_class, seed):
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=entry["id"], repo_type="dataset")
    rng = np.random.default_rng(seed); rows = []
    for side, sub in (("real", entry["real_dir"]), ("fake", entry["fake_dir"])):
        folder = os.path.join(root, sub)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")))
        files = list(rng.permutation(files))[: (per_class or len(files))]
        for f in files:
            with open(os.path.join(folder, f), "rb") as fh:
                rows.append(clip_row(fh.read(), f, side))
    return rows, {"split": "folders", "sampling": "index (all files)" if per_class is None else "index",
                  "label_mapping": {entry["real_dir"] + "/": "real", entry["fake_dir"] + "/": "fake"}}


def quick_auc(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    X = df[C.CORE7].values; y = (df["label"] == "fake").astype(int).values
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_datasets.json"))
    ap.add_argument("--out_dir", default="/kaggle/working/audit_results")
    ap.add_argument("--per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_gb", type=float, default=4.0, help="download fully (sample by index) up to this size")
    ap.add_argument("--max_scan", type=int, default=60000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--keep_cache", action="store_true")
    args = ap.parse_args()
    import pandas as pd

    entries = json.load(open(args.datasets))
    if args.only:
        keep = set(args.only.split(",")); entries = [e for e in entries if e["id"] in keep]
    for d in ("per_clip", "meta"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    for entry in entries:
        safe = entry["id"].replace("/", "__")
        out_csv = os.path.join(args.out_dir, "per_clip", f"{safe}.csv")
        if os.path.exists(out_csv):
            print(f"\n=== {entry['id']}: done already, skipping"); continue
        print(f"\n=== {entry['id']}  ({entry.get('note', '')})")
        t0 = time.time()
        pc = None if entry.get("per_class") == "all" else args.per_class
        try:
            if entry.get("mode") == "folders":
                rows, meta = audit_folders(entry, pc, args.seed)
            else:
                rows, meta = audit_hf(entry, pc, args.seed, args.max_gb, args.max_scan)
            df = pd.DataFrame(rows)
            n_r, n_f = int((df.label == "real").sum()), int((df.label == "fake").sum())
            print(f"  LABEL MAPPING (check!): {meta.get('label_mapping')}")
            if n_r < 20 or n_f < 20:
                raise RuntimeError(f"too few clips: real={n_r} fake={n_f}")
            df.to_csv(out_csv, index=False)
            meta.update({"dataset": entry["id"], "group": entry.get("group"), "revision": revision(entry["id"]),
                         "n_real": n_r, "n_fake": n_f, "seconds": round(time.time() - t0, 1),
                         "native_sr_real": sorted(df[df.label == "real"].native_sr.unique().tolist()),
                         "native_sr_fake": sorted(df[df.label == "fake"].native_sr.unique().tolist())})
            json.dump(meta, open(os.path.join(args.out_dir, "meta", f"{safe}.json"), "w"), indent=1, default=str)
            print(f"  n={n_r}+{n_f} | {meta['sampling']} | quick 5-fold LR AUC (core7, standardised) "
                  f"{quick_auc(df):.4f} | sr real {meta['native_sr_real']} fake {meta['native_sr_fake']} | "
                  f"{meta['seconds']} s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}"); traceback.print_exc(limit=2)
            json.dump({"dataset": entry["id"], "error": f"{type(e).__name__}: {e}"},
                      open(os.path.join(args.out_dir, "meta", f"{safe}.json"), "w"), indent=1)
        if not args.keep_cache:            # keep Kaggle's disk from filling up
            import glob
            hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
            for d in [os.environ.get("HF_DATASETS_CACHE"), os.path.expanduser("~/.cache/huggingface/datasets")] + \
                     glob.glob(os.path.join(hub, "datasets--*")):
                if d and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
    zip_base = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), "audit_results")
    shutil.make_archive(zip_base, "zip", root_dir=os.path.dirname(os.path.abspath(args.out_dir)),
                        base_dir=os.path.basename(os.path.abspath(args.out_dir)))
    print("\nDONE. Wrote", zip_base + ".zip", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    # Streaming downloads can leave non-daemon threads that block interpreter exit; leave immediately.
    os._exit(0)


if __name__ == "__main__":
    main()
