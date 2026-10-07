#!/usr/bin/env python3
"""
ShortcutScope — an acoustic shortcut auditor for audio deepfake detection datasets.

Checks whether "real" and "fake" audio classes in a dataset are separable using
only low-level acoustic properties (silence timing, loudness, zero-crossing rate,
duration) rather than genuine speech content. If a simple classifier trained on
these 7 features alone achieves high accuracy, the dataset likely contains a
"shortcut" that lets models cheat instead of learning real deepfake detection.

Background / methodology:
    This tool implements the audit protocol developed in our paper "Shortcut
    Learning and Pretraining Lineage in Audio Deepfake Detection: A Multi-Dataset
    Causal Audit" (2026), extending the feature-ablation method introduced by
    Müller et al., "Speech is Silver, Silence is Golden" (2021), to the leading
    silence, trailing silence, duration, loudness (RMS), zero-crossing rate (ZCR),
    and peak amplitude features.

Usage:
    # Audit a HuggingFace dataset with 'audio' and 'label' columns (0=real, 1=fake)
    python shortcutscope.py --hf_dataset garystafford/deepfake-audio-detection

    # Audit a dataset laid out as two folders of audio files
    python shortcutscope.py --real_dir path/to/real --fake_dir path/to/fake

    # Save a full report to CSV
    python shortcutscope.py --hf_dataset <name> --output report.csv

Requirements:
    pip install librosa numpy scikit-learn datasets soundfile
"""

import argparse
import os
import sys
import json
import numpy as np

FEATURE_NAMES = [
    "leading_silence_ms",
    "trailing_silence_ms",
    "duration_sec",
    "rms",
    "zcr",
    "peak_amplitude",
    "silence_ratio",
]


def extract_features(y: np.ndarray, sr: int, silence_threshold: float = 0.01) -> dict:
    """Extract the 7 shortcut-candidate acoustic features from a single audio array."""
    y = np.asarray(y, dtype=np.float32)
    non_silent = np.where(np.abs(y) > silence_threshold)[0]

    if len(non_silent) > 0:
        leading_silence_ms = non_silent[0] / sr * 1000
        trailing_silence_ms = (len(y) - non_silent[-1]) / sr * 1000
        silence_ratio = 1.0 - (len(non_silent) / len(y))
    else:
        leading_silence_ms = trailing_silence_ms = 0.0
        silence_ratio = 1.0

    duration_sec = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak_amplitude = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    zcr = len(zero_crossings) / len(y) if len(y) > 0 else 0.0

    return {
        "leading_silence_ms": leading_silence_ms,
        "trailing_silence_ms": trailing_silence_ms,
        "duration_sec": duration_sec,
        "rms": rms,
        "zcr": zcr,
        "peak_amplitude": peak_amplitude,
        "silence_ratio": silence_ratio,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size between two groups. |d| > 1.2 = huge, > 0.8 = large, > 0.5 = medium."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def interpret_d(d: float) -> str:
    ad = abs(d)
    if ad > 1.2:
        return "HUGE"
    if ad > 0.8:
        return "large"
    if ad > 0.5:
        return "medium"
    if ad > 0.2:
        return "small"
    return "negligible"


def load_from_hf(dataset_name: str, label_field: str = "label", audio_field: str = "audio"):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    real_feats, fake_feats = [], []
    for i, sample in enumerate(ds):
        audio = sample[audio_field]
        feats = extract_features(audio["array"], audio["sampling_rate"])
        target = fake_feats if sample[label_field] == 1 else real_feats
        target.append(feats)
        if i % 300 == 0:
            print(f"  processed {i}/{len(ds)}", file=sys.stderr)
    return real_feats, fake_feats


def load_from_folders(real_dir: str, fake_dir: str):
    import librosa

    def load_folder(folder):
        feats = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                y, sr = librosa.load(os.path.join(folder, fname), sr=None)
                feats.append(extract_features(y, sr))
        return feats

    print(f"Loading real audio from {real_dir} ...", file=sys.stderr)
    real_feats = load_folder(real_dir)
    print(f"Loading fake audio from {fake_dir} ...", file=sys.stderr)
    fake_feats = load_folder(fake_dir)
    return real_feats, fake_feats


def run_shortcut_classifier(real_feats: list, fake_feats: list) -> dict:
    """Train a classifier on ONLY the 7 shortcut features (no audio content at all).
    High accuracy here means the dataset is separable without real speech understanding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    X = np.array([[f[name] for name in FEATURE_NAMES] for f in real_feats + fake_feats])
    y = np.array([0] * len(real_feats) + [1] * len(fake_feats))

    if len(set(y)) < 2 or len(y) < 10:
        return {"error": "Not enough samples in both classes to train a classifier."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    coefs = dict(zip(FEATURE_NAMES, clf.coef_[0].tolist()))
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "feature_coefficients": dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1]))),
    }


def build_report(real_feats: list, fake_feats: list) -> dict:
    real_arr = {name: np.array([f[name] for f in real_feats]) for name in FEATURE_NAMES}
    fake_arr = {name: np.array([f[name] for f in fake_feats]) for name in FEATURE_NAMES}

    effect_sizes = {}
    for name in FEATURE_NAMES:
        d = cohens_d(fake_arr[name], real_arr[name])
        effect_sizes[name] = {"cohens_d": round(d, 3), "interpretation": interpret_d(d)}

    classifier_result = run_shortcut_classifier(real_feats, fake_feats)

    return {
        "n_real": len(real_feats),
        "n_fake": len(fake_feats),
        "effect_sizes": effect_sizes,
        "shortcut_only_classifier": classifier_result,
    }


def print_report(report: dict):
    print("\n" + "=" * 60)
    print("SHORTCUTSCOPE AUDIT REPORT")
    print("=" * 60)
    print(f"Real samples: {report['n_real']} | Fake samples: {report['n_fake']}\n")

    print(f"{'Feature':<22}{'Cohens d':<12}{'Interpretation'}")
    print("-" * 50)
    for name, info in report["effect_sizes"].items():
        print(f"{name:<22}{info['cohens_d']:<12}{info['interpretation']}")

    clf = report["shortcut_only_classifier"]
    print("\n--- Shortcut-only classifier (NO audio content used) ---")
    if "error" in clf:
        print(clf["error"])
    else:
        print(f"Accuracy : {clf['accuracy']:.4f}")
        print(f"F1 Score : {clf['f1']:.4f}")
        print(f"AUC-ROC  : {clf['auc_roc']:.4f}")
        print("\nTop features driving the shortcut:")
        for name, coef in list(clf["feature_coefficients"].items())[:3]:
            print(f"  {name:<22}{coef:+.4f}")

    verdict_auc = clf.get("auc_roc", 0)
    print("\n--- Verdict ---")
    if verdict_auc > 0.85:
        print("HIGH RISK: dataset is highly separable using acoustic artifacts alone.")
        print("A model trained on this dataset may learn shortcuts instead of real")
        print("deepfake-detection cues. Recommend pipeline-matching audit before use.")
    elif verdict_auc > 0.65:
        print("MODERATE RISK: some shortcut signal present. Investigate further")
        print("before trusting in-domain accuracy numbers at face value.")
    else:
        print("LOW RISK: acoustic artifacts alone do not strongly separate classes.")
        print("(This does not guarantee the dataset is free of other shortcuts.)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit an audio dataset for acoustic shortcuts.")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name (needs 'audio' + 'label' columns, label: 0=real, 1=fake)")
    parser.add_argument("--real_dir", type=str, help="Folder of real/bonafide audio files")
    parser.add_argument("--fake_dir", type=str, help="Folder of fake/spoof audio files")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the report as JSON")
    args = parser.parse_args()

    if args.hf_dataset:
        real_feats, fake_feats = load_from_hf(args.hf_dataset)
    elif args.real_dir and args.fake_dir:
        real_feats, fake_feats = load_from_folders(args.real_dir, args.fake_dir)
    else:
        parser.error("Provide either --hf_dataset, or both --real_dir and --fake_dir.")
        return

    report = build_report(real_feats, fake_feats)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()
