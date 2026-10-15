"""
common.py - shared audio handling for the v2 experiments.

Every number in the revised paper is produced through the functions in this
file, so training, in-domain testing and cross-dataset testing always use the
same preprocessing (reviewer points #2, #5, #12).

Label convention everywhere in v2:  1 = FAKE / spoof,  0 = REAL / bona fide.
Scores are always P(fake).
"""

import io
import zlib

import numpy as np

SR = 16000
FIVE_S = 5 * SR                 # length used by the original training pipeline
CROP_S = 2.0                    # fixed-length crop for the 'crop' / 'all' conditions
WHISPER_SAMPLES = 30 * SR       # Whisper always sees a 30 s window
SAMPLES_PER_FRAME = 320         # 30 s -> 1500 encoder frames
TARGET_RMS_DBFS = -23.0         # loudness normalisation target
CONDITIONS = ["raw", "sil", "loud", "crop", "all"]


# --------------------------------------------------------------------- utils
def clip_rng(clip_id, salt=0):
    """Deterministic per-clip random generator (same draw on every run)."""
    seed = zlib.crc32(f"{clip_id}|{salt}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def decode_bytes(data):
    """bytes -> (mono float32 array, native sr, info dict)."""
    import soundfile as sf
    info = {}
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            info = {"format": f.format, "subtype": f.subtype, "channels": f.channels}
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        import librosa
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        info = {"format": "via_librosa", "subtype": "unknown", "channels": -1}
    return np.asarray(y, dtype=np.float32), int(sr), info


def to_16k(y, sr):
    if sr == SR:
        return y.astype(np.float32)
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR).astype(np.float32)


def fit_length(y, n):
    """Truncate or zero-pad at the end to exactly n samples."""
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])


def trim(y, top_db=30):
    import librosa
    t, _ = librosa.effects.trim(y, top_db=top_db)
    return t if len(t) > 0 else y


def rms_normalise(y, target_dbfs=TARGET_RMS_DBFS):
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    g = 10 ** (target_dbfs / 20) / rms
    out = y * g
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.999:                       # avoid clipping; rare for -23 dBFS
        out = out * (0.999 / peak)
    return out.astype(np.float32)


def random_crop(y, seconds, rng):
    """Fixed-length crop from inside the signal; loops short clips instead of zero-padding."""
    n = int(seconds * SR)
    if len(y) < n:
        reps = int(np.ceil(n / max(len(y), 1)))
        y = np.tile(y, reps)
    start = int(rng.integers(0, len(y) - n + 1))
    return y[start:start + n]


# ------------------------------------------------------------ preprocessing
def preprocess(y16, condition, clip_id):
    """
    Apply one experimental condition to a 16 kHz mono clip.

    Returns (waveform, n_valid): the waveform that is then zero-padded to 30 s
    for Whisper, and the number of samples that carry audio rather than
    padding (used for valid-frame pooling, reviewer point #4).

      raw   original audio, cut/zero-padded to 5 s           (no cue removed)
      sil   trim silence + random 0-150 ms lead, cut/pad 5 s  (= original training pipeline)
      loud  RMS-normalise to -23 dBFS, cut/pad 5 s            (removes loudness)
      crop  trim silence, fixed 2 s crop, no padding          (removes duration + silence)
      all   trim + RMS-normalise + fixed 2 s crop             (removes all four)
    """
    rng = clip_rng(clip_id, salt=condition)
    if condition == "raw":
        y = y16
        n_valid = min(len(y), FIVE_S)
        return fit_length(y, FIVE_S), n_valid
    if condition == "sil":
        t = trim(y16)
        lead = int(rng.integers(0, int(SR * 0.150) + 1))
        y = np.concatenate([np.zeros(lead, dtype=np.float32), t])
        n_valid = min(len(y), FIVE_S)
        return fit_length(y, FIVE_S), n_valid
    if condition == "loud":
        y = rms_normalise(y16)
        n_valid = min(len(y), FIVE_S)
        return fit_length(y, FIVE_S), n_valid
    if condition == "crop":
        y = random_crop(trim(y16), CROP_S, rng)
        return y, len(y)
    if condition == "all":
        y = random_crop(rms_normalise(trim(y16)), CROP_S, rng)
        return y, len(y)
    if condition == "native":          # old cross-dataset script: whole clip, no 5 s cut
        y = y16[:WHISPER_SAMPLES]
        return y, len(y)
    raise ValueError(condition)


# ------------------------------------------------------------ hand features
CORE7 = ["leading_silence_ms", "trailing_silence_ms", "duration_sec", "rms", "zcr",
         "peak_amplitude", "silence_ratio"]
REL3 = ["leading_silence_rel_ms", "trailing_silence_rel_ms", "silence_ratio_rel"]


def _silence_feats(y, thr, sr):
    """Same definitions as ShortcutScope v1 (shortcutscope.extract_features)."""
    nz = np.where(np.abs(y) > thr)[0]
    if len(nz) == 0:
        return 0.0, 0.0, 1.0
    lead = nz[0] / sr * 1000
    trail = (len(y) - nz[-1]) / sr * 1000
    return float(lead), float(trail), float(1.0 - len(nz) / len(y))


def core_features(y, sr=SR, abs_thr=0.01, rel_db=-40.0):
    """The 7 ShortcutScope features (absolute 0.01 threshold, as in v1) plus the same
    silence features with a threshold relative to the clip's own peak (reviewer #8)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        y = np.zeros(1, dtype=np.float32)
    lead, trail, sratio = _silence_feats(y, abs_thr, sr)
    peak = float(np.max(np.abs(y)))
    rel_thr = peak * 10 ** (rel_db / 20)
    lead_r, trail_r, sratio_r = _silence_feats(y, rel_thr, sr)
    zcr = float(len(np.where(np.diff(np.sign(y)))[0]) / len(y))   # v1 definition
    return {
        "leading_silence_ms": lead, "trailing_silence_ms": trail,
        "duration_sec": len(y) / sr, "rms": float(np.sqrt(np.mean(y ** 2))),
        "zcr": zcr, "peak_amplitude": peak, "silence_ratio": sratio,
        "leading_silence_rel_ms": lead_r, "trailing_silence_rel_ms": trail_r,
        "silence_ratio_rel": sratio_r,
    }


NATIVE = ["native_sr", "rolloff99_hz", "bandwidth60_hz", "centroid_hz", "noise_floor_db",
          "dynamic_range_db", "dc_offset", "clip_rate"]


def native_features(y, sr):
    """Features computed at the NATIVE sampling rate, before any resampling (reviewer #9):
    bandwidth / roll-off (codec and sampling-rate traces), noise floor, DC offset, clipping."""
    y = np.asarray(y, dtype=np.float64)
    n_fft = 2048 if sr > 24000 else 1024
    hop = n_fft // 2
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    win = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    psd = spec.mean(axis=0) + 1e-20
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    cum = np.cumsum(psd) / np.sum(psd)
    rolloff = float(freqs[min(np.searchsorted(cum, 0.99), len(freqs) - 1)])
    db = 10 * np.log10(psd)
    above = np.where(db > db.max() - 60)[0]
    bw = float(freqs[above[-1]]) if len(above) else 0.0
    centroid = float(np.sum(freqs * psd) / np.sum(psd))
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10
    fdb = 20 * np.log10(frame_rms)
    noise = float(np.percentile(fdb, 10))
    dyn = float(np.percentile(fdb, 95) - noise)
    return {
        "native_sr": float(sr), "rolloff99_hz": rolloff, "bandwidth60_hz": bw,
        "centroid_hz": centroid, "noise_floor_db": noise, "dynamic_range_db": dyn,
        "dc_offset": float(np.mean(y)), "clip_rate": float(np.mean(np.abs(y) >= 0.999)),
    }


def mfcc_stats(y, sr=SR, n=13):
    import librosa
    m = librosa.feature.mfcc(y=np.asarray(y, dtype=np.float32), sr=sr, n_mfcc=n)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)
