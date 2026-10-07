# ShortcutScope

A lightweight auditing tool that checks whether an audio deepfake detection
dataset can be "solved" using simple acoustic artifacts (silence timing,
loudness, duration) instead of genuine speech content — a known failure mode
called **shortcut learning**.

If your dataset's real and fake audio classes are separable using only these
7 low-level features, any model trained on it risks learning the artifact
instead of actual deepfake detection — producing inflated accuracy numbers
that collapse on new data.

## Background

This tool implements the audit protocol from our paper *"Shortcut Learning
and Pretraining Lineage in Audio Deepfake Detection: A Multi-Dataset Causal
Audit"* (2026), which extends the feature-ablation method introduced by
Müller et al. in *"Speech is Silver, Silence is Golden"* (2021,
[arXiv:2106.12914](https://arxiv.org/abs/2106.12914)) to a broader feature
set and applies it to community-built datasets.

## Install

```bash
pip install librosa numpy scikit-learn datasets soundfile
```

## Usage

**Audit a HuggingFace dataset** (expects `audio` and `label` columns, where
`label: 0 = real/bonafide, 1 = fake/spoof`):

```bash
python shortcutscope.py --hf_dataset garystafford/deepfake-audio-detection
```

**Audit a dataset laid out as two folders**:

```bash
python shortcutscope.py --real_dir ./data/real --fake_dir ./data/fake
```

**Save the full report as JSON**:

```bash
python shortcutscope.py --hf_dataset <name> --output report.json
```

## What it checks

| Feature | What it measures |
|---|---|
| `leading_silence_ms` | Silence duration before speech starts |
| `trailing_silence_ms` | Silence duration after speech ends |
| `duration_sec` | Total clip length |
| `rms` | Overall loudness |
| `zcr` | Zero-crossing rate (texture/noisiness) |
| `peak_amplitude` | Maximum amplitude in the clip |
| `silence_ratio` | Fraction of the clip below the silence threshold |

For each feature, the tool reports **Cohen's d** (effect size between real
and fake classes) and trains a simple logistic regression using *only*
these 7 numbers — no audio content, no spectral features, no model of
speech at all — to see how separable the classes are from artifacts alone.

## Interpreting the verdict

- **AUC > 0.85** — high risk. The dataset is very likely contaminated by a
  shortcut. Investigate before training a model you intend to trust.
- **AUC 0.65–0.85** — moderate risk. Some signal present; worth a closer look.
- **AUC < 0.65** — low risk from *these* specific features. Does not
  guarantee the dataset is free of other, unmeasured shortcuts.

## Citation

If you use this tool, please cite:

```bibtex
@article{Ammara2026shortcutscope,
  title={Shortcut Learning and Pretraining Lineage in Audio Deepfake Detection: A Multi-Dataset Causal Audit},
  author={Tahir, Ammara},
  year={2026}
}
```

## License

MIT
