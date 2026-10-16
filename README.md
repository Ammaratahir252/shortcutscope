# ShortcutScope

A lightweight tool that checks whether an audio deepfake detection dataset can
be "solved" from simple acoustic properties (silence timing, loudness,
zero-crossing rate, duration) instead of genuine speech content. Relying on
such cues is a known failure mode called **shortcut learning**.

If the real and fake classes of a dataset are separable using only these seven
low-level features, a model trained on it may learn the artifact instead of
real deepfake cues, producing inflated in-domain scores that do not hold up on
other data.

## Background

ShortcutScope implements the acoustic audit described in:

> A. Tahir, *Acoustic Shortcuts in Audio Deepfake Datasets: An Audit of Ten
> Corpora and a Cross-Dataset Stress Test*, 2026.

It extends the silence analysis of Müller et al., *"Speech is Silver, Silence
is Golden: What do ASVspoof-trained Models Really Learn?"* (ASVspoof 2021
Workshop, [arXiv:2106.12914](https://arxiv.org/abs/2106.12914)), to a broader
set of features and applies it to community-built datasets.

Across the ten datasets audited in the paper (see `datasets.json`), these seven
features alone separate real from fake clips with an AUC above 0.85 in five
and between 0.65 and 0.85 in three more. Always resample to a common rate
before comparing zero-crossing rates: in `garystafford/deepfake-audio-detection`
real clips are stored at 44.1 kHz and fake clips at 16 kHz, which inflates the
ZCR difference if it is measured at native rates. The batch mode does this
automatically.

## Install

```bash
pip install -r requirements.txt
```

## Usage

**Audit a Hugging Face dataset.** The dataset needs an audio column and a
binary label column. Label conventions differ between datasets, so **check the
dataset card and pass the label value that means FAKE** with `--fake_label`
(default: 1). The tool prints the label names it finds, so you can confirm the
mapping.

```bash
# garystafford/deepfake-audio-detection: 0 = real, 1 = fake
python shortcutscope.py --hf_dataset garystafford/deepfake-audio-detection --fake_label 1

# DeepFake-Audio-Rangers/Arabic_Audio_Deepfake: 0 = fake, 1 = real
python shortcutscope.py --hf_dataset DeepFake-Audio-Rangers/Arabic_Audio_Deepfake --fake_label 0
```

Other options: `--label_field`, `--audio_field` and `--split` (default: `train`).

**Audit a dataset stored as two folders:**

```bash
python shortcutscope.py --real_dir ./data/real --fake_dir ./data/fake
```

**Save the full report as JSON:**

```bash
python shortcutscope.py --hf_dataset <name> --fake_label 1 --output report.json
```

## Auditing many datasets at once

`shortcutscope_batch.py` runs the same audit on every dataset listed in
`datasets.json`. It streams a class-balanced random sample (default 500 real
and 500 fake clips per dataset), resamples to 16 kHz so that zero-crossing
rates are comparable across datasets, and writes:

- `summary.csv`: one row per dataset with Cohen's d and per-feature AUC for all
  seven features, the features-only logistic-regression AUC, the verdict, the
  label mapping used, and the native sample rates of each class;
- `per_clip/<dataset>.csv`: the seven features for every sampled clip.

```bash
pip install -r requirements.txt
python shortcutscope_batch.py --datasets datasets.json --per_class 500 --out_dir audit_results
```

For every dataset the script prints the **label mapping** it used (for example
`{'0 (real)': 'real', '1 (fake)': 'fake'}`). Check it. If the label names are
ambiguous, the dataset is skipped with an error instead of being guessed; add
`"fake_values": [...]` for that dataset in `datasets.json`. Results are saved
after each dataset, and datasets already in `summary.csv` are skipped on a
re-run.

## Reproducing the paper (`experiments/`)

The `experiments/` folder contains the code behind every number in the paper.

| File | What it does | Hardware |
|---|---|---|
| `audit.py` | Per-clip features for all datasets in `audit_datasets.json`: the 7 features, relative-threshold silence, and 8 features at the **native** sampling rate (bandwidth, roll-off, noise floor, DC offset, clipping), plus format, bit depth and the Hugging Face revision hash. Samples by index where the dataset fits on disk. | CPU |
| `analyze_audit.py` | Repeated stratified 5-fold CV with standardised features, 95% bootstrap CIs, 200-permutation test, gradient-boosted trees, and the cross-dataset transfer matrix. | CPU |
| `encode.py` | Runs the frozen Whisper-small encoder once per preprocessing condition and stores the pooled vectors (all frames and audio frames only). | GPU |
| `heads.py` | Trains and tests all detector heads: 5 preprocessing conditions x 2 poolings x random/speaker-grouped splits x 5 seeds, matched ASVspoof 2019 LA testing, counterfactual injections, MFCC/feature baselines and the reverse direction. | CPU |
| `make_paper_assets.py` | Builds the paper's tables, figures and in-text numbers directly from the outputs. | CPU |
| `common.py` | Shared audio loading, preprocessing conditions and feature definitions. | |
| `notebooks/` | Ready-to-run Kaggle notebooks: `kaggle_audit_cpu.ipynb` and `kaggle_detector_gpu.ipynb`. | |

**Steps**

1. Run `experiments/notebooks/kaggle_audit_cpu.ipynb` on Kaggle (CPU) and download `audit_results.zip`.
2. Run `experiments/notebooks/kaggle_detector_gpu.ipynb` on Kaggle (GPU T4, with the
   `awsaf49/asvpoof-2019-dataset` input) and download `detector_results.zip`.
3. Unzip both and run:

```bash
python experiments/analyze_audit.py --audit_dir audit_results --out_dir audit_analysis
python experiments/make_paper_assets.py --results detector_out/results \
       --audit_analysis audit_analysis --out paper_assets
```

Label convention in `experiments/`: 1 = fake, 0 = real; every score is P(fake).

## What it checks

| Feature | What it measures |
|---|---|
| `leading_silence_ms` | Silence before speech starts |
| `trailing_silence_ms` | Silence after speech ends |
| `duration_sec` | Total clip length |
| `rms` | Overall loudness |
| `zcr` | Zero-crossing rate (texture / noisiness) |
| `peak_amplitude` | Maximum amplitude in the clip |
| `silence_ratio` | Fraction of samples below the silence threshold |

Silence is any sample with absolute amplitude at or below 0.01.

For each feature the tool reports **Cohen's d, computed as fake minus real**:
a positive value means the feature is larger in fake clips. As a rough guide,
|d| ≥ 0.8 is a large effect. It then trains a plain logistic regression on
*only* these seven numbers (stratified 80/20 split, seed 42), with no audio
content, spectral features or speech model, to show how separable the classes
are from artifacts alone. For publication-grade numbers use `experiments/`, which
standardises features, cross-validates and reports confidence intervals.

## Interpreting the verdict

- **AUC > 0.85**: high risk. The classes are strongly separable from
  acoustic artifacts alone. Investigate before trusting a model trained on it.
- **AUC 0.65–0.85**: moderate risk. Some shortcut signal is present.
- **AUC < 0.65**: low risk from *these* features. This does not rule out
  other, unmeasured shortcuts.

A low score is not proof that a dataset is clean, and a high score does not
by itself prove that a particular model uses the shortcut. Pair this audit with
at least one cross-dataset test.

## Citation

If you use this tool, please cite:

```bibtex
@misc{tahir2026acoustic,
  title  = {Acoustic Shortcuts in Audio Deepfake Datasets: An Audit of Ten Corpora and a Cross-Dataset Stress Test},
  author = {Tahir, Ammara},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT (see `LICENSE`).
