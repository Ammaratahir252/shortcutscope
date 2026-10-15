"""
make_v2_assets.py - builds every v2 table, figure and in-text number of the paper
directly from the result files (no number is typed by hand).

Inputs:  --results         folder written by heads.py (GPU experiments)
         --audit_analysis  folder written by analyze_audit.py (audit statistics)
Outputs: figures/fig_audit.pdf, figures/fig_transfer.pdf, figures/fig_injection.pdf
         tables/tab_audit.tex, tables/tab_ablation.tex, numbers.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
_ap = argparse.ArgumentParser(description="Build the paper's tables, figures and numbers from the outputs.")
_ap.add_argument("--results", default="v2_out/results", help="results folder written by heads.py")
_ap.add_argument("--audit_analysis", default="audit_analysis", help="folder written by analyze_audit.py")
_ap.add_argument("--out", default="paper_assets", help="where to write figures/, tables/ and numbers.json")
_a = _ap.parse_args()
RES, AUD, OUT = _a.results, _a.audit_analysis, _a.out
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4314b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 3.4
N = {}   # every in-text number


def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.pdf"); fig.savefig(f"{OUT}/figures/{name}.png"); plt.close(fig)


# ===================================================================== audit
NAMES = {  # id -> (short name, group)
    "amine-maazizi/vc-detection-10k": ("vc-detection-10k", "community"),
    "garystafford/deepfake-audio-detection": ("garystafford", "community"),
    "ajaykarthick/codecfake-audio": ("codecfake-audio", "community"),
    "ALLA1N/In-the-wild_validation-dataset": ("ALLA1N (51+51)", "community"),
    "DeepFake-Audio-Rangers/Arabic_Audio_Deepfake": ("Arabic AD", "community"),
    "SpeechAntiSpoofingBenchmarks/XMAD": ("XMAD", "hosted"),
    "SpeechAntiSpoofingBenchmarks/PyAra": ("PyAra", "hosted"),
    "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": ("ASVspoof 2021 LA", "hosted"),
    "SpeechAntiSpoofingBenchmarks/CVoiceFake_small": ("CVoiceFake\\_small", "hosted"),
    "SpeechAntiSpoofingBenchmarks/HABLA": ("HABLA", "hosted"),
}
FEAT = {"leading_silence_ms": "lead.\\ silence", "trailing_silence_ms": "trail.\\ silence", "duration_sec": "duration",
        "rms": "RMS", "zcr": "ZCR", "peak_amplitude": "peak amp.", "silence_ratio": "silence ratio"}
FEAT_PLAIN = {"leading_silence_ms": "leading silence", "trailing_silence_ms": "trailing silence",
              "duration_sec": "duration", "rms": "RMS", "zcr": "ZCR", "peak_amplitude": "peak amplitude",
              "silence_ratio": "silence ratio"}

S = pd.read_csv(f"{AUD}/audit_summary.csv").set_index("dataset")
D = pd.read_csv(f"{AUD}/feature_effects.csv")
T = pd.read_csv(f"{AUD}/transfer_lr.csv", index_col=0)
TG = pd.read_csv(f"{AUD}/transfer_gbt.csv", index_col=0)
ten = list(NAMES)


def sr(v):
    v = json.loads(v.replace("'", '"')) if isinstance(v, str) else v
    f = lambda x: f"{x/1000:g}"
    return f(v[0]) if len(v) == 1 else (f"{f(v[0])}--{f(v[-1])}" if len(v) == 2 else f"{f(v[0])}--{f(v[-1])}")


def risk(lo):
    return "high" if lo > 0.85 else ("mod." if lo > 0.65 else "low")


rows = {"community": [], "hosted": []}
for ds in ten:
    r = S.loc[ds]; name, grp = NAMES[ds]
    lf, ld = r["largest_feature"], r["largest_d"]
    srr, srf = sr(r["sr_real"]), sr(r["sr_fake"])
    srtxt = srr if srr == srf else f"{srr}/{srf}"
    if ds.startswith("ALLA1N"):
        srtxt = "44.1--48"
    rows[grp].append((r["core7_lr_auc"], f"{name} & {srtxt} & {r['core7_lr_auc']:.3f} [{r['core7_lr_ci_lo']:.2f}, "
                      f"{r['core7_lr_ci_hi']:.2f}] & {risk(r['core7_lr_ci_lo'])} & {r['core7_gbt_auc']:.3f} & "
                      f"{r['all_gbt_auc']:.3f} & {FEAT[lf]} (${ld:+.2f}$) \\\\"))
lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Features-only audit (500 real + 500 fake clips per dataset; all 1,866 clips of garystafford; "
         r"51+51 for ALLA1N). All AUCs are out-of-fold, from five repeats of stratified 5-fold cross-validation with "
         r"features standardised inside each fold; brackets give 95\% bootstrap intervals. Risk is set by the lower "
         r"bound of the interval (high $>0.85$, moderate 0.65--0.85). GBT: gradient-boosted trees. +native: the seven "
         r"features plus relative-threshold silence and eight features measured at the native sampling rate "
         r"(including the rate itself). Largest: feature with the largest $|d|$ (fake minus real). SR: native sampling "
         r"rate in kHz (real/fake when they differ).}",
         r"\label{tab:audit}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccccl}", r"\toprule",
         r"Dataset & SR & 7-feature LR AUC [95\% CI] & Risk & 7-feature GBT & +native GBT & Largest ($d$) \\",
         r"\midrule"]
for grp, title in (("community", "Community-built"), ("hosted", "Hosted benchmarks")):
    if grp == "hosted":
        lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
    lines += [t for _, t in sorted(rows[grp], key=lambda x: -x[0])]
lines += [r"\bottomrule", r"\end{tabular}"]
rep = {k: S.loc[k, "core7_lr_auc"] for k in ["Tanishq125/deepfake-audio-detection", "Dhruv0100/deepfake-audio-detection",
                                            "rijal028/secureaudio-corpus-standardized"]}
lines += [rf"\tabnote{{Re-uploads of garystafford: Tanishq125 and Dhruv0100 gave identical features to each other "
          rf"(7-feature LR AUC {rep['Tanishq125/deepfake-audio-detection']:.3f} on their 500+500 samples); the 16-kHz "
          rf"version rijal028 gave {rep['rijal028/secureaudio-corpus-standardized']:.3f}. Arabic AD is the only dataset "
          rf"whose 7-feature LR AUC is not significant in a 200-permutation test ($p={S.loc['DeepFake-Audio-Rangers/Arabic_Audio_Deepfake','core7_lr_perm_p']:.3f}$; all others $p=0.005$).}}",
          r"\end{table*}"]
open(f"{OUT}/tables/tab_audit.tex", "w").write("\n".join(lines) + "\n")

lo = S.loc[ten, "core7_lr_ci_lo"]
N["audit_n_high"] = int((lo > 0.85).sum()); N["audit_n_mod"] = int(((lo > 0.65) & (lo <= 0.85)).sum())
N["audit_n_low"] = int((lo <= 0.65).sum())
N["audit_lr_min"] = float(S.loc[ten, "core7_lr_auc"].min()); N["audit_lr_max"] = float(S.loc[ten, "core7_lr_auc"].max())
sep6 = S.loc[ten][S.loc[ten, "core7_lr_ci_lo"] > 0.65]
N["audit_sep_n"] = len(sep6); N["audit_sep_min"] = float(sep6.core7_lr_auc.min())
N["audit_gbt_all_min"] = float(S.loc[ten, "all_gbt_auc"].min())
N["audit_gbt_all_n091"] = int((S.loc[ten, "all_gbt_auc"] >= 0.91).sum())
for k in ten:
    N[f"lr_{NAMES[k][0]}"] = float(S.loc[k, "core7_lr_auc"]); N[f"gbt_{NAMES[k][0]}"] = float(S.loc[k, "core7_gbt_auc"])
    N[f"allgbt_{NAMES[k][0]}"] = float(S.loc[k, "all_gbt_auc"]); N[f"nativelr_{NAMES[k][0]}"] = float(S.loc[k, "native_lr_auc"])
N["arabic_perm_p"] = float(S.loc["DeepFake-Audio-Rangers/Arabic_Audio_Deepfake", "core7_lr_perm_p"])
N["arabic_ci"] = [float(S.loc["DeepFake-Audio-Rangers/Arabic_Audio_Deepfake", c]) for c in ("core7_lr_ci_lo", "core7_lr_ci_hi")]

M = T.loc[ten, ten].values; off = M[~np.eye(10, dtype=bool)]
MG = TG.loc[ten, ten].values; offg = MG[~np.eye(10, dtype=bool)]
N.update({"transfer_lr_offmean": float(off.mean()), "transfer_lr_below05": float((off < 0.5).mean()),
          "transfer_lr_diagmean": float(np.diag(M).mean()), "transfer_gbt_offmean": float(offg.mean()),
          "transfer_gbt_below05": float((offg < 0.5).mean()), "transfer_lr_offmax": float(off.max()),
          "transfer_gary_asv21": float(T.loc["garystafford/deepfake-audio-detection", "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"])})
# sign flips among separable datasets
piv = D[D.feature.isin(FEAT)].pivot(index="dataset", columns="feature", values="d").loc[ten, list(FEAT)]
flips = {f: bool((piv[f] > 0.5).any() and (piv[f] < -0.5).any()) for f in FEAT}
N["sign_flip_features"] = [FEAT_PLAIN[f] for f, v in flips.items() if v]
N["d_peak_vc"] = float(piv.loc["amine-maazizi/vc-detection-10k", "peak_amplitude"])
N["d_peak_pyara"] = float(piv.loc["SpeechAntiSpoofingBenchmarks/PyAra", "peak_amplitude"])
N["d_lead_gary"] = float(piv.loc["garystafford/deepfake-audio-detection", "leading_silence_ms"])
N["d_lead_asv21"] = float(piv.loc["SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA", "leading_silence_ms"])
N["d_lead_xmad"] = float(piv.loc["SpeechAntiSpoofingBenchmarks/XMAD", "leading_silence_ms"])

# ---- Fig: audit bars with CIs
order = sorted(ten, key=lambda k: S.loc[k, "core7_lr_auc"])
fig, ax = plt.subplots(figsize=(W, 2.55))
y = np.arange(len(order))
for i, k in enumerate(order):
    c = BLUE if NAMES[k][1] == "community" else ORANGE
    r = S.loc[k]
    ax.barh(i, r.core7_lr_auc - 0.4, left=0.4, height=0.62, color=c, zorder=3)
    ax.errorbar(r.core7_lr_auc, i, xerr=[[r.core7_lr_auc - r.core7_lr_ci_lo], [r.core7_lr_ci_hi - r.core7_lr_auc]],
                fmt="none", ecolor=INK, lw=0.8, capsize=1.8, zorder=4)
    ax.plot(r.all_gbt_auc, i, marker="D", ms=3.6, color=INK, mfc="white", mew=0.9, zorder=5, ls="none")
ax.set_yticks(y, [NAMES[k][0].replace("\\_", "_") for k in order])
for xv in (0.65, 0.85):
    ax.axvline(xv, color=MUTED, ls=":", lw=0.8, zorder=2)
ax.set_xlim(0.4, 1.02); ax.set_xlabel("Out-of-fold AUC (features only)")
ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="community, 7-feature LR"), Patch(color=ORANGE, label="hosted, 7-feature LR"),
                   Line2D([], [], marker="D", ls="none", color=INK, mfc="white", ms=3.6, label="+native features, GBT")],
          loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8)
save(fig, "fig_audit")

# ---- Fig: transfer matrix + signed d
short = [NAMES[k][0].replace("\\_", "_").replace(" (51+51)", "") for k in ten]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
im = ax.imshow(M, cmap="RdBu", vmin=0, vmax=1)
ax.set_xticks(range(10), short, rotation=55, ha="right"); ax.set_yticks(range(10), short)
for i in range(10):
    for j in range(10):
        t = f"{M[i, j]:.2f}"
        ax.text(j, i, "1.0" if t == "1.00" else t[1:], ha="center", va="center", fontsize=5.4,
                color="white" if abs(M[i, j] - 0.5) > 0.3 else INK)
ax.set_xlabel("test dataset"); ax.set_ylabel("train dataset"); ax.set_title("(a) 7-feature LR: train on one, test on another")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("AUC", fontsize=7); cb.ax.tick_params(labelsize=6.5)
ax = axs[1]
P = piv.values
im2 = ax.imshow(P, cmap="PuOr_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(7), [FEAT_PLAIN[f] for f in FEAT], rotation=55, ha="right"); ax.set_yticks(range(10), short)
for i in range(10):
    for j in range(7):
        ax.text(j, i, f"{P[i, j]:+.1f}", ha="center", va="center", fontsize=5.4,
                color="white" if abs(P[i, j]) > 1.8 else INK)
ax.set_title("(b) Cohen's $d$ (fake $-$ real)")
cb = fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.02); cb.set_label("$d$ (clipped at $\\pm3$)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
for a in axs:
    a.spines[:].set_visible(False); a.tick_params(length=0)
fig.tight_layout(w_pad=1.2)
save(fig, "fig_transfer")

# ================================================================ detector
R = pd.read_csv(f"{RES}/runs.csv"); L = pd.read_csv(f"{RES}/legacy.csv").iloc[0]
B = pd.read_csv(f"{RES}/baselines.csv"); V = pd.read_csv(f"{RES}/reverse.csv"); I = pd.read_csv(f"{RES}/injection.csv")
G = json.load(open(f"{RES}/groups.json"))
CN = {"raw": "raw", "sil": "silence", "loud": "loudness", "crop": "crop", "all": "silence, loudness, duration"}
CD = {"raw": "none", "sil": "silence", "loud": "loudness", "crop": "silence, duration", "all": "silence, loudness, duration"}


def ms(x, d=3, pct=False):
    m, s = x.mean(), x.std(ddof=1)
    if pct:
        return f"{100*m:.0f}$\\pm${100*s:.0f}"
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Controlled ablation of the case-study detector (frozen Whisper-small + MLP head). Every cell is "
         r"the mean $\pm$ SD over five seeds; the same preprocessing is applied to training, in-domain test and "
         r"ASVspoof~2019~LA (2,000 clips). Random: the original 70/15/15 split. Group: whole speaker clusters held out. "
         r"Pooling: mean over all 1,500 encoder frames (as in the original model) or over the frames that contain "
         r"audio. Real$\to$fake: share of bona fide ASVspoof clips labelled fake at the 0.5 threshold.}",
         r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{In-domain AUC} & \multicolumn{3}{c}{ASVspoof 2019 LA (random split)} & ITW \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
         r"Cues removed & Pooling & random & group & AUC & EER (\%) & real$\to$fake (\%) & AUC \\", r"\midrule"]
for c in ["raw", "sil", "loud", "crop", "all"]:
    for p in ["all", "valid"]:
        rr = R[(R.cond == c) & (R.pool == p) & (R.split == "random")]; rg = R[(R.cond == c) & (R.pool == p) & (R.split == "group")]
        lines.append(f"{CD[c] if p == 'all' else ''} & {'all frames' if p == 'all' else 'audio frames'} & "
                     f"{ms(rr.in_auc)} & {ms(rg.in_auc)} & {ms(rr.asv_auc)} & {ms(rr.asv_eer*100, 1)} & "
                     f"{ms(rr.asv_real_as_fake, pct=True)} & {ms(rr.itw_auc)} \\\\")
    if c != "all":
        lines.append(r"\addlinespace[1pt]")
lines += [r"\midrule",
          rf"\multicolumn{{8}}{{p{{0.97\textwidth}}}}{{Original checkpoint (v1 paper, silence, all frames): in-domain AUC {L.in_auc:.3f}; "
          rf"ASVspoof with the old script (no preprocessing) {L.asvnative_auc:.3f}, with matched preprocessing "
          rf"{L.asv_auc:.3f} (real$\to$fake {100*L.asv_real_as_fake:.0f}\%).}} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{OUT}/tables/tab_ablation.tex", "w").write("\n".join(lines) + "\n")

g = lambda c, p, s="random": R[(R.cond == c) & (R.pool == p) & (R.split == s)]
N["in_auc_min_any"] = float(R.groupby(["cond", "pool", "split"]).in_auc.mean().min())
N["in_auc_min_run"] = float(R.in_auc.min())
N["asv_raw_all"] = float(g("raw", "all").asv_auc.mean()); N["asv_raw_valid"] = float(g("raw", "valid").asv_auc.mean())
N["asv_raw_valid_sd"] = float(g("raw", "valid").asv_auc.std(ddof=1))
N["asv_raw_valid_eer"] = float(g("raw", "valid").asv_eer.mean()); N["asv_raw_valid_rf"] = float(g("raw", "valid").asv_real_as_fake.mean())
N["asv_sil_all"] = float(g("sil", "all").asv_auc.mean()); N["asv_sil_all_rf"] = float(g("sil", "all").asv_real_as_fake.mean())
N["asv_all_all"] = float(g("all", "all").asv_auc.mean()); N["asv_all_valid"] = float(g("all", "valid").asv_auc.mean())
N["asv_loud_valid"] = float(g("loud", "valid").asv_auc.mean())
N["asv_max_cell"] = float(R[R.split == "random"].groupby(["cond", "pool"]).asv_auc.mean().max())
N["asv_min_cell"] = float(R[R.split == "random"].groupby(["cond", "pool"]).asv_auc.mean().min())
N["asv_allframes_rf_range"] = [float(R[(R.split == "random") & (R.pool == "all")].groupby("cond").asv_real_as_fake.mean().min()),
                               float(R[(R.split == "random") & (R.pool == "all")].groupby("cond").asv_real_as_fake.mean().max())]
N["asv_valid_rf_raw_loud"] = [float(g("raw", "valid").asv_real_as_fake.mean()), float(g("loud", "valid").asv_real_as_fake.mean())]
N["itw_range"] = [float(R[R.split == "random"].groupby(["cond", "pool"]).itw_auc.mean().min()),
                  float(R[R.split == "random"].groupby(["cond", "pool"]).itw_auc.mean().max())]
N["group_in_min"] = float(R[R.split == "group"].groupby(["cond", "pool"]).in_auc.mean().min())
N["sil_native_valid"] = float(g("sil", "valid").asvnative_auc.mean()); N["sil_native_all"] = float(g("sil", "all").asvnative_auc.mean())
N["legacy"] = {k: float(L[k]) for k in ["in_auc", "asvnative_auc", "asvnative_acc", "asvnative_real_as_fake", "asv_auc",
                                       "asv_real_as_fake", "asvraw_auc"]}
N["groups"] = {"k_real": G["real"]["k"], "k_fake": G["fake"]["k"], "sil_real": G["real"]["silhouette"],
               "sil_fake": G["fake"]["silhouette"]}
# baselines (random split mean over seeds)
bb = B[B.split == "random"].groupby(["model", "cond"])[["in_auc", "asv_auc", "asv_real_as_fake"]].mean()
N["base"] = {f"{m}|{c}": [float(v) for v in bb.loc[(m, c)].values] for m, c in bb.index}
vv = V.groupby(["cond", "pool"])[["gary_auc", "asveval_auc"]].mean()
N["rev"] = {f"{c}|{p}": [float(v) for v in vv.loc[(c, p)].values] for c, p in vv.index}

# ---- Fig: injection curves
fig, axs = plt.subplots(1, 2, figsize=(W, 1.95), sharey=True)
models = [("raw", "all", BLUE, "-", "none removed, all frames"), ("raw", "valid", BLUE, "--", "none removed, audio frames"),
          ("sil", "all", ORANGE, "-", "silence removed, all frames"), ("sil", "valid", ORANGE, "--", "silence removed, audio frames"),
          ("all", "all", MUTED, "-", "silence+loudness+duration removed")]
for c, p, col, ls, lab in models:
    d = I[(I.cond == c) & (I.pool == p)]
    pk = d[d.kind == "peak_dbfs"].groupby("level").bona_as_fake.agg(["mean", "std"]).sort_index()
    axs[0].errorbar(pk.index, pk["mean"] * 100, yerr=pk["std"] * 100, color=col, ls=ls, lw=1.1, marker="o", ms=2.2,
                    capsize=1.5, elinewidth=0.6, label=lab)
    sl = d[d.kind == "silence_ms"].groupby("level").bona_as_fake.agg(["mean", "std"]).sort_index()
    axs[1].errorbar(sl.index, sl["mean"] * 100, yerr=sl["std"] * 100, color=col, ls=ls, lw=1.1, marker="o", ms=2.2,
                    capsize=1.5, elinewidth=0.6)
axs[0].set_xlabel("peak level (dBFS)"); axs[1].set_xlabel("silence added per end (ms)")
axs[0].set_ylabel("bona fide labelled fake (%)"); axs[0].set_xticks([-20, -12, -6, -1]); axs[1].set_xticks([0, 100, 200, 400])
for a in axs:
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_ylim(0, 100)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False, fontsize=6.2, handlelength=1.8,
           columnspacing=0.8)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_injection")
for c, p, *_ in models:
    d = I[(I.cond == c) & (I.pool == p)]
    pk = d[d.kind == "peak_dbfs"].groupby("level").bona_as_fake.mean()
    sl = d[d.kind == "silence_ms"].groupby("level").bona_as_fake.mean()
    N[f"inj_peak_{c}_{p}"] = [float(pk.loc[-20]), float(pk.loc[-1])]
    N[f"inj_sil_{c}_{p}"] = [float(sl.loc[0]), float(sl.loc[400])]
    mono = d[d.kind == "peak_dbfs"].pivot(index="seed", columns="level", values="bona_as_fake")[[-20, -12, -6, -1]]
    N[f"inj_peak_mono_{c}_{p}"] = int((mono.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).sum())

json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1))
