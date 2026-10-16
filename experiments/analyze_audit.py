#!/usr/bin/env python3
"""
analyze_audit.py - statistics for the audit (runs on the CSVs written by audit.py).

  - repeated stratified 5-fold CV (5 repeats), 95% bootstrap CI, label-permutation p-value
  - gradient-boosted trees next to logistic regression
  - features standardised inside every fold
  - per-feature rank AUC next to Cohen's d
  - cross-dataset transfer matrix and signed Cohen's d table

Usage: python analyze_audit.py --audit_dir audit_results --out_dir audit_analysis
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

NATIVE_NO_SR = [f for f in C.NATIVE if f != "native_sr"]
SETS = {"core7": C.CORE7, "core7+rel": C.CORE7 + C.REL3, "native": NATIVE_NO_SR,
        "all": C.CORE7 + C.REL3 + C.NATIVE}
REUPLOADS = {"Tanishq125/deepfake-audio-detection", "Dhruv0100/deepfake-audio-detection",
             "rijal028/secureaudio-corpus-standardized"}


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def model(kind, seed=0):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return HistGradientBoostingClassifier(random_state=seed)


def oof(X, y, kind, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[te] = model(kind, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cv_stats(X, y, kind, repeats=5, n_boot=2000, n_perm=200, seed=42):
    rng = np.random.default_rng(seed)
    P = np.array([oof(X, y, kind, seed + r) for r in range(repeats)])
    rep_aucs = [roc_auc_score(y, p) for p in P]
    pm = P.mean(axis=0)
    auc = roc_auc_score(y, pm)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], pm[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {"auc": auc, "ci_lo": lo, "ci_hi": hi, "rep_mean": float(np.mean(rep_aucs)),
           "rep_sd": float(np.std(rep_aucs))}
    if n_perm:
        obs = rep_aucs[0]
        null = [roc_auc_score(yp, oof(X, yp, kind, seed)) for yp in (rng.permutation(y) for _ in range(n_perm))]
        out["perm_p"] = (1 + sum(n >= obs for n in null)) / (1 + n_perm)
        out["perm_null_95"] = float(np.percentile(null, 95))
    return out


def risk(lo):
    return "high" if lo > 0.85 else ("moderate" if lo > 0.65 else "low")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--out_dir", default="audit_analysis")
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data, metas = {}, {}
    for f in sorted(glob.glob(os.path.join(args.audit_dir, "per_clip", "*.csv"))):
        name = os.path.basename(f)[:-4].replace("__", "/")
        data[name] = pd.read_csv(f)
        mp = os.path.join(args.audit_dir, "meta", os.path.basename(f)[:-4] + ".json")
        metas[name] = json.load(open(mp)) if os.path.exists(mp) else {}

    rows, drows = [], []
    for name, df in data.items():
        y = (df["label"] == "fake").astype(int).values
        m = metas.get(name, {})
        row = {"dataset": name, "group": m.get("group"), "n_real": int((y == 0).sum()), "n_fake": int(y.sum()),
               "sampling": m.get("sampling"), "split": m.get("split"), "revision": m.get("revision"),
               "sr_real": m.get("native_sr_real"), "sr_fake": m.get("native_sr_fake"),
               "subtype_real": sorted(df[y == 0]["subtype"].astype(str).unique().tolist()),
               "subtype_fake": sorted(df[y == 1]["subtype"].astype(str).unique().tolist())}
        for set_name, cols in SETS.items():
            X = df[cols].values.astype(float)
            for kind in ("lr", "gbt"):
                if set_name != "core7" and kind == "gbt" and set_name != "all":
                    continue
                st = cv_stats(X, y, kind, n_perm=args.n_perm if (set_name == "core7" and kind == "lr") else 0)
                for k, v in st.items():
                    row[f"{set_name}_{kind}_{k}"] = v
            print(f"{name:55s} {set_name:10s} LR AUC {row[f'{set_name}_lr_auc']:.4f} "
                  f"[{row[f'{set_name}_lr_ci_lo']:.3f}, {row[f'{set_name}_lr_ci_hi']:.3f}]")
        row["risk_core7_lr"] = risk(row["core7_lr_ci_lo"])
        for feat in C.CORE7 + C.REL3 + C.NATIVE:
            a, b = df.loc[y == 1, feat].values, df.loc[y == 0, feat].values
            d = cohens_d(a, b)
            fa = roc_auc_score(y, df[feat].values) if df[feat].nunique() > 1 else 0.5
            drows.append({"dataset": name, "feature": feat, "d": d, "feature_auc": fa,
                          "mean_real": b.mean(), "mean_fake": a.mean()})
        dd = {r["feature"]: r["d"] for r in drows if r["dataset"] == name and r["feature"] in C.CORE7}
        row["largest_feature"] = max(dd, key=lambda k: abs(dd[k])); row["largest_d"] = dd[row["largest_feature"]]
        rows.append(row)
    S = pd.DataFrame(rows); D = pd.DataFrame(drows)
    S.to_csv(os.path.join(args.out_dir, "audit_summary.csv"), index=False)
    D.to_csv(os.path.join(args.out_dir, "feature_effects.csv"), index=False)

    # ---- transfer matrix (one copy of every source only) ----
    names = [n for n in data if n not in REUPLOADS]
    for kind in ("lr", "gbt"):
        M = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            Xa = data[a][C.CORE7].values; ya = (data[a]["label"] == "fake").astype(int).values
            mdl = model(kind, 42).fit(Xa, ya)
            for b in names:
                if a == b:
                    M.loc[a, b] = S.set_index("dataset").loc[a, f"core7_{kind}_auc"]
                else:
                    Xb = data[b][C.CORE7].values; yb = (data[b]["label"] == "fake").astype(int).values
                    M.loc[a, b] = roc_auc_score(yb, mdl.predict_proba(Xb)[:, 1])
        M.to_csv(os.path.join(args.out_dir, f"transfer_{kind}.csv"))
        off = M.values[~np.eye(len(names), dtype=bool)]
        print(f"\ntransfer {kind}: diagonal mean {np.nanmean(np.diag(M.values)):.3f}, "
              f"off-diagonal mean {np.nanmean(off):.3f}, share <= 0.55: {np.mean(off <= 0.55):.2f}, "
              f"share < 0.5: {np.mean(off < 0.5):.2f}")
    piv = D[D.feature.isin(C.CORE7)].pivot(index="dataset", columns="feature", values="d")[C.CORE7]
    piv.to_csv(os.path.join(args.out_dir, "signed_d.csv"))
    print("\nsigned d (fake - real):\n", piv.round(2).to_string())
    print(f"\nsaved to {args.out_dir}")


if __name__ == "__main__":
    main()
