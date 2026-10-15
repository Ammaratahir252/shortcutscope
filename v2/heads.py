#!/usr/bin/env python3
"""
heads.py - trains and evaluates every detector head of the v2 experiments
on the vectors saved by encode.py. CPU is enough (about 15-30 min in total).

Experiments (reviewer numbers in brackets):
  E1  controlled ablation: 5 conditions x 2 poolings x 2 split types x 5 seeds,
      each head tested in-domain AND on ASVspoof with the SAME preprocessing
      (#2, #3, #4, #5, #12); the 'all'+group cells are the cleaned-dataset retrain.
  E2  legacy check: the paper's checkpoint head on the new vectors, ASVspoof
      'native' (old script) vs 'sil' (matched) preprocessing (#2).
  E3  counterfactual injection: added silence / peak level on ASVspoof clips (#6).
  E4  simple baselines in-domain and on ASVspoof: MFCC+LR, 7-feature LR and
      gradient-boosted trees (#10, 'run simple baselines out-of-domain').
  E5  reverse direction: train on ASVspoof 2019 LA train, test on garystafford.

Label convention: 1 = fake. Every score is P(fake).
Outputs in <out_dir>/results/: runs.csv, injection.csv, baselines.csv,
groups.json, scores.npz, summary.txt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEEDS = [42, 123, 2024, 7, 99]
POOLS = ["all", "valid"]
SPLITS = ["random", "group"]


# ------------------------------------------------------------------ metrics
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y, s, prefix=""):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    p = (s > 0.5).astype(int)
    out = {"auc": roc_auc_score(y, s) if len(set(y)) > 1 else np.nan,
           "eer": eer(y, s) if len(set(y)) > 1 else np.nan,
           "acc": accuracy_score(y, p), "f1_fake": f1_score(y, p, zero_division=0),
           "real_as_fake": float(np.mean(p[y == 0] == 1)) if np.any(y == 0) else np.nan,
           "fake_as_fake": float(np.mean(p[y == 1] == 1)) if np.any(y == 1) else np.nan}
    return {prefix + k: v for k, v in out.items()}


# ------------------------------------------------------------------ data
class Store:
    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, "emb"); self.cache = {}

    def has(self, name, cond):
        return os.path.exists(os.path.join(self.dir, f"{name}__{cond}.npz"))

    def get(self, name, cond):
        k = (name, cond)
        if k not in self.cache:
            d = np.load(os.path.join(self.dir, f"{name}__{cond}.npz"), allow_pickle=False)
            self.cache[k] = {f: d[f] for f in d.files}
        return self.cache[k]


def legacy_split(n, seed, val=0.15, test=0.15):
    """Exactly dataset.get_splits() from the original code (unstratified 70/15/15)."""
    idx = list(range(n))
    tr, ho = train_test_split(idx, test_size=val + test, random_state=seed, shuffle=True)
    va, te = train_test_split(ho, test_size=test / (val + test), random_state=seed, shuffle=True)
    return np.array(tr), np.array(va), np.array(te)


def make_groups(spk_path, ids, y, k_real=14, k_fake=12):
    """Approximate source groups with speaker clustering (the dataset has no video/platform field)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    d = np.load(spk_path)
    pos = {c: i for i, c in enumerate(d["ids"])}
    E = d["emb"][[pos[c] for c in ids]]
    g = np.zeros(len(ids), dtype=int); info = {}
    offset = 0
    for lab, k in ((0, k_real), (1, k_fake)):
        m = np.where(y == lab)[0]
        cl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(E[m])
        g[m] = cl + offset
        sizes = np.bincount(cl).tolist()
        info["real" if lab == 0 else "fake"] = {"k": k, "sizes": sorted(sizes, reverse=True),
                                                "silhouette": float(silhouette_score(E[m], cl))}
        offset += k
    return g, info


def group_split(y, groups, seed, val=0.15, test=0.15):
    """Whole speaker clusters go to test/val (per class, so both classes appear everywhere)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        gs = np.unique(groups[idx]); rng.shuffle(gs)
        n = len(idx); cnt = 0; bucket = te; limit = test * n
        for g in gs:
            members = idx[groups[idx] == g]
            if bucket is te and cnt >= limit:
                bucket = va; cnt = 0; limit = val * n
            elif bucket is va and cnt >= limit:
                bucket = tr
            bucket.extend(members.tolist()); cnt += len(members)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


# ------------------------------------------------------------------ head
def build_head(torch):
    nn = torch.nn
    return nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


def train_head(Xtr, ytr, Xva, yva, seed, epochs=20, bs=8, lr=1e-4, wd=1e-4):
    """Same head, optimiser, schedule, clipping and model selection as the original train.py."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    head = build_head(torch)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep = np.inf, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        for b in torch.randperm(len(Xt), generator=g).split(bs):
            opt.zero_grad()
            loss = lossf(head(Xt[b]).squeeze(1), yt[b]); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            sv = torch.sigmoid(head(Xv).squeeze(1)).numpy()
        e = eer(yva, sv)
        if e < best:
            best, best_ep = e, ep
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    return head, best, best_ep


def score(head, X):
    import torch
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor(X)).squeeze(1)).numpy()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/kaggle/working/v2_out")
    ap.add_argument("--legacy_ckpt", default=None, help="optional: the paper's best_model.pt")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    res_dir = os.path.join(args.out_dir, "results"); os.makedirs(res_dir, exist_ok=True)
    S = Store(args.out_dir)
    T0 = time.time()
    runs, inj_rows, base_rows, scores = [], [], [], {}

    gref = S.get("gary", "raw"); ids = gref["ids"]; y = gref["y"].astype(int)
    spk = os.path.join(args.out_dir, "spk", "gary_wavlm.npz")
    groups, ginfo = make_groups(spk, ids, y) if os.path.exists(spk) else (None, {"error": "no speaker file"})
    json.dump(ginfo, open(os.path.join(res_dir, "groups.json"), "w"), indent=1)
    print("speaker groups:", ginfo)

    split_sets = {}
    for seed in seeds:
        split_sets[("random", seed)] = legacy_split(len(y), seed)
        if groups is not None:
            split_sets[("group", seed)] = group_split(y, groups, seed)
    for (st, sd), (tr, va, te) in split_sets.items():
        json.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
                  open(os.path.join(res_dir, f"split_{st}_{sd}.json"), "w"))

    # ---------------- E1 + E3 ----------------
    inj_names = sorted({f.split("__")[0] for f in os.listdir(S.dir) if f.startswith("inj_")})
    for cond in C.CONDITIONS:
        G = S.get("gary", cond)
        assert (G["ids"] == ids).all()
        asv = S.get("asv_eval", cond) if S.has("asv_eval", cond) else None
        nat = S.get("asv_eval", "native") if (cond == "sil" and S.has("asv_eval", "native")) else None
        itw = S.get("itw", cond) if S.has("itw", cond) else None
        for pool in POOLS:
            X = G[f"emb_{pool}"]
            for (st, sd), (tr, va, te) in split_sets.items():
                head, vbest, ep = train_head(X[tr], y[tr], X[va], y[va], sd, epochs=args.epochs)
                key = f"{cond}|{pool}|{st}|{sd}"
                row = {"cond": cond, "pool": pool, "split": st, "seed": sd, "best_epoch": ep, "val_eer": vbest,
                       "n_train": len(tr), "n_test": len(te)}
                s_te = score(head, X[te]); row.update(metrics(y[te], s_te, "in_"))
                scores[key + "|in"] = s_te
                if asv is not None:
                    s = score(head, asv[f"emb_{pool}"]); row.update(metrics(asv["y"], s, "asv_"))
                    scores[key + "|asv"] = s
                if nat is not None:
                    s = score(head, nat[f"emb_{pool}"]); row.update(metrics(nat["y"], s, "asvnative_"))
                if itw is not None:
                    s = score(head, itw[f"emb_{pool}"]); row.update(metrics(itw["y"], s, "itw_"))
                runs.append(row)
                if st == "random" and cond in ("raw", "sil", "all"):
                    for name in inj_names:
                        if not S.has(name, cond):
                            continue
                        I = S.get(name, cond); s = score(head, I[f"emb_{pool}"]); yy = I["y"].astype(int)
                        kind, level = ("silence_ms", int(name[7:])) if name.startswith("inj_sil") else \
                                      ("peak_dbfs", -int(name[8:]))
                        inj_rows.append({"cond": cond, "pool": pool, "seed": sd, "kind": kind, "level": level,
                                         "bona_as_fake": float(np.mean(s[yy == 0] > 0.5)),
                                         "spoof_as_fake": float(np.mean(s[yy == 1] > 0.5)),
                                         "bona_mean_pfake": float(np.mean(s[yy == 0])),
                                         "spoof_mean_pfake": float(np.mean(s[yy == 1])),
                                         "auc": roc_auc_score(yy, s)})
                print(f"[E1] {key:28s} in AUC {row['in_auc']:.4f} | ASV AUC {row.get('asv_auc', np.nan):.4f} "
                      f"EER {row.get('asv_eer', np.nan)*100:.1f}% real->fake {row.get('asv_real_as_fake', np.nan)*100:.1f}% "
                      f"| {time.time()-T0:.0f}s")
        pd.DataFrame(runs).to_csv(os.path.join(res_dir, "runs.csv"), index=False)
        pd.DataFrame(inj_rows).to_csv(os.path.join(res_dir, "injection.csv"), index=False)

    # ---------------- E2 legacy ----------------
    if args.legacy_ckpt and os.path.exists(args.legacy_ckpt):
        import torch
        ck = torch.load(args.legacy_ckpt, map_location="cpu", weights_only=False)
        head = build_head(torch)
        head.load_state_dict({k[len("classifier."):]: v for k, v in ck["model_state"].items()
                              if k.startswith("classifier.")})
        head.eval()
        tr, va, te = legacy_split(len(y), 42)
        Gs = S.get("gary", "sil")
        row = {"cond": "sil", "pool": "all", "split": "random", "seed": 42, "legacy": True,
               "ckpt_epoch": ck.get("epoch"), "ckpt_test_metrics": json.dumps(ck.get("test_metrics"))}
        row.update(metrics(y[te], score(head, Gs["emb_all"][te]), "in_"))
        for c, pre in (("native", "asvnative_"), ("sil", "asv_"), ("raw", "asvraw_")):
            if S.has("asv_eval", c):
                A = S.get("asv_eval", c); row.update(metrics(A["y"], score(head, A["emb_all"]), pre))
        pd.DataFrame([row]).to_csv(os.path.join(res_dir, "legacy.csv"), index=False)
        print("[E2] legacy checkpoint:", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    # ---------------- E4 baselines ----------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    feat_sets = {"mfcc_lr": ("mfcc", None), "core7_lr": ("hand", slice(0, 7)), "core7_gbt": ("hand", slice(0, 7))}
    for cond in C.CONDITIONS:
        G = S.get("gary", cond)
        asv = S.get("asv_eval", cond) if S.has("asv_eval", cond) else None
        for (st, sd), (tr, va, te) in split_sets.items():
            trv = np.concatenate([tr, va])
            for name, (field, sl) in feat_sets.items():
                Xg = G[field] if sl is None else G[field][:, sl]
                model = (HistGradientBoostingClassifier(random_state=sd) if name.endswith("gbt")
                         else make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)))
                model.fit(Xg[trv], y[trv])
                row = {"cond": cond, "split": st, "seed": sd, "model": name}
                row.update(metrics(y[te], model.predict_proba(Xg[te])[:, 1], "in_"))
                if asv is not None:
                    Xa = asv[field] if sl is None else asv[field][:, sl]
                    row.update(metrics(asv["y"], model.predict_proba(Xa)[:, 1], "asv_"))
                base_rows.append(row)
    pd.DataFrame(base_rows).to_csv(os.path.join(res_dir, "baselines.csv"), index=False)
    print(f"[E4] baselines done | {time.time()-T0:.0f}s")

    # ---------------- E5 reverse direction ----------------
    rev = []
    if S.has("asv_train", "raw"):
        for cond in C.CONDITIONS:
            A = S.get("asv_train", cond); ya = A["y"].astype(int)
            G = S.get("gary", cond)
            AE = S.get("asv_eval", cond) if S.has("asv_eval", cond) else None
            for pool in POOLS:
                for sd in seeds:
                    tr, va = train_test_split(np.arange(len(ya)), test_size=0.15, random_state=sd, stratify=ya)
                    head, vbest, ep = train_head(A[f"emb_{pool}"][tr], ya[tr], A[f"emb_{pool}"][va], ya[va], sd,
                                                 epochs=args.epochs)
                    row = {"cond": cond, "pool": pool, "seed": sd, "best_epoch": ep, "val_eer": vbest}
                    s = score(head, G[f"emb_{pool}"]); row.update(metrics(y, s, "gary_"))
                    scores[f"rev|{cond}|{pool}|{sd}|gary"] = s
                    if AE is not None:
                        row.update(metrics(AE["y"], score(head, AE[f"emb_{pool}"]), "asveval_"))
                    rev.append(row)
                    print(f"[E5] {cond}|{pool}|{sd}: gary AUC {row['gary_auc']:.4f} | asv-eval AUC "
                          f"{row.get('asveval_auc', np.nan):.4f}")
        pd.DataFrame(rev).to_csv(os.path.join(res_dir, "reverse.csv"), index=False)

    np.savez_compressed(os.path.join(res_dir, "scores.npz"), **{k.replace("|", "__"): v for k, v in scores.items()})

    # ---------------- summary ----------------
    R = pd.DataFrame(runs)
    cols = [c for c in ["in_auc", "in_eer", "asv_auc", "asv_eer", "asv_real_as_fake", "asvnative_auc", "itw_auc"]
            if c in R.columns]
    summ = R.groupby(["cond", "pool", "split"])[cols].agg(["mean", "std"]).round(4)
    with open(os.path.join(res_dir, "summary.txt"), "w") as f:
        f.write(f"speaker groups: {json.dumps(ginfo)}\n\nE1 (mean, sd over {len(seeds)} seeds)\n")
        f.write(summ.to_string() + "\n\n")
        if inj_rows:
            f.write("E3 injection (mean over seeds)\n")
            f.write(pd.DataFrame(inj_rows).groupby(["cond", "pool", "kind", "level"])[
                ["bona_as_fake", "spoof_as_fake", "bona_mean_pfake", "auc"]].mean().round(4).to_string() + "\n\n")
        B = pd.DataFrame(base_rows)
        f.write("E4 baselines (mean over seeds)\n" + B.groupby(["model", "cond", "split"])[
            [c for c in ["in_auc", "asv_auc", "asv_real_as_fake"] if c in B.columns]].mean().round(4).to_string() + "\n\n")
        if rev:
            f.write("E5 reverse (mean over seeds)\n" + pd.DataFrame(rev).groupby(["cond", "pool"])[
                [c for c in ["gary_auc", "gary_eer", "asveval_auc"] if c in pd.DataFrame(rev).columns]
            ].mean().round(4).to_string() + "\n")
    print(open(os.path.join(res_dir, "summary.txt")).read())
    print(f"DONE in {(time.time()-T0)/60:.1f} min -> {res_dir}")


if __name__ == "__main__":
    main()
