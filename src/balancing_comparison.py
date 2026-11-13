"""Does CTGAN balancing help?  Protein-level 5-fold CV, XGBoost on the reduced per-residue features.

Arms (all see the identical 82 encoded inputs, all evaluated on the untouched natural-ratio test fold):
  none            no balancing
  weight          scale_pos_weight = n_neg / n_pos
  ros1.0          random oversampling (duplicate minority rows) to minority:majority = 1.0
  smotenc0.5/1.0  SMOTE-NC synthetic rows to ratio 0.5 / 1.0
  ctgan0.5/1.0    CTGAN synthetic rows to ratio 0.5 / 1.0
Leak-free by construction: inside each fold, PCA, CTGAN and SMOTE see only that fold's TRAIN proteins (minus a 15% protein
hold-out used for early stopping and for the MCC-optimal threshold). The hold-out and test rows are never resampled.

Usage: python -X utf8 src/balancing_comparison.py [--epochs 200] [--folds 0 1 2 3 4] [--tag name] [--k 32]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from balancing import (ctgan_sample, encode, fidelity, fit_context_pcas, fit_ctgan, fit_pca, make_table,
                       make_table_fused, n_synthetic, random_oversample, smote_nc)
from features import PER_RES, global_features, window_features
from evaluate import best_mcc_threshold, metrics_at

root = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="binary:logistic", max_depth=7, min_child_weight=5, learning_rate=0.05,
              colsample_bytree=0.7, tree_method="hist", n_jobs=-1, eval_metric="aucpr")


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, weighted=False, seed=0):
    spw = (ytr == 0).sum() / (ytr == 1).sum() if weighted else 1.0
    m = xgb.XGBClassifier(**PARAMS, n_estimators=1000, early_stopping_rounds=60, scale_pos_weight=spw, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    thr = best_mcc_threshold(yva, pva)
    return {"best_iteration": int(m.best_iteration), "thr": thr,
            "test_at_0.5": metrics_at(yte, pte, 0.5), "test_at_val_threshold": metrics_at(yte, pte, thr),
            "n_train": int(len(ytr)), "train_pos_rate": float(ytr.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tag", default="35M")
    ap.add_argument("--esm", default="35M")
    ap.add_argument("--fused", action="store_true", help="add compact global + neighbour context (100 inputs)")
    a = ap.parse_args()

    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    esm = np.load(root / "data" / "processed" / f"esm2_{a.esm}.npy").astype(np.float32)
    y, g = res["label"].to_numpy(), res["protein_id"].to_numpy()
    if a.fused:
        G = global_features(res)
        W = window_features(res, 3).reshape(len(res), 7, PER_RES)
        Wm = np.delete(W, 3, axis=1)[:, :, 20:40].mean(1)          # mean PSSM of the 6 neighbours (zeros outside protein)
        del W
    out_path = root / "results" / f"balancing_comparison_{a.tag}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"config": vars(a), "folds": {}}

    for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(res, y, g)):
        if k not in a.folds or str(k) in out["folds"]:
            continue
        t_fold = time.time()
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=k).split(tr_i, y[tr_i], g[tr_i]))
        i_tr, i_va = tr_i[tr], tr_i[va]
        pca = fit_pca(esm[i_tr], a.k, seed=k)                               # TRAIN rows only
        if a.fused:
            gp, wp = fit_context_pcas(G[i_tr], Wm[i_tr], 8, seed=k)                # TRAIN rows only
            T_tr, T_va, T_te = (make_table_fused(res.iloc[i], esm[i], pca, gp, wp, G[i], Wm[i]) for i in (i_tr, i_va, te_i))
        else:
            T_tr, T_va, T_te = (make_table(res.iloc[i], esm[i], pca) for i in (i_tr, i_va, te_i))
        y_tr, y_va, y_te = y[i_tr], y[i_va], y[te_i]
        X_va, X_te = encode(T_va), encode(T_te)
        n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
        print(f"\n=== fold {k}: train {len(y_tr):,} (pos {n_pos:,} = {y_tr.mean():.3f}) val {len(y_va):,} test {len(y_te):,} "
              f"| PCA var {pca.explained_variance_ratio_.sum():.2f}", flush=True)
        fold = {"pca_explained_variance": float(pca.explained_variance_ratio_.sum()), "arms": {}, "fidelity": {}}
        X_tr = encode(T_tr)
        fold['n_inputs'] = int(X_tr.shape[1])
        print(f"  inputs per residue: {X_tr.shape[1]}", flush=True)

        def add(name, df_syn):
            Xs = encode(df_syn)
            Xa, ya = np.vstack([X_tr, Xs]), np.r_[y_tr, np.ones(len(Xs), dtype=y_tr.dtype)]
            t0 = time.time()
            fold["arms"][name] = fit_eval(Xa, ya, X_va, y_va, X_te, y_te, seed=k)
            report(name, t0)

        def report(name, t0):
            r, r5 = fold["arms"][name]["test_at_val_threshold"], fold["arms"][name]["test_at_0.5"]
            print(f"  {name:11s} {time.time()-t0:4.0f}s  @valthr mcc {r['mcc']:.3f} prauc {r['pr_auc']:.3f} rec {r['recall']:.3f} "
                  f"| @0.5 mcc {r5['mcc']:.3f} rec {r5['recall']:.3f} prec {r5['precision']:.3f}", flush=True)

        t0 = time.time(); fold["arms"]["none"] = fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=k); report("none", t0)
        t0 = time.time(); fold["arms"]["weight"] = fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, weighted=True, seed=k); report("weight", t0)
        add("ros1.0", random_oversample(T_tr, y_tr, 1.0, seed=k))

        syn_smote = {}
        for r in (0.5, 1.0):
            syn_smote[r] = smote_nc(T_tr, y_tr, r, seed=k)
            add(f"smotenc{r}", syn_smote[r])

        t0 = time.time()
        gen = fit_ctgan(T_tr[y_tr == 1].reset_index(drop=True), epochs=a.epochs, seed=k)
        print(f"  CTGAN fit {time.time()-t0:.0f}s", flush=True)
        syn_ct = ctgan_sample(gen, n_synthetic(n_pos, n_neg, 1.0))
        for r in (0.5, 1.0):
            add(f"ctgan{r}", syn_ct.iloc[: n_synthetic(n_pos, n_neg, r)])

        real_min = T_tr[y_tr == 1].reset_index(drop=True)
        fold["fidelity"]["ctgan"] = fidelity(real_min, syn_ct, seed=k)
        fold["fidelity"]["smotenc"] = fidelity(real_min, syn_smote[1.0], seed=k)
        print("  fidelity ctgan :", {kk: round(v, 3) for kk, v in fold["fidelity"]["ctgan"].items()})
        print("  fidelity smote :", {kk: round(v, 3) for kk, v in fold["fidelity"]["smotenc"].items()}, flush=True)
        fold["seconds"] = round(time.time() - t_fold, 1)
        out["folds"][str(k)] = fold
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")      # incremental save

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
