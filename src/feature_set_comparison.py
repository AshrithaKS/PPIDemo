"""Do ESM-2 embeddings help?  Protein-level 5-fold CV, XGBoost, unbalanced (no CTGAN yet).

Feature sets
  H   handcrafted baseline: 7-residue window (one-hot+PSSM+DSSP) + global block        (394)
  E   ESM-2 embedding of the residue only                                               (480)
  E+G ESM-2 + global block                                                              (531)
  F   fused: ESM-2 + handcrafted window + global block                                  (874)
Protocol (identical for every set): GroupKFold(5) over all 422 proteins; inside each fold 15% of the
training PROTEINS are held out for early stopping (aucpr) and for choosing the MCC-optimal threshold.
XGBoost: depth 9, min_child_weight 5, lr 0.03, colsample_bytree 0.5, <=1500 trees, early stop 100.
(lr/colsample differ from the baseline scripts so that 874-feature runs finish in reasonable time; H is re-run under
the same settings so the comparison is apples-to-apples.)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from evaluate import best_mcc_threshold, metrics_at
from features import PER_RES, global_features, window_features

root = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="binary:logistic", max_depth=9, min_child_weight=5, learning_rate=0.03,
              colsample_bytree=0.5, tree_method="hist", n_jobs=-1, eval_metric="aucpr")


def load(model="35M"):
    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    esm = np.load(root / "data" / "processed" / f"esm2_{model}.npy").astype(np.float32)
    assert len(esm) == len(res)
    loc, glo = window_features(res, 3), global_features(res)
    sets = {"H": np.hstack([loc, glo]), "E": esm, "E+G": np.hstack([esm, glo]), "F": np.hstack([esm, loc, glo])}
    groups = {  # column-group labels per feature set (for gain-importance by family)
        "F": ["ESM"] * esm.shape[1] + [f"window:{['aa','pssm','dssp'][0 if j < 20 else 1 if j < 40 else 2]}"
              for j in (np.arange(loc.shape[1]) % PER_RES)] + ["global"] * glo.shape[1],
    }
    return res, sets, groups


def run(model="35M", only=None):
    res, sets, groups = load(model)
    y, g = res["label"].to_numpy(), res["protein_id"].to_numpy()
    out = {}
    for name, X in sets.items():
        if only and name not in only:
            continue
        runs, imps = [], []
        for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(X, y, g)):
            t0 = time.time()
            tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=k).split(X[tr_i], y[tr_i], g[tr_i]))
            m = xgb.XGBClassifier(**PARAMS, n_estimators=1500, early_stopping_rounds=100, random_state=k)
            m.fit(X[tr_i][tr], y[tr_i][tr], eval_set=[(X[tr_i][va], y[tr_i][va])], verbose=False)
            pva = m.predict_proba(X[tr_i][va])[:, 1]
            pte = m.predict_proba(X[te_i])[:, 1]
            thr = best_mcc_threshold(y[tr_i][va], pva)
            runs.append({"fold": k, "best_iteration": int(m.best_iteration), "thr": thr,
                         "test_at_0.5": metrics_at(y[te_i], pte, 0.5),
                         "test_at_val_threshold": metrics_at(y[te_i], pte, thr)})
            if name in groups:
                gain = m.get_booster().get_score(importance_type="total_gain")
                tot = sum(gain.values()) or 1.0
                fam = {}
                for fk, v in gain.items():
                    fam[groups[name][int(fk[1:])]] = fam.get(groups[name][int(fk[1:])], 0) + v / tot
                imps.append(fam)
            r = runs[-1]["test_at_val_threshold"]
            print(f"[{name} fold {k}] iters {m.best_iteration} {time.time()-t0:.0f}s  mcc {r['mcc']:.3f} "
                  f"prauc {r['pr_auc']:.3f} recall {r['recall']:.3f}", flush=True)
        df = pd.DataFrame([r["test_at_val_threshold"] for r in runs])
        d5 = pd.DataFrame([r["test_at_0.5"] for r in runs])
        out[name] = {"n_features": int(X.shape[1]),
                     "val_threshold": {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in df.columns},
                     "at_0.5": {c: {"mean": float(d5[c].mean()), "std": float(d5[c].std(ddof=0))} for c in d5.columns},
                     "runs": runs}
        if imps:
            fam = pd.DataFrame(imps).fillna(0).mean().sort_values(ascending=False)
            out[name]["gain_share_by_family"] = {k: float(v) for k, v in fam.items()}
        s = out[name]["val_threshold"]
        print(f"=== {name} ({X.shape[1]} feats)  MCC {s['mcc']['mean']:.3f}±{s['mcc']['std']:.3f}  "
              f"PR-AUC {s['pr_auc']['mean']:.3f}±{s['pr_auc']['std']:.3f}  recall {s['recall']['mean']:.3f}  "
              f"precision {s['precision']['mean']:.3f}  acc {s['accuracy']['mean']:.3f}", flush=True)
    p = root / "results" / f"feature_set_comparison_{model}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "35M")
