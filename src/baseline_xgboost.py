"""Handcrafted-feature XGBoost baselines on the protein-level split.

Arms (all use the same 7-residue window of one-hot + PSSM + DSSP = 343 features):
  unbalanced        paper XGBoost settings, no imbalance handling
  weighted          same, scale_pos_weight = n_neg / n_pos   (free imbalance baseline)
Each arm is evaluated on the fixed 70-protein test split at threshold 0.5 and at the threshold
that maximises MCC on a held-out validation set of TRAIN proteins (never tuned on test).
Repeated over several seeds (different validation proteins + XGBoost seed) for mean +- std.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from evaluate import best_mcc_threshold, metrics_at, trivial_baseline
from features import window_features

root = Path(__file__).resolve().parents[1]
PAPER_PARAMS = dict(objective="binary:logistic", max_depth=9, learning_rate=0.01, min_child_weight=5)


def run(seeds, n_estimators, half, n_jobs):
    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    X = window_features(res, half)
    y = res["label"].to_numpy()
    groups = res["protein_id"].to_numpy()
    is_train = (res["split"] == "train").to_numpy()
    Xtr_all, ytr_all, gtr_all = X[is_train], y[is_train], groups[is_train]
    Xte, yte = X[~is_train], y[~is_train]
    print(f"features {X.shape[1]}  train {Xtr_all.shape[0]:,}  test {Xte.shape[0]:,}  "
          f"pos train {ytr_all.mean():.3f} test {yte.mean():.3f}")

    out = {"config": {"window_half": half, "n_features": int(X.shape[1]), "n_estimators": n_estimators,
                      "seeds": seeds, "params": PAPER_PARAMS},
           "trivial_all_negative_test": trivial_baseline(yte), "arms": {}}
    arms = {"unbalanced": False, "weighted": True}
    for name, weighted in arms.items():
        runs = []
        for seed in seeds:
            t0 = time.time()
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            tr, va = next(gss.split(Xtr_all, ytr_all, gtr_all))
            spw = (ytr_all[tr] == 0).sum() / (ytr_all[tr] == 1).sum() if weighted else 1.0
            model = xgb.XGBClassifier(**PAPER_PARAMS, n_estimators=n_estimators, scale_pos_weight=spw,
                                      tree_method="hist", n_jobs=n_jobs, random_state=seed)
            model.fit(Xtr_all[tr], ytr_all[tr])
            pva = model.predict_proba(Xtr_all[va])[:, 1]
            pte = model.predict_proba(Xte)[:, 1]
            thr = best_mcc_threshold(ytr_all[va], pva)
            runs.append({"seed": seed, "val_tuned_threshold": thr,
                         "test_at_0.5": metrics_at(yte, pte, 0.5),
                         "test_at_val_threshold": metrics_at(yte, pte, thr),
                         "val_at_val_threshold": metrics_at(ytr_all[va], pva, thr)})
            m = runs[-1]["test_at_0.5"]
            print(f"[{name} seed {seed}] {time.time()-t0:.0f}s  test@0.5 acc {m['accuracy']:.3f} "
                  f"rec {m['recall']:.3f} mcc {m['mcc']:.3f} prauc {m['pr_auc']:.3f} | thr {thr:.2f} "
                  f"mcc {runs[-1]['test_at_val_threshold']['mcc']:.3f}", flush=True)
        agg = {}
        for key in ("test_at_0.5", "test_at_val_threshold"):
            df = pd.DataFrame([r[key] for r in runs])
            agg[key] = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in df.columns}
        out["arms"][name] = {"runs": runs, "summary": agg}

    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "baseline_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # compact markdown table
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "pr_auc"]
    lines = ["| arm | threshold | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    t = out["trivial_all_negative_test"]
    lines.append("| all-negative | - | " + " | ".join(f"{t[c]:.3f}" for c in cols) + " |")
    for name, a in out["arms"].items():
        for key, label in (("test_at_0.5", "0.5"), ("test_at_val_threshold", "val-tuned")):
            s = a["summary"][key]
            lines.append(f"| {name} | {label} | " + " | ".join(f"{s[c]['mean']:.3f}±{s[c]['std']:.3f}" for c in cols) + " |")
    table = "\n".join(lines)
    (root / "results" / "baseline_results.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--half", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    run(a.seeds, a.n_estimators, a.half, a.n_jobs)
