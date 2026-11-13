"""Baseline diagnostics (run:  python -X utf8 src/baseline_diagnostics.py <sweep|cv|leak|dnn|all>).

sweep  Is the paper-settings XGBoost undertrained?  Early stopping on validation PR-AUC (lr 0.01).
cv     5-fold protein-level GroupKFold over all 422 proteins (fold-to-fold std, not just seeds).
leak   Same models under a RANDOM RESIDUE-LEVEL 80/20 split (what the paper describes) vs protein-level.
dnn    Feed-forward network baseline (paper: 1024-256-2, dropout 0.3, lr 1e-3).
Feature sets: local = 7-residue window (343);  local+global = local + 51 whole-chain features.
Thresholds are always chosen on held-out validation data, never on the evaluated split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from evaluate import best_mcc_threshold, metrics_at
from features import global_features, window_features

root = Path(__file__).resolve().parents[1]
RES = root / "results"
PARAMS = dict(objective="binary:logistic", max_depth=9, learning_rate=0.01, min_child_weight=5)
SEEDS = [0, 1, 2]


def load():
    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    loc = window_features(res, 3)
    glo = global_features(res)
    return dict(
        y=res["label"].to_numpy(),
        g=res["protein_id"].to_numpy(),
        train=(res["split"] == "train").to_numpy(),
        feats={"local": loc, "local+global": np.hstack([loc, glo])},
    )


def agg(runs, key):
    df = pd.DataFrame([r[key] for r in runs])
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in df.columns}


def fit_xgb(Xtr, ytr, weighted, seed, n_est, eval_set=None, early=None):
    spw = (ytr == 0).sum() / (ytr == 1).sum() if weighted else 1.0
    kw = dict(early_stopping_rounds=early, eval_metric="aucpr") if early else {}
    m = xgb.XGBClassifier(**PARAMS, n_estimators=n_est, scale_pos_weight=spw, tree_method="hist",
                          n_jobs=-1, random_state=seed, **kw)
    m.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
    return m


def _save(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sweep(D):
    out = {}
    for fs, X in D["feats"].items():
        Xtr, ytr, gtr = X[D["train"]], D["y"][D["train"]], D["g"][D["train"]]
        Xte, yte = X[~D["train"]], D["y"][~D["train"]]
        for arm, w in (("unbalanced", False), ("weighted", True)):
            runs = []
            for seed in SEEDS:
                t0 = time.time()
                tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
                m = fit_xgb(Xtr[tr], ytr[tr], w, seed, 3000, [(Xtr[va], ytr[va])], early=200)
                bi = int(m.best_iteration)
                pva, pte = m.predict_proba(Xtr[va])[:, 1], m.predict_proba(Xte)[:, 1]
                thr = best_mcc_threshold(ytr[va], pva)
                runs.append({"seed": seed, "best_iteration": bi, "thr": thr,
                             "test_at_0.5": metrics_at(yte, pte, 0.5), "test_at_val_threshold": metrics_at(yte, pte, thr)})
                print(f"[sweep {fs}/{arm}/s{seed}] best_iter {bi}  {time.time()-t0:.0f}s  "
                      f"prauc {runs[-1]['test_at_0.5']['pr_auc']:.3f} mcc@thr {runs[-1]['test_at_val_threshold']['mcc']:.3f}", flush=True)
            out[f"{fs}|{arm}"] = {"runs": runs, "median_best_iteration": int(np.median([r["best_iteration"] for r in runs])),
                                  "test_at_0.5": agg(runs, "test_at_0.5"), "test_at_val_threshold": agg(runs, "test_at_val_threshold")}
    _save("baseline_sweep.json", out)
    return out


def _iters():
    p = RES / "baseline_sweep.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    return {k: max(100, v["median_best_iteration"]) for k, v in s.items()}


def cv(D):
    its = _iters()
    out = {}
    for fs, X in D["feats"].items():
        for arm, w in (("unbalanced", False), ("weighted", True)):
            n_est = its[f"{fs}|{arm}"]
            runs = []
            for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(X, D["y"], D["g"])):
                tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=k).split(X[tr_i], D["y"][tr_i], D["g"][tr_i]))
                m = fit_xgb(X[tr_i][tr], D["y"][tr_i][tr], w, k, n_est)
                pva = m.predict_proba(X[tr_i][va])[:, 1]
                pte = m.predict_proba(X[te_i])[:, 1]
                thr = best_mcc_threshold(D["y"][tr_i][va], pva)
                runs.append({"fold": k, "n_est": n_est, "test_at_0.5": metrics_at(D["y"][te_i], pte, 0.5),
                             "test_at_val_threshold": metrics_at(D["y"][te_i], pte, thr)})
            out[f"{fs}|{arm}"] = {"test_at_0.5": agg(runs, "test_at_0.5"), "test_at_val_threshold": agg(runs, "test_at_val_threshold"), "runs": runs}
            s = out[f"{fs}|{arm}"]["test_at_val_threshold"]
            print(f"[cv {fs}/{arm}] n_est {n_est}  mcc {s['mcc']['mean']:.3f}±{s['mcc']['std']:.3f} "
                  f"prauc {s['pr_auc']['mean']:.3f}±{s['pr_auc']['std']:.3f} recall {s['recall']['mean']:.3f}", flush=True)
    _save("baseline_cv.json", out)
    return out


def leak(D):
    its = _iters()
    out = {}
    n = len(D["y"])
    for fs, X in D["feats"].items():
        for arm, w in (("unbalanced", False), ("weighted", True)):
            n_est = its[f"{fs}|{arm}"]
            runs = []
            for seed in SEEDS:
                idx = np.arange(n)
                tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=seed, stratify=D["y"])
                tr, va = train_test_split(np.arange(len(tr_i)), test_size=0.15, random_state=seed, stratify=D["y"][tr_i])
                m = fit_xgb(X[tr_i][tr], D["y"][tr_i][tr], w, seed, n_est)
                pva = m.predict_proba(X[tr_i][va])[:, 1]
                pte = m.predict_proba(X[te_i])[:, 1]
                thr = best_mcc_threshold(D["y"][tr_i][va], pva)
                runs.append({"seed": seed, "test_at_0.5": metrics_at(D["y"][te_i], pte, 0.5),
                             "test_at_val_threshold": metrics_at(D["y"][te_i], pte, thr)})
            out[f"{fs}|{arm}"] = {"test_at_0.5": agg(runs, "test_at_0.5"), "test_at_val_threshold": agg(runs, "test_at_val_threshold")}
            s = out[f"{fs}|{arm}"]["test_at_val_threshold"]
            print(f"[leak(random residue split) {fs}/{arm}] mcc {s['mcc']['mean']:.3f}±{s['mcc']['std']:.3f} "
                  f"prauc {s['pr_auc']['mean']:.3f} recall {s['recall']['mean']:.3f}", flush=True)
    _save("baseline_leakage_test.json", out)
    return out


def dnn(D):
    import torch
    from torch import nn
    torch.set_num_threads(8)
    out = {}
    for fs, X in D["feats"].items():
        Xtr_all, ytr_all, gtr_all = X[D["train"]], D["y"][D["train"]], D["g"][D["train"]]
        Xte, yte = X[~D["train"]], D["y"][~D["train"]]
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr_all, ytr_all, gtr_all))
            mu, sd = Xtr_all[tr].mean(0), Xtr_all[tr].std(0) + 1e-6
            f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
            Xt, Xv, Xs = f(Xtr_all[tr]), f(Xtr_all[va]), f(Xte)
            yt = torch.tensor(ytr_all[tr], dtype=torch.float32)
            net = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
            pos_w = torch.tensor((ytr_all[tr] == 0).sum() / (ytr_all[tr] == 1).sum(), dtype=torch.float32)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            best, best_state, bad = -1, None, 0
            for ep in range(60):
                net.train()
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 512):
                    b = perm[i:i + 512]
                    opt.zero_grad(); lossf(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(net(Xv).squeeze(1)).numpy()
                score = metrics_at(ytr_all[va], pv, 0.5)["pr_auc"]
                if score > best: best, bad, best_state = score, 0, {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8: break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xv).squeeze(1)).numpy(); pt = torch.sigmoid(net(Xs).squeeze(1)).numpy()
            thr = best_mcc_threshold(ytr_all[va], pv)
            runs.append({"seed": seed, "epochs": ep + 1, "test_at_0.5": metrics_at(yte, pt, 0.5),
                         "test_at_val_threshold": metrics_at(yte, pt, thr)})
            print(f"[dnn {fs}/s{seed}] {time.time()-t0:.0f}s ep {ep+1} prauc {runs[-1]['test_at_0.5']['pr_auc']:.3f} "
                  f"rec@0.5 {runs[-1]['test_at_0.5']['recall']:.3f} mcc@thr {runs[-1]['test_at_val_threshold']['mcc']:.3f}", flush=True)
        out[fs] = {"test_at_0.5": agg(runs, "test_at_0.5"), "test_at_val_threshold": agg(runs, "test_at_val_threshold"), "runs": runs}
    _save("baseline_dnn.json", out)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    D = load()
    steps = {"sweep": sweep, "cv": cv, "leak": leak, "dnn": dnn}
    for k in (steps if what == "all" else [what]):
        steps[k](D)
