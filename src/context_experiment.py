"""Can wider sequence context raise accuracy?  Two cheap ideas, tested with the final_evaluation protocol
(protein-level split, 15% validation proteins for early stopping, threshold AND smoothing chosen on validation only,
3 seeds, test = the 70 held-out proteins). ESM-2 150M embeddings throughout.

  base   S150      ESM-2 150M (640) + composition + log length + relative position                         662
  ctx7   + mean ESM-2 embedding of residues within +-7                                                      1302
  ctx3_15 + means within +-3 and +-15                                                                       1942
Each model is also scored with "patch smoothing": p' = a*p + (1-a)*mean(p within +-h), a and h picked on the validation proteins.
Optional --fused adds the handcrafted window + PSSM/DSSP global block (feature set F) to the best context variant.
Output: results/context_experiment.json (+ .log). Nothing in models/ is touched.
Usage: PPI_N_JOBS=6 python -X utf8 src/context_experiment.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from evaluate import best_mcc_threshold, metrics_at
from features import global_features, smooth_probs, window_features, window_mean
from final_evaluation import DEFAULT
from model_utils import S_GLOBAL_COLS, make_model, protein_bootstrap, root

SMOOTH_GRID = [(0, 1.0)] + [(h, a) for h in (2, 4, 7) for a in (0.3, 0.5, 0.7)]


def build(esm_name: str = "150M"):
    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    esm = np.load(root / "data" / "processed" / f"esm2_{esm_name}.npy").astype(np.float32)
    g = res["protein_id"].to_numpy()
    glo = global_features(res)
    s = np.hstack([esm, glo[:, S_GLOBAL_COLS]])
    sets = {"base": s,
            "ctx7": np.hstack([s, window_mean(esm, g, 7)]),
            "ctx3_15": np.hstack([s, window_mean(esm, g, 3), window_mean(esm, g, 15)])}
    return res, sets, glo, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=["base", "ctx7", "ctx3_15"])
    ap.add_argument("--fused", default="", help="context variant to also run fused with the handcrafted features, e.g. ctx7")
    a = ap.parse_args()
    res, sets, glo, g = build()
    if a.fused:
        sets[f"F_{a.fused}"] = np.hstack([sets[a.fused], window_features(res, 3), glo[:, 20:49]])
        a.configs = a.configs + [f"F_{a.fused}"]
    y, is_train = res["label"].to_numpy(), (res["split"] == "train").to_numpy()
    yt, gt, yte, gte = y[is_train], g[is_train], y[~is_train], g[~is_train]
    out, probs, thrs = {}, {}, {}
    for name in a.configs:
        X = sets[name]
        Xtr, Xte = X[is_train], X[~is_train]
        rows = []
        for seed in a.seeds:
            t0 = time.time()
            tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, yt, gt))
            m = make_model(DEFAULT, yt[tr], seed=seed)
            m.fit(Xtr[tr], yt[tr], eval_set=[(Xtr[va], yt[va])], verbose=False)
            pva, pte = m.predict_proba(Xtr[va])[:, 1], m.predict_proba(Xte)[:, 1]
            # raw model
            thr = best_mcc_threshold(yt[va], pva)
            raw = metrics_at(yte, pte, thr)
            # patch smoothing, (h, a) chosen by validation MCC
            best = max(((metrics_at(yt[va], sv := smooth_probs(pva, gt[va], h, al), best_mcc_threshold(yt[va], sv))["mcc"], h, al)
                        for h, al in SMOOTH_GRID))
            _, h, al = best
            sva, ste = smooth_probs(pva, gt[va], h, al), smooth_probs(pte, gte, h, al)
            sthr = best_mcc_threshold(yt[va], sva)
            sm = metrics_at(yte, ste, sthr)
            rows.append({"seed": seed, "iters": int(m.best_iteration), "raw": raw, "smooth": sm, "smooth_h": h, "smooth_a": al})
            probs.setdefault(name, []).append(pte); probs.setdefault(name + "+smooth", []).append(ste)
            thrs.setdefault(name, []).append(thr); thrs.setdefault(name + "+smooth", []).append(sthr)
            print(f"[{name} seed {seed}] {time.time()-t0:4.0f}s  raw mcc {raw['mcc']:.3f} prauc {raw['pr_auc']:.3f} acc {raw['accuracy']:.3f} | "
                  f"smooth(h={h},a={al}) mcc {sm['mcc']:.3f} prauc {sm['pr_auc']:.3f} acc {sm['accuracy']:.3f}", flush=True)
        summ = {}
        for kind in ("raw", "smooth"):
            df = pd.DataFrame([r[kind] for r in rows])
            summ[kind] = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in df.columns}
            print(f"=== {name} {kind:6s}: MCC {df['mcc'].mean():.3f}±{df['mcc'].std(ddof=0):.3f}  PR-AUC {df['pr_auc'].mean():.3f}  "
                  f"acc {df['accuracy'].mean():.3f}  prec {df['precision'].mean():.3f}  recall {df['recall'].mean():.3f}", flush=True)
        out[name] = {"n_features": int(X.shape[1]), "runs": rows, "summary": summ}
    # paired protein-level bootstrap: every variant against the plain 150M base model
    order = ["base"] + [k for k in probs if k != "base"]
    ci, diffs = protein_bootstrap(yte, {k: np.mean(probs[k], axis=0) for k in order}, gte, {k: float(np.mean(thrs[k])) for k in order}, n_boot=1000)
    out["bootstrap_95ci"], out["paired_differences_base_minus"] = ci, diffs
    for k, d in diffs.items():
        print(f"  {k:28s} dMCC {d['mcc']['mean']:+.3f} CI {d['mcc']['ci'][0]:+.3f}..{d['mcc']['ci'][1]:+.3f} | "
              f"dPR-AUC {d['pr_auc']['mean']:+.3f} CI {d['pr_auc']['ci'][0]:+.3f}..{d['pr_auc']['ci'][1]:+.3f}   (negative = variant better)")
    (root / "results" / "context_experiment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/context_experiment.json")


if __name__ == "__main__":
    main()
