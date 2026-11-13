"""Does a bigger ESM-2 help?  Same protocol as final_evaluation.py (protein-level split, threshold chosen on validation proteins,
3 seeds, test = the 70 held-out proteins) for feature sets F (fused) and S (sequence-only), once per ESM-2 size.
Nothing in models/ is overwritten. Output: results/esm_size_comparison.json
Usage: PPI_N_JOBS=6 python -X utf8 src/esm_size_comparison.py --sizes 35M 150M
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from evaluate import best_mcc_threshold, metrics_at
from final_evaluation import DEFAULT
from model_utils import load_sets, make_model, protein_bootstrap, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["35M", "150M"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sets", nargs="+", default=["F", "S"])
    a = ap.parse_args()
    out, probs, thrs = {}, {}, {}
    for size in a.sizes:
        res, sets, y, g, is_train = load_sets(size)
        yt, gt, yte, gte = y[is_train], g[is_train], y[~is_train], g[~is_train]
        for sname in a.sets:
            Xtr, Xte = sets[sname][is_train], sets[sname][~is_train]
            runs, pl, tl = [], [], []
            for seed in a.seeds:
                t0 = time.time()
                tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, yt, gt))
                m = make_model(DEFAULT, yt[tr], seed=seed)
                m.fit(Xtr[tr], yt[tr], eval_set=[(Xtr[va], yt[va])], verbose=False)
                thr = best_mcc_threshold(yt[va], m.predict_proba(Xtr[va])[:, 1])
                pte = m.predict_proba(Xte)[:, 1]
                r = metrics_at(yte, pte, thr)
                runs.append(r); pl.append(pte); tl.append(thr)
                print(f"[{size} {sname} seed {seed}] {time.time()-t0:4.0f}s thr {thr:.2f} mcc {r['mcc']:.3f} prauc {r['pr_auc']:.3f} "
                      f"recall {r['recall']:.3f} prec {r['precision']:.3f} acc {r['accuracy']:.3f}", flush=True)
            df = pd.DataFrame(runs)
            key = f"{sname}_{size}"
            out[key] = {"n_features": int(Xtr.shape[1]), "mean": {c: float(df[c].mean()) for c in df.columns},
                        "std": {c: float(df[c].std(ddof=0)) for c in df.columns}, "runs": runs}
            probs[key], thrs[key] = np.mean(pl, axis=0), float(np.mean(tl))
            print(f"=== {key}: MCC {df['mcc'].mean():.3f}±{df['mcc'].std(ddof=0):.3f}  PR-AUC {df['pr_auc'].mean():.3f}  "
                  f"acc {df['accuracy'].mean():.3f}", flush=True)
    names = list(probs)
    ci, diffs = protein_bootstrap(yte, probs, gte, thrs, n_boot=1000)
    out["bootstrap_95ci"], out["paired_differences"] = ci, diffs
    for n, d in diffs.items():
        print(f"  {n:22s} dMCC {d['mcc']['mean']:+.3f} CI {d['mcc']['ci'][0]:+.3f}..{d['mcc']['ci'][1]:+.3f} | "
              f"dPR-AUC {d['pr_auc']['mean']:+.3f} CI {d['pr_auc']['ci'][0]:+.3f}..{d['pr_auc']['ci'][1]:+.3f}")
    (root / "results" / "esm_size_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/esm_size_comparison.json")


if __name__ == "__main__":
    main()
