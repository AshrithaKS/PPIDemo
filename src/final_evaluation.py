"""Final evaluation on the 70 held-out test proteins + save final / deployable models.

Configs
  H_baseline  handcrafted features (window + global), baseline parameters
  F_final     fused features (ESM-2 + handcrafted + global), parameters from best_params_F.json   <- final model
  S_demo      sequence-only features (ESM-2 + composition + length + position), same parameters    <- demo model
  F_default   only added if the tuning search found parameters different from the defaults (otherwise F_final already IS the default)
Each config is trained with several seeds (seed changes the 15% validation proteins used for early stopping and for the MCC-optimal
threshold, and the XGBoost seed). Test proteins never influence training, early stopping, thresholds or tuning.
Reports per-seed mean±std, plus a seed-ensemble with protein-level bootstrap 95% CIs and paired differences.
Set PPI_N_JOBS=6 to cap XGBoost at 6 CPU threads (cooler laptop, slower run).
Usage: python -X utf8 src/final_evaluation.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from evaluate import best_mcc_threshold, metrics_at
from features import GLOBAL_NAMES, feature_names
from model_utils import CTX_HALVES, ESM_SIZE, load_sets, make_model, protein_bootstrap, root

MODELS = {"35M": "facebook/esm2_t12_35M_UR50D", "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}  # as embed_esm.MODELS (no torch import here)

DEFAULT = dict(max_depth=9, min_child_weight=5, subsample=1.0, colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=0.001, gamma=0.0, spw="none")


def names_for(setname: str, esm_dim: int = 480, ctx_halves=()) -> list[str]:
    """Feature names in the exact column order of model_utils.load_sets."""
    esm = [f"esm_{i}" for i in range(esm_dim)]
    ctx = [f"ctx{h}_esm_{i}" for h in ctx_halves for i in range(esm_dim)]
    if setname == "H":
        return feature_names(3) + GLOBAL_NAMES
    if setname == "F":
        return esm + ctx + feature_names(3) + GLOBAL_NAMES
    return esm + GLOBAL_NAMES[:20] + ["g_log_len", "g_rel_pos"] + ctx          # S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    bp = json.loads((root / "results" / "best_params_F.json").read_text(encoding="utf-8"))
    tuned = bp["best_params"]
    improved = bool(bp["best_value"] > bp["reference_trial_value"] + 1e-9)
    same_as_default = (tuned == DEFAULT)
    print(f"tuning: {bp['n_complete']} trials, best PR-AUC {bp['best_value']:.4f} vs reference {bp['reference_trial_value']:.4f} "
          f"-> search improved on defaults: {improved}; final parameters equal defaults: {same_as_default}", flush=True)
    cfgs = {"H_baseline": ("H", DEFAULT), "F_final": ("F", tuned), "S_demo": ("S", tuned)}
    if not same_as_default:
        cfgs["F_default"] = ("F", DEFAULT)

    res, sets, y, g, is_train = load_sets()
    yt, gt, yte, gte = y[is_train], g[is_train], y[~is_train], g[~is_train]
    print(f"train {is_train.sum():,} residues / {len(np.unique(gt))} proteins; test {(~is_train).sum():,} residues / "
          f"{len(np.unique(gte))} proteins, test positive rate {yte.mean():.3f}", flush=True)
    (root / "models").mkdir(exist_ok=True)

    out = {"config": {"seeds": a.seeds, "final_params": tuned, "default_params": DEFAULT, "tuning_trials_complete": bp["n_complete"],
                      "tuning_improved_over_default": improved, "final_equals_default": same_as_default,
                      "tuning_learning_rate": bp.get("tuning_learning_rate"), "final_learning_rate": 0.05}, "configs": {}}
    probs_by_cfg, thr_by_cfg = {}, {}
    for name, (sname, params) in cfgs.items():
        Xtr, Xte = sets[sname][is_train], sets[sname][~is_train]
        runs, plist, tlist = [], [], []
        for seed in a.seeds:
            t0 = time.time()
            tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, yt, gt))
            m = make_model(params, yt[tr], seed=seed)
            m.fit(Xtr[tr], yt[tr], eval_set=[(Xtr[va], yt[va])], verbose=False)
            pva, pte = m.predict_proba(Xtr[va])[:, 1], m.predict_proba(Xte)[:, 1]
            thr = best_mcc_threshold(yt[va], pva)
            r = {"seed": seed, "best_iteration": int(m.best_iteration), "thr": thr,
                 "test_at_val_threshold": metrics_at(yte, pte, thr), "test_at_0.5": metrics_at(yte, pte, 0.5)}
            runs.append(r); plist.append(pte); tlist.append(thr)
            if seed == a.seeds[0] and name in ("F_final", "S_demo"):
                path = root / "models" / ("final_model_F.json" if name == "F_final" else "demo_model_S.json")
                m.save_model(str(path))
                esm_dim = {"35M": 480, "150M": 640, "650M": 1280}[ESM_SIZE]
                meta = {"config": name, "feature_set": sname, "n_features": int(Xtr.shape[1]),
                        "feature_names": names_for(sname, esm_dim, CTX_HALVES), "esm_model": MODELS[ESM_SIZE], "esm_size": ESM_SIZE,
                        "esm_dim": esm_dim, "context_halves": list(CTX_HALVES), "params": params, "best_iteration": int(m.best_iteration),
                        "threshold": thr, "seed": seed,
                        "trained_on": "352 DeepPPISP training proteins (85% fit / 15% early-stopping + threshold)",
                        "test_metrics_at_threshold": r["test_at_val_threshold"]}
                (root / "models" / (path.stem + "_meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            m1 = r["test_at_val_threshold"]
            print(f"[{name} seed {seed}] {time.time()-t0:4.0f}s iters {m.best_iteration:4d} thr {thr:.2f}  "
                  f"mcc {m1['mcc']:.3f} prauc {m1['pr_auc']:.3f} recall {m1['recall']:.3f} prec {m1['precision']:.3f} acc {m1['accuracy']:.3f}", flush=True)
        df = pd.DataFrame([r["test_at_val_threshold"] for r in runs])
        out["configs"][name] = {"set": sname, "n_features": int(Xtr.shape[1]), "runs": runs,
                                "mean_std": {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))} for c in df.columns}}
        probs_by_cfg[name], thr_by_cfg[name] = np.mean(plist, axis=0), float(np.mean(tlist))
        out["configs"][name]["ensemble"] = metrics_at(yte, probs_by_cfg[name], thr_by_cfg[name])
        out["configs"][name]["ensemble_threshold"] = thr_by_cfg[name]
        print(f"=== {name}: per-seed MCC {df['mcc'].mean():.3f}±{df['mcc'].std(ddof=0):.3f}  PR-AUC {df['pr_auc'].mean():.3f}±{df['pr_auc'].std(ddof=0):.3f}"
              f"  | ensemble MCC {out['configs'][name]['ensemble']['mcc']:.3f} PR-AUC {out['configs'][name]['ensemble']['pr_auc']:.3f}", flush=True)

    order = ["F_final", "H_baseline"] + (["F_default"] if "F_default" in cfgs else []) + ["S_demo"]     # F_final first -> paired differences are F_final minus others
    ci, diffs = protein_bootstrap(yte, {k: probs_by_cfg[k] for k in order}, gte, {k: thr_by_cfg[k] for k in order}, n_boot=1000)
    out["bootstrap_95ci"], out["paired_differences"] = ci, diffs
    for n, d in ci.items():
        print(f"  {n:10s} 95% CI (70 test proteins)  MCC {d['mcc'][0]:.3f}-{d['mcc'][1]:.3f}   PR-AUC {d['pr_auc'][0]:.3f}-{d['pr_auc'][1]:.3f}")
    for n, d in diffs.items():
        print(f"  {n:25s} dMCC {d['mcc']['mean']:+.3f} CI {d['mcc']['ci'][0]:+.3f}..{d['mcc']['ci'][1]:+.3f} P(better) {d['mcc']['p_better']:.2f} | "
              f"dPR-AUC {d['pr_auc']['mean']:+.3f} CI {d['pr_auc']['ci'][0]:+.3f}..{d['pr_auc']['ci'][1]:+.3f} P(better) {d['pr_auc']['p_better']:.2f}")

    te = res.loc[~is_train, ["protein_id", "position", "aa", "label"]].reset_index(drop=True)
    for k, v in probs_by_cfg.items():
        te[f"prob_{k}"] = v.astype(np.float32)
    te.to_parquet(root / "results" / "test_predictions.parquet", index=False)
    (root / "results" / "final_evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsaved results/final_evaluation.json, results/test_predictions.parquet, models/*")


if __name__ == "__main__":
    main()
