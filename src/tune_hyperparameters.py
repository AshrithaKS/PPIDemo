"""Hyper-parameter tuning of the fused model (set F) with Optuna.

Tuning uses ONLY the 352 training proteins (inner 3-fold GroupKFold by protein); the 70 held-out test proteins are never seen.
Objective = mean PR-AUC over inner folds (threshold-free, so not affected by threshold noise). Early stopping uses the inner
validation fold, which is slightly optimistic but only ranks trials; the final evaluation (final_evaluation.py) uses clean splits.
The search uses a faster learning rate (default 0.1) with median pruning to keep it affordable on CPU; the final models are
then trained at learning rate 0.05 with early stopping (final_evaluation.py). Progress is saved after every trial.
Usage: python -X utf8 src/tune_hyperparameters.py [--trials 16] [--lr 0.1] [--set F]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from model_utils import load_sets, make_model, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.1, help="faster learning rate used only during the search")
    ap.add_argument("--set", default="F")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    res, sets, y, g, is_train = load_sets()
    X = sets[a.set][is_train]
    yt, gt = y[is_train], g[is_train]
    folds = list(GroupKFold(3).split(X, yt, gt))
    print(f"tuning set {a.set}: {X.shape[1]} features, {len(yt):,} training residues, {len(np.unique(gt))} proteins, 3 inner folds", flush=True)

    def objective(trial):
        t0 = time.time()
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 9),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            spw=trial.suggest_categorical("spw", ["none", "sqrt", "full"]),
        )
        scores, iters = [], []
        for k, (tr, va) in enumerate(folds):
            m = make_model(params, yt[tr], seed=a.seed + k, n_estimators=800, early=40, lr=a.lr)
            m.fit(X[tr], yt[tr], eval_set=[(X[va], yt[va])], verbose=False)
            scores.append(average_precision_score(yt[va], m.predict_proba(X[va])[:, 1]))
            iters.append(int(m.best_iteration))
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():                       # clearly worse than earlier trials after fold k -> skip the rest
                print(f"  trial {trial.number:2d} {time.time()-t0:4.0f}s  PRUNED after fold {k+1}  PR-AUC so far {np.mean(scores):.4f}  "
                      f"depth {params['max_depth']} mcw {params['min_child_weight']} col {params['colsample_bytree']:.2f} spw {params['spw']}", flush=True)
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iterations", iters)
        print(f"  trial {trial.number:2d} {time.time()-t0:4.0f}s  PR-AUC {np.mean(scores):.4f}  "
              f"depth {params['max_depth']} mcw {params['min_child_weight']} col {params['colsample_bytree']:.2f} spw {params['spw']} iters {iters}", flush=True)
        return float(np.mean(scores))

    def save_progress(study, trial):                       # written after EVERY trial, so an interruption loses nothing
        out = root / "results"
        study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration")).to_csv(
            out / f"hyperparameter_tuning_{a.set}.csv", index=False)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if done:
            b = study.best_trial
            (out / f"best_params_{a.set}.json").write_text(json.dumps({
                "set": a.set, "best_value": b.value, "best_params": b.params, "tuning_learning_rate": a.lr,
                "reference_trial_value": study.trials[0].value, "n_trials_done": len(study.trials),
                "n_complete": len(done), "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                "best_iterations": b.user_attrs.get("best_iterations")}, indent=2), encoding="utf-8")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0))
    # reference trial: the earlier fused-model settings (depth 9, mcw 5, colsample 0.5) so tuning is judged against a known baseline
    study.enqueue_trial({"max_depth": 9, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 0.5,
                         "reg_lambda": 1.0, "reg_alpha": 0.001, "gamma": 0.0, "spw": "none"})
    study.optimize(objective, n_trials=a.trials, callbacks=[save_progress])

    print(f"\nbest PR-AUC {study.best_value:.4f} (reference trial with the earlier fused-model settings: {study.trials[0].value:.4f}; "
          f"tuning learning rate {a.lr})")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
