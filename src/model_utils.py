"""Shared modelling pieces: feature sets, XGBoost construction, protein-level bootstrap.

Feature sets (rows aligned with data/processed/residues.parquet)
  H  handcrafted: 7-residue window (one-hot + PSSM + DSSP) + global block                     394
  F  fused: ESM-2 + handcrafted window + global block   (final model)        1034 with 150M (874 with 35M)
  S  sequence-only: ESM-2 + amino-acid composition + log length + relative position     662 with 150M (502 with 35M)
     (S needs no PSSM/DSSP, so it is the only set that can score a brand-new FASTA sequence -> demo model)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, matthews_corrcoef

from features import global_features, window_features, window_mean

import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
FIXED = dict(objective="binary:logistic", tree_method="hist", n_jobs=int(os.environ.get("PPI_N_JOBS", "-1")), eval_metric="aucpr", learning_rate=0.05)
S_GLOBAL_COLS = list(range(20)) + [49, 50]          # composition (20), log length, relative position

# ---- the one place that decides the language model and the wider-context features used by the final + demo models
ESM_SIZE = "150M"           # embeddings in data/processed/esm2_<ESM_SIZE>.npy (35M: 480-d, 150M: 640-d); 150M beat 35M (esm_size_comparison)
CTX_HALVES: tuple = ()       # mean ESM-2 embedding of residues within +-h, one block per h; () = none (context_experiment: no gain)


def context_blocks(esm: np.ndarray, groups: np.ndarray, halves=None) -> list:
    return [window_mean(esm, groups, h) for h in (CTX_HALVES if halves is None else halves)]


def load_sets(esm_name: str | None = None, ctx_halves=None):
    """Feature sets (column order is fixed and mirrored by final_evaluation.names_for and inference.features_S):
    H = window + global;  F = ESM + context + window + global;  S = ESM + composition/length/position + context."""
    esm_name = esm_name or ESM_SIZE
    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
    esm = np.load(root / "data" / "processed" / f"esm2_{esm_name}.npy").astype(np.float32)
    assert len(esm) == len(res)
    g = res["protein_id"].to_numpy()
    ctx = context_blocks(esm, g, ctx_halves)
    loc, glo = window_features(res, 3), global_features(res)
    sets = {"H": np.hstack([loc, glo]), "F": np.hstack([esm, *ctx, loc, glo]), "S": np.hstack([esm, glo[:, S_GLOBAL_COLS], *ctx])}
    y = res["label"].to_numpy()
    g = res["protein_id"].to_numpy()
    is_train = (res["split"] == "train").to_numpy()
    return res, sets, y, g, is_train


def spw_value(mode: str, y: np.ndarray) -> float:
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": ratio}[mode]


def make_model(params: dict, y_train: np.ndarray, seed: int = 0, n_estimators: int = 1000, early: int = 50,
               lr: float | None = None):
    """XGBoost classifier. `lr` overrides the default learning rate (tuning uses a faster one, final models use FIXED's 0.05)."""
    p = {k: v for k, v in params.items() if k != "spw"}
    fixed = {**FIXED, **({"learning_rate": lr} if lr is not None else {})}
    return xgb.XGBClassifier(**fixed, **p, n_estimators=n_estimators, early_stopping_rounds=early,
                             scale_pos_weight=spw_value(params.get("spw", "none"), y_train), random_state=seed)


def protein_bootstrap(y, probs: dict, groups, thr: dict, n_boot: int = 1000, seed: int = 0):
    """Protein-level bootstrap (resample whole proteins) of MCC and PR-AUC.

    probs/thr: {config name: probability array / threshold}. Resamples are shared across configs, so the
    returned paired differences (first config minus each other) are valid. Returns {name: {metric: (lo, hi)}} and diffs.
    """
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == i) for i in range(len(ug))]
    names = list(probs)
    stat = {n: {"mcc": [], "pr_auc": []} for n in names}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for n in names:
            pb = probs[n][idx]
            stat[n]["pr_auc"].append(average_precision_score(yb, pb))
            pred = (pb >= thr[n]).astype(int)
            stat[n]["mcc"].append(matthews_corrcoef(yb, pred) if pred.any() else 0.0)
    ci = {n: {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in d.items()} for n, d in stat.items()}
    diffs = {}
    base = names[0]
    for n in names[1:]:
        diffs[f"{base} - {n}"] = {m: {"mean": float(np.mean(np.array(stat[base][m]) - np.array(stat[n][m]))),
                                      "ci": (float(np.percentile(np.array(stat[base][m]) - np.array(stat[n][m]), 2.5)),
                                             float(np.percentile(np.array(stat[base][m]) - np.array(stat[n][m]), 97.5))),
                                      "p_better": float(np.mean(np.array(stat[base][m]) > np.array(stat[n][m])))}
                                  for m in ("mcc", "pr_auc")}
    return ci, diffs
